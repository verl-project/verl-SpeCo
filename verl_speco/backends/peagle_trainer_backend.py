# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P-EAGLE (parallel-drafting EAGLE) drafter training backend.

Logic follows NeMo AutoModel's P-EAGLE trainer (``peagle_trainer.py``): the draft
predicts all ``num_depths`` tokens in a single parallel forward over a flat,
COD-subsampled sequence, supervised by a count-normalized ``KL(target || draft)``
over the draft vocabulary. There is no EAGLE-3 test-time-training recurrence and
no per-depth loss decay.

Integration: P-EAGLE reuses the EAGLE-3 aux + last-hidden collection with the
reference target-wrapper shift (AutoModel ``target.py`` ``_shift_left_with_zero``):
row ``p`` pairs the unshifted aux feature ``f[p]`` with the NEXT token ``x[p+1]``,
supervised against the distribution of ``x[p+2]`` from ``last_hidden[p+1]`` and
gated by ``loss_mask[p+1]``. ``base_trainer`` applies that shift during batch
assembly (ids/last_hidden/loss_mask by +1, aux unshifted), so this trainer stays a
verbatim port of the reference ``_peagle_position_loss``. The frozen target head
turns ``last_hidden_states`` into the full-vocab target logits, which are then
restricted to the draft vocab. Only ``build_model`` and ``compute_loss`` differ
from the EAGLE-3 backend; preprocess/optimizer/target-head are inherited.
"""

import logging
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig
from verl_speco.models.peagle.cod_sampling import generate_cod_sample_indices
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step
from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import get_device_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

_TARGET_CONFIG_DROP_KEYS = (
    "architectures",
    "model_type",
    "auto_map",
    "_name_or_path",
    "torch_dtype",
    "tie_word_embeddings",
)


def _kl_div_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Per-position KL(target || draft) over the draft vocab. Shapes [*, V] -> [*]."""
    log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(
        log_p, target_p, reduction="none", log_target=False
    ).sum(dim=-1)


class PEagleTrainingModel(nn.Module):
    """Training wrapper around ``LlamaForCausalLMPeagle``.

    The whole training step (COD sampling, the flat multi-depth forward and the
    count-normalized KL) runs inside ``forward`` because FSDP2 unshards a wrapped
    module's parameters in its pre-forward hook: driving the draft through its
    submodules from the backend leaves the parameters as sharded DTensors and
    fails with ``got mixed torch.Tensor and DTensor`` on every rank > 1. This
    mirrors ``DFlashTrainingModel``, which is why the DFlash family already
    trains under FSDP.

    Training-only behavior (sampling, masking, loss) stays here rather than in
    the model package, again mirroring the DFlash wrapper.
    """

    _no_split_modules = ["PeagleFusedLayer", "PeagleVanillaLayer"]

    def __init__(
        self,
        draft_model: LlamaForCausalLMPeagle,
        num_depths: int = 8,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
        sequence_partitions: int = 1,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.config = draft_model.config
        self.num_depths = int(num_depths)
        self.down_sample_ratio = float(down_sample_ratio)
        self.down_sample_ratio_min = float(down_sample_ratio_min)
        if sequence_partitions < 1:
            raise ValueError("sequence_partitions must be positive")
        self.sequence_partitions = sequence_partitions

    def forward(
        self,
        input_ids: torch.Tensor,
        aux_hidden: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        target_logits: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ):
        """Return ``(loss_sum, loss_tokens, correct)`` over the sampled positions.

        ``target_logits`` comes from the frozen target head and is computed by the
        backend outside this module, the way ``DFlashTrainingModel`` receives
        ``lm_head_weight``, so the target head never becomes an FSDP parameter.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if seq_lengths is not None:
            # seq_lengths is a flat list of document lengths for ONE packed
            # sequence, which is what base_trainer builds for P-EAGLE. There is no
            # per-row structure to index, so a multi-row batch would silently
            # apply one row's document layout to all of them.
            if batch_size > 1:
                raise ValueError(
                    "P-EAGLE seq_lengths describe a single packed sequence, but the batch "
                    f"has {batch_size} rows; pack the documents into one row or drop seq_lengths"
                )
            # Any tail past sum(seq_lengths) gets document id -1 in the COD mask,
            # which makes those queries attend to nothing at all rather than
            # failing, so check the invariant instead of drafting on garbage.
            total_length = int(seq_lengths.sum())
            if total_length != seq_len:
                raise ValueError(
                    f"P-EAGLE seq_lengths sum to {total_length} but the packed sequence is "
                    f"{seq_len} tokens long"
                )
        loss_num = torch.zeros((), device=device, dtype=torch.float32)
        loss_den = torch.zeros((), device=device, dtype=torch.float32)
        correct = torch.zeros((), device=device, dtype=torch.float32)

        for b in range(batch_size):
            row_loss_mask = loss_mask[b : b + 1].long()
            anchor_pos, depth = generate_cod_sample_indices(
                seq_length=seq_len,
                loss_mask=row_loss_mask,
                num_depths=self.num_depths,
                down_sample_ratio=self.down_sample_ratio,
                down_sample_ratio_min=self.down_sample_ratio_min,
            )
            orig_positions = anchor_pos + depth
            if seq_lengths is not None:
                document_lengths = seq_lengths.to(device)
            else:
                document_lengths = (
                    attention_mask[b].sum().clamp_min(1).reshape(1).to(device)
                )
            loss_positions = row_loss_mask[0, orig_positions].bool()

            if self.sequence_partitions == 1:
                num, den, hits = self._position_loss(
                    input_ids[b],
                    aux_hidden[b : b + 1],
                    target_logits[b],
                    anchor_pos,
                    depth,
                    loss_positions,
                    document_lengths,
                )
                loss_num = loss_num + num
                loss_den = loss_den + den
                correct = correct + hits
                continue

            # Algorithm 1: descendants inherit their depth-1 ancestor's segment.
            owners = (anchor_pos + (depth > 0)) * self.sequence_partitions // seq_len
            owners = owners.clamp_max(self.sequence_partitions - 1)
            for segment in range(self.sequence_partitions):
                owned = owners == segment
                indices = torch.where(owned | ((depth == 0) & (owners <= segment)))[0]
                if indices.numel() == 0:
                    continue
                args = (
                    input_ids[b],
                    aux_hidden[b : b + 1],
                    target_logits[b],
                    anchor_pos[indices],
                    depth[indices],
                    loss_positions[indices] & owned[indices],
                    document_lengths,
                )
                if self.training:
                    # Retain only inputs/scalars until backward, then recompute
                    # one segment at a time through the existing wrapped forward.
                    num, den, hits = checkpoint(
                        self._position_loss, *args, use_reentrant=False
                    )
                else:
                    num, den, hits = self._position_loss(*args)
                loss_num = loss_num + num
                loss_den = loss_den + den
                correct = correct + hits

        return loss_num, loss_den, correct

    def _position_loss(
        self,
        input_ids: torch.Tensor,
        aux_hidden: torch.Tensor,
        target_logits: torch.Tensor,
        anchor_pos: torch.Tensor,
        depth: torch.Tensor,
        loss_positions: torch.Tensor,
        document_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        draft = self.draft_model
        orig_positions = anchor_pos + depth
        is_depth0 = depth == 0
        flat_ids = torch.where(
            is_depth0,
            input_ids[orig_positions],
            torch.full_like(orig_positions, draft.config.mask_token_id),
        ).unsqueeze(0)
        real_proj = draft.project_hidden_states(aux_hidden[:, orig_positions])[0]
        flat_hidden = torch.where(
            is_depth0.unsqueeze(-1),
            real_proj,
            draft.masked_projected_hidden().expand(orig_positions.shape[0], -1),
        ).unsqueeze(0)
        block_mask = draft.build_peagle_block_mask(
            anchor_pos=anchor_pos,
            depth=depth,
            lengths=document_lengths,
            total_seq_len=input_ids.shape[0],
        )
        hidden = draft.forward_peagle(
            sampled_input_ids=flat_ids,
            sampled_projected_hidden=flat_hidden,
            position_ids=orig_positions.unsqueeze(0),
            block_mask=block_mask,
        )
        # Context-only positions participate in attention, but need no vocab head.
        logits = draft.compute_logits(hidden[:, loss_positions])[0]
        targets = target_logits[orig_positions[loss_positions]].index_select(
            -1, draft.selected_token_ids()
        )
        elementwise = _kl_div_loss(logits, targets)
        correct = (logits.argmax(-1) == targets.argmax(-1)).float().sum()
        return elementwise.sum(), loss_positions.float().sum(), correct


class PEagleTrainerBackend(Eagle3TrainerBackend):
    """Drafter trainer backend for P-EAGLE (parallel drafting)."""

    @property
    def model_type(self):
        return "peagle"

    # P-EAGLE trains on full local sequences and does not implement the SP loss.
    supports_ulysses_sp = False

    # Unlike EAGLE-3, P-EAGLE fine-tunes the draft embedding instead of freezing
    # the target-seeded copy (speculators sets embed_requires_grad=True), so hot
    # publish has to carry it along with the draft's own lm_head.
    trains_draft_embeddings = True

    def _training_cfg(self):
        return self.config.rollout.drafter.training

    def _build_draft_config(self, spec_model_path, target_hf_config):
        config_path = (
            os.path.join(spec_model_path, "config.json") if spec_model_path else None
        )
        if config_path and os.path.exists(config_path):
            return PeagleConfig.from_pretrained(spec_model_path)

        training_cfg = self._training_cfg()
        cfg_dict = target_hf_config.to_dict()
        for key in _TARGET_CONFIG_DROP_KEYS:
            cfg_dict.pop(key, None)
        draft_vocab_size = training_cfg.get("peagle_draft_vocab_size", None)
        draft_config = PeagleConfig(
            num_draft_layers=int(training_cfg.get("peagle_num_draft_layers", 4)),
            target_hidden_size=int(target_hf_config.hidden_size),
            num_aux_hidden_states=int(
                training_cfg.get("peagle_num_aux_hidden_states", 3)
            ),
            draft_vocab_size=int(draft_vocab_size)
            if draft_vocab_size is not None
            else int(target_hf_config.vocab_size),
            num_depths=int(training_cfg.get("peagle_num_depths", 8)),
            down_sample_ratio=float(training_cfg.get("peagle_down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(
                training_cfg.get("peagle_down_sample_ratio_min", 0.2)
            ),
            mask_token_id=training_cfg.get("peagle_mask_token_id", None),
            fc_norm=bool(training_cfg.get("peagle_fc_norm", False)),
            parallel_drafting=True,
            **cfg_dict,
        )
        draft_config.num_hidden_layers = int(
            training_cfg.get("peagle_num_draft_layers", 4)
        )
        draft_config.torch_dtype = torch.bfloat16
        draft_config.tie_word_embeddings = False
        draft_config.architectures = ["LlamaForCausalLMPeagle"]
        return draft_config

    def build_model(self):
        if bool(self._training_cfg().get("use_logits", False)):
            raise ValueError(
                "P-EAGLE distills against the frozen target head; set "
                "actor_rollout_ref.rollout.drafter.training.use_logits=False"
            )
        spec_model_path = self.config.rollout.drafter.model_path
        target_hf_config = self._get_target_hf_config()
        draft_config = self._build_draft_config(spec_model_path, target_hf_config)
        self.vocab_size = draft_config.vocab_size

        checkpoint_has_vocab_mapping = False
        if spec_model_path and os.path.exists(
            os.path.join(spec_model_path, "config.json")
        ):
            log_drafter_checkpoint_step(
                logger, spec_model_path, action="Loading P-EAGLE drafter weights"
            )
            loaded = LlamaForCausalLMPeagle.from_pretrained(
                spec_model_path,
                config=draft_config,
                output_loading_info=True,
            )
            if isinstance(loaded, tuple):
                drafter_module, loading_info = loaded
                missing_keys = set(loading_info.get("missing_keys", []))
                checkpoint_has_vocab_mapping = not {"t2d", "d2t"}.intersection(
                    missing_keys
                )
            else:
                drafter_module = loaded
                checkpoint_has_vocab_mapping = self._has_valid_vocab_mapping(
                    drafter_module
                )
        else:
            drafter_module = LlamaForCausalLMPeagle(draft_config)

        # A reduced draft vocabulary is only meaningful with a real t2d/d2t pair
        # derived from token frequency. The model's constructor falls back to
        # "the first draft_vocab_size target ids", which is an arbitrary slice of
        # any real tokenizer, so refuse it exactly like the EAGLE-3 backend does
        # instead of silently training a draft that can never emit the rest.
        if drafter_module.draft_vocab_size != drafter_module.vocab_size:
            if checkpoint_has_vocab_mapping and self._has_valid_vocab_mapping(
                drafter_module
            ):
                logger.debug("Using P-EAGLE vocab mapping loaded from draft checkpoint")
            else:
                raise ValueError(
                    "PEAGLE draft_vocab_size differs from target vocab_size, but the draft "
                    "checkpoint does not provide valid t2d/d2t vocab mapping buffers"
                )
        self._validate_vocab_mapping(drafter_module)

        # P-EAGLE trains the draft embeddings (speculators sets embed_requires_grad=True),
        # so seed them from the target but do NOT freeze.
        drafter_module.load_embedding(self.config.model.path)

        target_device = (
            torch.device(f"{device_name}:{get_device_id()}")
            if device_name != "cpu"
            else torch.device("cpu")
        )
        self.target_model = (
            self._build_target_model(self.config.model.path, target_hf_config)
            .to(target_device)
            .eval()
        )
        for param in self.target_model.parameters():
            param.requires_grad_(False)

        training_cfg = self._training_cfg()
        training_model = PEagleTrainingModel(
            draft_model=drafter_module,
            sequence_partitions=int(training_cfg.get("peagle_sequence_partitions", 1)),
            num_depths=int(
                training_cfg.get(
                    "peagle_num_depths", getattr(draft_config, "num_depths", 8)
                )
            ),
            down_sample_ratio=float(
                training_cfg.get(
                    "peagle_down_sample_ratio",
                    getattr(draft_config, "down_sample_ratio", 0.7),
                )
            ),
            down_sample_ratio_min=float(
                training_cfg.get(
                    "peagle_down_sample_ratio_min",
                    getattr(draft_config, "down_sample_ratio_min", 0.2),
                )
            ),
        )
        return training_model, draft_config

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError(
                "P-EAGLE drafter training does not support Ulysses sequence parallel yet"
            )
        last_hidden_states = batch.get("last_hidden_states", None)
        if last_hidden_states is None:
            raise ValueError(
                "P-EAGLE requires last_hidden_states; use_logits must be False"
            )

        device = batch["input_ids"].device
        with torch.no_grad():
            target_logits = self.target_model(
                last_hidden_states
            ).float()  # [B, S, vocab]

        loss_num, loss_den, correct = model(
            input_ids=batch["input_ids"],
            aux_hidden=batch["hidden_states"],
            loss_mask=batch["loss_mask"],
            attention_mask=batch["attention_mask"],
            target_logits=target_logits,
            # Per-document chunk lengths for COD document isolation. base_trainer
            # concatenates every document into one flat batch-1 sequence, so the
            # all-ones attention_mask no longer marks document boundaries; fall
            # back to a single document only when the lengths are unavailable.
            seq_lengths=batch.get("seq_lengths", None),
        )

        accuracy = (correct / loss_den.clamp_min(1.0)).detach()
        return {
            "total_local_vloss": torch.tensor(0.0, device=device),
            "total_local_ploss": loss_num,
            "local_num_tokens": loss_den,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "accuracy": accuracy,
        }
