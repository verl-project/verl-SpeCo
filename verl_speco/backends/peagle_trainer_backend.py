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

from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig
from verl_speco.models.peagle.cod_sampling import generate_cod_sample_indices
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step
from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import get_device_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

_TARGET_CONFIG_DROP_KEYS = ("architectures", "model_type", "auto_map", "_name_or_path", "torch_dtype", "tie_word_embeddings")


def _kl_div_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Per-position KL(target || draft) over the draft vocab. Shapes [*, V] -> [*]."""
    log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(log_p, target_p, reduction="none", log_target=False).sum(dim=-1)


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
    ):
        super().__init__()
        self.draft_model = draft_model
        self.config = draft_model.config
        self.num_depths = int(num_depths)
        self.down_sample_ratio = float(down_sample_ratio)
        self.down_sample_ratio_min = float(down_sample_ratio_min)

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
        draft = self.draft_model
        mask_token_id = int(getattr(draft.config, "mask_token_id", draft.vocab_size - 1))
        selected_token_ids = draft.selected_token_ids().to(input_ids.device)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
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
                row_length = seq_lengths.to(device)
            else:
                row_length = attention_mask[b].sum().clamp_min(1).reshape(1).to(device)
            loss_positions = row_loss_mask[0, orig_positions].bool()

            is_depth0 = depth == 0
            mask_hidden_proj = draft.masked_projected_hidden()  # [1, H]
            flat_ids = torch.where(
                is_depth0, input_ids[b][orig_positions], torch.full_like(orig_positions, mask_token_id)
            ).unsqueeze(0)
            real_proj = draft.project_hidden_states(aux_hidden[b : b + 1][:, orig_positions])[0]  # [n, H]
            flat_hidden = torch.where(
                is_depth0.unsqueeze(-1), real_proj, mask_hidden_proj.expand(orig_positions.shape[0], -1)
            ).unsqueeze(0)

            block_mask = draft.build_peagle_block_mask(
                anchor_pos=anchor_pos, depth=depth, lengths=row_length, total_seq_len=seq_len
            )
            hidden = draft.forward_peagle(
                sampled_input_ids=flat_ids,
                sampled_projected_hidden=flat_hidden,
                position_ids=orig_positions.unsqueeze(0),
                block_mask=block_mask,
            )
            logits = draft.compute_logits(hidden)[0]  # [n, draft_vocab]
            draft_target_logits = target_logits[b][orig_positions].index_select(dim=-1, index=selected_token_ids)

            elementwise = _kl_div_loss(logits, draft_target_logits)
            mask_f = loss_positions.to(elementwise.dtype)
            loss_num = loss_num + (elementwise * mask_f).sum()
            loss_den = loss_den + mask_f.sum()
            with torch.no_grad():
                correct = correct + (
                    (logits.argmax(dim=-1) == draft_target_logits.argmax(dim=-1)) & loss_positions
                ).float().sum()

        return loss_num, loss_den, correct


class PEagleTrainerBackend(Eagle3TrainerBackend):
    """Drafter trainer backend for P-EAGLE (parallel drafting)."""

    @property
    def model_type(self):
        return "peagle"

    # P-EAGLE trains on full local sequences and does not implement the SP loss.
    supports_ulysses_sp = False

    def _training_cfg(self):
        return self.config.rollout.drafter.training

    def _build_draft_config(self, spec_model_path, target_hf_config):
        config_path = os.path.join(spec_model_path, "config.json") if spec_model_path else None
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
            num_aux_hidden_states=int(training_cfg.get("peagle_num_aux_hidden_states", 3)),
            draft_vocab_size=int(draft_vocab_size) if draft_vocab_size is not None else int(target_hf_config.vocab_size),
            num_depths=int(training_cfg.get("peagle_num_depths", 8)),
            down_sample_ratio=float(training_cfg.get("peagle_down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(training_cfg.get("peagle_down_sample_ratio_min", 0.2)),
            mask_token_id=training_cfg.get("peagle_mask_token_id", None),
            fc_norm=bool(training_cfg.get("peagle_fc_norm", False)),
            parallel_drafting=True,
            **cfg_dict,
        )
        draft_config.num_hidden_layers = int(training_cfg.get("peagle_num_draft_layers", 4))
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

        if spec_model_path and os.path.exists(os.path.join(spec_model_path, "config.json")):
            log_drafter_checkpoint_step(logger, spec_model_path, action="Loading P-EAGLE drafter weights")
            drafter_module = LlamaForCausalLMPeagle.from_pretrained(spec_model_path, config=draft_config)
        else:
            drafter_module = LlamaForCausalLMPeagle(draft_config)

        # P-EAGLE trains the draft embeddings (speculators sets embed_requires_grad=True),
        # so seed them from the target but do NOT freeze.
        drafter_module.load_embedding(self.config.model.path)

        target_device = torch.device(f"{device_name}:{get_device_id()}") if device_name != "cpu" else torch.device("cpu")
        self.target_model = self._build_target_model(self.config.model.path, target_hf_config).to(target_device).eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)

        training_cfg = self._training_cfg()
        training_model = PEagleTrainingModel(
            draft_model=drafter_module,
            num_depths=int(training_cfg.get("peagle_num_depths", getattr(draft_config, "num_depths", 8))),
            down_sample_ratio=float(
                training_cfg.get("peagle_down_sample_ratio", getattr(draft_config, "down_sample_ratio", 0.7))
            ),
            down_sample_ratio_min=float(
                training_cfg.get("peagle_down_sample_ratio_min", getattr(draft_config, "down_sample_ratio_min", 0.2))
            ),
        )
        return training_model, draft_config

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError("P-EAGLE drafter training does not support Ulysses sequence parallel yet")
        last_hidden_states = batch.get("last_hidden_states", None)
        if last_hidden_states is None:
            raise ValueError("P-EAGLE requires last_hidden_states; use_logits must be False")

        device = batch["input_ids"].device
        with torch.no_grad():
            target_logits = self.target_model(last_hidden_states).float()  # [B, S, vocab]

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
