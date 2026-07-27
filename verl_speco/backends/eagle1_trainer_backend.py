"""EAGLE-1 / EAGLE-2 drafter training backend.

Logic follows NeMo AutoModel's EAGLE-1/2 training path (``core_v12.py`` /
``draft_llama_v12.py``): a single ``fc`` fuses the token embedding with the
target's last-layer hidden state, one standard decoder layer predicts the
next-step hidden state, and the loss combines a SmoothL1 feature-regression term
with a full-vocabulary soft cross-entropy distillation term against the frozen
target head. EAGLE-1 and EAGLE-2 share this training path; they differ only in
the inference-time speculative tree policy (EAGLE-2 uses a dynamic/context-aware
tree), which lives in the rollout engine, not here.

The backend reuses the EAGLE-3 online data-collection plumbing by reporting
``model_type == "eagle3"`` (so ``base_trainer`` assembles the shifted inputs and
``last_hidden_states`` target). Only ``build_model`` and ``compute_loss`` differ;
optimizer/scheduler/preprocess are inherited from ``Eagle3TrainerBackend``.
"""

import logging
import os
from copy import deepcopy

import torch

from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend, _masked_soft_cross_entropy
from verl_speco.models.eagle1 import Eagle1Config, LlamaForCausalLMEagle1
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()

# Config keys that must not be forwarded verbatim from the target config into the
# draft config (they are set explicitly below).
_TARGET_CONFIG_DROP_KEYS = (
    "architectures",
    "model_type",
    "auto_map",
    "_name_or_path",
    "torch_dtype",
    "tie_word_embeddings",
)


class Eagle1TrainerBackend(Eagle3TrainerBackend):
    """Drafter trainer backend for EAGLE-1 and EAGLE-2."""

    @property
    def model_type(self):
        # Reuse the EAGLE-3 shifted-input / last_hidden_states data plumbing.
        return "eagle3"

    # EAGLE-1/2 trains on full local sequences and does not implement the
    # Ulysses sequence-parallel loss path yet; base_trainer honours this flag and
    # forces ulysses_sequence_parallel_size = 1 instead of aborting mid-run.
    supports_ulysses_sp = False

    def _build_draft_config(self, spec_model_path, target_hf_config):
        config_path = os.path.join(spec_model_path, "config.json") if spec_model_path else None
        if config_path and os.path.exists(config_path):
            return Eagle1Config.from_pretrained(spec_model_path)

        training_cfg = self.config.rollout.drafter.training
        draft_layers = int(training_cfg.get("eagle1_num_hidden_layers", 1))
        cfg_dict = deepcopy(target_hf_config).to_dict()
        for key in _TARGET_CONFIG_DROP_KEYS:
            cfg_dict.pop(key, None)
        draft_config = Eagle1Config(
            draft_num_hidden_layers=draft_layers,
            target_hidden_size=int(target_hf_config.hidden_size),
            num_aux_hidden_states=1,
            **cfg_dict,
        )
        # The draft is a shallow model; advertise its real depth (not the target's)
        # so the exported config.json matches the single-layer checkpoint.
        draft_config.num_hidden_layers = draft_layers
        draft_config.torch_dtype = torch.bfloat16
        draft_config.tie_word_embeddings = False
        draft_config.architectures = ["LlamaForCausalLMEagle1"]
        return draft_config

    def build_model(self):
        """Build the EAGLE-1/2 dense draft model and the frozen target head."""
        logger.debug(
            "Initializing EAGLE-1/2 draft with target model_type: %s",
            getattr(self.target_model_config, "model_type", None),
        )
        if bool(self.config.rollout.drafter.training.get("use_logits", False)):
            raise ValueError(
                "EAGLE-1/2 only supports hidden-state distillation against the frozen target head; "
                "set actor_rollout_ref.rollout.drafter.training.use_logits=False"
            )
        spec_model_path = self.config.rollout.drafter.model_path
        target_hf_config = self._get_target_hf_config()

        draft_config = self._build_draft_config(spec_model_path, target_hf_config)
        if not hasattr(draft_config, "target_hidden_size"):
            draft_config.target_hidden_size = int(target_hf_config.hidden_size)
        self.vocab_size = draft_config.vocab_size

        if spec_model_path and os.path.exists(os.path.join(spec_model_path, "config.json")):
            log_drafter_checkpoint_step(logger, spec_model_path, action="Loading EAGLE-1/2 drafter weights")
            drafter_module = LlamaForCausalLMEagle1.from_pretrained(spec_model_path, config=draft_config)
        else:
            drafter_module = LlamaForCausalLMEagle1(draft_config)

        # Seed and freeze the draft token embeddings from the target model.
        target_model_path = self.config.model.path
        drafter_module.load_embedding(target_model_path)
        drafter_module.freeze_embedding()

        # EAGLE-1/2 always distills against the frozen target head (weight tying):
        # the draft carries no lm_head of its own, so token logits are produced by
        # the target head applied to the predicted hidden states.
        target_device = (
            torch.device(f"{device_name}:{get_device_id()}") if device_name != "cpu" else torch.device("cpu")
        )
        self.target_model = self._build_target_model(target_model_path, target_hf_config).to(target_device).eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)

        return drafter_module, draft_config

    def compute_loss(self, model, batch, _current_pad_size):
        """SmoothL1 feature regression + soft-CE token distillation (EAGLE-1/2)."""
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError("EAGLE-1/2 drafter training does not support Ulysses sequence parallel yet")

        input_ids = batch["input_ids"]
        hidden_states = batch["hidden_states"]
        last_hidden_states = batch.get("last_hidden_states", None)
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        position_ids = batch["position_ids"]
        if last_hidden_states is None:
            raise ValueError("EAGLE-1/2 requires last_hidden_states; use_logits must be False")

        training_cfg = self.config.rollout.drafter.training
        hidden_loss_weight = float(training_cfg.get("eagle1_hidden_loss_weight", 1.0))
        token_loss_weight = float(training_cfg.get("eagle1_token_loss_weight", 0.1))
        feature_noise = float(training_cfg.get("eagle1_feature_noise", 0.0))

        draft_model = model.module if hasattr(model, "module") else model

        # EAGLE feature-noise augmentation on the draft input feature only
        # (U(-fn, fn); train mode only). The SmoothL1 target stays clean.
        fc_input = hidden_states
        if draft_model.training and feature_noise > 0:
            fc_input = fc_input + (torch.rand_like(fc_input) - 0.5) * (2.0 * feature_noise)

        predicted_hidden = model(
            input_ids=input_ids,
            hidden_states=fc_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Weight-tied logits from the frozen target head. Gradients flow into the
        # draft through the frozen linear; the head itself is not updated.
        predicted_logits = self.target_model(predicted_hidden).float()
        with torch.no_grad():
            target_logits = self.target_model(last_hidden_states).float()
            target_probs = torch.softmax(target_logits, dim=-1)

        # Full-vocabulary soft cross-entropy distillation. Reuse the EAGLE-3
        # helper, which sanitizes non-finite logits/targets BEFORE log_softmax so
        # masked positions cannot leak 0*NaN gradients into the draft.
        per_token_ploss, valid_position = _masked_soft_cross_entropy(predicted_logits, target_probs, loss_mask)
        finite_hidden = torch.isfinite(predicted_hidden).all(dim=-1) & torch.isfinite(last_hidden_states).all(dim=-1)
        valid_mask = valid_position & finite_hidden
        num_tokens = valid_mask.float().sum()

        # SmoothL1 hidden regression against the (shifted) target last hidden.
        # Sanitize before SmoothL1 for the same NaN-gradient reason.
        safe_pred_hidden = torch.where(
            torch.isfinite(predicted_hidden), predicted_hidden, torch.zeros_like(predicted_hidden)
        ).float()
        safe_target_hidden = torch.where(
            torch.isfinite(last_hidden_states), last_hidden_states, torch.zeros_like(last_hidden_states)
        ).float()
        hidden_per_token = self.criterion(safe_pred_hidden, safe_target_hidden).mean(dim=-1)
        hidden_per_token = torch.where(valid_mask, hidden_per_token, torch.zeros_like(hidden_per_token))
        total_local_vloss = hidden_per_token.sum()

        token_per_token = torch.where(valid_mask, per_token_ploss, torch.zeros_like(per_token_ploss))
        total_local_ploss = token_per_token.sum()

        # Gate on DEBUG so the .item() host-device syncs are skipped at INFO+.
        if logger.isEnabledFor(logging.DEBUG) and num_tokens.detach().item() > 0:
            with torch.no_grad():
                draft_top1 = predicted_logits.argmax(dim=-1)
                target_top1 = target_probs.argmax(dim=-1)
                acc = ((draft_top1 == target_top1) & valid_mask).float().sum() / num_tokens.clamp_min(1)
            logger.debug(
                "[eagle1 loss] tokens=%s hidden_loss=%.6f token_loss=%.6f top1_acc=%.6f",
                int(num_tokens.detach().cpu().item()),
                float((total_local_vloss / num_tokens.clamp_min(1)).detach().cpu().item()),
                float((total_local_ploss / num_tokens.clamp_min(1)).detach().cpu().item()),
                float(acc.detach().cpu().item()),
            )

        return {
            "total_local_vloss": total_local_vloss,
            "total_local_ploss": total_local_ploss,
            "local_num_tokens": num_tokens,
            "v_weight": hidden_loss_weight,
            "p_weight": token_loss_weight,
        }
