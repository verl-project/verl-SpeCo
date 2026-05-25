import logging
import glob
import json
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig

from .model.dflash import DFlashConfig, DFlashDraftModel, build_target_layer_ids
from .model.dflash.flex_attention import compile_friendly_create_block_mask
from .model.target.target_head import TargetHead
from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from verl.utils.device import get_device_name
from verl.utils.fsdp_utils import get_device_id


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()


def _create_dflash_mask_mod(anchor_positions: torch.Tensor, block_keep_mask: torch.Tensor, ctx_len: int, block_size: int):
    """Create DFlash block attention mask.

    A query block can attend to context tokens before its anchor and draft
    tokens inside the same block. Different sampled blocks are isolated.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]
        is_context = kv_idx < ctx_len
        mask_context = is_context & (kv_idx < anchor_pos)
        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        return (mask_context | mask_draft) & block_keep_mask[b, q_block_id]

    dflash_mask_mod.__name__ = f"dflash_mask_A{anchor_positions.shape[1]}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashTrainingModel(nn.Module):
    """Training wrapper around DFlashDraftModel.

    This class is deliberately kept in the backend module rather than the model
    package because it contains training-only behavior: anchor sampling, block
    mask construction, label gathering, loss weighting, and metrics.
    """

    _no_split_modules = ["DFlashDecoderLayer"]

    def __init__(self, draft_model: DFlashDraftModel, block_size: int = 16, num_anchors: int = 512, loss_decay_gamma: float = 7.0):
        super().__init__()
        self.draft_model = draft_model
        self.config = draft_model.config
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma

    def _sample_anchor_positions(self, seq_len: int, loss_mask: torch.Tensor, device: torch.device):
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - self.block_size, 0)
        if max_anchor == 0:
            anchors = torch.zeros(bsz, self.num_anchors, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, self.num_anchors, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, seq_len + 1)
        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, 2.0)
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        take_n = min(self.num_anchors, gathered.shape[1])
        selected = gathered[:, :take_n].sort(dim=1).values
        if take_n < self.num_anchors:
            selected = torch.cat(
                [selected, torch.zeros(bsz, self.num_anchors - take_n, dtype=torch.long, device=device)],
                dim=1,
            )
        keep_mask = torch.arange(self.num_anchors, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=self.num_anchors)
        return torch.where(keep_mask, selected, 0), keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor, seq_len: int):
        bsz = anchor_positions.shape[0]
        device = anchor_positions.device
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        return context_position_ids, draft_position_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids: torch.Tensor, anchor_positions: torch.Tensor, block_keep_mask: torch.Tensor):
        bsz, seq_len = input_ids.shape
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device
        noise_ids = torch.full(
            (bsz, n_blocks * self.block_size),
            self.draft_model.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = (torch.arange(n_blocks, device=device) * self.block_size).unsqueeze(0).expand(bsz, -1)
        anchor_tokens = torch.gather(input_ids, 1, anchor_positions.clamp(0, seq_len - 1))
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n_blocks)
        noise_ids[batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )
        return self.draft_model.embed_tokens(noise_ids)

    def forward(self, input_ids: torch.Tensor, hidden_states_list: list[torch.Tensor], loss_mask: torch.Tensor, lm_head_weight: torch.Tensor):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        anchor_positions, block_keep_mask = self._sample_anchor_positions(seq_len, loss_mask, device)
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids, draft_position_ids = self._create_position_ids(anchor_positions, seq_len)
        draft_len = n_blocks * self.block_size

        block_mask = None
        if device.type == "cuda":
            block_mask = compile_friendly_create_block_mask(
                mask_mod=_create_dflash_mask_mod(anchor_positions, block_keep_mask, seq_len, self.block_size),
                B=bsz,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=seq_len + draft_len,
                device=device,
            )

        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )
        logits = F.linear(draft_hidden, lm_head_weight)

        label_offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices)

        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices)
        weight_mask = weight_mask * original_loss_mask
        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)
        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum().clamp(min=1e-6)
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)
            loss_per_position = (loss_per_token.view(bsz, n_blocks, self.block_size) * binary_weights).sum(dim=(0, 1)) / count_per_pos
            acc_per_position = correct.view(bsz, n_blocks, self.block_size).float().sum(dim=(0, 1)) / count_per_pos

        return loss, accuracy, loss_per_position, acc_per_position, count_per_position


class DFlashTrainerBackend:
    def __init__(self, config, target_model_config):
        self.config = config
        self.target_model_config = target_model_config
        self.target_lm_head = None

    @property
    def model_type(self):
        return "dflash"

    def setup_optimizer(self, drafter_model, drafter_train_config):
        trainable_params = [p for p in drafter_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=drafter_train_config.lr,
            betas=(0.9, 0.95),
            weight_decay=drafter_train_config.get("weight_decay", 1e-2),
        )

    def setup_scheduler(self, optimizer, train_cfg):
        total_steps = train_cfg.get("step", 0)
        num_warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))
        warmup_style = train_cfg.get("warmup_style", "constant")

        if warmup_style == "constant":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        if warmup_style == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=train_cfg.get("min_lr_ratio", 0.0),
                num_cycles=train_cfg.get("num_cycles", 0.5),
            )
        raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

    def _get_target_hf_config(self):
        target_hf_config = getattr(self.target_model_config, "hf_config", None)
        if target_hf_config is not None:
            return target_hf_config
        if hasattr(self.target_model_config, "hidden_size") and hasattr(self.target_model_config, "vocab_size"):
            return self.target_model_config
        config_path = (
            getattr(self.target_model_config, "local_hf_config_path", None)
            or getattr(self.target_model_config, "hf_config_path", None)
            or getattr(self.target_model_config, "path", None)
        )
        if config_path is None:
            raise ValueError("Cannot resolve target HF config for DFlash drafter")
        return AutoConfig.from_pretrained(
            config_path,
            trust_remote_code=bool(getattr(self.target_model_config, "trust_remote_code", False)),
        )

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = training_cfg.get("dflash_hidden_size", None)
        hidden_size = int(hidden_size_cfg if hidden_size_cfg is not None else target_text_config.hidden_size)
        num_context_layers = int(training_cfg.get("dflash_num_target_layers", 5))
        target_num_hidden_layers = int(getattr(target_text_config, "num_hidden_layers", 36))
        mask_token_id_cfg = training_cfg.get("dflash_mask_token_id", None)
        mask_token_id = int(mask_token_id_cfg if mask_token_id_cfg is not None else target_text_config.vocab_size - 1)
        target_layer_ids = training_cfg.get("dflash_target_layer_ids", None)
        if target_layer_ids is None:
            target_layer_ids = build_target_layer_ids(num_context_layers, target_num_hidden_layers)
        return DFlashConfig(
            hidden_size=hidden_size,
            intermediate_size=int(getattr(target_text_config, "intermediate_size", hidden_size * 4)),
            num_hidden_layers=int(training_cfg.get("dflash_num_hidden_layers", 1)),
            num_attention_heads=int(getattr(target_text_config, "num_attention_heads")),
            num_key_value_heads=int(getattr(target_text_config, "num_key_value_heads", getattr(target_text_config, "num_attention_heads"))),
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(getattr(target_text_config, "max_position_embeddings", 32768)),
            rope_theta=float(getattr(target_text_config, "rope_theta", 10000.0)),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            architectures=["DFlashDraftModel"],
        )

    def _load_state_file(self, path: str) -> dict:
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                return {key: f.get_tensor(key) for key in f.keys()}
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_draft_state_dict(self, model_path: str) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}
        index_paths = glob.glob(os.path.join(model_path, "*.index.json"))
        if index_paths:
            if len(index_paths) > 1:
                raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
            with open(index_paths[0], "r", encoding="utf-8") as f:
                index_json = json.load(f)
            for shard_file in sorted(set(index_json.get("weight_map", {}).values())):
                state_dict.update(self._load_state_file(os.path.join(model_path, shard_file)))
        else:
            safetensors_path = os.path.join(model_path, "model.safetensors")
            pytorch_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                state_dict = self._load_state_file(safetensors_path)
            elif os.path.exists(pytorch_path):
                state_dict = self._load_state_file(pytorch_path)
            else:
                raise FileNotFoundError(f"No model index, model.safetensors or pytorch_model.bin found in {model_path}")
        return state_dict

    def _normalize_draft_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key_remap = {
            "fc.weight": "context_proj.weight",
            "hidden_norm.weight": "context_norm.weight",
            "norm.weight": "final_norm.weight",
        }
        normalized_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("_orig_mod.draft_model.", "module.draft_model.", "draft_model.", "module."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    break
            normalized_key = key_remap.get(normalized_key, normalized_key)
            normalized_state[normalized_key] = value
        return normalized_state

    def _infer_num_context_layers_from_state(self, normalized_state: dict[str, torch.Tensor], target_hidden_size: int) -> int | None:
        context_proj = normalized_state.get("context_proj.weight")
        if context_proj is None:
            return None
        if context_proj.ndim != 2:
            raise ValueError(f"DFlash context_proj.weight must be rank-2, got shape {tuple(context_proj.shape)}")
        input_dim = int(context_proj.shape[1])
        if input_dim % int(target_hidden_size) != 0:
            raise ValueError(
                f"DFlash context_proj.weight input dim {input_dim} is not divisible by "
                f"target_hidden_size={target_hidden_size}"
            )
        return input_dim // int(target_hidden_size)

    def _normalize_dflash_config(
        self,
        drafter_config: DFlashConfig,
        target_hf_config,
        normalized_state: dict[str, torch.Tensor] | None,
        spec_model_path: str,
    ) -> DFlashConfig:
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        target_hidden_size = int(
            getattr(target_text_config, "hidden_size", None)
            or getattr(drafter_config, "target_hidden_size")
        )
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", None)
            or getattr(drafter_config, "target_num_hidden_layers", 36)
        )

        nested_dflash_config = getattr(drafter_config, "dflash_config", None)
        nested_target_layer_ids = None
        if isinstance(nested_dflash_config, dict):
            nested_target_layer_ids = nested_dflash_config.get("target_layer_ids")

        target_layer_ids = getattr(drafter_config, "target_layer_ids", None)
        if target_layer_ids is None and nested_target_layer_ids is not None:
            target_layer_ids = nested_target_layer_ids
        if target_layer_ids is not None:
            target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]

        state_num_context_layers = None
        if normalized_state is not None:
            state_num_context_layers = self._infer_num_context_layers_from_state(normalized_state, target_hidden_size)

        ids_num_context_layers = len(target_layer_ids) if target_layer_ids is not None else None
        configured_num_target_layers = int(getattr(drafter_config, "num_target_layers", target_num_hidden_layers))
        configured_num_context_layers = getattr(drafter_config, "num_context_layers", None)
        if configured_num_context_layers is not None:
            configured_num_context_layers = int(configured_num_context_layers)

        if state_num_context_layers is not None and ids_num_context_layers is not None and state_num_context_layers != ids_num_context_layers:
            raise ValueError(
                f"DFlash checkpoint/config mismatch in {spec_model_path}: context_proj.weight implies "
                f"{state_num_context_layers} context layers, but target_layer_ids has {ids_num_context_layers} entries"
            )

        num_context_layers = state_num_context_layers or ids_num_context_layers or configured_num_context_layers
        if num_context_layers is None:
            if configured_num_target_layers == target_num_hidden_layers:
                num_context_layers = int(self.config.rollout.drafter.training.get("dflash_num_target_layers", 5))
            else:
                # Backward compatibility for older local configs that used
                # num_target_layers as the concatenated hidden-state count.
                num_context_layers = configured_num_target_layers
        if target_layer_ids is None:
            target_layer_ids = build_target_layer_ids(int(num_context_layers), target_num_hidden_layers)
        if len(target_layer_ids) != int(num_context_layers):
            raise ValueError(
                f"DFlash expected {num_context_layers} target layer ids, got {len(target_layer_ids)} "
                f"in {spec_model_path}"
            )

        if configured_num_target_layers != target_num_hidden_layers or configured_num_context_layers != int(num_context_layers):
            logger.warning(
                "Normalizing DFlash training config: num_target_layers=%s->%s "
                "num_context_layers=%s->%s (state_context_layers=%s target_layer_ids=%s model_path=%s)",
                configured_num_target_layers,
                target_num_hidden_layers,
                configured_num_context_layers,
                num_context_layers,
                state_num_context_layers,
                target_layer_ids,
                spec_model_path,
            )

        drafter_config.num_target_layers = target_num_hidden_layers
        drafter_config.num_context_layers = int(num_context_layers)
        drafter_config.target_hidden_size = target_hidden_size
        drafter_config.target_num_hidden_layers = target_num_hidden_layers
        drafter_config.target_layer_ids = target_layer_ids
        return drafter_config

    def _load_draft_checkpoint(
        self,
        draft_model: DFlashDraftModel,
        model_path: str,
        normalized_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if normalized_state is None:
            normalized_state = self._normalize_draft_state_dict(self._load_draft_state_dict(model_path))

        model_state = draft_model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        unexpected = []
        mismatched = []
        for key, value in normalized_state.items():
            if key not in model_state:
                unexpected.append(key)
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                if key == "context_proj.weight":
                    raise ValueError(
                        "DFlash context_proj.weight shape mismatch after config normalization: "
                        f"checkpoint={tuple(value.shape)} model={tuple(model_state[key].shape)}"
                    )
                continue
            filtered_state[key] = value

        missing, _ = draft_model.load_state_dict(filtered_state, strict=False)
        if unexpected or missing or mismatched:
            logger.warning(
                "DFlash draft checkpoint load report from %s: loaded=%s missing=%s unexpected=%s mismatched=%s",
                model_path,
                len(filtered_state),
                list(missing),
                unexpected,
                mismatched,
            )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")
        target_hf_config = self._get_target_hf_config()
        normalized_state = None

        if config_path and os.path.exists(config_path):
            drafter_config = DFlashConfig.from_dflash_pretrained(spec_model_path)
            if spec_model_path and os.path.exists(spec_model_path):
                normalized_state = self._normalize_draft_state_dict(self._load_draft_state_dict(spec_model_path))
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DFlashConfig):
            raise TypeError(f"DFlash config is not a DFlashConfig: {type(drafter_config)}")
        drafter_config = self._normalize_dflash_config(drafter_config, target_hf_config, normalized_state, spec_model_path)

        if spec_model_path and os.path.exists(spec_model_path) and os.path.exists(config_path):
            draft_model = DFlashDraftModel(deepcopy(drafter_config))
            self._load_draft_checkpoint(draft_model, spec_model_path, normalized_state=normalized_state)
        else:
            draft_model = DFlashDraftModel(deepcopy(drafter_config))
        draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        self.target_lm_head = self._build_target_lm_head(target_model_path)
        training_cfg = self.config.rollout.drafter.training
        return DFlashTrainingModel(
            draft_model=draft_model,
            block_size=int(training_cfg.get("dflash_block_size", 16)),
            num_anchors=int(training_cfg.get("dflash_num_anchors", 512)),
            loss_decay_gamma=float(training_cfg.get("dflash_loss_decay_gamma", 7.0)),
        ), drafter_config

    def _build_target_lm_head(self, target_model_path: str):
        target_device = torch.device(f"{device_name}:{get_device_id()}") if device_name != "cpu" else torch.device("cpu")
        target_lm_head = TargetHead.from_pretrained(model_path=target_model_path).to(target_device).eval()
        for param in target_lm_head.parameters():
            param.requires_grad_(False)
        return target_lm_head

    def preprocess_individual_items(self, items, device, model_config):
        res = {"ids": [], "h_states": [], "masks": []}
        max_window = int(self.config.rollout.drafter.training.get("dflash_max_window", 512))
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)
        h_dim = int(getattr(model_config, "target_hidden_size", model_config.hidden_size))
        num_context_layers = int(getattr(model_config, "num_context_layers", getattr(model_config, "num_target_layers", 5)))
        expected_hidden_dim = h_dim * num_context_layers

        for item in items:
            ids = item["input_ids"].to(device, non_blocking=True)
            raw_h = item["hidden_states"]
            full_h = torch.cat(raw_h, dim=-1) if isinstance(raw_h, (list, tuple)) else raw_h
            full_h = full_h.to(device, dtype=torch.bfloat16)
            if full_h.size(-1) < expected_hidden_dim:
                raise ValueError(
                    f"DFlash expected at least {expected_hidden_dim} hidden dims "
                    f"({num_context_layers} context layers of size {h_dim}), got {full_h.size(-1)}"
                )
            
            if item.get("loss_mask") is not None:
                item_loss_mask = item["loss_mask"].to(device, dtype=torch.float32, non_blocking=True)
            elif "prompts" in item and "responses" in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)    
                prompt_len = item["prompts"].size(0)
                responses = item["responses"]
                item_loss_mask[prompt_len : prompt_len + responses.size(0)] = (responses != pad_id).float()[: max(0, ids.size(0) - prompt_len)]
               
            else:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32) 
                item_loss_mask[:] = 1.0
            valid_len = min(ids.size(0), full_h.size(0), item_loss_mask.size(0))
            ids = ids[:valid_len]
            full_h = full_h[:valid_len]
            item_loss_mask = item_loss_mask[:valid_len]
            nonzero = torch.nonzero(item_loss_mask)
            if nonzero.numel() > 0:
                r_start = nonzero[0, 0]
                start = torch.clamp(r_start - (max_window // 2), min=0, max=max(0, ids.size(0) - max_window)).item()
                end = min(start + max_window, ids.size(0))
            else:
                start, end = max(0, ids.size(0) - max_window), ids.size(0)

            res["ids"].append(ids[start:end])
            res["h_states"].append(full_h[start:end, :expected_hidden_dim])
            res["masks"].append(item_loss_mask[start:end])
        return res

    def compute_loss(self, model, batch, _current_pad_size):
        if getattr(self, "use_ulysses_sp", False):
            raise NotImplementedError("DFlash drafter training does not support Ulysses sequence parallel yet")
        if self.target_lm_head is None:
            raise ValueError("DFlash target_lm_head is not initialized")

        draft_model = model.module if hasattr(model, "module") else model
        hidden_states = batch["hidden_states"]
        num_context_layers = draft_model.draft_model.num_context_layers
        per_layer_dim = hidden_states.shape[-1] // num_context_layers
        hidden_states_list = list(hidden_states.split(per_layer_dim, dim=-1))

        loss, accuracy, loss_pp, acc_pp, count_pp = model(
            input_ids=batch["input_ids"],
            hidden_states_list=hidden_states_list,
            loss_mask=batch["loss_mask"],
            lm_head_weight=self.target_lm_head.fc.weight,
        )
        local_num_tokens = count_pp.sum().to(loss.device, dtype=loss.dtype)
        return {
            "total_local_vloss": torch.tensor(0.0, device=batch["input_ids"].device),
            "total_local_ploss": loss * local_num_tokens,
            "local_num_tokens": local_num_tokens,
            "v_weight": 0.0,
            "p_weight": 1.0,
            "accuracy": accuracy.detach(),
            "loss_per_position": loss_pp.detach(),
            "acc_per_position": acc_pp.detach(),
            "count_per_position": count_pp.detach(),
        }
