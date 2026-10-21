# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer backend for the DSV4-Flash DSpark drafter.

Extends :class:`DSparkTrainerBackend` and swaps the DFlash transformer stack for
the DeepSeek-V4-Flash sparse backbone (latent attention + sink + MoE + mHC).
Independent (standalone / offline feature-store) training only; online
co-training is intentionally out of scope.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from copy import deepcopy

import torch

from verl_speco.backends.dspark_trainer_backend import (
    DSparkTrainerBackend,
    DSparkTrainingModel,
)
from verl_speco.models.dsv4_dspark import DSV4DSparkConfig, DSV4DSparkDraftModel
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class _DSV4TrainingCfgView:
    """Read ``dspark_*`` keys through their ``dsv4_dspark_*`` spelling.

    The shared DSpark confidence-head resolver in
    :class:`DSparkTrainerBackend` looks up ``dspark_confidence_*`` keys. DSV4
    stores the same knobs under ``dsv4_dspark_confidence_*``; this view lets the
    shared resolver run unchanged while reading the DSV4 keys when present.
    """

    def __init__(self, source):
        self._source = source

    def get(self, key, default=None):
        if isinstance(key, str) and key.startswith("dspark_"):
            dsv4_value = self._source.get("dsv4_" + key, None)
            if dsv4_value is not None:
                return dsv4_value
        return self._source.get(key, default)


_SAFETENSORS_DTYPE_TO_TORCH = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
}


def _to_torch_dtype(dtype):
    """Map a safetensors dtype (string or already a torch dtype) to torch."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return _SAFETENSORS_DTYPE_TO_TORCH.get(str(dtype), dtype)


class _TensorMeta:
    """Shape/dtype-only stand-in for a tensors read from a safetensors header.

    Supports the small surface the config-normalisation code touches
    (``shape``/``ndim``/``size``/``numel``) without materialising the tensor.
    """

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


class DSV4DSparkTrainerBackend(DSparkTrainerBackend):
    """DSV4-Flash DSpark sparse-backbone drafter backend."""

    @property
    def model_type(self):
        return "dsv4_dspark"

    def _released_load_ep_context(self, drafter_config):
        """Expert-parallel context for :func:`load_released_draft`.

        ``DrafterBaseTrainer`` configures ``moe_ep`` immediately before
        ``build_model``, so the ambient ``moe_ep._EP`` context is authoritative
        and fresh; this only validates the training config against the
        checkpoint's ``dsv4_n_routed_experts`` and cross-checks the ambient
        slice size before handing out the rank/slice pair.
        """
        from verl_speco.backends.dsv4_ep_utils import (
            ensure_ep_divisible,
            is_ep_active,
            resolve_routed_expert_count,
        )
        from verl_speco.models.dsv4_dspark.backbone import moe_ep

        if not is_ep_active(self.config):
            return None, None
        n_experts = resolve_routed_expert_count(self.config, drafter_config)
        ambient = moe_ep._EP
        if ambient is None or ambient.size <= 1:
            return None, None
        experts_per_rank = ensure_ep_divisible(n_experts, ambient.size)
        if ambient.experts_per_rank != experts_per_rank:
            raise ValueError(
                f"DSV4 DSpark ambient EP context (experts_per_rank="
                f"{ambient.experts_per_rank}) disagrees with the training config "
                f"(n_routed_experts={n_experts} over {ambient.size} ranks -> "
                f"{experts_per_rank})"
            )
        return ambient.rank, ambient.experts_per_rank

    def _scan_drafter_index_shards(
        self, model_path: str
    ) -> tuple[dict[str, str], set[str], list[str]] | None:
        """Scan every ``*.index.json`` for drafter (``mtp.*``/embed/head) keys.

        Returns ``(key -> shard file, shards, index_paths)`` or ``None`` when
        the directory has no ``*.index.json`` at all (callers fall back to
        their single-file logic).
        """
        index_paths = sorted(glob.glob(os.path.join(model_path, "*.index.json")))
        if not index_paths:
            return None
        key_to_shard: dict[str, str] = {}
        for path in index_paths:
            try:
                with open(path, encoding="utf-8") as f:
                    weight_map = json.load(f).get("weight_map", {})
            except (OSError, json.JSONDecodeError):
                continue
            for key in weight_map:
                if str(key).startswith("mtp.") or key in (
                    "embed.weight",
                    "head.weight",
                ):
                    key_to_shard[key] = weight_map[key]
        return key_to_shard, set(key_to_shard.values()), index_paths

    def _load_draft_state_dict(self, model_path: str) -> dict:
        """Load only the drafter (``mtp.*``) tensors, whatever the index layout.

        Quantized DSV4 checkpoints may ship a separate main-model index, or a
        single combined index holding all ~100k target tensors plus ~7k drafter
        tensors across 75 shards (only ~11 carry drafter keys). The base loader
        rejects multiple ``*.index.json`` and otherwise reads every shard, so we
        select only the shards that contain drafter keys and merge across every
        index that carries them.
        """
        scanned = self._scan_drafter_index_shards(model_path)
        if scanned is None:
            return super()._load_draft_state_dict(model_path)
        key_to_shard, drafter_shards, index_paths = scanned

        if not key_to_shard:
            if len(index_paths) == 1:
                # Single index without drafter keys: defer to the base loader so
                # non-mtp checkpoints keep working; build_model then falls back to
                # the DFlash checkpoint path.
                return super()._load_draft_state_dict(model_path)
            raise FileNotFoundError(
                f"Multiple index.json files found in {model_path} and none "
                "contains mtp.* drafter weights"
            )

        state_dict: dict = {}
        for shard_file in sorted(drafter_shards):
            state_dict.update(
                self._load_state_file(os.path.join(model_path, shard_file))
            )
        return state_dict

    def _load_draft_state_metadata(self, model_path: str) -> dict:
        """Return ``{key: _TensorMeta}`` for the drafter keys (keys + shapes only).

        Reads the safetensors headers instead of the tensors, so a combined
        checkpoint with ~100k target tensors does not get materialised just to
        infer the drafter topology. Returns ``{}`` when the checkpoint carries no
        drafter (``mtp.*``) keys, letting the caller fall back to a full load.
        """
        from safetensors import safe_open

        scanned = self._scan_drafter_index_shards(model_path)
        if scanned is None:
            candidate = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(candidate):
                candidate = os.path.join(model_path, "quant_model_weights.safetensors")
            if not os.path.exists(candidate):
                return {}
            shards = {os.path.basename(candidate)}
            base_dir = model_path
        else:
            _, shards, _ = scanned
            if not shards:
                return {}
            base_dir = model_path

        metadata: dict = {}
        for shard in sorted(shards):
            shard_path = os.path.join(base_dir, shard)
            if not os.path.exists(shard_path):
                # Match load_released_draft: tolerate a listed-but-missing shard.
                logger.warning(
                    "DSV4 DSpark: skipping missing shard %s while reading drafter "
                    "metadata",
                    shard_path,
                )
                continue
            with safe_open(shard_path, framework="pt") as f:
                keys = list(f.keys())
                use_slice = hasattr(f, "get_slice")
                for key in keys:
                    if not (
                        key.startswith("mtp.") or key in ("embed.weight", "head.weight")
                    ):
                        continue
                    if use_slice:
                        sl = f.get_slice(key)
                        shape, dtype = sl.get_shape(), sl.get_dtype()
                    else:
                        tensor = f.get_tensor(key)
                        shape, dtype = tuple(tensor.shape), tensor.dtype
                    metadata[key] = _TensorMeta(shape, _to_torch_dtype(dtype))
        return metadata

    def _normalize_dspark_confidence_config(
        self, drafter_config, normalized_state, spec_model_path, training_cfg
    ):
        # Released drafter keys live under ``mtp.*``; the shared resolver looks
        # for the bare ``confidence_head.proj.*`` names, so surface them by
        # suffix before delegating.
        state = dict(normalized_state) if normalized_state else {}
        for key, value in list(state.items()):
            for suffix in ("confidence_head.proj.weight", "confidence_head.proj.bias"):
                if key.endswith(suffix) and suffix not in state:
                    state[suffix] = value
        # The released DSV4 DSpark confidence head carries only the weight (no
        # bias key). The shared resolver requires both to recognise the head, so
        # synthesize a zero bias here; ``load_released_draft`` then zeroes the
        # model's bias so it matches the bias-less checkpoint.
        if (
            "confidence_head.proj.weight" in state
            and "confidence_head.proj.bias" not in state
        ):
            state["confidence_head.proj.bias"] = torch.zeros(1)
        return super()._normalize_dspark_confidence_config(
            drafter_config,
            state,
            spec_model_path,
            _DSV4TrainingCfgView(training_cfg),
        )

    def _normalize_dflash_config(
        self, drafter_config, target_hf_config, normalized_state, spec_model_path
    ):
        training_cfg = self.config.rollout.drafter.training
        source_cfg = getattr(drafter_config, "_source_checkpoint_config", None) or {}
        checkpoint_provides_mask = any(
            source_cfg.get(key) is not None
            for key in (
                "mask_token_id",
                "dspark_noise_token_id",
                "dsv4_dspark_mask_token_id",
            )
        )
        mask_token_id_cfg = training_cfg.get("dsv4_dspark_mask_token_id", None)
        if mask_token_id_cfg is not None and not checkpoint_provides_mask:
            # Fill in the training default only when the checkpoint config does
            # not already carry a noise token; the checkpoint's value wins.
            drafter_config.mask_token_id = int(mask_token_id_cfg)
        target_layer_ids_cfg = training_cfg.get("dsv4_dspark_target_layer_ids", None)
        if (
            target_layer_ids_cfg is not None
            and getattr(drafter_config, "target_layer_ids", None) is None
        ):
            # Only fill in the default selection when the checkpoint config does
            # not already carry one; a checkpoint's own selection wins.
            drafter_config.target_layer_ids = [
                int(layer_id) for layer_id in target_layer_ids_cfg
            ]
        if (
            training_cfg.get("dsv4_dspark_num_target_layers", None) is not None
            and getattr(drafter_config, "num_context_layers", None) is None
        ):
            drafter_config.num_context_layers = int(
                training_cfg["dsv4_dspark_num_target_layers"]
            )
        if normalized_state:
            fc_key = next(
                (k for k in normalized_state if k.endswith("main_proj.weight")), None
            )
            if fc_key and fc_key in normalized_state:
                fc_w = normalized_state[fc_key]
                if fc_w.ndim == 2 and fc_w.size(0) > 0:
                    target_hs = int(
                        getattr(
                            getattr(target_hf_config, "text_config", target_hf_config),
                            "hidden_size",
                            getattr(
                                drafter_config,
                                "target_hidden_size",
                                drafter_config.hidden_size,
                            ),
                        )
                    )
                    inferred = fc_w.size(1) // target_hs
                    if fc_w.size(1) % target_hs != 0:
                        raise ValueError(
                            f"DSV4 DSpark {fc_key} input dim "
                            f"{int(fc_w.size(1))} is not divisible by "
                            f"target_hidden_size={target_hs}; cannot infer "
                            "num_context_layers."
                        )
                    if inferred > 0 and inferred != getattr(
                        drafter_config, "num_context_layers", 0
                    ):
                        logger.info(
                            "DSV4 DSpark: inferring num_context_layers=%d from %s shape %s",
                            inferred,
                            fc_key,
                            tuple(fc_w.shape),
                        )
                        drafter_config.num_context_layers = inferred
        return super()._normalize_dflash_config(
            drafter_config,
            target_hf_config,
            normalized_state,
            spec_model_path,
        )

    def _resolve_draft_depth(self, training_cfg) -> int:
        """Resolve the number of draft layers.

        The released checkpoint config is the *target* config (``deepseek_v4``)
        plus a ``dspark_*`` draft section; its ``num_hidden_layers`` is the
        TARGET depth (e.g. 43), so it must never be used as the draft depth. The
        draft depth always comes from the training config.
        """
        return int(training_cfg.get("dsv4_dspark_num_hidden_layers", 3))

    def _build_fallback_config(self, target_hf_config):
        training_cfg = self.config.rollout.drafter.training
        target_text_config = getattr(target_hf_config, "text_config", target_hf_config)
        hidden_size_cfg = training_cfg.get("dsv4_dspark_hidden_size", None)
        hidden_size = int(
            hidden_size_cfg
            if hidden_size_cfg is not None
            else target_text_config.hidden_size
        )
        num_context_layers = int(training_cfg.get("dsv4_dspark_num_target_layers", 3))
        target_num_hidden_layers = int(
            getattr(target_text_config, "num_hidden_layers", 43)
        )
        mask_token_id_cfg = training_cfg.get("dsv4_dspark_mask_token_id", None)
        if mask_token_id_cfg is None:
            # The released drafter uses a dedicated noise token (128799 for
            # DeepSeek-V4-Flash). Falling back to ``vocab_size - 1`` silently
            # produces the wrong noise embedding, so fail closed instead.
            raise ValueError(
                "dsv4_dspark_mask_token_id must be set (128799 for "
                "DeepSeek-V4-Flash DSpark); refusing to fall back to vocab_size-1."
            )
        mask_token_id = int(mask_token_id_cfg)
        target_layer_ids = training_cfg.get("dsv4_dspark_target_layer_ids", None)
        if target_layer_ids is None:
            from verl_speco.models.dflash import build_target_layer_ids

            target_layer_ids = build_target_layer_ids(
                num_context_layers, target_num_hidden_layers
            )
        return DSV4DSparkConfig(
            hidden_size=hidden_size,
            intermediate_size=int(
                getattr(target_text_config, "intermediate_size", hidden_size * 4)
            ),
            num_hidden_layers=int(training_cfg.get("dsv4_dspark_num_hidden_layers", 3)),
            num_attention_heads=int(target_text_config.num_attention_heads),
            num_key_value_heads=int(
                getattr(
                    target_text_config,
                    "num_key_value_heads",
                    target_text_config.num_attention_heads,
                )
            ),
            vocab_size=int(target_text_config.vocab_size),
            rms_norm_eps=float(getattr(target_text_config, "rms_norm_eps", 1e-6)),
            max_position_embeddings=int(
                getattr(target_text_config, "max_position_embeddings", 32768)
            ),
            rope_theta=float(getattr(target_text_config, "rope_theta", 10000.0)),
            num_target_layers=target_num_hidden_layers,
            num_context_layers=num_context_layers,
            target_hidden_size=int(target_text_config.hidden_size),
            target_num_hidden_layers=target_num_hidden_layers,
            target_layer_ids=target_layer_ids,
            mask_token_id=mask_token_id,
            block_size=int(training_cfg.get("dsv4_dspark_block_size", 6)),
            num_anchors=int(training_cfg.get("dsv4_dspark_num_anchors", 512)),
            markov_rank=int(training_cfg.get("dsv4_dspark_markov_rank", 256)),
            markov_head_type=str(
                training_cfg.get("dsv4_dspark_markov_head_type", "vanilla")
            ),
            confidence_head_alpha=float(
                training_cfg.get("dsv4_dspark_confidence_head_alpha", 0.0)
            ),
            confidence_head_with_markov=bool(
                training_cfg.get("dsv4_dspark_confidence_head_with_markov", True)
            ),
            ce_loss_alpha=float(training_cfg.get("dsv4_dspark_ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(training_cfg.get("dsv4_dspark_l1_loss_alpha", 0.0)),
            loss_decay_gamma=float(
                training_cfg.get("dsv4_dspark_loss_decay_gamma", 7.0)
            ),
            dsv4_num_heads=int(training_cfg.get("dsv4_dspark_backbone_num_heads", 64)),
            dsv4_head_dim=int(training_cfg.get("dsv4_dspark_backbone_head_dim", 512)),
            dsv4_rope_head_dim=int(
                training_cfg.get("dsv4_dspark_backbone_rope_head_dim", 64)
            ),
            dsv4_q_lora_rank=int(
                training_cfg.get("dsv4_dspark_backbone_q_lora_rank", 1024)
            ),
            dsv4_o_lora_rank=int(
                training_cfg.get("dsv4_dspark_backbone_o_lora_rank", 1024)
            ),
            dsv4_o_groups=int(training_cfg.get("dsv4_dspark_backbone_o_groups", 8)),
            dsv4_window_size=int(
                training_cfg.get("dsv4_dspark_backbone_window_size", 128)
            ),
            dsv4_rope_theta=float(
                training_cfg.get("dsv4_dspark_backbone_rope_theta", 10000.0)
            ),
            dsv4_rope_factor=float(
                training_cfg.get("dsv4_dspark_backbone_rope_factor", 16.0)
            ),
            dsv4_original_seq_len=int(
                training_cfg.get("dsv4_dspark_backbone_original_seq_len", 65536)
            ),
            dsv4_beta_fast=float(
                training_cfg.get("dsv4_dspark_backbone_beta_fast", 32.0)
            ),
            dsv4_beta_slow=float(
                training_cfg.get("dsv4_dspark_backbone_beta_slow", 1.0)
            ),
            dsv4_n_routed_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_routed_experts", 256)
            ),
            dsv4_n_shared_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_shared_experts", 1)
            ),
            dsv4_n_activated_experts=int(
                training_cfg.get("dsv4_dspark_backbone_n_activated_experts", 6)
            ),
            dsv4_moe_inter_dim=int(
                training_cfg.get("dsv4_dspark_backbone_moe_inter_dim", 2048)
            ),
            dsv4_score_func=str(
                training_cfg.get("dsv4_dspark_backbone_score_func", "sqrtsoftplus")
            ),
            dsv4_route_scale=float(
                training_cfg.get("dsv4_dspark_backbone_route_scale", 1.5)
            ),
            dsv4_swiglu_limit=float(
                training_cfg.get("dsv4_dspark_backbone_swiglu_limit", 10.0)
            ),
            dsv4_hc_mult=int(training_cfg.get("dsv4_dspark_backbone_hc_mult", 4)),
            dsv4_hc_sinkhorn_iters=int(
                training_cfg.get("dsv4_dspark_backbone_hc_sinkhorn_iters", 20)
            ),
            dsv4_hc_eps=float(training_cfg.get("dsv4_dspark_backbone_hc_eps", 1e-6)),
        )

    def build_model(self):
        target_model_path = self.config.model.path
        spec_model_path = self.config.rollout.drafter.model_path
        # ``os.path.join("", "config.json")`` would resolve to a CWD-relative
        # path, so only look for a checkpoint config when a path is configured.
        config_path = (
            os.path.join(spec_model_path, "config.json") if spec_model_path else ""
        )
        target_hf_config = self._get_target_hf_config()
        training_cfg = self.config.rollout.drafter.training
        normalized_state = None
        skip_loading = getattr(self, "_skip_weight_loading", False)

        if config_path and os.path.exists(config_path):
            drafter_config = DSV4DSparkConfig.from_dsv4_dspark_pretrained(
                spec_model_path
            )
            drafter_config.num_hidden_layers = self._resolve_draft_depth(training_cfg)
            source_cfg = (
                getattr(drafter_config, "_source_checkpoint_config", None) or {}
            )
            logger.info(
                "DSV4 DSpark config: draft-source checkpoint keys=%s "
                "(remaining backbone params use DSV4DSparkConfig defaults)",
                sorted(
                    key
                    for key in source_cfg
                    if key.startswith(("dsv4_", "hc_", "dspark_"))
                ),
            )
            if not skip_loading and spec_model_path and os.path.exists(spec_model_path):
                log_drafter_checkpoint_step(
                    logger,
                    spec_model_path,
                    action="Loading DSV4 DSpark drafter weights",
                )
                # Only keys + shapes are needed here (for mtp detection and
                # num_context_layers/confidence inference); the released-weight
                # loader re-reads the tensors, so avoid materialising ~40GB.
                normalized_state = self._normalize_draft_state_dict(
                    self._load_draft_state_metadata(spec_model_path)
                )
        else:
            drafter_config = self._build_fallback_config(target_hf_config)

        if not isinstance(drafter_config, DSV4DSparkConfig):
            raise TypeError(
                f"DSV4 DSpark config is not a DSV4DSparkConfig: {type(drafter_config)}"
            )
        drafter_config = self._normalize_dflash_config(
            drafter_config, target_hf_config, normalized_state, spec_model_path
        )

        used_released_weights = False
        if (
            not skip_loading
            and spec_model_path
            and os.path.exists(spec_model_path)
            and os.path.exists(config_path)
        ):
            draft_model = DSV4DSparkDraftModel(deepcopy(drafter_config))
            state_keys = set(normalized_state.keys()) if normalized_state else set()
            if any(k.startswith("mtp.") for k in state_keys):
                from verl_speco.models.dsv4_dspark.weights import load_released_draft

                n_layers = int(getattr(drafter_config, "num_hidden_layers", 3))
                target_dtype_str = str(
                    training_cfg.get("dsv4_dspark_weight_dtype", "bfloat16")
                )
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    # No int8: target_dtype is the POST-dequantization dtype, so
                    # int8 would re-quantize the dequantized values (and int8
                    # params cannot train).
                }
                target_dtype = dtype_map.get(target_dtype_str, torch.bfloat16)
                ep_rank, experts_per_rank = self._released_load_ep_context(
                    drafter_config
                )
                load_result = load_released_draft(
                    draft_model,
                    spec_model_path,
                    n_layers,
                    verbose=True,
                    target_dtype=target_dtype,
                    ep_rank=ep_rank,
                    experts_per_rank=experts_per_rank,
                )
                logger.info(
                    "DSV4 DSpark: released weight load coverage loaded=%d missing=%d "
                    "(ep_rank=%s experts_per_rank=%s)",
                    len(load_result.loaded),
                    len(load_result.missing),
                    ep_rank,
                    experts_per_rank,
                )
                used_released_weights = True
            else:
                # Non-mtp (DFlash-style) checkpoint: this path needs the actual
                # tensors, so load them here rather than relying on the metadata.
                self._load_draft_checkpoint(
                    draft_model,
                    spec_model_path,
                    normalized_state=self._normalize_draft_state_dict(
                        self._load_draft_state_dict(spec_model_path)
                    ),
                )
        else:
            draft_model = DSV4DSparkDraftModel(deepcopy(drafter_config))
        if not skip_loading and not used_released_weights:
            draft_model.load_embedding(target_model_path)
        draft_model.freeze_embedding()

        if not skip_loading:
            self.target_lm_head = self._build_target_lm_head(
                target_model_path, target_hf_config
            )
        else:
            # skip_loading (random init) still needs a REAL target head whenever
            # the loss reads target logits (target-argmax CE / distribution TV).
            # An empty placeholder would make those losses produce zero tokens.
            needs_real_head = (
                str(training_cfg.get("ce_target_source", "labels"))
                == "target_argmax"
                or float(training_cfg.get("dsv4_dspark_tv_loss_alpha", 0) or 0) > 0
                or float(training_cfg.get("dsv4_dspark_l1_loss_alpha", 0) or 0) > 0
            )
            if needs_real_head:
                self.target_lm_head = self._build_target_lm_head(
                    target_model_path, target_hf_config
                )
            else:
                from verl_speco.models.target.target_head import TargetHead

                target_text_config = getattr(
                    target_hf_config, "text_config", target_hf_config
                )
                hidden_size = int(
                    getattr(
                        target_text_config, "hidden_size", drafter_config.hidden_size
                    )
                )
                vocab_size = int(
                    getattr(
                        target_text_config, "vocab_size", drafter_config.vocab_size
                    )
                )
                lm_head_weight = torch.empty(
                    vocab_size, hidden_size, dtype=torch.bfloat16
                )
                self.target_lm_head = TargetHead(lm_head_weight)

        return DSparkTrainingModel(
            draft_model=draft_model,
            block_size=int(
                training_cfg.get(
                    "dsv4_dspark_block_size",
                    getattr(drafter_config, "block_size", 6),
                )
            ),
            num_anchors=int(
                training_cfg.get(
                    "dsv4_dspark_num_anchors",
                    getattr(drafter_config, "num_anchors", 512),
                )
            ),
            loss_decay_gamma=float(
                training_cfg.get(
                    "dsv4_dspark_loss_decay_gamma",
                    getattr(drafter_config, "loss_decay_gamma", 7.0),
                )
            ),
            loss_mode=str(training_cfg.get("dsv4_dspark_loss_mode", "full_vocab")),
            sampled_ce_negatives=int(
                training_cfg.get("dsv4_dspark_sampled_ce_negatives", 0)
            ),
            ce_loss_alpha=float(
                training_cfg.get(
                    "dsv4_dspark_ce_loss_alpha",
                    getattr(drafter_config, "ce_loss_alpha", 0.1),
                )
            ),
            l1_loss_alpha=(
                float(
                    training_cfg.get(
                        "dsv4_dspark_l1_loss_alpha",
                        0.0,
                    )
                )
                # The DSpark distribution term is the L1 distance between draft
                # and target probabilities, which is 2 * total variation. The
                # DSV4 release names the coefficient in TV units, so fold it in
                # as 0.5 * tv_loss_alpha so ``l1 * alpha`` reproduces the TV loss.
                + 0.5 * float(training_cfg.get("dsv4_dspark_tv_loss_alpha", 0.0))
            ),
            confidence_head_alpha=float(
                training_cfg.get("dsv4_dspark_confidence_loss_alpha", 0.0)
            ),
            l1_chunk_size=int(training_cfg.get("dsv4_dspark_l1_chunk_size", 0)),
            distribution_loss_impl=str(
                training_cfg.get("dsv4_dspark_distribution_loss_impl", "auto")
            ),
            # Without an explicit ``ce_target_source`` the training model defaults
            # to dataset labels, silently ignoring a configured target_argmax CE
            # even though the target head and aux-plus-last layout are set up.
            ce_target_source=str(training_cfg.get("ce_target_source", "labels")),
            confidence_target_mode=str(
                training_cfg.get(
                    "dsv4_dspark_confidence_target_mode",
                    training_cfg.get("dspark_confidence_target_mode", "greedy"),
                )
            ),
            debug_log=bool(training_cfg.get("dsv4_dspark_debug_log", False)),
            debug_log_first_n=int(training_cfg.get("dsv4_dspark_debug_log_first_n", 2)),
            debug_log_interval=int(
                training_cfg.get("dsv4_dspark_debug_log_interval", 100)
            ),
        ), drafter_config

    def _preprocess_label(self) -> str:
        return "DSV4 DSpark"

    def _preprocess_window_key(self) -> str:
        return "dsv4_dspark_max_window"

    def _preprocess_default_num_target_layers(self) -> int:
        return 3

    def _preprocess_requires_target_hidden(self, training_cfg) -> bool:
        # The l1/tv/confidence losses and target-argmax CE read the final target
        # hidden state, so aux-only feature items cannot serve them.
        return (
            float(training_cfg.get("dsv4_dspark_l1_loss_alpha", 0.0) or 0.0) > 0
            or float(training_cfg.get("dsv4_dspark_tv_loss_alpha", 0.0) or 0.0) > 0
            or float(training_cfg.get("dsv4_dspark_confidence_loss_alpha", 0.0) or 0.0)
            > 0
            or str(training_cfg.get("ce_target_source", "labels") or "labels").lower()
            == "target_argmax"
        )
