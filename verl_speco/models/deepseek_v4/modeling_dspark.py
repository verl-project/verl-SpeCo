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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

import math
import os
import re

# Adapted from training_common/modeling/ds4_dspark.py.
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from verl_speco.ops.deepseek_v4_loss import (
    all_reduce_gradient,
    differentiable_all_reduce_sum,
    differentiable_vocab_parallel_log_probs,
    scale_gradient,
)

_RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from verl_speco.models.deepseek_v4.runtime import DeepseekV4ServingTrainingWrapper
    from verl_speco.ops import deepseek_v4_runtime as _ds4_runtime
    from verl_speco.ops.deepseek_v4_runtime import (
        _install_fused_eh_projection_collective,
        _install_tp_sharded_attention_projections,
        _install_tp_sharded_eh_projections,
        _profile_vllm_ascend_backward_phase,
        _register_dsa_backward_host_metadata,
        consume_vllm_ascend_backward_profile,
        set_vllm_ascend_backward_profiling,
    )
except ModuleNotFoundError as error:
    # Keep the pure tensor-contract helpers importable in lightweight CPU
    # environments that do not carry DeepSeek V4 Transformers/vLLM packages.
    _RUNTIME_IMPORT_ERROR = error
    _ds4_runtime = None
    DeepseekV4ServingTrainingWrapper = nn.Module
    _install_fused_eh_projection_collective = None
    _install_tp_sharded_attention_projections = None
    _install_tp_sharded_eh_projections = None
    _profile_vllm_ascend_backward_phase = None
    _register_dsa_backward_host_metadata = None
    consume_vllm_ascend_backward_profile = None
    set_vllm_ascend_backward_profiling = None


def _install_dspark_dsa_device_operator_training_aliases() -> bool:
    """Bind DSpark DeviceOperator dispatch to the shared DSA replay."""

    if _ds4_runtime is None:
        return False
    record = _ds4_runtime._increment_training_diagnostic
    record("dspark_device_alias_install_attempts")
    scatter = _ds4_runtime._DSA_TRAINING_SCATTER_WRAPPER
    sparse_attention = _ds4_runtime._DSA_TRAINING_SPARSE_ATTENTION_WRAPPER
    if scatter is None or sparse_attention is None:
        record("dspark_device_alias_missing_wrappers")
        return False
    from verl_speco.backends.vllm_ascend_compat import kv_plan_type
    from verl_speco.ops.ascend_dsa_compat import patch_kv_plan

    plan = kv_plan_type()
    if plan is not None:
        return patch_kv_plan(plan, scatter, sparse_attention)
    try:
        from vllm_ascend.device.device_op import DeviceOperator
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        record("dspark_device_alias_import_failures")
        return False
    if getattr(DeviceOperator, "_speco_deepseek_v4_dspark_dsa_training_patch", False):
        return True

    def dspark_scatter(cache, x, slot_mapping):
        record("dspark_device_alias_scatter_calls")
        return scatter(cache, slot_mapping, x)

    def dspark_sparse_attention_op():
        record("dspark_device_alias_sparse_attention_selector_calls")
        return sparse_attention

    DeviceOperator.dsa_kv_compress_scatter = staticmethod(dspark_scatter)
    DeviceOperator.get_dsa_sparse_attn_op = staticmethod(dspark_sparse_attention_op)
    DeviceOperator._speco_deepseek_v4_dspark_dsa_training_patch = True
    record("dspark_device_alias_install_successes")
    return True


def _training_safe_rope_output(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_mul,
) -> torch.Tensor:
    """Replay the vLLM-Ascend DSpark RoPE without an in-place write."""

    original_shape = x.shape
    working_x = x
    if working_x.dim() == 2:
        working_x = working_x.unsqueeze(-2)
    if working_x.dim() == 3:
        working_x = working_x.unsqueeze(1)
    return rotary_mul(
        working_x,
        cos,
        sin,
        rotary_mode="interleave",
    ).reshape(original_shape)


class _VllmForwardTorchBackwardRoPE(torch.autograd.Function):
    """Use the exact vLLM RoPE value with an out-of-place Torch backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        module,
        original_forward,
        rotary_mul,
    ) -> torch.Tensor:
        if cos.requires_grad or sin.requires_grad:
            raise RuntimeError(
                "DSpark hybrid RoPE expects precomputed cos/sin without gradients"
            )
        ctx.save_for_backward(cos, sin)
        ctx.rotary_mul = rotary_mul

        # A custom autograd Function.forward runs with grad tracking disabled.
        # The serving implementation writes its result back into x, so give it
        # an independent tensor rather than the split/view retained by backward.
        inference_x = x.detach().clone()
        output = original_forward(
            module,
            inference_x,
            cos.detach(),
            sin.detach(),
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "vLLM-Ascend ComplexExpRotaryEmbedding.forward must return a Tensor"
            )
        if output.shape != x.shape:
            raise RuntimeError(
                "vLLM-Ascend RoPE changed the input shape: "
                f"input={x.shape}, output={output.shape}"
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            # RoPE is linear in x, so its input value does not affect dY/dX.
            # Reusing grad_output's shape/layout avoids retaining every forward
            # activation solely for this surrogate backward.
            replay_x = grad_output.detach().requires_grad_(True)
            replay_output = _training_safe_rope_output(
                replay_x,
                cos,
                sin,
                ctx.rotary_mul,
            )
            (grad_x,) = torch.autograd.grad(
                replay_output,
                replay_x,
                grad_outputs=grad_output,
                create_graph=create_graph,
                retain_graph=False,
            )
        return grad_x, None, None, None, None, None


def _install_dspark_training_safe_rope() -> bool:
    """Install exact vLLM forward values with a safe Torch backward."""

    import torch_npu
    from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

    original_forward = ComplexExpRotaryEmbedding.forward
    if getattr(original_forward, "_speco_deepseek_v4_training_safe_rope", False):
        return True

    @wraps(original_forward)
    def training_safe_rope(self, x, cos, sin):
        if not torch.is_grad_enabled() or not x.requires_grad:
            return original_forward(self, x, cos, sin)
        return _VllmForwardTorchBackwardRoPE.apply(
            x,
            cos,
            sin,
            self,
            original_forward,
            torch_npu.npu_rotary_mul,
        )

    training_safe_rope._speco_deepseek_v4_training_safe_rope = True
    ComplexExpRotaryEmbedding.forward = training_safe_rope
    return True


def _unwrap_projection_output(output: Any, *, name: str) -> torch.Tensor:
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError(f"{name} returned an empty tuple")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"{name} must return a Tensor, got {type(output).__name__}")
    return output


def _replay_dspark_dsa_query(
    attention_impl: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_mul,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the differentiable DSpark DSA query projection path."""

    q_a = _unwrap_projection_output(
        attention_impl.wq_a(hidden_states),
        name="DSpark DSA wq_a",
    )
    qr = _unwrap_projection_output(
        attention_impl.q_norm(q_a),
        name="DSpark DSA q_norm",
    )
    q = _unwrap_projection_output(
        attention_impl.wq_b(qr),
        name="DSpark DSA wq_b",
    ).unflatten(
        -1,
        (
            int(attention_impl.n_local_heads),
            int(attention_impl.head_dim),
        ),
    )

    # Match DeviceOperator.apply_dsa_q_rms without depending on an
    # inference-oriented Triton/custom-op autograd registration.
    q_dtype = q.dtype
    q_float = q.float()
    q = (
        q_float
        * torch.rsqrt(
            q_float.square().mean(dim=-1, keepdim=True) + float(attention_impl.eps)
        )
    ).to(q_dtype)

    nope_head_dim = int(attention_impl.nope_head_dim)
    if not 0 < nope_head_dim < q.shape[-1]:
        raise RuntimeError(
            "Invalid DSpark DSA partial-RoPE split: "
            f"nope_head_dim={nope_head_dim}, head_dim={q.shape[-1]}"
        )
    q_nope, q_rope = q.split(
        (nope_head_dim, q.shape[-1] - nope_head_dim),
        dim=-1,
    )
    q_rope = _training_safe_rope_output(
        q_rope,
        cos,
        sin,
        rotary_mul,
    )
    return torch.cat((q_nope, q_rope), dim=-1), qr


class _NativeForwardReplayBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        native_value: torch.Tensor,
        replay_value: torch.Tensor,
    ) -> torch.Tensor:
        return native_value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


class _ProfiledMarkovLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, bias):
        ctx.has_bias = bias is not None
        tensors = (features, weight) if bias is None else (features, weight, bias)
        ctx.save_for_backward(*tensors)
        return F.linear(features, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        profile = (
            _profile_vllm_ascend_backward_phase("backward_markov_head")
            if _profile_vllm_ascend_backward_phase is not None
            else nullcontext()
        )
        with profile:
            saved = ctx.saved_tensors
            features, weight = saved[:2]
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            feature_flat = features.reshape(-1, features.shape[-1])
            grad_features = (
                grad_flat.matmul(weight).reshape_as(features)
                if ctx.needs_input_grad[0]
                else None
            )
            grad_weight = (
                grad_flat.transpose(0, 1).matmul(feature_flat).to(weight.dtype)
                if ctx.needs_input_grad[1]
                else None
            )
            grad_bias = None
            if ctx.has_bias and ctx.needs_input_grad[2]:
                bias = saved[2]
                grad_bias = grad_flat.sum(dim=0).to(bias.dtype)
            return grad_features, grad_weight, grad_bias


def _local_markov_linear(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    vocab_start_index: int,
    local_vocab_size: int,
    vocab_size: int,
    process_group=None,
) -> torch.Tensor:
    """Support both vocab-sharded and full replicated runtime Markov heads."""
    if vocab_start_index < 0 or local_vocab_size <= 0:
        raise ValueError("Invalid local Markov vocabulary range")
    if weight.shape[0] >= vocab_size:
        # Each rank contributes gradients for its own rows. Reduce the full
        # gradient so replicated optimizer states update the same parameters.
        weight = all_reduce_gradient(weight, process_group)
        if bias is not None:
            bias = all_reduce_gradient(bias, process_group)
        end = min(vocab_start_index + local_vocab_size, vocab_size)
        if end <= vocab_start_index:
            raise ValueError("Local Markov vocabulary range is outside the vocabulary")
        weight = weight[vocab_start_index:end]
        bias = None if bias is None else bias[vocab_start_index:end]
    elif weight.shape[0] >= local_vocab_size:
        weight = weight[:local_vocab_size]
        bias = None if bias is None else bias[:local_vocab_size]
    else:
        raise ValueError("Markov weight has fewer rows than the local vocabulary")
    logits = _ProfiledMarkovLinear.apply(features, weight, bias)
    if logits.shape[-1] < local_vocab_size:
        logits = F.pad(logits, (0, local_vocab_size - logits.shape[-1]))
    return logits


def _straight_through_training_value(
    native_value: torch.Tensor,
    replay_value: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if native_value.shape != replay_value.shape:
        raise RuntimeError(
            f"{name} replay shape mismatch: "
            f"native={tuple(native_value.shape)}, "
            f"replay={tuple(replay_value.shape)}"
        )
    # Forward is bit-for-bit the serving result; backward follows replay_value.
    return _NativeForwardReplayBackward.apply(native_value, replay_value)


def _install_dspark_dsa_query_training_replay() -> bool:
    """Reconnect DSA query parameters across in-place serving operators."""

    import torch_npu
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

    patched_any = False
    for method_name in (
        "_mla_prolog_single_stream",
        "_mla_prolog_multistream",
    ):
        original_method = getattr(AscendDSAImpl, method_name, None)
        if original_method is None:
            continue
        if getattr(
            original_method,
            "_speco_deepseek_v4_training_query_replay",
            False,
        ):
            patched_any = True
            continue

        def make_training_prolog(original_prolog):
            @wraps(original_prolog)
            def training_prolog(
                self,
                hidden_states,
                cos,
                sin,
                *args,
                **kwargs,
            ):
                native_output = original_prolog(
                    self,
                    hidden_states,
                    cos,
                    sin,
                    *args,
                    **kwargs,
                )
                if not torch.is_grad_enabled():
                    return native_output
                if not isinstance(native_output, tuple) or len(native_output) != 3:
                    raise RuntimeError(
                        "DSpark DSA MLA prolog must return (q, qr, qr_pertoken_scale)"
                    )

                native_q, native_qr, qr_pertoken_scale = native_output
                replay_q, replay_qr = _replay_dspark_dsa_query(
                    self,
                    hidden_states,
                    cos,
                    sin,
                    torch_npu.npu_rotary_mul,
                )
                return (
                    _straight_through_training_value(
                        native_q,
                        replay_q,
                        name="DSpark DSA q",
                    ),
                    _straight_through_training_value(
                        native_qr,
                        replay_qr,
                        name="DSpark DSA qr",
                    ),
                    qr_pertoken_scale,
                )

            training_prolog._speco_deepseek_v4_training_query_replay = True
            return training_prolog

        setattr(
            AscendDSAImpl,
            method_name,
            make_training_prolog(original_method),
        )
        patched_any = True
    return patched_any


@dataclass(frozen=True)
class DSparkRuntimeContract:
    block_size: int
    num_draft_layers: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    hidden_size: int
    hc_mult: int
    vocab_size: int
    noise_token_id: int | None
    target_offset: int = 2
    bonus_anchor: bool = False

    @classmethod
    def from_config(cls, config: Any) -> DSparkRuntimeContract:
        target_layer_ids = tuple(
            int(value) for value in getattr(config, "dspark_target_layer_ids", ())
        )
        num_draft_layers = getattr(config, "n_mtp_layers", None)
        if num_draft_layers is None:
            num_draft_layers = getattr(config, "dspark_num_mtp_layers", 3)
        dflash_config = getattr(config, "dflash_config", None) or {}
        noise_token_id = None
        for candidate in (
            dflash_config.get("mask_token_id"),
            getattr(config, "mask_token_id", None),
            getattr(config, "dspark_noise_token_id", None),
            getattr(config, "pard_token", None),
            getattr(config, "ptd_token_id", None),
        ):
            if candidate is not None:
                noise_token_id = int(candidate)
                break
        contract = cls(
            block_size=int(getattr(config, "dspark_block_size", 0)),
            num_draft_layers=int(num_draft_layers or 0),
            target_layer_ids=target_layer_ids,
            markov_rank=int(getattr(config, "dspark_markov_rank", 0)),
            hidden_size=int(getattr(config, "hidden_size", 0)),
            hc_mult=int(getattr(config, "hc_mult", 0)),
            vocab_size=int(getattr(config, "vocab_size", 0)),
            noise_token_id=noise_token_id,
            # vLLM-Ascend seeds DeepSeek V4 DSpark with the target model's
            # already sampled next token. Draft slot zero therefore predicts
            # anchor + 2, not anchor + 1.
            target_offset=2,
            bonus_anchor=bool(getattr(config, "dspark_bonus_anchor", False)),
        )
        contract.validate(
            num_hidden_layers=int(getattr(config, "num_hidden_layers", 0))
        )
        return contract

    @property
    def target_hidden_size(self) -> int:
        return self.hidden_size * len(self.target_layer_ids)

    def validate(self, *, num_hidden_layers: int) -> None:
        positive = {
            "dspark_block_size": self.block_size,
            "num_draft_layers": self.num_draft_layers,
            "dspark_markov_rank": self.markov_rank,
            "hidden_size": self.hidden_size,
            "hc_mult": self.hc_mult,
            "vocab_size": self.vocab_size,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Invalid DeepSeek V4 DSpark dimensions: {invalid}")
        if not self.target_layer_ids:
            raise ValueError("dspark_target_layer_ids must not be empty")
        if tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids:
            raise ValueError(
                "dspark_target_layer_ids must be strictly increasing and unique"
            )
        invalid_layers = [
            layer_id
            for layer_id in self.target_layer_ids
            if layer_id < 0 or layer_id >= num_hidden_layers
        ]
        if invalid_layers:
            raise ValueError(
                "DSpark target layers are outside the target model: "
                f"invalid={invalid_layers}, num_hidden_layers={num_hidden_layers}"
            )
        if self.noise_token_id is None:
            raise ValueError(
                "DeepSeek V4 DSpark config must define a parallel-drafting "
                "mask token (mask_token_id, dspark_noise_token_id, pard_token, "
                "ptd_token_id, or dflash_config.mask_token_id)"
            )


@dataclass(frozen=True)
class DSparkBlockTargets:
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    label_indices: torch.Tensor
    target_ids: torch.Tensor
    previous_token_ids: torch.Tensor
    eval_mask: torch.Tensor


@dataclass(frozen=True)
class DSparkRuntimeBlockInputs:
    context_lengths: torch.Tensor
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor


@dataclass(frozen=True)
class DSparkPackedRuntimeInputs:
    """Flattened serving requests produced from valid training anchors."""

    batch_indices: torch.Tensor
    anchor_positions: torch.Tensor
    context_lengths: torch.Tensor
    context_hidden_states: torch.Tensor
    context_positions: torch.Tensor
    query_input_ids: torch.Tensor
    query_positions: torch.Tensor
    previous_token_ids: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    context_lengths_cpu: torch.Tensor | None = None
    query_positions_cpu: torch.Tensor | None = None
    shared_context_block_size: int | None = None


@dataclass(frozen=True)
class _DSparkPhaseRecord:
    name: str
    host_seconds: float
    start_event: Any | None
    end_event: Any | None


class DSparkPhaseProfiler:
    """Collect asynchronous NPU timings without synchronizing the hot path."""

    def __init__(self) -> None:
        self.enabled = False
        self._records: list[_DSparkPhaseRecord] = []

    def reset(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._records.clear()

    @contextmanager
    def record(self, name: str):
        if not self.enabled:
            yield
            return
        start_event = None
        end_event = None
        npu = getattr(torch, "npu", None)
        event_type = getattr(npu, "Event", None)
        if event_type is not None:
            try:
                start_event = event_type(enable_timing=True)
                end_event = event_type(enable_timing=True)
                start_event.record()
            except (RuntimeError, TypeError):
                # Profiling must never make an otherwise valid training run
                # depend on a particular torch_npu Event API revision.
                start_event = None
                end_event = None
        host_started = perf_counter()
        try:
            yield
        finally:
            host_seconds = perf_counter() - host_started
            if end_event is not None:
                try:
                    end_event.record()
                except (RuntimeError, TypeError):
                    start_event = None
                    end_event = None
            self._records.append(
                _DSparkPhaseRecord(
                    name=name,
                    host_seconds=host_seconds,
                    start_event=start_event,
                    end_event=end_event,
                )
            )

    def consume(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for record in self._records:
            values = summary.setdefault(
                record.name,
                {"calls": 0, "host_seconds": 0.0, "device_seconds": 0.0},
            )
            values["calls"] = int(values["calls"]) + 1
            values["host_seconds"] = float(values["host_seconds"]) + record.host_seconds
            if record.start_event is not None and record.end_event is not None:
                try:
                    elapsed_seconds = (
                        float(record.start_event.elapsed_time(record.end_event))
                        / 1000.0
                    )
                except (RuntimeError, TypeError):
                    elapsed_seconds = 0.0
                values["device_seconds"] = (
                    float(values["device_seconds"]) + elapsed_seconds
                )
        self._records.clear()
        self.enabled = False
        return summary


def register_drafting_backward_host_metadata(
    built_metadata: Any,
    common_metadata: Any,
    compress_ratio: int,
    *,
    enabled: bool,
    register_fn: Any | None = None,
) -> bool:
    """Register drafting metadata for the shared analytical DSA backward."""

    if not enabled:
        return False
    effective_register_fn = (
        _register_dsa_backward_host_metadata if register_fn is None else register_fn
    )
    if effective_register_fn is None:
        return False
    return bool(
        effective_register_fn(
            built_metadata,
            common_metadata,
            int(compress_ratio),
        )
    )


@dataclass(frozen=True)
class DSparkPagedCacheLayout:
    """One cache-group layout matching the DSpark serving proposer."""

    block_size: int
    block_table: torch.Tensor
    context_slot_mapping: torch.Tensor
    query_slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_blocks: int = 0


def _shared_context_plan(
    context_lengths: list[int],
    batch_indices: list[int],
    shared_block_size: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return each sample's canonical prefix and each request's private tail start."""

    if shared_block_size <= 0:
        raise ValueError("shared_block_size must be positive")
    if len(context_lengths) != len(batch_indices):
        raise ValueError("context lengths and batch indices must agree")
    canonical_lengths: dict[int, int] = {}
    for batch_index, length in zip(batch_indices, context_lengths):
        if length < 0:
            raise ValueError("context lengths must be non-negative")
        full_length = length // shared_block_size * shared_block_size
        canonical_lengths[batch_index] = max(
            canonical_lengths.get(batch_index, 0), full_length
        )
    return list(canonical_lengths.items()), [
        length // shared_block_size * shared_block_size for length in context_lengths
    ]


def build_anchor_candidate_mask(
    loss_mask: torch.Tensor,
    *,
    target_offset: int = 2,
) -> torch.Tensor:
    if loss_mask.ndim != 2:
        raise ValueError(
            f"DSpark loss_mask must have shape [batch, sequence], got {loss_mask.shape}"
        )
    if target_offset < 1:
        raise ValueError(f"target_offset must be positive, got {target_offset}")
    if loss_mask.shape[1] <= target_offset:
        return loss_mask[:, :0].bool()
    enabled = loss_mask > 0
    # The seed token and first predicted token must both be supervised.
    return enabled[:, target_offset - 1 : -1] & enabled[:, target_offset:]


def select_anchor_positions(
    loss_mask: torch.Tensor,
    *,
    num_anchors: int,
    generator: torch.Generator | None = None,
    target_offset: int = 2,
    min_anchor_spacing: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_anchors <= 0:
        raise ValueError(f"num_anchors must be positive, got {num_anchors}")
    if min_anchor_spacing <= 0:
        raise ValueError(
            f"min_anchor_spacing must be positive, got {min_anchor_spacing}"
        )
    candidates = build_anchor_candidate_mask(
        loss_mask,
        target_offset=target_offset,
    )
    batch_size, _num_candidates = candidates.shape
    device = loss_mask.device
    anchors = torch.zeros(
        batch_size,
        num_anchors,
        dtype=torch.long,
        device=device,
    )
    keep_mask = torch.zeros(
        batch_size,
        num_anchors,
        dtype=torch.bool,
        device=device,
    )
    for batch_index in range(batch_size):
        valid = torch.nonzero(candidates[batch_index], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        # Anchor selection is tiny compared with a DSpark forward. Perform the
        # greedy spacing check on CPU so it cannot introduce one NPU sync per
        # candidate position.
        valid_cpu = valid.detach().cpu()
        order = torch.randperm(
            valid_cpu.numel(),
            generator=generator,
        )
        selected: list[int] = []
        for candidate in valid_cpu.index_select(0, order).tolist():
            if selected and any(
                abs(candidate - other) < min_anchor_spacing for other in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == num_anchors:
                break
        if not selected:
            continue
        selected_tensor = torch.tensor(
            sorted(selected),
            dtype=torch.long,
            device=device,
        )
        count = int(selected_tensor.numel())
        anchors[batch_index, :count] = selected_tensor
        keep_mask[batch_index, :count] = True
    return anchors, keep_mask


def build_block_targets(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    block_size: int,
    num_anchors: int,
    generator: torch.Generator | None = None,
    target_offset: int = 2,
    min_anchor_spacing: int = 1,
) -> DSparkBlockTargets:
    if input_ids.ndim != 2 or input_ids.shape != loss_mask.shape:
        raise ValueError(
            "DSpark input_ids and loss_mask must have the same [batch, sequence] "
            f"shape, got ids={input_ids.shape}, mask={loss_mask.shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    _batch_size, sequence_length = input_ids.shape
    if sequence_length == 0:
        raise ValueError("DSpark sequences must contain at least one token")
    anchors, keep_mask = select_anchor_positions(
        loss_mask,
        num_anchors=num_anchors,
        generator=generator,
        target_offset=target_offset,
        min_anchor_spacing=min_anchor_spacing,
    )
    offsets = torch.arange(
        target_offset,
        target_offset + block_size,
        device=input_ids.device,
    ).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=sequence_length - 1)
    safe_indices = torch.where(
        keep_mask.unsqueeze(-1),
        safe_indices,
        torch.zeros_like(safe_indices),
    )
    expanded_ids = input_ids.unsqueeze(1).expand(-1, num_anchors, -1)
    target_ids = torch.gather(expanded_ids, 2, safe_indices)
    expanded_mask = loss_mask.unsqueeze(1).expand(-1, num_anchors, -1)
    target_mask = torch.gather(expanded_mask, 2, safe_indices) > 0
    eval_mask = (
        (label_indices < sequence_length) & target_mask & keep_mask.unsqueeze(-1)
    )
    # A rejected/invalid position invalidates the remainder of the block.
    eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
    previous_indices = (label_indices - 1).clamp(min=0, max=sequence_length - 1)
    previous_indices = torch.where(
        keep_mask.unsqueeze(-1),
        previous_indices,
        torch.zeros_like(previous_indices),
    )
    previous_token_ids = torch.gather(expanded_ids, 2, previous_indices)
    return DSparkBlockTargets(
        anchor_positions=anchors,
        block_keep_mask=keep_mask,
        label_indices=label_indices,
        target_ids=target_ids,
        previous_token_ids=previous_token_ids,
        eval_mask=eval_mask,
    )


def build_runtime_block_inputs(
    input_ids: torch.Tensor,
    targets: DSparkBlockTargets,
    *,
    noise_token_id: int,
    positions: torch.Tensor | None = None,
) -> DSparkRuntimeBlockInputs:
    """Build the default DeepSeek V4 DSpark serving query layout."""

    if input_ids.ndim != 2:
        raise ValueError(
            f"DSpark input_ids must have shape [batch, sequence], got {input_ids.shape}"
        )
    batch_size, sequence_length = input_ids.shape
    if positions is None:
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=input_ids.device,
        ).expand(batch_size, -1)
    elif positions.shape != input_ids.shape:
        raise ValueError("DSpark positions and input_ids shapes disagree")
    if targets.anchor_positions.shape[0] != batch_size:
        raise ValueError("DSpark anchors and input batch sizes disagree")
    block_size = targets.target_ids.shape[-1]
    seed_indices = targets.anchor_positions + 1
    safe_seed_indices = seed_indices.clamp(max=sequence_length - 1)
    seeds = torch.gather(input_ids, 1, safe_seed_indices)
    query_input_ids = torch.full(
        (*targets.anchor_positions.shape, block_size),
        int(noise_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    query_input_ids[..., 0] = seeds
    query_input_ids = torch.where(
        targets.block_keep_mask.unsqueeze(-1),
        query_input_ids,
        torch.full_like(query_input_ids, int(noise_token_id)),
    )
    seed_positions = torch.gather(positions, 1, safe_seed_indices)
    offsets = torch.arange(
        0,
        block_size,
        dtype=targets.anchor_positions.dtype,
        device=input_ids.device,
    ).view(1, 1, -1)
    query_positions = seed_positions.unsqueeze(-1) + offsets
    query_positions = torch.where(
        targets.block_keep_mask.unsqueeze(-1),
        query_positions,
        torch.zeros_like(query_positions),
    )
    return DSparkRuntimeBlockInputs(
        context_lengths=targets.anchor_positions + 1,
        query_input_ids=query_input_ids,
        query_positions=query_positions,
    )


def pack_runtime_block_inputs(
    target_hidden_states: torch.Tensor,
    targets: DSparkBlockTargets,
    runtime_inputs: DSparkRuntimeBlockInputs,
    *,
    positions: torch.Tensor | None = None,
    active_anchors: torch.Tensor | None = None,
    active_context_lengths_cpu: torch.Tensor | None = None,
    active_query_positions_cpu: torch.Tensor | None = None,
    validate_active_anchors: bool = True,
    shared_context_block_size: int | None = None,
) -> DSparkPackedRuntimeInputs:
    """Pack serving queries with optional read-only shared context blocks."""

    if target_hidden_states.ndim != 3:
        raise ValueError(
            "DSpark target_hidden_states must have shape [batch, sequence, width]"
        )
    batch_size, sequence_length, hidden_width = target_hidden_states.shape
    if positions is None:
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=target_hidden_states.device,
        ).expand(batch_size, -1)
    elif positions.shape != (batch_size, sequence_length):
        raise ValueError("DSpark positions and hidden-state shapes disagree")
    if targets.block_keep_mask.shape[0] != batch_size:
        raise ValueError("DSpark targets and hidden-state batch sizes disagree")
    if active_anchors is None:
        active = torch.nonzero(targets.block_keep_mask, as_tuple=False)
    else:
        if active_anchors.ndim != 2 or active_anchors.shape[1] != 2:
            raise ValueError("DSpark active_anchors must have shape [num_anchors, 2]")
        active = active_anchors.to(device=targets.anchor_positions.device)
        if (
            validate_active_anchors
            and active.numel()
            and not bool(targets.block_keep_mask[active[:, 0], active[:, 1]].all())
        ):
            raise ValueError("DSpark active_anchors contains an inactive anchor")
    if active.numel() == 0:
        raise ValueError("DSpark batch does not contain a valid training anchor")

    batch_indices = active[:, 0]
    anchor_indices = active[:, 1]
    anchors = targets.anchor_positions[batch_indices, anchor_indices]
    context_lengths = runtime_inputs.context_lengths[batch_indices, anchor_indices]
    if validate_active_anchors and int(context_lengths.max().item()) > sequence_length:
        raise ValueError("DSpark context extends beyond captured hidden states")

    if shared_context_block_size is None:
        token_positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=target_hidden_states.device,
        ).unsqueeze(0)
        context_mask = token_positions < context_lengths.unsqueeze(1)
        flat_context_indices = (
            batch_indices.unsqueeze(1) * sequence_length + token_positions
        ).masked_select(context_mask)
    else:
        host_lengths = (
            active_context_lengths_cpu.tolist()
            if active_context_lengths_cpu is not None
            else context_lengths.detach().to("cpu").tolist()
        )
        host_batches = batch_indices.detach().to("cpu").tolist()
        canonical, tail_starts = _shared_context_plan(
            host_lengths, host_batches, shared_context_block_size
        )
        ranges = [(batch_index, 0, length) for batch_index, length in canonical] + [
            (batch_index, tail_start, length)
            for batch_index, tail_start, length in zip(
                host_batches, tail_starts, host_lengths
            )
        ]
        flat_context_indices = torch.cat(
            [
                batch_index * sequence_length
                + torch.arange(start, end, device=target_hidden_states.device)
                for batch_index, start, end in ranges
            ]
        )
    context_hidden_states = target_hidden_states.reshape(
        batch_size * sequence_length,
        hidden_width,
    ).index_select(0, flat_context_indices)
    context_positions = positions.reshape(batch_size * sequence_length).index_select(
        0,
        flat_context_indices,
    )
    expected_host_requests = int(active.shape[0])
    if active_context_lengths_cpu is not None:
        if active_context_lengths_cpu.shape != (expected_host_requests,):
            raise ValueError(
                "DSpark host context lengths must match active anchors: "
                f"got={active_context_lengths_cpu.shape}, "
                f"expected=({expected_host_requests},)"
            )
        active_context_lengths_cpu = active_context_lengths_cpu.to(
            device="cpu",
            dtype=torch.int32,
        )
    if active_query_positions_cpu is not None:
        expected_query_shape = (
            expected_host_requests,
            int(runtime_inputs.query_input_ids.shape[-1]),
        )
        if active_query_positions_cpu.shape != expected_query_shape:
            raise ValueError(
                "DSpark host query positions must match packed queries: "
                f"got={active_query_positions_cpu.shape}, "
                f"expected={expected_query_shape}"
            )
        active_query_positions_cpu = active_query_positions_cpu.to(device="cpu")
    return DSparkPackedRuntimeInputs(
        batch_indices=batch_indices,
        anchor_positions=anchors,
        context_lengths=context_lengths,
        context_hidden_states=context_hidden_states,
        context_positions=context_positions,
        query_input_ids=runtime_inputs.query_input_ids[batch_indices, anchor_indices],
        query_positions=runtime_inputs.query_positions[batch_indices, anchor_indices],
        previous_token_ids=targets.previous_token_ids[batch_indices, anchor_indices],
        target_ids=targets.target_ids[batch_indices, anchor_indices],
        eval_mask=targets.eval_mask[batch_indices, anchor_indices],
        context_lengths_cpu=active_context_lengths_cpu,
        query_positions_cpu=active_query_positions_cpu,
        shared_context_block_size=shared_context_block_size,
    )


def gather_active_teacher_hidden_states(
    target_head_hidden_states: torch.Tensor,
    label_indices: torch.Tensor,
    active_anchors: torch.Tensor,
) -> torch.Tensor:
    """Gather teacher states only for the anchors in the current chunk."""

    if target_head_hidden_states.ndim != 3:
        raise ValueError(
            "target_head_hidden_states must have shape [batch, sequence, width]"
        )
    if label_indices.ndim != 3:
        raise ValueError("DSpark label_indices must have shape [batch, anchors, block]")
    if active_anchors.ndim != 2 or active_anchors.shape[1] != 2:
        raise ValueError("active_anchors must have shape [num_anchors, 2]")
    if label_indices.shape[0] != target_head_hidden_states.shape[0]:
        raise ValueError("Teacher states and DSpark labels have different batches")

    batch_size, sequence_length, hidden_width = target_head_hidden_states.shape
    batch_indices = active_anchors[:, 0].to(
        device=target_head_hidden_states.device,
        dtype=torch.long,
    )
    anchor_indices = active_anchors[:, 1].to(
        device=label_indices.device,
        dtype=torch.long,
    )
    selected_labels = label_indices[
        batch_indices,
        anchor_indices,
    ].clamp(min=0, max=sequence_length - 1)
    flat_indices = (
        batch_indices.unsqueeze(1) * sequence_length + selected_labels
    ).reshape(-1)
    return (
        target_head_hidden_states.reshape(
            batch_size * sequence_length,
            hidden_width,
        )
        .index_select(0, flat_indices)
        .reshape(
            active_anchors.shape[0],
            label_indices.shape[-1],
            hidden_width,
        )
    )


def build_combined_target_replay_loss(
    combined_target_hidden_states: torch.Tensor,
    reusable_combined_target_hidden_states: torch.Tensor,
    *,
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    """Replay cached-context gradients through ``combine_hidden_states`` once.

    Anchor chunks backpropagate into the detached reusable tensor. DeepSpeed
    already scales those gradients by the configured accumulation count, so
    the replay loss compensates for the second automatic DeepSpeed scaling.
    """

    if combined_target_hidden_states.shape != (
        reusable_combined_target_hidden_states.shape
    ):
        raise ValueError(
            "DSpark combined hidden-state replay shapes disagree: "
            f"combined={combined_target_hidden_states.shape}, "
            f"reusable={reusable_combined_target_hidden_states.shape}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    reusable_gradient = reusable_combined_target_hidden_states.grad
    if reusable_gradient is None:
        raise RuntimeError(
            "DSpark reusable combined hidden states did not receive gradients"
        )
    return (combined_target_hidden_states * reusable_gradient.detach()).sum() * int(
        gradient_accumulation_steps
    )


def should_synchronize_dspark_backward(
    *,
    is_accumulation_boundary: bool,
    is_last_chunk: bool,
    replay_combined_gradients: bool,
) -> bool:
    """Return whether this chunk backward must perform DP gradient reduction."""

    return bool(
        is_accumulation_boundary and is_last_chunk and not replay_combined_gradients
    )


def dspark_backward_sync_context(
    model_engine,
    *,
    synchronize: bool,
    coalesce_gradient_communication: bool,
):
    """Select a normal or deferred-reduction DeepSpeed backward context."""

    if synchronize or not coalesce_gradient_communication:
        return nullcontext()
    return model_engine.no_sync()


def build_paged_cache_layout(
    context_lengths: torch.Tensor,
    *,
    query_length: int,
    block_size: int,
    context_lengths_cpu: torch.Tensor | None = None,
) -> DSparkPagedCacheLayout:
    """Allocate deterministic, non-overlapping cache blocks per request."""

    if context_lengths.ndim != 1 or context_lengths.numel() == 0:
        raise ValueError("context_lengths must be a non-empty rank-1 tensor")
    if query_length <= 0 or block_size <= 0:
        raise ValueError("query_length and block_size must be positive")
    device = context_lengths.device
    lengths = context_lengths.to(torch.long)
    if context_lengths_cpu is None:
        host_lengths = lengths.detach().to("cpu")
    else:
        if context_lengths_cpu.shape != context_lengths.shape:
            raise ValueError(
                "context_lengths_cpu and context_lengths must have the same shape"
            )
        host_lengths = context_lengths_cpu.to(
            device="cpu",
            dtype=torch.long,
        )
    total_lengths = lengths + int(query_length)
    blocks_per_request = torch.div(
        total_lengths + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    max_context_length = int(host_lengths.max().item())
    max_blocks = math.ceil((max_context_length + int(query_length)) / int(block_size))
    block_offsets = torch.cumsum(blocks_per_request, dim=0) - blocks_per_request
    block_columns = torch.arange(
        max_blocks,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    block_table = block_offsets.unsqueeze(1) + block_columns
    block_table = torch.where(
        block_columns < blocks_per_request.unsqueeze(1),
        block_table,
        torch.zeros_like(block_table),
    ).to(torch.int32)

    context_token_positions = torch.arange(
        max_context_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    context_mask = context_token_positions < lengths.unsqueeze(1)
    context_slots = (
        block_offsets.unsqueeze(1)
        + torch.div(
            context_token_positions,
            block_size,
            rounding_mode="floor",
        )
    ) * block_size + context_token_positions.remainder(block_size)
    context_slot_mapping = context_slots.masked_select(context_mask)

    query_offsets = torch.arange(
        query_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    query_token_positions = lengths.unsqueeze(1) + query_offsets
    query_slot_mapping = (
        block_offsets.unsqueeze(1)
        + torch.div(
            query_token_positions,
            block_size,
            rounding_mode="floor",
        )
    ) * block_size + query_token_positions.remainder(block_size)
    query_start_loc = torch.arange(
        lengths.numel() + 1,
        dtype=torch.int32,
        device=device,
    ) * int(query_length)
    return DSparkPagedCacheLayout(
        block_size=int(block_size),
        block_table=block_table,
        context_slot_mapping=context_slot_mapping.to(torch.int64),
        query_slot_mapping=query_slot_mapping.reshape(-1).to(torch.int64),
        query_start_loc=query_start_loc,
        seq_lens=total_lengths.to(torch.int32),
    )


def build_shared_paged_cache_layout(
    context_lengths: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    query_length: int,
    block_size: int,
    shared_block_size: int,
    context_lengths_cpu: torch.Tensor | None = None,
) -> DSparkPagedCacheLayout:
    """Alias full context blocks while keeping every query/tail writable only once."""

    if context_lengths.ndim != 1 or context_lengths.numel() == 0:
        raise ValueError("context_lengths must be a non-empty rank-1 tensor")
    if batch_indices.shape != context_lengths.shape:
        raise ValueError("batch_indices and context_lengths must agree")
    if query_length <= 0 or block_size <= 0:
        raise ValueError("query_length and block_size must be positive")
    if shared_block_size <= 0:
        raise ValueError("shared_block_size must be positive")
    if shared_block_size % block_size:
        raise ValueError("shared_block_size must be a multiple of block_size")
    if (
        context_lengths_cpu is not None
        and context_lengths_cpu.shape != context_lengths.shape
    ):
        raise ValueError(
            "context_lengths_cpu and context_lengths must have the same shape"
        )
    device = context_lengths.device
    host_lengths = (
        context_lengths_cpu.to(device="cpu", dtype=torch.long).tolist()
        if context_lengths_cpu is not None
        else context_lengths.detach().to("cpu").tolist()
    )
    host_batches = batch_indices.detach().to("cpu").tolist()
    canonical, tail_starts = _shared_context_plan(
        host_lengths, host_batches, shared_block_size
    )
    next_block = 0
    shared_blocks: dict[int, list[int]] = {}
    context_slot_parts: list[torch.Tensor] = []
    for batch_index, length in canonical:
        count = length // block_size
        blocks = list(range(next_block, next_block + count))
        shared_blocks[batch_index] = blocks
        next_block += count
        if blocks:
            context_slot_parts.append(
                torch.arange(
                    blocks[0] * block_size,
                    (blocks[-1] + 1) * block_size,
                    dtype=torch.int64,
                    device=device,
                )
            )

    block_rows: list[list[int]] = []
    query_slot_parts: list[torch.Tensor] = []
    for batch_index, length, tail_start in zip(host_batches, host_lengths, tail_starts):
        tail_length = length - tail_start
        private_count = math.ceil((tail_length + query_length) / block_size)
        private_blocks = list(range(next_block, next_block + private_count))
        next_block += private_count
        block_rows.append(
            shared_blocks[batch_index][: tail_start // block_size] + private_blocks
        )
        if tail_length:
            context_slot_parts.append(
                torch.arange(
                    private_blocks[0] * block_size,
                    private_blocks[0] * block_size + tail_length,
                    dtype=torch.int64,
                    device=device,
                )
            )
        query_slot_parts.append(
            torch.arange(
                private_blocks[0] * block_size + tail_length,
                private_blocks[0] * block_size + tail_length + query_length,
                dtype=torch.int64,
                device=device,
            )
        )

    max_blocks = max(map(len, block_rows))
    block_table = torch.tensor(
        [row + [0] * (max_blocks - len(row)) for row in block_rows],
        dtype=torch.int32,
        device=device,
    )
    return DSparkPagedCacheLayout(
        block_size=block_size,
        block_table=block_table,
        context_slot_mapping=(
            torch.cat(context_slot_parts)
            if context_slot_parts
            else torch.empty(0, dtype=torch.int64, device=device)
        ),
        query_slot_mapping=torch.cat(query_slot_parts),
        query_start_loc=torch.arange(
            len(host_lengths) + 1, dtype=torch.int32, device=device
        )
        * query_length,
        seq_lens=(context_lengths + query_length).to(torch.int32),
        num_blocks=next_block,
    )


def apply_markov_teacher_forcing(
    base_logits: torch.Tensor,
    previous_token_ids: torch.Tensor,
    *,
    markov_embedding: torch.nn.Module,
    markov_projection: torch.nn.Module,
) -> torch.Tensor:
    if base_logits.shape[:-1] != previous_token_ids.shape:
        raise ValueError(
            "Markov teacher-forcing shapes disagree: "
            f"logits={base_logits.shape}, previous_ids={previous_token_ids.shape}"
        )
    markov_bias = markov_projection(markov_embedding(previous_token_ids))
    if markov_bias.shape != base_logits.shape:
        raise ValueError(
            "Markov projection must produce one vocabulary bias per draft slot: "
            f"bias={markov_bias.shape}, logits={base_logits.shape}"
        )
    return base_logits + markov_bias


def compute_dspark_confidence_logits(
    hidden_states: torch.Tensor,
    previous_token_ids: torch.Tensor,
    *,
    markov_embedding: torch.nn.Module,
    confidence_head: torch.nn.Module,
) -> torch.Tensor:
    """Predict per-position acceptance logits from DSpark draft features."""

    if hidden_states.shape[:-1] != previous_token_ids.shape:
        raise ValueError(
            "DSpark confidence hidden states and Markov tokens disagree: "
            f"hidden={hidden_states.shape}, previous={previous_token_ids.shape}"
        )
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_previous = previous_token_ids.reshape(-1).to(hidden_states.device)
    markov_features = markov_embedding(flat_previous).to(flat_hidden.dtype)
    features = torch.cat((flat_hidden, markov_features), dim=-1)
    return confidence_head(features).reshape(previous_token_ids.shape)


@dataclass(frozen=True)
class DSparkLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    distribution_l1: torch.Tensor
    distribution_kl: torch.Tensor
    confidence: torch.Tensor
    valid_weight: torch.Tensor
    distribution_position_counts: torch.Tensor | None = None
    distribution_l1_position_sums: torch.Tensor | None = None
    top1_correct_counts: torch.Tensor | None = None


@dataclass(frozen=True)
class DSparkTrainingOutput:
    loss: DSparkLoss
    draft_logits: torch.Tensor
    targets: DSparkBlockTargets
    packed: DSparkPackedRuntimeInputs
    prefix_correct_counts: torch.Tensor | None = None
    prefix_total_counts: torch.Tensor | None = None


def build_dspark_loss_weights(
    eval_mask: torch.Tensor,
    *,
    loss_decay_gamma: float | None = None,
) -> torch.Tensor:
    """Return the per-token weights shared by full and chunked DSpark loss."""

    weights = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(
            eval_mask.shape[-1],
            device=eval_mask.device,
            dtype=torch.float32,
        )
        weights = weights * torch.exp(-positions / float(loss_decay_gamma))
    return weights


def compute_dspark_loss(
    draft_logits: torch.Tensor,
    targets: DSparkBlockTargets,
    *,
    target_logits: torch.Tensor | None = None,
    confidence_logits: torch.Tensor | None = None,
    loss_decay_gamma: float | None = None,
    cross_entropy_weight: float = 0.1,
    distribution_l1_weight: float = 0.9,
    distribution_kl_weight: float = 0.0,
    confidence_weight: float = 0.0,
    compute_distribution_metrics: bool = False,
) -> DSparkLoss:
    if draft_logits.ndim != 4:
        raise ValueError(
            "DSpark draft_logits must have shape [batch, anchors, block, vocab]"
        )
    if draft_logits.shape[:-1] != targets.target_ids.shape:
        raise ValueError(
            f"Draft logits and targets disagree: {draft_logits.shape} vs "
            f"{targets.target_ids.shape}"
        )
    weights = build_dspark_loss_weights(
        targets.eval_mask,
        loss_decay_gamma=loss_decay_gamma,
    )
    denominator = weights.sum().clamp_min(1.0)
    ce_per_token = F.cross_entropy(
        draft_logits.reshape(-1, draft_logits.shape[-1]).float(),
        targets.target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    cross_entropy = (ce_per_token * weights).sum() / denominator

    zero = cross_entropy.new_zeros(())
    distribution_l1 = zero
    distribution_kl = zero
    accept_probability = None
    distribution_position_counts = None
    distribution_l1_position_sums = None
    top1_correct_counts = None
    if target_logits is not None:
        if target_logits.shape != draft_logits.shape:
            raise ValueError(
                f"Target logits must match draft logits, got {target_logits.shape} "
                f"and {draft_logits.shape}"
            )
        draft_probability = draft_logits.float().softmax(dim=-1)
        target_probability = target_logits.float().softmax(dim=-1)
        l1_per_token = (draft_probability - target_probability).abs().sum(dim=-1)
        distribution_l1 = (l1_per_token * weights).sum() / denominator
        if distribution_kl_weight > 0:
            draft_log_probability = draft_logits.float().log_softmax(dim=-1)
            target_log_probability = target_logits.float().log_softmax(dim=-1)
            kl_per_token = (
                target_probability * (target_log_probability - draft_log_probability)
            ).sum(dim=-1)
            distribution_kl = (kl_per_token * weights).sum() / denominator
        accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
        if compute_distribution_metrics:
            metric_mask = targets.eval_mask.to(torch.float32)
            distribution_position_counts = metric_mask.sum(dim=(0, 1))
            distribution_l1_position_sums = (l1_per_token.detach() * metric_mask).sum(
                dim=(0, 1)
            )
            top1_correct_counts = (
                draft_logits.detach()
                .argmax(dim=-1)
                .eq(target_logits.detach().argmax(dim=-1))
                .to(torch.float32)
                .mul(metric_mask)
                .sum(dim=(0, 1))
            )
    elif (
        distribution_l1_weight > 0
        or distribution_kl_weight > 0
        or confidence_logits is not None
    ):
        raise ValueError(
            "target_logits are required for DSpark L1/KL distribution matching "
            "and confidence supervision"
        )

    confidence = zero
    if confidence_logits is not None:
        if confidence_logits.shape != targets.target_ids.shape:
            raise ValueError(
                "confidence_logits must have shape [batch, anchors, block], got "
                f"{confidence_logits.shape}"
            )
        confidence_error = F.binary_cross_entropy_with_logits(
            confidence_logits.float(),
            accept_probability.detach(),
            reduction="none",
        )
        confidence = (confidence_error * weights).sum() / denominator
    elif confidence_weight > 0:
        raise ValueError(
            "confidence_logits are required when confidence_weight is positive"
        )

    total = (
        float(cross_entropy_weight) * cross_entropy
        + float(distribution_l1_weight) * distribution_l1
        + float(distribution_kl_weight) * distribution_kl
        + float(confidence_weight) * confidence
    )
    return DSparkLoss(
        total=total,
        cross_entropy=cross_entropy,
        distribution_l1=distribution_l1,
        distribution_kl=distribution_kl,
        confidence=confidence,
        valid_weight=denominator,
        distribution_position_counts=distribution_position_counts,
        distribution_l1_position_sums=distribution_l1_position_sums,
        top1_correct_counts=top1_correct_counts,
    )


def compute_vocab_parallel_dspark_loss(
    draft_local_logits: torch.Tensor,
    targets: DSparkBlockTargets,
    *,
    target_local_logits: torch.Tensor,
    vocab_start_index: int,
    process_group=None,
    confidence_logits: torch.Tensor | None = None,
    loss_decay_gamma: float | None = None,
    cross_entropy_weight: float = 0.1,
    distribution_l1_weight: float = 0.9,
    distribution_kl_weight: float = 0.0,
    confidence_weight: float = 0.0,
    compute_distribution_metrics: bool = False,
) -> DSparkLoss:
    """Compute exact DSpark CE/L1/KL on local vocabulary shards."""

    if draft_local_logits.shape != target_local_logits.shape:
        raise ValueError(
            "DSpark local draft and target logits must have identical shapes: "
            f"draft={draft_local_logits.shape}, "
            f"target={target_local_logits.shape}"
        )
    if draft_local_logits.shape[:-1] != targets.target_ids.shape:
        raise ValueError(
            "DSpark local logits and target IDs disagree: "
            f"logits={draft_local_logits.shape}, "
            f"targets={targets.target_ids.shape}"
        )
    if target_local_logits.requires_grad:
        raise ValueError("DSpark target logits must remain detached")

    weights = build_dspark_loss_weights(
        targets.eval_mask,
        loss_decay_gamma=loss_decay_gamma,
    )
    denominator = weights.sum().clamp_min(1.0)
    draft_log_probability = differentiable_vocab_parallel_log_probs(
        draft_local_logits,
        process_group,
    )
    with torch.no_grad():
        target_log_probability = differentiable_vocab_parallel_log_probs(
            target_local_logits,
            process_group,
        )

    local_vocab_size = int(draft_local_logits.shape[-1])
    if local_vocab_size <= 0:
        raise ValueError("DSpark local vocabulary shard must not be empty")
    local_target_ids = targets.target_ids - int(vocab_start_index)
    owns_target = (local_target_ids >= 0) & (local_target_ids < local_vocab_size)
    safe_target_ids = local_target_ids.clamp(
        min=0,
        max=local_vocab_size - 1,
    )
    local_target_log_probability = draft_log_probability.gather(
        -1,
        safe_target_ids.unsqueeze(-1),
    ).squeeze(-1)
    local_target_log_probability = torch.where(
        owns_target,
        local_target_log_probability,
        torch.zeros_like(local_target_log_probability),
    )
    target_token_log_probability = differentiable_all_reduce_sum(
        local_target_log_probability,
        process_group,
    )
    cross_entropy = (-target_token_log_probability * weights).sum() / denominator

    draft_probability = draft_log_probability.exp()
    target_probability = target_log_probability.exp()
    local_l1 = (draft_probability - target_probability).abs().sum(dim=-1)
    l1_per_token = differentiable_all_reduce_sum(local_l1, process_group)
    distribution_l1 = (l1_per_token * weights).sum() / denominator
    zero = cross_entropy.new_zeros(())
    distribution_kl = zero
    if distribution_kl_weight > 0:
        local_kl = (
            target_probability * (target_log_probability - draft_log_probability)
        ).sum(dim=-1)
        kl_per_token = differentiable_all_reduce_sum(local_kl, process_group)
        distribution_kl = (kl_per_token * weights).sum() / denominator
    accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
    distribution_position_counts = None
    distribution_l1_position_sums = None
    top1_correct_counts = None
    if compute_distribution_metrics:
        metric_mask = targets.eval_mask.to(torch.float32)
        distribution_position_counts = metric_mask.sum(dim=(0, 1))
        distribution_l1_position_sums = (l1_per_token.detach() * metric_mask).sum(
            dim=(0, 1)
        )
        draft_greedy_ids = distributed_vocab_argmax(
            draft_local_logits.detach(),
            vocab_start_index=vocab_start_index,
            process_group=process_group,
        )
        target_greedy_ids = distributed_vocab_argmax(
            target_local_logits.detach(),
            vocab_start_index=vocab_start_index,
            process_group=process_group,
        )
        top1_correct_counts = (
            draft_greedy_ids.eq(target_greedy_ids)
            .to(torch.float32)
            .mul(metric_mask)
            .sum(dim=(0, 1))
        )

    confidence = zero
    if confidence_logits is not None:
        if confidence_logits.shape != targets.target_ids.shape:
            raise ValueError(
                "confidence_logits must match DSpark target IDs, got "
                f"{confidence_logits.shape} and {targets.target_ids.shape}"
            )
        confidence_error = F.binary_cross_entropy_with_logits(
            confidence_logits.float(),
            accept_probability.detach(),
            reduction="none",
        )
        confidence = (confidence_error * weights).sum() / denominator
    elif confidence_weight > 0:
        raise ValueError(
            "confidence_logits are required when confidence_weight is positive"
        )

    tp_world_size = (
        dist.get_world_size(group=process_group) if dist.is_initialized() else 1
    )
    gradient_scale = 1.0 / float(tp_world_size)
    cross_entropy = scale_gradient(cross_entropy, gradient_scale)
    distribution_l1 = scale_gradient(distribution_l1, gradient_scale)
    distribution_kl = scale_gradient(distribution_kl, gradient_scale)
    total = (
        float(cross_entropy_weight) * cross_entropy
        + float(distribution_l1_weight) * distribution_l1
        + float(distribution_kl_weight) * distribution_kl
        + float(confidence_weight) * confidence
    )
    return DSparkLoss(
        total=total,
        cross_entropy=cross_entropy,
        distribution_l1=distribution_l1,
        distribution_kl=distribution_kl,
        confidence=confidence,
        valid_weight=denominator,
        distribution_position_counts=distribution_position_counts,
        distribution_l1_position_sums=distribution_l1_position_sums,
        top1_correct_counts=top1_correct_counts,
    )


@torch.no_grad()
def distributed_vocab_argmax(
    local_logits: torch.Tensor,
    *,
    vocab_start_index: int,
    process_group=None,
) -> torch.Tensor:
    """Return exact global argmax IDs from TP-local vocabulary logits."""

    local_values, local_indices = local_logits.float().max(dim=-1)
    local_indices = local_indices + int(vocab_start_index)
    if not dist.is_initialized() or dist.get_world_size(group=process_group) == 1:
        return local_indices
    world_size = dist.get_world_size(group=process_group)
    gathered_values = [torch.empty_like(local_values) for _ in range(world_size)]
    gathered_indices = [torch.empty_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered_values, local_values, group=process_group)
    dist.all_gather(gathered_indices, local_indices, group=process_group)
    values = torch.stack(gathered_values, dim=0)
    indices = torch.stack(gathered_indices, dim=0)
    global_values = values.max(dim=0).values
    sentinel = torch.full_like(indices, torch.iinfo(indices.dtype).max)
    candidates = torch.where(
        values == global_values.unsqueeze(0),
        indices,
        sentinel,
    )
    return candidates.min(dim=0).values


class DeepseekV4DSparkServingWrapper(DeepseekV4ServingTrainingWrapper):
    """Trainability shell around the vLLM-Ascend DSpark serving model.

    This class deliberately exposes a low-level runtime-query boundary first.
    Anchor packing and cache metadata remain caller-owned so a validation test
    can compare exactly the same context/query layout used by serving.
    """

    def _load_draft_model_class(self) -> type[nn.Module]:
        from verl_speco.backends.vllm_ascend_compat import model_symbol

        return model_symbol("dspark", "DSparkDeepseekV4ForCausalLM")

    def _router_module_names(self) -> tuple[str, ...]:
        return (
            "vllm_ascend.models.deepseek_v4.model",
            "vllm_ascend.models.deepseek_v4",
            "vllm_ascend.models.deepseek_v4_dspark",
        )

    def _install_variant_runtime_adapters(self) -> bool:
        if not self.using_model_internal_router_training_forward:
            raise RuntimeError(
                "The installed vLLM-Ascend runtime exposes no supported "
                "DeepseekV4MoE class for trainable internal-router logits"
            )
        return _install_dspark_dsa_device_operator_training_aliases()

    def __init__(self, config: Any) -> None:
        if _RUNTIME_IMPORT_ERROR is not None:
            raise RuntimeError(
                "DeepSeek V4 DSpark runtime training requires the repository's "
                "DeepSeek V4 Transformers and vLLM-Ascend environment"
            ) from _RUNTIME_IMPORT_ERROR
        self.runtime_contract = DSparkRuntimeContract.from_config(config)
        if self.runtime_contract.bonus_anchor:
            raise ValueError(
                "DeepSeek V4 DSpark training currently supports the default "
                "serving query layout only; dspark_bonus_anchor must be false"
            )
        from vllm_ascend.utils import (
            AscendDeviceType,
            get_ascend_device_type,
        )

        if get_ascend_device_type() == AscendDeviceType.A5:
            raise RuntimeError(
                "DSpark training on Ascend A5 requires an autograd adapter "
                "for kv_compress_epilog context-cache writes; the audited "
                "npu_scatter_nd_update_v2 replay currently supports non-A5 "
                "devices only"
            )
        config.vllm_speculative_config = {
            "method": "dspark",
            "num_speculative_tokens": self.runtime_contract.block_size,
        }
        # The DSpark training path retains custom autograd context and is not
        # compatible with vLLM's serving-time AOT/CUDAGraph wrappers.
        config.vllm_enforce_eager = True
        config.vllm_compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
        }
        super().__init__(config)
        self._training_config = config
        self._cached_shared_context_block_size: int | None = None
        self.confidence_head: nn.Linear | None = None
        if bool(getattr(config, "dspark_enable_confidence_head", False)):
            # In current vLLM-Ascend DSpark, markov_embed is a callable helper,
            # not an nn.Module. Use a real serving-model parameter only to
            # place the training-only head on the matching device and dtype.
            model_parameter = next(self.model.parameters())
            self.confidence_head = nn.Linear(
                self.runtime_contract.hidden_size + self.runtime_contract.markov_rank,
                1,
                device=model_parameter.device,
                dtype=model_parameter.dtype,
            )
        self.using_training_safe_rope = _install_dspark_training_safe_rope()
        self.using_training_dsa_query_replay = (
            _install_dspark_dsa_query_training_replay()
        )
        if not self.using_training_dsa_query_replay:
            raise RuntimeError(
                "The installed vLLM-Ascend DSA runtime exposes no supported "
                "MLA prolog for DSpark query-gradient replay"
            )
        self._local_lm_head_projection_validated = False
        self._local_markov_projection_validated = False
        self._phase_profiler = DSparkPhaseProfiler()

    def set_phase_profiling(self, enabled: bool) -> None:
        self._phase_profiler.reset(enabled=enabled)
        if set_vllm_ascend_backward_profiling is not None:
            set_vllm_ascend_backward_profiling(enabled)

    def profile_phase(self, name: str):
        return self._phase_profiler.record(name)

    def consume_phase_profile(self) -> dict[str, dict[str, float | int]]:
        summary = self._phase_profiler.consume()
        if consume_vllm_ascend_backward_profile is not None:
            internal_summary = consume_vllm_ascend_backward_profile()
            for name, values in internal_summary.items():
                if name in summary:
                    raise RuntimeError(f"Duplicate DSpark phase profile name: {name}")
                summary[name] = values
        return summary

    def combine_target_hidden_states(
        self,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.runtime_contract.target_hidden_size
        if target_hidden_states.shape[-1] != expected:
            raise ValueError(
                "DSpark selected-layer hidden width does not match config: "
                f"got={target_hidden_states.shape[-1]}, expected={expected}"
            )
        return self.model.combine_hidden_states(
            target_hidden_states.to(self.runtime_device())
        )

    def _shared_context_block_size(self) -> int:
        if self._cached_shared_context_block_size is not None:
            return self._cached_shared_context_block_size
        block_sizes = []
        for layer in self.model.model.layers.values():
            swa = layer.self_attn.dsa_attn.swa_cache_layer
            spec = swa.get_kv_cache_spec(self.vllm_config)
            block_sizes.append(
                int(getattr(spec, "block_size", getattr(swa, "block_size", 64)))
            )
        if not block_sizes:
            raise RuntimeError("DSpark model does not expose a draft SWA layer")
        self._cached_shared_context_block_size = math.lcm(*block_sizes)
        return self._cached_shared_context_block_size

    def _build_packed_runtime_metadata(
        self,
        packed: DSparkPackedRuntimeInputs,
    ) -> tuple[dict[str, Any], list[torch.Tensor | None]]:
        """Build the same per-layer SWA metadata used by DSpark drafting."""

        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        device = self.runtime_device()
        context_lengths = packed.context_lengths.to(device=device)
        query_positions = packed.query_positions.reshape(-1).to(device=device)
        query_positions_cpu = packed.query_positions_cpu
        if query_positions_cpu is None:
            query_positions_cpu = query_positions.detach().to("cpu")
        else:
            query_positions_cpu = query_positions_cpu.reshape(-1)
        context_lengths_cpu = packed.context_lengths_cpu
        if context_lengths_cpu is None:
            context_lengths_cpu = context_lengths.detach().to(
                "cpu",
                dtype=torch.int32,
            )
        num_reqs = int(context_lengths.numel())
        query_length = int(packed.query_input_ids.shape[-1])
        num_query_tokens = num_reqs * query_length
        seq_lens_cpu = context_lengths_cpu + query_length
        query_start_loc_cpu = (
            torch.arange(
                num_reqs + 1,
                dtype=torch.int32,
            )
            * query_length
        )
        max_total_length = int(seq_lens_cpu.max().item())
        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        metadata: dict[str, Any] = {}
        context_slot_mappings: list[torch.Tensor | None] = []
        layouts_by_block_size: dict[int, DSparkPagedCacheLayout] = {}

        for layer in self.model.model.layers.values():
            attn = layer.self_attn
            module = attn.dsa_attn
            swa = module.swa_cache_layer
            swa_spec = swa.get_kv_cache_spec(self.vllm_config)
            block_size = int(
                getattr(swa_spec, "block_size", getattr(swa, "block_size", 64))
            )
            layout = layouts_by_block_size.get(block_size)
            if layout is None:
                # DSpark cache slots are layer-local, but their request/block
                # geometry is identical for every layer using this block size.
                # Reuse the immutable layout tensors within one anchor chunk.
                if packed.shared_context_block_size is None:
                    layout = build_paged_cache_layout(
                        context_lengths,
                        query_length=query_length,
                        block_size=block_size,
                        context_lengths_cpu=context_lengths_cpu,
                    )
                else:
                    layout = build_shared_paged_cache_layout(
                        context_lengths,
                        packed.batch_indices.to(device=device),
                        query_length=query_length,
                        block_size=block_size,
                        shared_block_size=packed.shared_context_block_size,
                        context_lengths_cpu=context_lengths_cpu,
                    )
                layouts_by_block_size[block_size] = layout
            self._ensure_dummy_dsa_kv_cache_for_layer(
                attn_module=module,
                batch_shape=(num_reqs, max_total_length),
                device=device,
                swa_num_blocks=(
                    layout.num_blocks
                    if packed.shared_context_block_size is not None
                    else None
                ),
            )
            common_metadata = AscendCommonAttentionMetadata(
                query_start_loc=layout.query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=layout.seq_lens,
                _seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                num_computed_tokens_cpu=context_lengths_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_query_tokens,
                max_query_len=query_length,
                max_seq_len=max_total_length,
                block_table_tensor=layout.block_table,
                slot_mapping=layout.query_slot_mapping,
                causal=False,
                is_prefilling=torch.zeros(
                    num_reqs,
                    dtype=torch.bool,
                    device=device,
                ),
                num_input_tokens=num_query_tokens,
                actual_seq_lengths_q=actual_seq_lengths_q,
                positions=query_positions,
                positions_cpu=query_positions_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                decode_token_per_req=query_length,
            )
            prefix = module.prefix
            builder_key = (
                "dspark",
                prefix,
                block_size,
                str(device),
            )
            builder = self._dsa_metadata_builder_cache.get(builder_key)
            if builder is None:
                builder = AscendDSAMetadataBuilder(
                    kv_cache_spec=swa_spec,
                    layer_names=[f"{prefix}.swa_cache"],
                    vllm_config=self.vllm_config,
                    device=device,
                )
                if self.cache_dsa_metadata_structures:
                    self._dsa_metadata_builder_cache[builder_key] = builder
            built = builder.build_for_drafting(
                common_metadata,
                draft_index=1,
                prefill_ratio_to_sas_metadata={},
                decode_ratio_to_sas_metadata={},
                common_ratio_to_sas_metadata={},
                block_size=block_size,
            )
            register_drafting_backward_host_metadata(
                built,
                common_metadata,
                int(getattr(swa_spec, "compress_ratio", 1)),
                enabled=self.reuse_dsa_backward_host_metadata,
            )
            metadata[f"{prefix}.swa_cache"] = built
            context_slot_mappings.append(layout.context_slot_mapping)
        if not metadata:
            raise RuntimeError("DSpark model does not expose a draft SWA layer")
        return metadata, context_slot_mappings

    def forward_packed_runtime(
        self,
        packed: DSparkPackedRuntimeInputs,
        *,
        context_states_are_combined: bool = False,
    ) -> torch.Tensor:
        """Run one packed anchor batch through the imported serving model."""

        with self.profile_phase("metadata"):
            metadata, context_slot_mappings = self._build_packed_runtime_metadata(
                packed
            )
        hidden_states = self.forward_runtime_queries(
            query_input_ids=packed.query_input_ids,
            query_positions=packed.query_positions,
            target_hidden_states=packed.context_hidden_states,
            context_positions=packed.context_positions,
            context_slot_mapping=context_slot_mappings,
            attn_metadata=metadata,
            target_hidden_states_combined=context_states_are_combined,
        )
        return hidden_states.reshape(
            packed.query_input_ids.shape[0],
            packed.query_input_ids.shape[1],
            -1,
        )

    def forward_runtime_queries(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None],
        attn_metadata: dict[str, Any],
        target_hidden_states_combined: bool = False,
    ) -> torch.Tensor:
        """Execute the same context-KV/query split as the serving proposer."""

        with self.profile_phase("runtime_input"):
            self._ensure_vllm_ascend_tp_equivalent(stage="dspark_forward")
            device = self.runtime_device()
            query_input_ids = query_input_ids.reshape(-1).to(device)
            query_positions = query_positions.reshape(-1).to(device)
            context_positions = context_positions.reshape(-1).to(device)
        with self.profile_phase("context_combine"):
            if target_hidden_states_combined:
                context_states = target_hidden_states.reshape(
                    -1,
                    self.runtime_contract.hidden_size,
                ).to(device)
            else:
                context_states = self.combine_target_hidden_states(
                    target_hidden_states.reshape(
                        -1,
                        self.runtime_contract.target_hidden_size,
                    )
                )
        num_query_tokens = int(query_input_ids.numel())
        with self._set_current_vllm_config(self.vllm_config):
            from verl_speco.backends.vllm_ascend_compat import (
                ascend_forward_context as set_ascend_forward_context,
            )

            with self.profile_phase("context_kv"):
                self.model.precompute_and_store_context_kv(
                    context_states,
                    context_positions,
                    context_slot_mapping,
                )
            with (
                set_ascend_forward_context(
                    attn_metadata=attn_metadata,
                    vllm_config=self.vllm_config,
                    num_tokens=num_query_tokens,
                    model_instance=self.model,
                    is_draft_model=True,
                    input_ids=query_input_ids,
                ),
                self.profile_phase("draft_model"),
            ):
                return self.model(
                    input_ids=query_input_ids,
                    positions=query_positions,
                )

    def compute_base_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        with self._set_current_vllm_config(self.vllm_config):
            base_logits = self.model.compute_logits(flat_hidden)
        return base_logits.reshape(*hidden_states.shape[:-1], -1)

    def compute_local_base_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        vocab_start_index: int,
        process_group=None,
        validate_projection: bool = False,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        normalized = self.model.model.norm(flat_hidden)
        normalized = all_reduce_gradient(normalized, process_group)
        local_logits = F.linear(
            normalized,
            self.model.lm_head.weight,
            getattr(self.model.lm_head, "bias", None),
        )
        local_logits = self._apply_local_logits_processor(
            local_logits,
            self.model.logits_processor,
        )
        valid_size = min(
            int(local_logits.shape[-1]),
            self.runtime_contract.vocab_size - int(vocab_start_index),
        )
        local_logits = local_logits[..., :valid_size].reshape(
            *hidden_states.shape[:-1],
            -1,
        )
        if validate_projection and not self._local_lm_head_projection_validated:
            with torch.no_grad():
                reference = self.compute_base_logits(hidden_states.detach())
                expected = reference[
                    ...,
                    vocab_start_index : (vocab_start_index + local_logits.shape[-1]),
                ]
                torch.testing.assert_close(
                    local_logits.detach(),
                    expected,
                    rtol=2e-3,
                    atol=2e-3,
                    msg=(
                        "TP-local DSpark LM-head projection does not match "
                        "the serving compute_logits path"
                    ),
                )
            self._local_lm_head_projection_validated = True
        return local_logits

    @staticmethod
    def _apply_local_logits_processor(
        logits: torch.Tensor,
        logits_processor: nn.Module,
    ) -> torch.Tensor:
        soft_cap = getattr(logits_processor, "soft_cap", None)
        if soft_cap is not None:
            logits = torch.tanh(logits / float(soft_cap)) * float(soft_cap)
        scale = float(getattr(logits_processor, "scale", 1.0))
        return logits if scale == 1.0 else logits * scale

    @staticmethod
    def _resolve_markov_logits_processor(
        markov_head: nn.Module,
        serving_model: nn.Module,
    ) -> nn.Module:
        """Resolve version-dependent ownership of Markov logit processing."""

        processor = getattr(markov_head, "logits_processor", None)
        if processor is not None:
            return processor
        processor = getattr(serving_model, "logits_processor", None)
        if processor is not None:
            return processor
        raise RuntimeError(
            "The installed DSpark runtime exposes no logits processor on "
            f"either {type(markov_head).__name__} or "
            f"{type(serving_model).__name__}; cannot reproduce the serving "
            "Markov projection semantics"
        )

    def compute_markov_bias(
        self,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        flat_previous = previous_token_ids.reshape(-1).to(self.runtime_device())
        with self._set_current_vllm_config(self.vllm_config):
            markov_bias = self.model.markov_bias(self.model.markov_embed(flat_previous))
        return markov_bias.reshape(*previous_token_ids.shape, -1)

    def compute_local_markov_bias(
        self,
        previous_token_ids: torch.Tensor,
        *,
        vocab_start_index: int,
        local_vocab_size: int,
        process_group=None,
        validate_projection: bool = False,
    ) -> torch.Tensor:
        flat_previous = previous_token_ids.reshape(-1).to(self.runtime_device())
        with self._set_current_vllm_config(self.vllm_config):
            markov_features = self.model.markov_embed(flat_previous)
        markov_features = all_reduce_gradient(markov_features, process_group)
        markov_head = self.model.model.markov_head
        projection = markov_head.markov_w2
        local_bias = _local_markov_linear(
            markov_features,
            projection.weight,
            getattr(projection, "bias", None),
            vocab_start_index=vocab_start_index,
            local_vocab_size=local_vocab_size,
            vocab_size=self.runtime_contract.vocab_size,
            process_group=process_group,
        )
        local_bias = self._apply_local_logits_processor(
            local_bias,
            self._resolve_markov_logits_processor(markov_head, self.model),
        )[..., :local_vocab_size]
        local_bias = local_bias.reshape(*previous_token_ids.shape, -1)
        if validate_projection and not self._local_markov_projection_validated:
            with torch.no_grad():
                reference = self.compute_markov_bias(previous_token_ids.detach())
                expected = reference[
                    ...,
                    vocab_start_index : vocab_start_index + local_vocab_size,
                ]
                torch.testing.assert_close(
                    local_bias.detach(),
                    expected,
                    rtol=2e-3,
                    atol=2e-3,
                    msg=(
                        "TP-local DSpark Markov projection does not match "
                        "the serving markov_bias path"
                    ),
                )
            self._local_markov_projection_validated = True
        return local_bias

    def apply_markov_bias(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape[:-1] != previous_token_ids.shape:
            raise ValueError(
                "DSpark base logits and Markov tokens disagree: "
                f"logits={base_logits.shape}, previous={previous_token_ids.shape}"
            )
        markov_bias = self.compute_markov_bias(previous_token_ids)
        if markov_bias.shape != base_logits.shape:
            raise RuntimeError(
                "DSpark Markov bias and base logits disagree: "
                f"bias={markov_bias.shape}, logits={base_logits.shape}"
            )
        return base_logits + markov_bias

    def compute_block_logits(
        self,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[:-1] != previous_token_ids.shape:
            raise ValueError(
                "DSpark hidden states and Markov teacher tokens disagree: "
                f"hidden={hidden_states.shape}, previous={previous_token_ids.shape}"
            )
        return self.apply_markov_bias(
            self.compute_base_logits(hidden_states),
            previous_token_ids,
        )

    @torch.no_grad()
    def compute_markov_rollout(
        self,
        base_logits: torch.Tensor,
        seed_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a block exactly as serving does after the DFlash forward.

        DFlash produces every position's base logits in parallel.  The Markov
        head is then applied autoregressively: the prediction at position k is
        the previous-token input used to score position k + 1.
        """

        if base_logits.ndim != 3:
            raise ValueError(
                "DSpark rollout expects base logits with shape [requests, block, vocab]"
            )
        if seed_token_ids.shape != base_logits.shape[:1]:
            raise ValueError(
                "DSpark rollout seed must contain one token per request: "
                f"seeds={seed_token_ids.shape}, logits={base_logits.shape}"
            )
        previous = seed_token_ids.to(
            device=self.runtime_device(),
            dtype=torch.long,
        )
        predictions = []
        for position in range(base_logits.shape[1]):
            position_logits = self.apply_markov_bias(
                base_logits[:, position, :],
                previous,
            )
            previous = position_logits.argmax(dim=-1)
            predictions.append(previous)
        return torch.stack(predictions, dim=-1)

    @torch.no_grad()
    def compute_vocab_parallel_markov_rollout(
        self,
        base_local_logits: torch.Tensor,
        seed_token_ids: torch.Tensor,
        *,
        vocab_start_index: int,
        process_group=None,
    ) -> torch.Tensor:
        """Decode greedy DSpark prefixes without gathering vocabulary logits."""

        if base_local_logits.ndim != 3:
            raise ValueError(
                "DSpark local rollout expects [requests, block, local_vocab]"
            )
        if seed_token_ids.shape != base_local_logits.shape[:1]:
            raise ValueError(
                "DSpark local rollout seed must contain one token per request"
            )
        previous = seed_token_ids.to(
            device=self.runtime_device(),
            dtype=torch.long,
        )
        predictions = []
        local_vocab_size = int(base_local_logits.shape[-1])
        for position in range(base_local_logits.shape[1]):
            markov_bias = self.compute_local_markov_bias(
                previous,
                vocab_start_index=vocab_start_index,
                local_vocab_size=local_vocab_size,
                process_group=process_group,
            )
            previous = distributed_vocab_argmax(
                base_local_logits[:, position, :] + markov_bias,
                vocab_start_index=vocab_start_index,
                process_group=process_group,
            )
            predictions.append(previous)
        return torch.stack(predictions, dim=-1)

    @staticmethod
    def compute_cumulative_prefix_counts(
        predicted_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count exact prefixes, not independent per-position matches."""

        if predicted_ids.shape != target_ids.shape:
            raise ValueError(
                "DSpark rollout predictions and targets disagree: "
                f"predictions={predicted_ids.shape}, targets={target_ids.shape}"
            )
        if eval_mask.shape != target_ids.shape:
            raise ValueError(
                "DSpark rollout mask and targets disagree: "
                f"mask={eval_mask.shape}, targets={target_ids.shape}"
            )
        # Ascend's cumprod kernel does not accept DT_BOOL. Use the same
        # int32 -> cumprod -> bool pattern as build_block_targets above.
        valid_prefix = eval_mask.to(torch.int32).cumprod(dim=-1).to(torch.bool)
        correct_prefix = (
            predicted_ids.eq(target_ids).to(torch.int32).cumprod(dim=-1).to(torch.bool)
        )
        correct_counts = (correct_prefix & valid_prefix).sum(
            dim=0,
            dtype=torch.float32,
        )
        total_counts = valid_prefix.sum(dim=0, dtype=torch.float32)
        return correct_counts, total_counts

    def compute_confidence_logits(
        self,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.confidence_head is None:
            return None
        with self._set_current_vllm_config(self.vllm_config):
            return compute_dspark_confidence_logits(
                hidden_states,
                previous_token_ids,
                markov_embedding=self.model.markov_embed,
                confidence_head=self.confidence_head,
            )

    def prepare_training_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        raw_mask: torch.Tensor,
        positions: torch.Tensor,
        num_anchors: int,
        generator: torch.Generator | None = None,
    ) -> tuple[DSparkBlockTargets, DSparkRuntimeBlockInputs]:
        """Select anchors once so training can execute them in chunks."""

        if input_ids.shape != raw_mask.shape:
            raise ValueError("DSpark input_ids and raw_mask shapes disagree")
        if positions.shape != input_ids.shape:
            raise ValueError("DSpark positions and input_ids shapes disagree")
        configured_spacing = getattr(
            self._training_config,
            "dspark_anchor_min_spacing",
            None,
        )
        anchor_min_spacing = (
            self.runtime_contract.block_size
            if configured_spacing is None
            else int(configured_spacing)
        )
        if anchor_min_spacing <= 0:
            raise ValueError(
                f"dspark_anchor_min_spacing must be positive, got {anchor_min_spacing}"
            )
        # SpeCo supervision: 1 enables loss, 0 marks context/padding.
        targets = build_block_targets(
            input_ids,
            raw_mask,
            block_size=self.runtime_contract.block_size,
            num_anchors=num_anchors,
            generator=generator,
            target_offset=self.runtime_contract.target_offset,
            min_anchor_spacing=anchor_min_spacing,
        )
        runtime_inputs = build_runtime_block_inputs(
            input_ids,
            targets,
            noise_token_id=int(self.runtime_contract.noise_token_id),
            positions=positions,
        )
        return targets, runtime_inputs

    def compute_training_batch(
        self,
        *,
        input_ids: torch.Tensor,
        raw_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_head_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        target_lm_head: nn.Module,
        num_anchors: int,
        generator: torch.Generator | None = None,
        prepared_targets: DSparkBlockTargets | None = None,
        prepared_runtime_inputs: DSparkRuntimeBlockInputs | None = None,
        active_anchors: torch.Tensor | None = None,
        active_context_lengths_cpu: torch.Tensor | None = None,
        active_query_positions_cpu: torch.Tensor | None = None,
        combined_target_hidden_states: torch.Tensor | None = None,
        cross_entropy_weight: float = 0.1,
        distribution_l1_weight: float = 0.9,
        distribution_kl_weight: float = 0.0,
        confidence_weight: float = 0.0,
        loss_decay_gamma: float | None = None,
        compute_prefix_accuracy: bool = False,
        compute_distribution_metrics: bool = False,
        vocab_parallel_loss: bool = False,
        validate_vocab_parallel_projection: bool = False,
    ) -> DSparkTrainingOutput:
        """Execute a serving-equivalent DSpark teacher-forcing batch."""

        device = self.runtime_device()
        input_ids = input_ids.to(device)
        raw_mask = raw_mask.to(device)
        target_hidden_states = target_hidden_states.to(device)
        target_head_hidden_states = target_head_hidden_states.to(device)
        positions = positions.to(device)
        if input_ids.shape != raw_mask.shape:
            raise ValueError("DSpark input_ids and raw_mask shapes disagree")
        if positions.shape != input_ids.shape:
            raise ValueError("DSpark positions and input_ids shapes disagree")
        if target_hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "DSpark selected-layer hidden states do not align with tokens"
            )
        if combined_target_hidden_states is not None:
            combined_target_hidden_states = combined_target_hidden_states.to(device)
            expected_combined_shape = (
                *input_ids.shape,
                self.runtime_contract.hidden_size,
            )
            if combined_target_hidden_states.shape != expected_combined_shape:
                raise ValueError(
                    "DSpark combined target hidden states have the wrong shape: "
                    f"got={combined_target_hidden_states.shape}, "
                    f"expected={expected_combined_shape}"
                )
        if target_head_hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "DSpark target-head hidden states do not align with tokens"
            )

        if (prepared_targets is None) != (prepared_runtime_inputs is None):
            raise ValueError(
                "DSpark prepared_targets and prepared_runtime_inputs must be "
                "provided together"
            )
        if prepared_targets is None:
            targets, runtime_inputs = self.prepare_training_inputs(
                input_ids=input_ids,
                raw_mask=raw_mask,
                positions=positions,
                num_anchors=num_anchors,
                generator=generator,
            )
        else:
            targets = prepared_targets
            runtime_inputs = prepared_runtime_inputs

        if active_anchors is None:
            active = torch.nonzero(targets.block_keep_mask, as_tuple=False)
        else:
            if active_anchors.ndim != 2 or active_anchors.shape[1] != 2:
                raise ValueError(
                    "DSpark active_anchors must have shape [num_anchors, 2]"
                )
            active = active_anchors.to(device=device)
        if active.numel() == 0:
            raise ValueError("DSpark batch does not contain a valid training anchor")
        states_for_packing = (
            combined_target_hidden_states
            if combined_target_hidden_states is not None
            else target_hidden_states
        )
        with self.profile_phase("pack"):
            packed = pack_runtime_block_inputs(
                states_for_packing,
                targets,
                runtime_inputs,
                positions=positions,
                active_anchors=active,
                active_context_lengths_cpu=active_context_lengths_cpu,
                active_query_positions_cpu=active_query_positions_cpu,
                validate_active_anchors=False,
                shared_context_block_size=(
                    self._shared_context_block_size()
                    if bool(
                        getattr(
                            self._training_config,
                            "dspark_share_context_kv",
                            False,
                        )
                    )
                    else None
                ),
            )
        draft_hidden_states = self.forward_packed_runtime(
            packed,
            context_states_are_combined=(combined_target_hidden_states is not None),
        )
        with self.profile_phase("draft_logits"):
            vocab_start_index = 0
            vocab_process_group = None
            base_local_logits = None
            if vocab_parallel_loss:
                if not callable(getattr(target_lm_head, "local_logits", None)):
                    raise RuntimeError(
                        "DSpark vocab_parallel_loss requires ParallelLMHead"
                    )
                vocab_process_group = target_lm_head.process_group
                vocab_rank = (
                    dist.get_rank(group=vocab_process_group)
                    if dist.is_initialized()
                    else 0
                )
                shard_indices = getattr(
                    self.model.lm_head,
                    "shard_indices",
                    None,
                )
                vocab_start_index = int(
                    getattr(
                        shard_indices,
                        "org_vocab_start_index",
                        vocab_rank * int(self.model.lm_head.weight.shape[0]),
                    )
                )
                base_local_logits = self.compute_local_base_logits(
                    draft_hidden_states,
                    vocab_start_index=vocab_start_index,
                    process_group=vocab_process_group,
                    validate_projection=validate_vocab_parallel_projection,
                )
                markov_bias = self.compute_local_markov_bias(
                    packed.previous_token_ids,
                    vocab_start_index=vocab_start_index,
                    local_vocab_size=int(base_local_logits.shape[-1]),
                    process_group=vocab_process_group,
                    validate_projection=validate_vocab_parallel_projection,
                )
                draft_logits = base_local_logits + markov_bias
                base_logits = None
            else:
                base_logits = self.compute_base_logits(draft_hidden_states)
                draft_logits = self.apply_markov_bias(
                    base_logits,
                    packed.previous_token_ids,
                )
        prefix_correct_counts = None
        prefix_total_counts = None
        with self.profile_phase("confidence"):
            confidence_logits = self.compute_confidence_logits(
                draft_hidden_states,
                packed.previous_token_ids,
            )
        if confidence_weight > 0 and confidence_logits is None:
            raise ValueError(
                "DSpark confidence_weight is positive but "
                "dspark_enable_confidence_head is false"
            )

        with self.profile_phase("teacher_gather"):
            packed_teacher = gather_active_teacher_hidden_states(
                target_head_hidden_states,
                targets.label_indices - 1,
                active,
            )
        loss_targets = DSparkBlockTargets(
            anchor_positions=packed.anchor_positions.unsqueeze(-1),
            block_keep_mask=torch.ones(
                (packed.anchor_positions.numel(), 1),
                dtype=torch.bool,
                device=device,
            ),
            label_indices=targets.label_indices[active[:, 0], active[:, 1]].unsqueeze(
                1
            ),
            target_ids=packed.target_ids.unsqueeze(1),
            previous_token_ids=packed.previous_token_ids.unsqueeze(1),
            eval_mask=packed.eval_mask.unsqueeze(1),
        )
        target_logits = None
        target_local_logits = None
        with self.profile_phase("teacher_logits"):
            if vocab_parallel_loss:
                with torch.no_grad():
                    target_local_logits = target_lm_head.local_logits(
                        packed_teacher
                    ).float()
                _local_vocab_size = int(target_local_logits.shape[-1])
                draft_local_logits = draft_logits
                if draft_local_logits.shape != target_local_logits.shape:
                    raise RuntimeError(
                        "DSpark draft logits do not align with target vocab shard: "
                        f"draft={draft_local_logits.shape}, "
                        f"target={target_local_logits.shape}, "
                        f"start={vocab_start_index}"
                    )
            else:
                with torch.no_grad():
                    target_logits = target_lm_head(packed_teacher).float()
        if compute_prefix_accuracy:
            with self.profile_phase("prefix_rollout"):
                rollout_ids = (
                    self.compute_vocab_parallel_markov_rollout(
                        base_local_logits.detach(),
                        packed.previous_token_ids[:, 0],
                        vocab_start_index=vocab_start_index,
                        process_group=vocab_process_group,
                    )
                    if base_local_logits is not None
                    else self.compute_markov_rollout(
                        base_logits.detach(),
                        packed.previous_token_ids[:, 0],
                    )
                )
                # Speculative-drafting accuracy measures agreement with the
                # target model's greedy token, not ordinary next-token accuracy
                # against the dataset token IDs used by the CE term.
                target_greedy_ids = (
                    distributed_vocab_argmax(
                        target_local_logits,
                        vocab_start_index=vocab_start_index,
                        process_group=vocab_process_group,
                    )
                    if target_local_logits is not None
                    else target_logits.argmax(dim=-1)
                )
                prefix_correct_counts, prefix_total_counts = (
                    self.compute_cumulative_prefix_counts(
                        rollout_ids,
                        target_greedy_ids,
                        packed.eval_mask,
                    )
                )
        loss_kwargs = {
            "confidence_logits": (
                None if confidence_logits is None else confidence_logits.unsqueeze(1)
            ),
            "loss_decay_gamma": loss_decay_gamma,
            "cross_entropy_weight": cross_entropy_weight,
            "distribution_l1_weight": distribution_l1_weight,
            "distribution_kl_weight": distribution_kl_weight,
            "confidence_weight": confidence_weight,
            "compute_distribution_metrics": compute_distribution_metrics,
        }
        with self.profile_phase("loss"):
            loss = (
                compute_vocab_parallel_dspark_loss(
                    draft_local_logits.unsqueeze(1),
                    loss_targets,
                    target_local_logits=target_local_logits.unsqueeze(1),
                    vocab_start_index=vocab_start_index,
                    process_group=vocab_process_group,
                    **loss_kwargs,
                )
                if target_local_logits is not None
                else compute_dspark_loss(
                    draft_logits.unsqueeze(1),
                    loss_targets,
                    target_logits=target_logits.unsqueeze(1),
                    **loss_kwargs,
                )
            )
        return DSparkTrainingOutput(
            loss=loss,
            draft_logits=draft_logits,
            targets=targets,
            packed=packed,
            prefix_correct_counts=prefix_correct_counts,
            prefix_total_counts=prefix_total_counts,
        )

    def freeze_shared_token_weights(self) -> None:
        """Freeze checkpoint-loaded token weights, not live Target aliases."""

        self.model.model.embed_tokens.requires_grad_(False)
        self.model.lm_head.requires_grad_(False)

    def freeze_params(self, cfg: Any) -> None:
        explicitly_frozen = set(cfg.train.params_to_freeze)
        freeze_mlp = bool(getattr(cfg.train, "dspark_freeze_mlp", False))
        frozen_mlp_tensors = 0
        frozen_mlp_parameters = 0
        for name, parameter in self.named_parameters():
            shared = "embed_tokens" in name or name.startswith("model.lm_head.")
            routing_state = name.endswith("e_score_correction_bias")
            # Newer serving runtimes own an additional confidence head. The
            # training loss uses this wrapper's optional confidence_head,
            # so retain the serving weights without optimizing an unused head.
            serving_confidence = name.startswith("model.model.confidence_head.")
            matched_mlp = freeze_mlp and "mlp" in name.lower()
            trainable = (
                parameter.is_floating_point()
                and not shared
                and not routing_state
                and not serving_confidence
                and name not in explicitly_frozen
                and not matched_mlp
            )
            parameter.requires_grad_(trainable)
            if matched_mlp:
                frozen_mlp_tensors += 1
                frozen_mlp_parameters += parameter.numel()
        self.freeze_shared_token_weights()
        self.dspark_mlp_freeze_summary = {
            "enabled": freeze_mlp,
            "tensors": frozen_mlp_tensors,
            "parameters": frozen_mlp_parameters,
        }

    def forward(self, *args, **kwargs):
        if not args and "target_head_hidden_states" in kwargs:
            return self.compute_training_batch(**kwargs)
        raise RuntimeError(
            "DeepSeek V4 DSpark requires packed context/query metadata. Use "
            "compute_training_batch() or forward_runtime_queries() with the "
            "serving-equivalent cache layout."
        )

    def load_weights(self, path: str, mode: str = "full") -> dict[str, Any]:
        if mode == "no":
            return {"loaded": 0, "skipped": 0, "updated_keys": []}
        if mode != "full":
            raise ValueError(
                "DeepSeek V4 DSpark supports load_weights='full' or 'no'; "
                f"partial mode {mode!r} cannot initialize the block drafter"
            )
        weights: list[tuple[str, torch.Tensor]] = []
        shared_embed: torch.Tensor | None = None
        shared_head: torch.Tensor | None = None
        has_dedicated_embed = False
        has_dedicated_head = False
        for root, _, files in os.walk(path):
            for filename in files:
                if not filename.endswith(".safetensors"):
                    continue
                with safe_open(
                    os.path.join(root, filename),
                    framework="pt",
                    device="cpu",
                ) as handle:
                    for name in handle.keys():  # noqa: SIM118 -- safetensors/runtime key API is explicit
                        if name.startswith("mtp."):
                            weights.append((name, handle.get_tensor(name)))
                            has_dedicated_embed |= name == "mtp.0.embed.weight"
                            has_dedicated_head |= bool(
                                re.fullmatch(r"mtp\.\d+\.head\.weight", name)
                            )
                        elif name in {
                            "model.embed_tokens.weight",
                            "embed_tokens.weight",
                            "embed.weight",
                        }:
                            shared_embed = handle.get_tensor(name)
                        elif name in {
                            "lm_head.weight",
                            "model.lm_head.weight",
                            "head.weight",
                        }:
                            shared_head = handle.get_tensor(name)
        if not has_dedicated_embed and shared_embed is not None:
            weights.append(("embed.weight", shared_embed))
        if not has_dedicated_head and shared_head is not None:
            weights.append(("head.weight", shared_head))
        token_weight_sources = {
            "embedding": "dedicated_mtp" if has_dedicated_embed else "target_fallback",
            "lm_head": "dedicated_mtp" if has_dedicated_head else "target_fallback",
            "checkpoint_path": os.path.realpath(path),
            "live_target_alias": False,
        }

        with self._set_current_vllm_config(self.vllm_config):
            loaded_params = self.model.load_weights(weights)
            required_groups = {
                "shared_embedding": {"model.embed_tokens.weight"},
                "shared_lm_head": {"lm_head.weight"},
                "main_projection": {
                    name
                    for name in loaded_params
                    if ".main_proj." in name or ".main_norm." in name
                },
                "markov_head": {
                    name for name in loaded_params if ".markov_head." in name
                },
            }
            missing_groups = [
                group
                for group, names in required_groups.items()
                if not names or names.isdisjoint(loaded_params)
            ]
            configured_draft_layers = int(self.runtime_contract.num_draft_layers)
            loaded_layer_prefixes = {
                name.rsplit(".", 1)[0]
                for name in loaded_params
                if name.startswith("model.layers.")
            }
            runtime_layer_indices = tuple(self.model.model.layers.keys())
            missing_layers = [
                layer_index
                for layer_index in runtime_layer_indices
                if not any(
                    prefix.startswith(f"model.layers.{layer_index}.")
                    for prefix in loaded_layer_prefixes
                )
            ]
            if missing_groups or missing_layers:
                raise RuntimeError(
                    "DSpark checkpoint did not initialize every required "
                    "serving component: "
                    f"missing_groups={missing_groups}, "
                    f"missing_runtime_layers={missing_layers}, "
                    f"configured_draft_layers={configured_draft_layers}"
                )
            self._dequantize_weights_for_training()
            self._validate_loaded_training_weight_format()
            self._process_weights_after_loading()
            self.using_tp_sharded_eh_projections = _install_tp_sharded_eh_projections(
                self.model
            )
            self.using_fused_eh_projection_collective = (
                _install_fused_eh_projection_collective(self.model)
            )
            self.using_tp_sharded_attention_projections = (
                _install_tp_sharded_attention_projections(self.model)
            )
        self._keep_hc_parameters_float32()
        self.freeze_shared_token_weights()
        self.token_weight_sources = token_weight_sources
        self._ensure_vllm_ascend_tp_equivalent(
            stage="dspark_load_weights",
            force=True,
        )
        return {
            "loaded": len(loaded_params),
            "skipped": max(len(weights) - len(loaded_params), 0),
            "updated_keys": sorted(loaded_params),
            "checkpoint_format": self.mtp_checkpoint_format,
            "checkpoint_format_report": self.mtp_checkpoint_format_report,
            "token_weight_sources": self.token_weight_sources,
        }
