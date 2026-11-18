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

# Adapted from training_common/modeling/lm_head.py.
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from verl_speco.backends.vllm_ascend_parallel import is_main_process
from verl_speco.checkpoint_tensor import _load_checkpoint_tensor


class DS4TargetLogitsHead(nn.Module):
    """DeepSeek V4 target output path without instantiating the target model."""

    def __init__(
        self,
        *,
        hc_head_fn: torch.Tensor,
        hc_head_base: torch.Tensor,
        hc_head_scale: torch.Tensor,
        norm_weight: torch.Tensor,
        lm_head: nn.Module,
        hidden_size: int,
        hc_mult: int,
        hc_eps: float,
        rms_norm_eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.hc_eps = float(hc_eps)
        self.rms_norm_eps = float(rms_norm_eps)
        self.dtype = dtype
        self.original_vocab_size = lm_head.original_vocab_size

        self.hc_head_fn = nn.Parameter(hc_head_fn, requires_grad=False)
        self.hc_head_base = nn.Parameter(hc_head_base, requires_grad=False)
        self.hc_head_scale = nn.Parameter(hc_head_scale, requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.lm_head = lm_head

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected_width = self.hc_mult * self.hidden_size
        if hidden_states.shape[-1] != expected_width:
            raise ValueError(
                "DS4 teacher logits require target pre-hc_head residual states: "
                f"expected last dimension {expected_width}, got "
                f"{hidden_states.shape[-1]}"
            )

        prefix_shape = hidden_states.shape[:-1]
        x = hidden_states.reshape(-1, expected_width).float()
        reciprocal_rms = torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
        )
        mixes = F.linear(x, self.hc_head_fn.float()) * reciprocal_rms
        coefficients = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float() + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        x = x.view(-1, self.hc_mult, self.hidden_size)
        hidden_states = (coefficients.unsqueeze(-1) * x).sum(dim=1)

        reciprocal_rms = torch.rsqrt(
            hidden_states.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
        )
        hidden_states = (hidden_states * reciprocal_rms * self.norm_weight.float()).to(
            self.dtype
        )
        return hidden_states.view(*prefix_shape, self.hidden_size)

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.lm_head, "local_logits"):
            raise RuntimeError(
                "DS4 vocab-parallel KL requires a sharded target LM head"
            )
        return self.lm_head.local_logits(self.project_hidden_states(hidden_states))

    @property
    def process_group(self):
        return getattr(self.lm_head, "process_group", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.project_hidden_states(hidden_states))


def _find_safetensor_weights(
    model_name_or_path: str,
    candidates_by_name: dict[str, tuple[str, ...]],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    candidate_keys = {
        key for candidates in candidates_by_name.values() for key in candidates
    }
    locations: dict[str, str] = {}
    for filename in sorted(os.listdir(model_name_or_path)):
        if not filename.endswith(".safetensors"):
            continue
        path = os.path.join(model_name_or_path, filename)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in set(handle.keys()) & candidate_keys:
                if key in locations:
                    raise ValueError(
                        f"Duplicate target output weight {key!r} in "
                        f"{locations[key]!r} and {path!r}"
                    )
                locations[key] = path

    found: dict[str, tuple[str, str]] = {}
    for logical_name, candidates in candidates_by_name.items():
        for key in candidates:
            if key in locations:
                found[logical_name] = (locations[key], key)
                break

    missing = sorted(set(candidates_by_name) - set(found))
    if missing:
        details = ", ".join(
            f"{name}={list(candidates_by_name[name])}" for name in missing
        )
        raise ValueError(
            "Could not find the complete DS4 target output head; refusing to "
            f"fall back to MTP output weights. Missing: {details}"
        )

    tensors: dict[str, torch.Tensor] = {}
    selected_keys: dict[str, str] = {}
    for logical_name, (path, key) in found.items():
        indexed = any(
            name.endswith(".index.json") for name in os.listdir(model_name_or_path)
        )
        standard = os.path.isfile(os.path.join(model_name_or_path, "model.safetensors"))
        if indexed or standard:
            # Reuse SpeCo's checkpoint index/key handling for normal HF layouts.
            tensors[logical_name] = _load_checkpoint_tensor(model_name_or_path, key)
        else:
            # Preserve support for legacy unindexed multi-shard captures.
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensors[logical_name] = handle.get_tensor(key)
        selected_keys[logical_name] = key
    return tensors, selected_keys


class DistributedParallelLMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, process_group=None):
        ctx.save_for_backward(weight)
        ctx.bias_is_none = bias is None
        ctx.process_group = process_group

        if not dist.is_initialized() or dist.get_world_size(group=process_group) == 1:
            return F.linear(x, weight, bias)

        world_size = dist.get_world_size(group=process_group)

        original_shape = x.shape
        x_flat = x.contiguous().view(-1, original_shape[-1])

        local_N = torch.tensor([x_flat.size(0)], dtype=torch.long, device=x.device)
        gathered_Ns = [
            torch.empty(1, dtype=torch.long, device=x.device) for _ in range(world_size)
        ]

        dist.all_gather(gathered_Ns, local_N, group=process_group)

        max_N = max(n.item() for n in gathered_Ns)

        ctx.max_N = max_N
        ctx.local_N = local_N.item()
        ctx.original_shape = original_shape

        if x_flat.size(0) < max_N:
            x_flat_padded = F.pad(x_flat, (0, 0, 0, max_N - x_flat.size(0)))
        else:
            x_flat_padded = x_flat

        gathered_x = [
            torch.empty((max_N, x_flat.size(-1)), dtype=x.dtype, device=x.device)
            for _ in range(world_size)
        ]

        dist.all_gather(gathered_x, x_flat_padded, group=process_group)
        X_full = torch.stack(gathered_x, dim=0)
        Y_local = F.linear(X_full, weight, bias)

        received_Y = [torch.empty_like(Y_local[0]) for _ in range(world_size)]
        send_Y = [Y_local[i].contiguous() for i in range(world_size)]

        dist.all_to_all(received_Y, send_Y, group=process_group)

        final_logits_padded = torch.cat(received_Y, dim=-1)

        final_logits_flat = final_logits_padded[: ctx.local_N, :]
        final_logits = final_logits_flat.view(*original_shape[:-1], -1)

        return final_logits

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        process_group = ctx.process_group
        world_size = (
            dist.get_world_size(group=process_group) if dist.is_initialized() else 1
        )

        if world_size == 1:
            return grad_output.matmul(weight), None, None, None

        vocab_shard_size = weight.size(0)
        padded_vocab_size = vocab_shard_size * world_size

        if grad_output.size(-1) < padded_vocab_size:
            pad_size = padded_vocab_size - grad_output.size(-1)
            grad_output = F.pad(grad_output, (0, pad_size))

        grad_output_flat = grad_output.contiguous().view(-1, padded_vocab_size)
        if grad_output_flat.size(0) < ctx.max_N:
            pad_N = ctx.max_N - grad_output_flat.size(0)
            grad_output_padded = F.pad(grad_output_flat, (0, 0, 0, pad_N))
        else:
            grad_output_padded = grad_output_flat

        grad_output_chunks = list(
            torch.split(grad_output_padded, vocab_shard_size, dim=-1)
        )

        send_grad_Y = [g.contiguous() for g in grad_output_chunks]
        received_grad_Y = [torch.empty_like(send_grad_Y[0]) for _ in range(world_size)]
        dist.all_to_all(received_grad_Y, send_grad_Y, group=process_group)

        grad_Y_local = torch.stack(received_grad_Y, dim=0)
        grad_X_full = grad_Y_local.matmul(weight)

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x_padded = torch.empty_like(grad_X_full[0])
            grad_X_full_list = [grad_X_full[i].contiguous() for i in range(world_size)]

            dist.reduce_scatter(
                grad_x_padded,
                grad_X_full_list,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )

            grad_x_flat = grad_x_padded[: ctx.local_N, :]
            grad_x = grad_x_flat.view(*ctx.original_shape)

        return grad_x, None, None, None


class ParallelLMHead(nn.Module):
    def __init__(
        self,
        weight_shard,
        original_vocab_size,
        dtype,
        bias_shard=None,
        process_group=None,
    ):
        super().__init__()
        self.original_vocab_size = original_vocab_size
        self.weight = nn.Parameter(weight_shard, requires_grad=False)
        self.dtype = dtype
        self.process_group = process_group
        if bias_shard is not None:
            self.bias = nn.Parameter(bias_shard, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        logits = DistributedParallelLMHeadFunction.apply(
            x,
            self.weight,
            self.bias,
            self.process_group,
        )
        if logits.size(-1) > self.original_vocab_size:
            logits = logits[..., : self.original_vocab_size]
        return logits

    def local_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.weight, self.bias)
        world_size = (
            dist.get_world_size(group=self.process_group)
            if dist.is_initialized()
            else 1
        )
        rank = dist.get_rank(group=self.process_group) if dist.is_initialized() else 0
        shard_start = rank * self.weight.shape[0]
        valid_size = max(
            0,
            min(self.weight.shape[0], self.original_vocab_size - shard_start),
        )
        if world_size == 1:
            valid_size = self.original_vocab_size
        return logits[..., :valid_size]


def load_target_lm_head(
    model_name_or_path: str,
    logger,
    device: torch.device,
    dtype: torch.dtype,
    parallel: bool = False,
    model_type: str = "qwen35",
    tensor_parallel_group=None,
    tensor_parallel_rank: int | None = None,
    tensor_parallel_size: int | None = None,
    target_checkpoint_id: str | None = None,
) -> nn.Module:

    rank = dist.get_rank() if dist.is_initialized() else 0

    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "text_config"):
        config = config.text_config

    vocab_size = config.vocab_size
    hidden_size = config.hidden_size

    ds4_target_weights = None
    ds4_target_keys = None
    if model_type == "ds4":
        ds4_target_weights, ds4_target_keys = _find_safetensor_weights(
            model_name_or_path,
            {
                "hc_head_fn": ("model.hc_head_fn", "hc_head_fn"),
                "hc_head_base": ("model.hc_head_base", "hc_head_base"),
                "hc_head_scale": ("model.hc_head_scale", "hc_head_scale"),
                "norm_weight": ("model.norm.weight", "norm.weight"),
                "lm_head_weight": (
                    "model.head.weight",
                    "model.lm_head.weight",
                    "lm_head.weight",
                    "head.weight",
                ),
            },
        )
        lm_head_weight = ds4_target_weights.pop("lm_head_weight")

    safetensors_files = [
        f for f in os.listdir(model_name_or_path) if f.endswith(".safetensors")
    ]

    if model_type == "glm5":
        expected_keys = ["model.layers.78.shared_head.head.weight"]
    elif model_type == "qwen35":
        expected_keys = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    elif model_type == "ds4":
        expected_keys = []
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if model_type != "ds4":
        found_keys = {}
        for safetensor in safetensors_files:
            file_path = os.path.join(model_name_or_path, safetensor)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in expected_keys:
                    if key in keys and key not in found_keys:
                        found_keys[key] = file_path

        target_file = None
        target_key = None
        for key in expected_keys:
            if key in found_keys:
                target_key = key
                target_file = found_keys[key]
                break

        if target_file is None:
            raise ValueError(
                f"Could not find any of the expected weights {expected_keys}"
            )

        with safe_open(target_file, framework="pt", device="cpu") as f:
            lm_head_weight = f.get_tensor(target_key)

    if tuple(lm_head_weight.shape) != (vocab_size, hidden_size):
        raise ValueError(
            "Target LM head shape mismatch: "
            f"expected {(vocab_size, hidden_size)}, got "
            f"{tuple(lm_head_weight.shape)}"
        )

    if parallel:
        world_size = (
            int(tensor_parallel_size)
            if tensor_parallel_size is not None
            else (
                dist.get_world_size(group=tensor_parallel_group)
                if dist.is_initialized()
                else 1
            )
        )
        rank = (
            int(tensor_parallel_rank)
            if tensor_parallel_rank is not None
            else (
                dist.get_rank(group=tensor_parallel_group)
                if dist.is_initialized()
                else 0
            )
        )

        shard_size = (vocab_size + world_size - 1) // world_size
        padded_vocab_size = shard_size * world_size

        if padded_vocab_size > vocab_size:
            pad_size = padded_vocab_size - vocab_size
            lm_head_weight = F.pad(lm_head_weight, (0, 0, 0, pad_size), value=0.0)

        start_idx = rank * shard_size
        end_idx = start_idx + shard_size

        weight_shard = (
            lm_head_weight[start_idx:end_idx].clone().to(dtype=dtype, device=device)
        )
        del lm_head_weight

        lm_head = ParallelLMHead(
            weight_shard,
            original_vocab_size=vocab_size,
            dtype=dtype,
            process_group=tensor_parallel_group,
        )
    else:
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

        with torch.no_grad():
            lm_head.weight.copy_(lm_head_weight.to(dtype=dtype))

        lm_head = lm_head.to(device=device)
        for param in lm_head.parameters():
            param.requires_grad_(False)

        lm_head.original_vocab_size = vocab_size
        del lm_head_weight

    if model_type == "ds4":
        assert ds4_target_weights is not None
        assert ds4_target_keys is not None
        hc_mult = int(config.hc_mult)
        expected_shapes = {
            "hc_head_fn": (hc_mult, hc_mult * hidden_size),
            "hc_head_base": (hc_mult,),
            "hc_head_scale": (1,),
            "norm_weight": (hidden_size,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(ds4_target_weights[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"DS4 target {name} shape mismatch: expected "
                    f"{expected_shape}, got {actual_shape}"
                )
        lm_head = DS4TargetLogitsHead(
            hc_head_fn=ds4_target_weights["hc_head_fn"].to(
                dtype=torch.float32, device=device
            ),
            hc_head_base=ds4_target_weights["hc_head_base"].to(
                dtype=torch.float32, device=device
            ),
            hc_head_scale=ds4_target_weights["hc_head_scale"].to(
                dtype=torch.float32, device=device
            ),
            norm_weight=ds4_target_weights["norm_weight"].to(
                dtype=dtype, device=device
            ),
            lm_head=lm_head,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
            dtype=dtype,
        ).to(device=device)
        if is_main_process():
            logger.info(
                "Loaded lightweight DS4 target logits head using keys: %s",
                ds4_target_keys,
            )

    lm_head.eval()
    lm_head.source_checkpoint_path = os.path.realpath(model_name_or_path)
    if target_checkpoint_id is not None:
        lm_head.target_checkpoint_id = target_checkpoint_id

    if dist.is_initialized():
        dist.barrier(group=tensor_parallel_group if parallel else None)

    return lm_head
