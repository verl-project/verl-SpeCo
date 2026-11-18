# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reshard BF16 Ascend runtime MoE matrices from training TP into serving EP."""

import torch


class RuntimeExpertAssembler:
    """Stage only locally owned experts, preserving the live runtime layout."""

    def __init__(self, model, tp_size):
        self.tp_size = tp_size
        self.specs = {}
        self.buffers = {}
        self.received = {}
        for module in model.modules():
            for name in ("w13_weight_list", "w2_weight_list"):
                values = getattr(module, name, None)
                if isinstance(values, (list, tuple)) and values:
                    raise ValueError("Expert weight-list caches are unsupported for EP hot updates")
        parameters = dict(model.named_parameters())
        for prefix, module in model.named_modules():
            if f"{prefix}.w13_weight" not in parameters:
                continue
            mapping = getattr(module, "expert_map", None)
            w13, w2 = (
                getattr(module, "w13_weight", None),
                getattr(module, "w2_weight", None),
            )
            if (
                not torch.is_tensor(mapping)
                or not torch.is_tensor(w13)
                or not torch.is_tensor(w2)
            ):
                continue
            if (
                not w13.is_floating_point()
                or w13.dtype != torch.bfloat16
                or w2.dtype != torch.bfloat16
            ):
                raise ValueError("EP updates require unquantized BF16 runtime experts")
            mapping = mapping.detach().cpu().clone()
            if mapping.ndim != 1 or mapping.dtype not in (torch.int32, torch.int64):
                raise ValueError("Invalid runtime expert_map")
            if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
                raise ValueError("Unsupported EP expert matrix layout")
            count, hidden, doubled = w13.shape
            intermediate = w2.shape[1]
            if (
                doubled != 2 * intermediate
                or w2.shape[2] != hidden
                or intermediate % tp_size
            ):
                raise ValueError(
                    "EP updates require input-major matrices and divisible TP intermediate size"
                )
            local = mapping[mapping >= 0].tolist()
            if any(x < -1 for x in mapping.tolist()) or sorted(local) != list(
                range(count)
            ):
                raise ValueError(
                    "Expert map must cover every local slot exactly once; redundant/shared experts are unsupported"
                )
            indices = torch.empty(count, dtype=torch.long)
            for global_id, local_id in enumerate(mapping.tolist()):
                if local_id >= 0:
                    indices[local_id] = global_id
            for suffix, target in [("w13_weight", w13), ("w2_weight", w2)]:
                name = f"{prefix}.{suffix}" if prefix else suffix
                self.specs[name] = (
                    module,
                    mapping,
                    indices,
                    tuple(target.shape),
                    suffix,
                )
                self.received[name] = set()
        if self.specs:
            for module in model.modules():
                if (
                    getattr(module, "enable_eplb", False)
                    or getattr(module, "dynamic_eplb", False)
                    or getattr(module, "global_redundant_expert_num", 0)
                ):
                    raise ValueError(
                        "EP draft updates do not support EPLB or redundant experts"
                    )

    def add(self, name, rank, value):
        _module, mapping, indices, target_shape, suffix = self.specs[name]
        if not 0 <= rank < self.tp_size or rank in self.received[name]:
            raise ValueError("Duplicate or invalid expert TP lane")
        if (
            value.dtype != torch.bfloat16
            or value.ndim != 3
            or value.shape[0] != len(mapping)
        ):
            raise ValueError(f"Invalid BF16 expert TP shard: {name}")
        width = (
            target_shape[2] // 2 if suffix == "w13_weight" else target_shape[1]
        ) // self.tp_size
        expected = (
            (len(mapping), target_shape[1], 2 * width)
            if suffix == "w13_weight"
            else (len(mapping), width, target_shape[2])
        )
        if tuple(value.shape) != expected:
            raise ValueError(
                f"Expert TP shard shape differs: {name}, expected {expected}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite expert TP shard: {name}")
        selected = value.detach().index_select(0, indices.to(value.device)).to("cpu")
        buffer = self.buffers.setdefault(name, None)
        if buffer is None:
            buffer = torch.empty(target_shape, dtype=value.dtype)
            self.buffers[name] = buffer
        start, end = rank * width, (rank + 1) * width
        if suffix == "w13_weight":
            half = target_shape[2] // 2
            buffer[:, :, start:end].copy_(selected[:, :, :width])
            buffer[:, :, half + start : half + end].copy_(selected[:, :, width:])
        else:
            buffer[:, start:end, :].copy_(selected)
        self.received[name].add(rank)
        return buffer if len(self.received[name]) == self.tp_size else None

    def validate(self):
        for name, (module, mapping, _, _, _) in self.specs.items():
            if not torch.equal(module.expert_map.detach().cpu(), mapping):
                raise RuntimeError("Runtime expert_map changed during draft update")
            if self.received[name] != set(range(self.tp_size)):
                raise RuntimeError(f"Incomplete expert TP lanes: {name}")

    def clear(self):
        self.buffers.clear()
