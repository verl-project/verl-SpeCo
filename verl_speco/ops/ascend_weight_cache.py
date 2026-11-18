# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Refresh Ascend's derived FP32 linear weights without replacing live storage."""

import torch


class FP32WeightCaches:
    """Track native linear caches across an in-place parameter transaction.

    Do not rerun ``process_weights_after_loading``: it can transform expert
    layouts and replace storage. Copying into the existing tensor also preserves
    its NPU format and any graph references.
    """

    def __init__(self, model):
        self.model = model
        self.entries = []
        for name, module in model.named_modules():
            cache = getattr(module, "weight_fp32", None)
            if cache is None and not getattr(module, "precast_fp32_weight", False):
                continue
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or not isinstance(cache, torch.Tensor)
                or cache.shape != weight.shape
                or cache.dtype != torch.float32
                or cache.device != weight.device
            ):
                raise ValueError(f"Invalid Ascend FP32 weight cache: {name}")
            self.entries.append(
                (name, module, weight, cache, cache.data_ptr(), weight.data_ptr())
            )

    def validate(self):
        current = dict(self.model.named_modules())
        for name, module, weight, cache, cache_ptr, weight_ptr in self.entries:
            if (
                current.get(name) is not module
                or getattr(module, "weight", None) is not weight
                or getattr(module, "weight_fp32", None) is not cache
                or cache.data_ptr() != cache_ptr
                or weight.data_ptr() != weight_ptr
                or cache.shape != weight.shape
                or cache.dtype != torch.float32
                or cache.device != weight.device
            ):
                raise RuntimeError(f"Ascend FP32 weight cache storage changed: {name}")

    def snapshot(self):
        self.validate()
        return [cache.detach().cpu().clone() for _, _, _, cache, _, _ in self.entries]

    @torch.inference_mode()
    def refresh(self):
        self.validate()
        for _, _, weight, cache, _, _ in self.entries:
            cache.copy_(weight.detach().to(dtype=torch.float32))

    @torch.inference_mode()
    def restore(self, previous):
        self.validate()
        if len(previous) != len(self.entries):
            raise ValueError("Ascend FP32 cache snapshot schema differs")
        for (_, _, _, cache, _, _), value in zip(self.entries, previous, strict=True):
            cache.copy_(value)


def refresh_fp32_weight_caches(model):
    """Refresh after optimizer/restore mutations; roll back caches on failure."""
    caches = FP32WeightCaches(model)
    previous = caches.snapshot()
    try:
        caches.refresh()
    except Exception:
        caches.restore(previous)
        raise
