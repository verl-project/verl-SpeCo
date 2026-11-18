# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Attach existing SpeCo DSA backward to the new KV-plan dispatch boundary."""

import torch


def patch_kv_plan(plan_type, scatter, sparse_attention):
    if getattr(plan_type, "_speco_dspark_training_patch", False):
        return True
    native_scatter = plan_type.dsa_kv_compress_scatter
    native_selector = plan_type.get_dsa_sparse_attn_op

    def require_training_layout(plan):
        if (
            plan.uses_sparse_flash_mla
            or plan.uses_kv_compress_epilog
            or plan.layout_kv != "PA_ND"
        ):
            raise RuntimeError(
                "DSpark training requires the uncompressed PA_ND DSA KV plan"
            )

    def training_scatter(self, cache, x, slot_mapping):
        if not torch.is_grad_enabled():
            return native_scatter(self, cache, x, slot_mapping)
        if x is None:
            return None
        require_training_layout(self)
        return scatter(cache, slot_mapping, x)

    def training_selector(self):
        if not torch.is_grad_enabled():
            return native_selector(self)
        require_training_layout(self)
        return sparse_attention

    plan_type.dsa_kv_compress_scatter = training_scatter
    plan_type.get_dsa_sparse_attn_op = training_selector
    plan_type._speco_dspark_training_patch = True
    return True
