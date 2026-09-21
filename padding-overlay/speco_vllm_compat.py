# SPDX-License-Identifier: Apache-2.0
"""vLLM API compatibility shims for the verl 0.8.0 rollout code.

verl 0.8.0 imports ``FusedMoE`` from ``vllm.model_executor.layers.fused_moe.layer``. vLLM 0.29
replaced that class with ``MoERunner`` (an ``nn.Module``) plus ``FusedMoEConfig`` and
``FusedMoEFactory``. The verl code only uses the symbol for ``isinstance`` checks while applying
its optional FP8 patches, so aliasing the relocated class keeps that path importable.

This module changes no behaviour on vLLM versions that still expose ``FusedMoE``.
"""

__all__ = ["install_fused_moe_alias"]


def install_fused_moe_alias() -> str:
    """Make ``...fused_moe.layer.FusedMoE`` importable; returns the resolved class name."""
    from vllm.model_executor.layers.fused_moe import layer as _fused_moe_layer

    existing = getattr(_fused_moe_layer, "FusedMoE", None)
    if existing is not None:
        return existing.__name__

    from vllm.model_executor.layers.fused_moe.layer import MoERunner

    _fused_moe_layer.FusedMoE = MoERunner
    return MoERunner.__name__
