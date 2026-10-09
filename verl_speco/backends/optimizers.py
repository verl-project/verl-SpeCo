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
"""Optimizer construction for drafter training.

``adamw`` keeps the historical single AdamW optimizer; ``muon`` drives 2D hidden
weights with Muon and the remaining parameters with AdamW. The drafter trainer
assumes one optimizer, so :class:`MuonAdamW` hosts both algorithms as separate
parameter groups while keeping all state in ``self.state``.
"""

import logging
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# 2D parameters outside the orthogonalized update (embeddings and output heads).
_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "codebook",
    "markov_w1",
    "markov_w2",
)

_MATRIX_NDIM = 2

# torch.optim.Muon defaults.
_NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
_MUON_EPS = 1e-7

_MUON_DEFAULTS: dict[str, object] = {
    # Only ``use_muon`` comes from the defaults; every other Muon option is set
    # explicitly when the group is built.
    "use_muon": True,
}

_ADAMW_DEFAULTS: dict[str, object] = {
    "use_muon": False,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": False,
}


def split_named_params_for_muon(
    model: Module,
    *,
    adamw_name_hints: Sequence[str] | None = None,
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups.

    A trainable 2D matrix goes to Muon unless its name contains one of the
    built-in ``_ADAMW_NAME_HINTS`` (embeddings, output heads, codebooks, Markov
    heads) or one of the extra ``adamw_name_hints``, or has a singleton
    dimension; every other parameter goes to AdamW.
    """
    hints = _ADAMW_NAME_HINTS + tuple(adamw_name_hints or ())
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1
            and not any(hint in name for hint in hints)
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


class MuonAdamW(Optimizer):
    """Apply Muon and AdamW to disjoint parameter groups in one optimizer.

    ``lr`` seeds both groups; ``muon_lr`` defaults to the same ``lr`` as AdamW,
    matching speculators (``adjust_lr_fn="match_rms_adamw"`` already rescales the
    orthogonalized update to AdamW's RMS).
    """

    def __init__(
        self,
        muon_params: list[Tensor],
        adamw_params: list[Tensor],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        muon_lr: float | None = None,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_weight_decay: float = 0.1,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str | None = "match_rms_adamw",
    ) -> None:
        if muon_adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError(
                f"Unsupported muon adjust_lr_fn: {muon_adjust_lr_fn!r}; "
                "expected 'original', 'match_rms_adamw', or None"
            )
        resolved_muon_lr = float(lr) if muon_lr is None else float(muon_lr)
        if resolved_muon_lr < 0.0 or float(lr) < 0.0:
            raise ValueError(
                f"Muon/AdamW lr must be non-negative, got muon_lr={resolved_muon_lr}, "
                f"lr={float(lr)}"
            )
        if not 0.0 <= float(muon_momentum) < 1.0:
            raise ValueError(
                f"muon_momentum must be in [0, 1), got {float(muon_momentum)}"
            )
        if not 1 <= int(muon_ns_steps) < 100:
            raise ValueError(
                "muon_ns_steps must be in [1, 100): PyTorch's Muon "
                "Newton-Schulz kernel rejects ns_steps >= 100, got "
                f"{int(muon_ns_steps)}"
            )
        for name, value in (
            ("weight_decay", weight_decay),
            ("muon_weight_decay", muon_weight_decay),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {float(value)}")
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"AdamW betas must be in [0, 1), got {betas!r}")
        if float(eps) <= 0.0:
            raise ValueError(f"AdamW eps must be positive, got {float(eps)}")

        param_groups: list[dict] = []
        if muon_params:
            group = dict(_MUON_DEFAULTS)
            group.update(
                {
                    "params": list(muon_params),
                    "lr": resolved_muon_lr,
                    "weight_decay": float(muon_weight_decay),
                    "momentum": float(muon_momentum),
                    "nesterov": bool(muon_nesterov),
                    "ns_coefficients": _NS_COEFFICIENTS,
                    "eps": _MUON_EPS,
                    "ns_steps": int(muon_ns_steps),
                    "adjust_lr_fn": muon_adjust_lr_fn,
                }
            )
            param_groups.append(group)
        if adamw_params:
            group = dict(_ADAMW_DEFAULTS)
            group.update(
                {
                    "params": list(adamw_params),
                    "lr": float(lr),
                    "betas": tuple(betas),
                    "eps": float(eps),
                    "weight_decay": float(weight_decay),
                }
            )
            param_groups.append(group)
        if not param_groups:
            raise ValueError("MuonAdamW received no trainable parameters.")

        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        from torch.optim._muon import muon as muon_update

        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []
        for param in group["params"]:
            if param.grad is None:
                continue
            if torch.is_complex(param):
                raise RuntimeError("Muon does not support complex parameters")
            if param.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            params.append(param)
            grads.append(param.grad)
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    param.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

        if not params:
            return
        muon_update(
            params,
            grads,
            momentum_bufs,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            # A checkpoint round-trip may deserialize this tuple as a list.
            ns_coefficients=tuple(group["ns_coefficients"]),
            eps=group["eps"],
            ns_steps=group["ns_steps"],
            adjust_lr_fn=group["adjust_lr_fn"],
            has_complex=False,
        )

    def _adamw_step(self, group: dict) -> None:
        from torch.optim.adamw import adamw as adamw_update

        params: list[Tensor] = []
        grads: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        max_exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        has_complex = False

        for param in group["params"]:
            if param.grad is None:
                continue
            has_complex |= torch.is_complex(param)
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            params.append(param)
            grads.append(param.grad)

            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            state_steps.append(state["step"])

        if not params:
            return
        beta1, beta2 = group["betas"]
        adamw_update(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            has_complex=has_complex,
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )


def _config_value(config, key: str, default):
    """Return a config value, treating an explicit ``null`` as the default."""
    value = config.get(key, default)
    return default if value is None else value


def _config_float(config, key: str, default: float) -> float:
    return float(_config_value(config, key, default))


def _config_int(config, key: str, default: int) -> int:
    return int(_config_value(config, key, default))


def _is_fsdp1_wrapped(model: Module) -> bool:
    """Whether ``model`` is wrapped in FSDP1 (``FullyShardedDataParallel``).

    FSDP1 exposes parameters as 1D shards under ``FULL_SHARD``/``HYBRID_SHARD``
    even with ``use_orig_params=true`` (and as a single 1D ``FlatParameter``
    with ``use_orig_params=false``), so the dimension-based Muon/AdamW split
    cannot recover the original hidden matrices and would silently route
    everything to AdamW. FSDP2 (``FSDPModule``) and DDP are not affected.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
    except Exception:  # pragma: no cover - torch build without FSDP1
        return False
    return isinstance(model, FSDP1)


def _muon_kernel_available() -> bool:
    try:
        from torch.optim import _muon  # noqa: F401
    except Exception:  # pragma: no cover - torch build without the Muon kernel
        return False
    return True


def _config_betas(config, default: tuple[float, float] = (0.9, 0.95)):
    value = _config_value(config, "adamw_betas", None)
    if value is None:
        return default
    betas = tuple(float(item) for item in value)
    if len(betas) != 2:
        raise ValueError(f"adamw_betas must have two values, got {value!r}")
    return betas


def build_drafter_optimizer(drafter_model: Module, drafter_train_config) -> Optimizer:
    """Build the drafter optimizer based on ``drafter_train_config.optimizer``."""
    optimizer_name = (
        str(drafter_train_config.get("optimizer", "adamw") or "adamw").strip().lower()
    )
    lr = float(drafter_train_config.lr)
    weight_decay = _config_float(drafter_train_config, "weight_decay", 1e-2)
    betas = _config_betas(drafter_train_config)

    if optimizer_name == "adamw":
        trainable_params = [
            param for param in drafter_model.parameters() if param.requires_grad
        ]
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name == "muon":
        if not _muon_kernel_available():  # pragma: no cover - torch version
            raise RuntimeError(
                "optimizer=muon requires a torch build with the Muon kernel "
                "(torch.optim._muon, added in torch>=2.9)."
            )
        if _is_fsdp1_wrapped(drafter_model):
            raise ValueError(
                "optimizer=muon is not supported with FSDP1: FSDP1 shards "
                "parameters to 1D (even with use_orig_params=true), so the "
                "Muon/AdamW split cannot identify the hidden matrices. Use the "
                "fsdp2 or ddp strategy, or keep optimizer=adamw."
            )
        muon_params, adamw_params = split_named_params_for_muon(
            drafter_model,
            adamw_name_hints=_config_value(
                drafter_train_config, "muon_adamw_name_hints", None
            ),
        )
        if not muon_params and not adamw_params:
            raise ValueError("No trainable parameters found to optimize.")
        muon_lr = _config_value(drafter_train_config, "muon_lr", None)
        resolved_muon_lr = lr if muon_lr is None else float(muon_lr)
        logger.info(
            "Muon optimizer: muon_lr=%.3e adamw_lr=%.3e 2D_params=%d adamw_params=%d",
            resolved_muon_lr,
            lr,
            len(muon_params),
            len(adamw_params),
        )
        return MuonAdamW(
            [param for _, param in muon_params],
            [param for _, param in adamw_params],
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            muon_lr=None if muon_lr is None else float(muon_lr),
            muon_momentum=_config_float(drafter_train_config, "muon_momentum", 0.95),
            muon_nesterov=_config_value(drafter_train_config, "muon_nesterov", True),
            muon_weight_decay=_config_float(
                drafter_train_config, "muon_weight_decay", 0.1
            ),
            muon_ns_steps=_config_int(drafter_train_config, "muon_ns_steps", 5),
            muon_adjust_lr_fn=drafter_train_config.get(
                "muon_adjust_lr_fn", "match_rms_adamw"
            ),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")
