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

_MUON_DEFAULTS = {
    "use_muon": True,
    "momentum": 0.95,
    "nesterov": True,
    "ns_coefficients": _NS_COEFFICIENTS,
    "eps": _MUON_EPS,
    "ns_steps": 5,
    "adjust_lr_fn": "match_rms_adamw",
}

_ADAMW_DEFAULTS = {
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
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split a model's trainable parameters into Muon and AdamW groups."""
    muon_params: list[tuple[str, Tensor]] = []
    adamw_params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == _MATRIX_NDIM
            and min(param.shape) > 1
            and not any(hint in name for hint in _ADAMW_NAME_HINTS)
        ):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    return muon_params, adamw_params


class MuonAdamW(Optimizer):
    """Apply Muon and AdamW to disjoint parameter groups in one optimizer.

    ``lr`` seeds both groups; ``muon_lr`` defaults to ``10 * lr``.
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
        resolved_muon_lr = 10.0 * lr if muon_lr is None else float(muon_lr)

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
            ns_coefficients=group["ns_coefficients"],
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


def build_drafter_optimizer(drafter_model: Module, drafter_train_config) -> Optimizer:
    """Build the drafter optimizer based on ``drafter_train_config.optimizer``."""
    optimizer_name = (
        str(drafter_train_config.get("optimizer", "adamw") or "adamw").strip().lower()
    )
    lr = float(drafter_train_config.lr)
    weight_decay = float(drafter_train_config.get("weight_decay", 1e-2) or 1e-2)

    if optimizer_name in ("adamw", "adam"):
        trainable_params = [
            param for param in drafter_model.parameters() if param.requires_grad
        ]
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

    if optimizer_name == "muon":
        if not hasattr(torch.optim, "Muon"):  # pragma: no cover - torch version
            raise RuntimeError(
                "optimizer=muon requires a torch build that ships Muon "
                "(torch.optim.Muon, added in torch>=2.9)."
            )
        muon_params, adamw_params = split_named_params_for_muon(drafter_model)
        if not muon_params and not adamw_params:
            raise ValueError("No trainable parameters found to optimize.")
        logger.info(
            "Muon optimizer: %d 2D params via Muon, %d params via AdamW.",
            len(muon_params),
            len(adamw_params),
        )
        muon_lr = drafter_train_config.get("muon_lr", None)
        return MuonAdamW(
            [param for _, param in muon_params],
            [param for _, param in adamw_params],
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            muon_lr=None if muon_lr is None else float(muon_lr),
            muon_momentum=float(
                drafter_train_config.get("muon_momentum", 0.95) or 0.95
            ),
            muon_nesterov=bool(drafter_train_config.get("muon_nesterov", True)),
            muon_weight_decay=float(
                drafter_train_config.get("muon_weight_decay", 0.1) or 0.1
            ),
            muon_ns_steps=int(drafter_train_config.get("muon_ns_steps", 5) or 5),
            muon_adjust_lr_fn=drafter_train_config.get(
                "muon_adjust_lr_fn", "match_rms_adamw"
            ),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")
