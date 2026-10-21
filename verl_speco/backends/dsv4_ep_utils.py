# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# SPDX-License-Identifier: Apache-2.0
"""DSV4 DSpark Expert Parallelism (EP) and FSDP2 setup utilities.

Extracted from `base_trainer.py` to keep the shared trainer file clean.
All functions are called only when `_is_ep_enabled()` returns True
(i.e., `speculative_algorithm == "DSV4_DSPARK"` and `dsv4_dspark_enable_ep == True`).
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

logger = logging.getLogger(__name__)


def _init_router_bias(model, world_size, experts_per_rank, rank):
    """Install a per-rank EP routing preference to break routing symmetry.

    A (near-)zero router makes the selection logits constant, so every token
    collapses onto the first ``topk`` experts and the other ranks receive no
    real rows in the all-to-all. We therefore give each rank a small transient
    preference (+boost for its own experts, -boost for the rest).

    The preference is stored on the module as ``_ep_rank_bias`` (a plain
    attribute, not a parameter/buffer) and applied at routing time only, so the
    shared ``router.bias`` buffer keeps its real, rank-independent value and the
    exported checkpoint is not polluted with rank-local routing state.
    """
    boost = 0.5
    n_experts = world_size * experts_per_rank
    for module in model.modules():
        router = getattr(module, "router", None)
        if router is None:
            continue
        # Bias is the global expert vector; shape[0] != n_experts rejects a
        # partially-sharded buffer we must not write a mismatched slice into.
        if not hasattr(router, "bias") or router.bias.shape[0] != n_experts:
            continue
        with torch.no_grad():
            local_start = rank * experts_per_rank
            local_end = min(local_start + experts_per_rank, n_experts)
            preference = torch.full(
                (n_experts,), -boost, dtype=torch.float32, device=router.bias.device
            )
            preference[local_start:local_end] = boost
            # Break the within-rank symmetry so ties do not all go to one expert.
            preference[local_start:local_end] += (
                torch.randn(local_end - local_start, device=router.bias.device) * 0.01
            )
            router._ep_rank_bias = preference
            # Mark the router's own symmetry-breaking as done: the per-rank
            # preference below supersedes update_load_balance_bias's first-step
            # random init, so the two mechanisms do not stack.
            if hasattr(router, "_bias_initialized"):
                router._bias_initialized = True
    if rank == 0:
        logger.info(
            f"[EP] Router EP preference set: boost=±{boost} for "
            f"{experts_per_rank} local experts/rank (shared bias preserved for export)"
        )


def _read_cfg(source, key, default=None):
    """Read a config value from either a mapping or an attribute object."""
    if source is None:
        return default
    if hasattr(source, "get"):
        return source.get(key, default)
    return getattr(source, key, default)


def is_ep_enabled(config) -> bool:
    rollout = getattr(config, "rollout", None)
    drafter = getattr(rollout, "drafter", None)
    algorithm = str(_read_cfg(drafter, "speculative_algorithm", "")).strip().upper()
    if algorithm != "DSV4_DSPARK":
        return False
    training = _read_cfg(drafter, "training", None)
    return bool(_read_cfg(training, "dsv4_dspark_enable_ep", False))


def is_ep_active(config) -> bool:
    """EP is both requested *and* possible (multi-rank process group).

    ``is_ep_enabled`` only inspects the config; a single-rank run cannot slice
    experts, so EP is treated as inactive there and training falls back to the
    full expert set instead of failing at build time.
    """
    if not is_ep_enabled(config):
        return False
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch always ships distributed
        return False
    return bool(
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )


def resolve_routed_expert_count(config, drafter_config=None) -> int:
    """Resolve the routed-expert count from the training config and checkpoint.

    The count is used to slice experts across ranks, so the training config and
    the drafter (checkpoint) config must agree; a mismatch is a hard error rather
    than a silent truncation.
    """
    training = _read_cfg(getattr(config, "rollout", None), "drafter", None)
    training_cfg = _read_cfg(training, "training", None)
    cfg_value = _read_cfg(training_cfg, "dsv4_dspark_backbone_n_routed_experts", None)
    ckpt_value = (
        getattr(drafter_config, "dsv4_n_routed_experts", None)
        if drafter_config is not None
        else None
    )
    if (
        cfg_value is not None
        and ckpt_value is not None
        and int(cfg_value) != int(ckpt_value)
    ):
        raise ValueError(
            "DSV4 DSpark routed-expert mismatch: training config "
            f"dsv4_dspark_backbone_n_routed_experts={int(cfg_value)} but the "
            f"checkpoint config dsv4_n_routed_experts={int(ckpt_value)}."
        )
    value = ckpt_value if ckpt_value is not None else cfg_value
    return int(value) if value is not None else 256


def ensure_ep_divisible(n_experts: int, world_size: int) -> int:
    """Return experts-per-rank, requiring the expert count to divide evenly."""
    if world_size <= 0:
        raise ValueError(f"EP world_size must be positive, got {world_size}")
    if n_experts % world_size != 0:
        raise ValueError(
            f"EP: n_routed_experts ({n_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    return n_experts // world_size


def _ep_preconfigured() -> bool:
    """Whether ``moe_ep`` was configured before model build (pre-sliced model)."""
    from verl_speco.models.dsv4_dspark.backbone import moe_ep

    return moe_ep._EP is not None and moe_ep._EP.size > 1


def _resolve_ep_layout(trainer) -> tuple[int, int, int]:
    """Shared EP preamble: fail closed unless moe_ep was pre-configured.

    Returns ``(world_size, rank, experts_per_rank)``.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    experts_per_rank = ensure_ep_divisible(
        resolve_routed_expert_count(trainer.config), world_size
    )
    if not _ep_preconfigured():
        raise RuntimeError(
            "DSV4 DSpark EP requires moe_ep.configure() before build_model "
            "(DrafterBaseTrainer configures it when EP is enabled and "
            "world_size > 1)"
        )
    logger.info(
        f"[Rank {rank}] EP pre-configured: model already has "
        f"{experts_per_rank} experts"
    )
    return world_size, rank, experts_per_rank


def _finish_ep_setup(
    raw_model: torch.nn.Module,
    world_size: int,
    rank: int,
    experts_per_rank: int,
    *,
    log_prefix: str,
) -> None:
    """Shared EP tail: rebuild the RoPE table, then break routing symmetry."""
    draft_model = getattr(raw_model, "draft_model", None) or raw_model
    if hasattr(draft_model, "_rebuild_freqs"):
        seq_len = (
            draft_model.freqs_cis.shape[0] if hasattr(draft_model, "freqs_cis") else 1
        )
        draft_model._rebuild_freqs(max(seq_len, 1))
        if rank == 0:
            logger.info(
                f"[{log_prefix}] Rebuilt freqs_cis "
                f"({draft_model.freqs_cis.device}, {draft_model.freqs_cis.shape})"
            )
    _init_router_bias(raw_model, world_size, experts_per_rank, rank)


def apply_ep_only(trainer, raw_model: torch.nn.Module):
    """Apply expert parallelism without FSDP2 (EP+DDP).

    Each rank keeps a disjoint ``n_routed_experts / world_size`` slice of the
    experts and non-expert parameters stay replicated (gradients are synced
    manually by :func:`sync_non_expert_grads`).

    Requires ``moe_ep.configure()`` to have run before ``build_model``
    (``DrafterBaseTrainer`` does this whenever EP is enabled and world_size > 1),
    so the model already carries its per-rank expert slice.
    """
    world_size, rank, experts_per_rank = _resolve_ep_layout(trainer)
    logger.info(
        f"[Rank {rank}] EP+DDP: {experts_per_rank} experts/rank, "
        f"non-expert grads synced via all-reduce"
    )

    # Move to device and set requires_grad
    raw_model.to(trainer.runtime_device)
    n_trainable = 0
    expert_param_ids: set[int] = set()
    for name, p in raw_model.named_parameters():
        if "embed_tokens" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            n_trainable += 1

    # Collect expert param ids (local, no gradient sync needed).
    for module in raw_model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or not hasattr(experts, "w1") or experts.w1.dim() < 3:
            continue
        expert_param_ids.update(id(p) for p in experts.parameters())
    trainer._ep_expert_param_ids = expert_param_ids
    trainer._ep_ddp_world_size = world_size

    if rank == 0:
        logger.info(
            f"[EP+DDP] {n_trainable} trainable params, "
            f"{len(expert_param_ids)} expert params (local), "
            f"{n_trainable - len(expert_param_ids)} non-expert params (DDP sync)"
        )

    _finish_ep_setup(
        raw_model, world_size, rank, experts_per_rank, log_prefix="EP-only"
    )


def apply_ep_fsdp2(
    trainer,
    raw_model: torch.nn.Module,
    fsdp_kwargs: dict,
    fsdp_config,
):
    """Apply expert parallelism together with FSDP2.

    Key: FSDP2-wrap the DRAFT MODEL (the inner model with layers), NOT the
    training wrapper. The wrapper calls ``draft_model.forward()``, which triggers
    the FSDP2 hooks; the wrapper's own parameters are not FSDP2-managed.

    Flow:
    1. Move to device (bf16 cast first to halve memory)
    2. shard_experts_as_dtensor: wrap expert params as Shard(0) DTensors
    3. FSDP2 wrap draft_model: per-layer (shared_experts, attn, block) + root
       with ignored_params=expert DTensors
    4. Set self.model = raw_model (training wrapper still calls draft_model)

    Requires ``moe_ep.configure()`` to have run before ``build_model``
    (``DrafterBaseTrainer`` does this whenever EP is enabled and world_size > 1),
    so the model already carries its per-rank expert slice.
    """
    world_size, rank, experts_per_rank = _resolve_ep_layout(trainer)
    # Expert DTensors are built on the raw (dp, sp) training mesh. The experts
    # are in ``ignored_params`` (never sharded/reduced), so the placements only
    # need to keep the optimizer/grad-norm from mixing plain tensors with
    # DTensors.
    mesh = trainer.training_device_mesh
    random_init = bool(
        trainer.config.rollout.drafter.training.get("dsv4_dspark_random_init", False)
    )

    # Move to device (cast to bf16 first to halve memory, matching speculators)
    raw_model.to(torch.bfloat16)
    raw_model.to(trainer.runtime_device)

    # shard_experts_as_dtensor — wrap the routed experts (one local EP slice
    # per rank) as Shard(0) DTensors. This keeps every parameter a DTensor so
    # FSDP2's gradient-norm reduction and the optimizer do not mix plain
    # tensors with DTensors (uniform with the released reference flow).
    draft_model = getattr(raw_model, "draft_model", None) or raw_model
    if hasattr(draft_model, "shard_experts_as_dtensor"):
        draft_model.shard_experts_as_dtensor(mesh)
        if rank == 0:
            logger.info("[EP] Experts wrapped as Shard(0) DTensors")

    # FSDP2 wrap draft_model per-layer + root
    mp_policy = fsdp_kwargs["mp_policy"]
    reshard = True

    ignored_params = None
    if hasattr(draft_model, "fsdp_ignored_params"):
        ignored_params = draft_model.fsdp_ignored_params() or None

    wrap_modules = []
    if hasattr(draft_model, "fsdp_wrap_plan"):
        wrap_modules = draft_model.fsdp_wrap_plan()

    # Random init (no EP): don't wrap GroupedExperts separately —
    # _moe_dispatch_torch accesses w1/w2/w3 directly (not via experts()),
    # so FSDP2 pre-forward hook on GroupedExperts won't fire.
    # Let the parent block's FSDP2 wrap handle unshard.
    if random_init:
        wrap_modules = [
            m
            for m in wrap_modules
            if not hasattr(m, "w1") or not hasattr(m, "num_local_experts")
        ]

    extra = {"mesh": mesh, "reshard_after_forward": reshard, "mp_policy": mp_policy}
    if ignored_params:
        extra["ignored_params"] = ignored_params

    for mod in wrap_modules:
        fully_shard(mod, **extra)

    fully_shard(
        raw_model,
        mesh=mesh,
        reshard_after_forward=False,
        mp_policy=mp_policy,
        ignored_params=ignored_params if ignored_params else None,
    )

    # Mark FSDP2 mode (skip _sync_non_expert_grads)
    trainer._ep_ddp_world_size = 0  # 0 = FSDP2 mode (no manual gradient sync)

    _finish_ep_setup(raw_model, world_size, rank, experts_per_rank, log_prefix="EP")
    logger.info(
        f"[Rank {rank}] EP+FSDP2 applied: EP={world_size}, "
        f"{experts_per_rank} experts/rank, "
        f"draft_model FSDP2-wrapped, "
        f"{len(ignored_params) if ignored_params else 0} expert params as DTensors"
    )


def sync_non_expert_grads(trainer) -> None:
    """All-reduce (average) gradients of non-expert params across the WORLD group.

    Expert params are local (EP): each rank owns a disjoint set of experts,
    so their gradients must NOT be reduced. Non-expert params (attention,
    norms, shared_experts, router, etc.) are replicated across all ranks;
    their gradients need averaging to produce a consistent update.
    """
    world_size = getattr(trainer, "_ep_ddp_world_size", 1)
    if world_size <= 1:
        return

    # An empty expert-id set means "no params excluded" (every grad is synced),
    # not "nothing to do".
    expert_ids = getattr(trainer, "_ep_expert_param_ids", None) or set()

    # Build a canonical, rank-invariant list: parameters that happen to miss a
    # gradient on one rank are still reduced (as zeros) so every rank issues the
    # same collective shapes/dtypes. A rank-dependent grad-presence skew would
    # otherwise desync the all-reduce and hang.
    entries: list[tuple[torch.nn.Parameter, torch.Tensor, bool]] = []
    for p in trainer.model.parameters():
        if id(p) in expert_ids or not p.requires_grad:
            continue
        if p.grad is not None:
            entries.append((p, p.grad, True))
        else:
            entries.append((p, torch.zeros_like(p), False))

    if not entries:
        return

    if os.environ.get("DSPARK_DEBUG_EP") == "1":
        total_numel = sum(g.numel() for _, g, _ in entries)
        print(
            f"[EPDEBUG gradsync rank={dist.get_rank()}] "
            f"n_grads={len(entries)} total_numel={total_numel} "
            f"dtypes={sorted({str(g.dtype) for _, g, _ in entries})}",
            flush=True,
        )

    # Group by dtype so mixed-precision non-expert params can be reduced
    # together; avoid torch private ``_flatten_dense_tensors`` helpers.
    grads_by_dtype: dict[
        torch.dtype, list[tuple[torch.nn.Parameter, torch.Tensor, bool]]
    ] = {}
    for entry in entries:
        grads_by_dtype.setdefault(entry[1].dtype, []).append(entry)

    for group in grads_by_dtype.values():
        flat_grad = torch.cat([grad.reshape(-1) for _, grad, _ in group])
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(float(world_size))
        offset = 0
        for _, grad, had_grad in group:
            numel = grad.numel()
            # Only write back params that owned a gradient; a synthesized zero
            # must not create a spurious optimizer update.
            if had_grad:
                grad.copy_(flat_grad[offset : offset + numel].view_as(grad))
            offset += numel


def dump_local_expert_slices(trainer, checkpoint_path: str) -> str | None:
    """Persist this EP rank's local routed-expert slices for offline merging.

    The standalone exporter writes a full state dict on the checkpoint leader
    only; under expert parallelism that file contains just the leader's expert
    slice (``n_routed_experts / world_size`` rows). Every rank dumps its own
    slice under ``<checkpoint_path>/ep_experts/rank_<r>.pt`` so a complete
    drafter can be reconstructed with

        python -m verl_speco.tools.convert_checkpoint --in <ckpt_dir> \
            --ep-experts-dir <ckpt_dir>/ep_experts --out <out_dir>

    (rank-ordered concat; rank r owns experts ``[r * local, (r + 1) * local)``).
    """
    if not is_ep_active(trainer.config):
        return None

    model = getattr(trainer, "model", None)
    if model is None:
        return None
    if hasattr(model, "module"):
        model = model.module
    draft_model = getattr(model, "draft_model", None) or model
    layers = getattr(draft_model, "layers", None)
    if layers is None:
        return None

    from torch.distributed.tensor import DTensor

    def _local(tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    payload: dict[str, torch.Tensor] = {}
    for layer_index, layer in enumerate(layers):
        experts = getattr(getattr(layer, "ffn", None), "experts", None)
        if experts is None:
            continue
        for weight_name in ("w1", "w2", "w3"):
            param = getattr(experts, weight_name, None)
            if param is None:
                continue
            data = param.data if hasattr(param, "data") else param
            payload[f"layers.{layer_index}.ffn.experts.{weight_name}"] = (
                _local(data).detach().to("cpu").clone()
            )
    if not payload:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    out_dir = os.path.join(checkpoint_path, "ep_experts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rank_{rank}.pt")
    torch.save(payload, out_path)
    return out_path
