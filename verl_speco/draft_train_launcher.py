"""User-friendly launcher for standalone SPECO draft model training.

This module lets examples keep the familiar ``python -m ...`` shape while still
starting one distributed training process per local device.  It delegates to
PyTorch's distributed launcher instead of requiring users to type ``torchrun``
directly.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


_NPROC_KEYS = (
    "speco.draft_training.nproc_per_node",
    "speco.draft_training.num_gpus_per_node",
    "actor_rollout_ref.rollout.drafter.training.nproc_per_node",
    "actor_rollout_ref.rollout.drafter.training.num_gpus_per_node",
)
_NNODES_KEYS = (
    "speco.draft_training.nnodes",
    "speco.draft_training.num_nodes",
    "actor_rollout_ref.rollout.drafter.training.nnodes",
    "actor_rollout_ref.rollout.drafter.training.num_nodes",
)
_NODE_RANK_KEYS = (
    "speco.draft_training.node_rank",
    "actor_rollout_ref.rollout.drafter.training.node_rank",
)
_MASTER_ADDR_KEYS = (
    "speco.draft_training.master_addr",
    "actor_rollout_ref.rollout.drafter.training.master_addr",
)
_MASTER_PORT_KEYS = (
    "speco.draft_training.master_port",
    "actor_rollout_ref.rollout.drafter.training.master_port",
)
_STANDALONE_KEYS = (
    "speco.draft_training.standalone",
    "actor_rollout_ref.rollout.drafter.training.standalone",
)

_LAUNCH_OVERRIDE_KEYS = frozenset(
    _NPROC_KEYS
    + _NNODES_KEYS
    + _NODE_RANK_KEYS
    + _MASTER_ADDR_KEYS
    + _MASTER_PORT_KEYS
    + _STANDALONE_KEYS
)


@dataclass(frozen=True)
class DraftTrainLaunchConfig:
    nproc_per_node: str
    nnodes: str
    node_rank: str | None
    master_addr: str | None
    master_port: str | None
    standalone: bool
    module: str


def _split_override(item: str) -> tuple[str, str] | None:
    if "=" not in item or item.startswith("-"):
        return None
    key, value = item.split("=", 1)
    return key, value


def _find_override(overrides: Iterable[str], keys: Iterable[str]) -> str | None:
    wanted = set(keys)
    for item in overrides:
        parsed = _split_override(item)
        if parsed is None:
            continue
        key, value = parsed
        if key in wanted:
            return value
    return None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for standalone: {value!r}")


def resolve_launch_config(
    overrides: list[str],
    *,
    module: str = "verl_speco.draft_train",
) -> DraftTrainLaunchConfig:
    """Resolve distributed-launch settings from Hydra-style CLI overrides."""

    nproc = _find_override(overrides, _NPROC_KEYS) or "1"
    nnodes = _find_override(overrides, _NNODES_KEYS) or "1"
    node_rank = _find_override(overrides, _NODE_RANK_KEYS)
    master_addr = _find_override(overrides, _MASTER_ADDR_KEYS)
    master_port = _find_override(overrides, _MASTER_PORT_KEYS)

    standalone_default = nnodes == "1" and master_addr is None and master_port is None
    standalone = _parse_bool(_find_override(overrides, _STANDALONE_KEYS), default=standalone_default)
    if standalone and nnodes != "1":
        raise ValueError("standalone=true requires nnodes=1")

    return DraftTrainLaunchConfig(
        nproc_per_node=nproc,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        standalone=standalone,
        module=module,
    )


def normalize_training_args(overrides: list[str], config: DraftTrainLaunchConfig) -> list[str]:
    """Replace launcher aliases with canonical Hydra configuration fields."""

    normalized = []
    for item in overrides:
        parsed = _split_override(item)
        if parsed is None or parsed[0] not in _LAUNCH_OVERRIDE_KEYS:
            normalized.append(item)
    normalized.extend(
        [
            f"speco.draft_training.nproc_per_node={config.nproc_per_node}",
            f"speco.draft_training.nnodes={config.nnodes}",
            f"speco.draft_training.standalone={str(config.standalone).lower()}",
        ]
    )
    if config.node_rank is not None:
        normalized.append(f"speco.draft_training.node_rank={config.node_rank}")
    if config.master_addr is not None:
        normalized.append(f"speco.draft_training.master_addr={config.master_addr}")
    if config.master_port is not None:
        normalized.append(f"speco.draft_training.master_port={config.master_port}")
    return normalized


def build_torch_distributed_command(
    config: DraftTrainLaunchConfig,
    training_args: list[str],
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the command that starts the distributed draft trainer."""

    command = [
        python_executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={config.nnodes}",
        f"--nproc_per_node={config.nproc_per_node}",
    ]
    if config.standalone:
        command.append("--standalone")
    else:
        if config.node_rank is not None:
            command.append(f"--node_rank={config.node_rank}")
        if config.master_addr is not None:
            command.append(f"--master_addr={config.master_addr}")
        if config.master_port is not None:
            command.append(f"--master_port={config.master_port}")
    command.extend(["-m", config.module])
    command.extend(training_args)
    return command


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch standalone SPECO draft training.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved launch command and exit.")
    parser.add_argument(
        "--module",
        default="verl_speco.draft_train",
        help="Training module passed to the distributed launcher.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to invoke torch.distributed.run.",
    )
    args, training_args = parser.parse_known_args(argv)

    launch_config = resolve_launch_config(training_args, module=args.module)
    normalized_training_args = normalize_training_args(training_args, launch_config)
    command = build_torch_distributed_command(
        launch_config,
        normalized_training_args,
        python_executable=args.python_executable,
    )
    if args.dry_run:
        print(_format_command(command))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
