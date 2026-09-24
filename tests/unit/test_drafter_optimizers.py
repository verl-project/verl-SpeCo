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
from __future__ import annotations

import contextlib
import copy
import os
import socket

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from verl_speco.backends.optimizers import (
    MuonAdamW,
    build_drafter_optimizer,
    split_named_params_for_muon,
)


class _DraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def forward(self, hidden):
        return self.fc2(torch.relu(self.fc1(self.norm(hidden))))


def test_split_routes_2d_hidden_weights_to_muon() -> None:
    model = _DraftModel()

    muon_params, adamw_params = split_named_params_for_muon(model)

    muon_names = {name for name, _ in muon_params}
    adamw_names = {name for name, _ in adamw_params}

    assert muon_names == {"fc1.weight", "fc2.weight"}
    assert "norm.weight" in adamw_names
    assert "norm.bias" in adamw_names
    assert "embed_tokens.weight" in adamw_names
    assert "lm_head.weight" in adamw_names
    assert not muon_names & adamw_names


def test_split_excludes_frozen_parameters() -> None:
    model = _DraftModel()
    model.embed_tokens.weight.requires_grad_(False)
    model.fc1.weight.requires_grad_(False)

    muon_params, adamw_params = split_named_params_for_muon(model)

    names = {name for name, _ in muon_params} | {name for name, _ in adamw_params}
    assert "embed_tokens.weight" not in names
    assert "fc1.weight" not in names


def test_build_optimizer_defaults_to_adamw() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"lr": 1e-4})

    optimizer = build_drafter_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1


def test_build_optimizer_muon_splits_lrs() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "muon", "lr": 1e-4, "weight_decay": 1e-2})

    optimizer = build_drafter_optimizer(model, config)

    assert isinstance(optimizer, MuonAdamW)
    assert len(optimizer.param_groups) == 2
    muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
    adamw_group = next(g for g in optimizer.param_groups if not g["use_muon"])
    # muon_lr defaults to the AdamW lr (speculators parity, not 10 * lr).
    assert muon_group["lr"] == pytest.approx(1e-4)
    assert adamw_group["lr"] == pytest.approx(1e-4)


def test_build_optimizer_muon_respects_explicit_muon_lr() -> None:
    model = _DraftModel()
    config = OmegaConf.create(
        {"optimizer": "muon", "lr": 1e-4, "muon_lr": 2e-4}
    )

    optimizer = build_drafter_optimizer(model, config)

    muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
    assert muon_group["lr"] == pytest.approx(2e-4)


def test_build_optimizer_preserves_zero_weight_decay() -> None:
    model = _DraftModel()
    config = OmegaConf.create(
        {
            "optimizer": "muon",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "muon_weight_decay": 0.0,
        }
    )

    optimizer = build_drafter_optimizer(model, config)

    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(0.0)


def test_muon_optimizer_steps_both_groups_and_round_trips_state() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "muon", "lr": 1e-3, "weight_decay": 0.0})
    optimizer = build_drafter_optimizer(model, config)

    muon_before = model.fc1.weight.detach().clone()
    adamw_before = model.norm.weight.detach().clone()
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()

    assert not torch.equal(muon_before, model.fc1.weight)
    assert not torch.equal(adamw_before, model.norm.weight)
    assert "momentum_buffer" in optimizer.state[model.fc1.weight]
    assert "exp_avg" in optimizer.state[model.norm.weight]

    state_dict = copy.deepcopy(optimizer.state_dict())
    restored = build_drafter_optimizer(model, config)
    restored.load_state_dict(state_dict)
    assert len(restored.state) == len(optimizer.state)


def test_unsupported_optimizer_raises() -> None:
    model = _DraftModel()
    config = OmegaConf.create({"optimizer": "sgd", "lr": 1e-4})

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_drafter_optimizer(model, config)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _single_process_gloo_group():
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the FSDP1 optimizer tests")
    if dist.is_initialized():
        if dist.get_world_size() > 1:
            pytest.skip("FSDP1 optimizer tests need a single-rank group")
        yield
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _wrap_fsdp1(use_orig_params: bool):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    try:
        return FSDP(
            _DraftModel(),
            use_orig_params=use_orig_params,
            device_id=None,
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
        )
    except RuntimeError as exc:
        # Pure-CPU torch builds cannot construct FSDP1 (it needs an accelerator).
        pytest.skip(f"FSDP1 is unavailable on this host: {exc}")


def _fsdp1_supported() -> bool:
    """Whether this host can construct an FSDP1 module at all."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        return False
    initialized = dist.is_initialized()
    if not initialized:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        FSDP(_DraftModel(), use_orig_params=True, device_id=None)
        return True
    except RuntimeError:
        return False
    finally:
        if not initialized:
            dist.destroy_process_group()


def test_muon_rejects_fsdp1_flat_parameters() -> None:
    with _single_process_gloo_group():
        model = _wrap_fsdp1(use_orig_params=False)

        with pytest.raises(ValueError, match="not supported with FSDP1"):
            build_drafter_optimizer(
                model, OmegaConf.create({"optimizer": "muon", "lr": 1e-4})
            )


def test_muon_rejects_fsdp1_original_parameters() -> None:
    # Muon is rejected for FSDP1 regardless of use_orig_params: a single-rank
    # NO_SHARD wrap may keep 2D views, but multi-rank FULL_SHARD does not, so the
    # strategy must not silently fall back to AdamW.
    with _single_process_gloo_group():
        model = _wrap_fsdp1(use_orig_params=True)

        with pytest.raises(ValueError, match="not supported with FSDP1"):
            build_drafter_optimizer(
                model, OmegaConf.create({"optimizer": "muon", "lr": 1e-4})
            )


def _multirank_fsdp1_muon_child(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = _wrap_fsdp1(use_orig_params=True)
        # Real FULL_SHARD exposes 1D shards even with use_orig_params=true, so the
        # dimension-based split would route fc1.weight to AdamW.
        fc1_ndims = [p.ndim for n, p in model.named_parameters() if n.endswith("fc1.weight")]
        assert fc1_ndims == [1], fc1_ndims
        build_drafter_optimizer(
            model, OmegaConf.create({"optimizer": "muon", "lr": 1e-4})
        )
    finally:
        dist.destroy_process_group()


def test_muon_rejects_multirank_fsdp1() -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the FSDP1 optimizer tests")
    if not _fsdp1_supported():
        pytest.skip("FSDP1 needs an accelerator device on this torch build")
    import torch.multiprocessing as mp

    # A missing rejection would silently build an all-AdamW optimizer; the child
    # raises so mp.spawn surfaces it here.
    with pytest.raises(Exception, match="not supported with FSDP1"):
        mp.spawn(
            _multirank_fsdp1_muon_child,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
        )


def test_muon_rejects_fsdp1_policy_without_fsdp(monkeypatch) -> None:
    # Independent of FSDP availability: the Muon split must refuse FSDP1 wrappers.
    import verl_speco.backends.optimizers as optimizers_module

    monkeypatch.setattr(optimizers_module, "_is_fsdp1_wrapped", lambda model: True)
    with pytest.raises(ValueError, match="not supported with FSDP1"):
        build_drafter_optimizer(
            _DraftModel(), OmegaConf.create({"optimizer": "muon", "lr": 1e-4})
        )


class _AllParamsModel(nn.Module):
    """Every parameter receives a gradient, so optimizer state is complete."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 8))

    def forward(self, hidden):
        return self.net(hidden)


@pytest.mark.parametrize(
    "config",
    [
        {"optimizer": "muon", "lr": -1e-4},
        {"optimizer": "muon", "lr": 1e-4, "muon_momentum": 1.0},
        {"optimizer": "muon", "lr": 1e-4, "muon_ns_steps": 0},
        {"optimizer": "muon", "lr": 1e-4, "muon_ns_steps": 100},
        {"optimizer": "muon", "lr": 1e-4, "muon_weight_decay": -0.1},
        {"optimizer": "muon", "lr": 1e-4, "weight_decay": -0.1},
        {"optimizer": "muon", "lr": 1e-4, "adamw_betas": [1.0, 0.9]},
        {"optimizer": "muon", "lr": 1e-4, "adamw_betas": [0.9]},
    ],
)
def test_muon_rejects_invalid_hyperparameters(config) -> None:
    with pytest.raises(ValueError):
        build_drafter_optimizer(_DraftModel(), OmegaConf.create(config))


def test_muon_rejects_ns_steps_at_kernel_limit() -> None:
    """PyTorch's Muon kernel raises for ns_steps >= 100; fail fast instead."""
    with pytest.raises(ValueError, match=r"muon_ns_steps must be in \[1, 100\)"):
        build_drafter_optimizer(
            _DraftModel(),
            OmegaConf.create(
                {"optimizer": "muon", "lr": 1e-4, "muon_ns_steps": 100}
            ),
        )


def test_unknown_optimizer_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_drafter_optimizer(
            _DraftModel(), OmegaConf.create({"optimizer": "adam", "lr": 1e-4})
        )


def test_muon_adamw_betas_are_configurable() -> None:
    optimizer = build_drafter_optimizer(
        _DraftModel(),
        OmegaConf.create({"optimizer": "muon", "lr": 1e-3, "adamw_betas": [0.8, 0.9]}),
    )
    adamw_group = next(g for g in optimizer.param_groups if not g["use_muon"])
    assert tuple(adamw_group["betas"]) == (0.8, 0.9)


def test_muon_adamw_name_hints_are_configurable() -> None:
    optimizer = build_drafter_optimizer(
        _DraftModel(),
        OmegaConf.create(
            {
                "optimizer": "muon",
                "lr": 1e-3,
                "muon_adamw_name_hints": ["fc1"],
            }
        ),
    )
    muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
    # fc1.weight is hinted back to AdamW, leaving only fc2.weight on Muon.
    assert len(muon_group["params"]) == 1


def test_muon_steps_with_ddp() -> None:
    with _single_process_gloo_group():
        model = torch.nn.parallel.DistributedDataParallel(_DraftModel())
        optimizer = build_drafter_optimizer(
            model, OmegaConf.create({"optimizer": "muon", "lr": 1e-3})
        )
        assert len(optimizer.param_groups) == 2
        model(torch.randn(4, 8)).sum().backward()
        optimizer.step()


def test_muon_steps_with_fsdp2_dtensor_parameters() -> None:
    with _single_process_gloo_group():
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cpu", (1,))
        model = _DraftModel()
        for module in model.children():
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        optimizer = build_drafter_optimizer(
            model, OmegaConf.create({"optimizer": "muon", "lr": 1e-3})
        )
        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        assert len(muon_group["params"]) == 2
        model(torch.randn(4, 8)).sum().backward()
        optimizer.step()


def _multirank_fsdp2_muon_child(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        mesh = init_device_mesh("cpu", (world_size,))
        model = _DraftModel()
        for module in model.children():
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)

        # Real FULL_SHARD splits each parameter across ranks even though the
        # logical DTensor keeps its original 2D shape, so the dimension-based
        # split must still route the hidden matrices to Muon.
        sharded = [
            name
            for name, param in model.named_parameters()
            if param.to_local().shape != param.shape
        ]
        assert sharded, "FSDP2 did not shard any parameter"

        optimizer = build_drafter_optimizer(
            model, OmegaConf.create({"optimizer": "muon", "lr": 1e-3})
        )
        assert len(optimizer.param_groups) == 2
        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        assert len(muon_group["params"]) == 2, len(muon_group["params"])

        model(torch.randn(4, 8)).sum().backward()
        optimizer.step()
    finally:
        dist.destroy_process_group()


def test_muon_steps_with_multirank_fsdp2() -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is required for the FSDP2 optimizer tests")
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError as exc:  # pragma: no cover - torch build without FSDP2
        pytest.skip(f"FSDP2 is unavailable on this torch build: {exc}")
    import torch.multiprocessing as mp

    mp.spawn(_multirank_fsdp2_muon_child, args=(2, _free_port()), nprocs=2, join=True)


def test_muon_optimizer_checkpoint_roundtrip(tmp_path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    model = _AllParamsModel()
    optimizer = build_drafter_optimizer(
        model, OmegaConf.create({"optimizer": "muon", "lr": 1e-3})
    )
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = copy.deepcopy(get_optimizer_state_dict(model, optimizer, options=options))

    # Wipe the Muon momentum, restore it from the captured state, then confirm a
    # further step and a full DCP save/load round-trip both work.
    for param in optimizer.param_groups[0]["params"]:
        optimizer.state[param]["momentum_buffer"].zero_()
    set_optimizer_state_dict(model, optimizer, state, options=options)
    assert any(
        float(optimizer.state[param]["momentum_buffer"].abs().sum()) > 0
        for param in optimizer.param_groups[0]["params"]
    )

    class _Adapter:
        def state_dict(self):
            return get_optimizer_state_dict(model, optimizer, options=options)

        def load_state_dict(self, loaded):
            set_optimizer_state_dict(model, optimizer, loaded, options=options)

    checkpoint = str(tmp_path / "optimizer")
    dcp.save({"optimizer": _Adapter()}, checkpoint_id=checkpoint)
    dcp.load({"optimizer": _Adapter()}, checkpoint_id=checkpoint)

    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()


