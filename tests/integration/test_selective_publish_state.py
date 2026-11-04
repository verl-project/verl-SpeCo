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
"""Compare the actual trainer's legacy and selective publication methods."""

import ast
import fnmatch
import logging
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from verl_speco.trainer.publish_state import export_fsdp2_publish_state


def _trainer_type():
    # Import actual method bodies without requiring the full verl/Ray stack.
    path = Path(__file__).parents[2] / "verl_speco" / "trainer" / "base_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DrafterBaseTrainer"
    )
    methods = {
        "_is_frozen_publish_param",
        "_get_full_model_state_dict",
        "_get_trainable_state_dict",
        "get_model_state_dict",
        "prepare_model_state_dict_for_publish",
        "pop_model_state_dict_for_publish",
        "clear_pending_publish_state_dict",
    }
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    cls.bases, cls.decorator_list = [], []
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)

    def legacy_full_state(model, **kwargs):
        state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        return state if dist.get_rank() == 0 else {}

    namespace = {
        "torch": torch,
        "dist": dist,
        "FSDP": FSDP,
        "fnmatch": fnmatch,
        "logger": logging.getLogger(__name__),
        "device_name": "cuda",
        "get_fsdp_full_state_dict": legacy_full_state,
        "load_fsdp_model_to_gpu": lambda model: None,
        "offload_fsdp_model_to_cpu": lambda model: None,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - repository methods under test
    return namespace["DrafterBaseTrainer"]


def _model(mesh):
    model = torch.nn.Module()
    for name, rows in [
        ("embed_tokens", 8),
        ("lm_head", 6),
        ("fc", 4),
        ("frozen", 7),
        ("confidence_head", 1),
        ("markov_head", 3),
    ]:
        layer = torch.nn.Linear(4, rows, bias=False)
        weight = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 37
        layer.weight = torch.nn.Parameter(
            distribute_tensor(weight, mesh, [Shard(0)])
            if mesh is not None
            else weight.clone()
        )
        setattr(model, name, layer)
    model.alias_fc = model.fc
    model.register_buffer("floating_buffer", torch.tensor([0.1, 0.2]))
    model.register_buffer("t2d", torch.arange(4))
    model.register_buffer("ephemeral", torch.tensor([9.0]), persistent=False)
    return model


def _distributed_check(rank, rendezvous, backend):
    device_type = "cuda" if backend == "nccl" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = dist.device_mesh.init_device_mesh(device_type, (2,))
        model = _model(mesh)
        trainer = _trainer_type()()
        trainer._record_gpu_transfer = lambda scope, stats: None
        trainer.rank = rank
        trainer.clear_pending_publish_state_dict()
        trainer.model = model
        trainer.training_device_mesh = mesh
        trainer._frozen_param_names = ["frozen."]
        trainer._is_ddp_wrapped = lambda: False
        config = {"publish_dtype": "bf16", "gpu_publish_cast_before_copy": True}
        trainer.config = SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(training=config))
        )
        cases = [
            (False, False, None, None),
            (False, False, "bf16", None),
            (True, False, "fp16", None),
            (True, True, "fp32", None),
            (False, False, "bf16", "*fc.weight"),
            (
                False,
                False,
                "bf16",
                ["confidence_head.*", "markov_head.*", "floating_buffer"],
            ),
            (False, False, "bf16", "does_not_match"),
        ]
        for trains_head, trains_embeddings, dtype, patterns in cases:
            trainer.backend = SimpleNamespace(
                trains_draft_lm_head=trains_head,
                trains_draft_embeddings=trains_embeddings,
            )
            config.update(
                publish_dtype=dtype,
                publish_param_name_patterns=patterns,
                publish_trainable_only_export=False,
            )
            reference = trainer.get_model_state_dict()
            if reference is not None:
                reference = {name: tensor.clone() for name, tensor in reference.items()}
            before = {
                name: value.detach().clone() for name, value in model.named_parameters()
            }
            gathered_shapes = []
            original = DTensor.full_tensor

            def track_gather(
                tensor, *args, _shapes=gathered_shapes, _original=original, **kwargs
            ):
                _shapes.append(tuple(tensor.shape))
                return _original(tensor, *args, **kwargs)

            DTensor.full_tensor = track_gather
            config["publish_trainable_only_export"] = True
            try:
                actual = trainer.get_model_state_dict()
            finally:
                DTensor.full_tensor = original
            assert (7, 4) not in gathered_shapes  # frozen entries never gather
            if not trains_embeddings:
                assert (8, 4) not in gathered_shapes
            if not trains_head:
                assert (6, 4) not in gathered_shapes
            if patterns == "does_not_match":
                assert gathered_shapes == []
            assert (actual is None) == (reference is None)
            if actual is not None:
                assert set(actual) == set(reference)
                assert "t2d" not in actual and "ephemeral" not in actual
                for name, tensor in actual.items():
                    assert tensor.dtype == reference[name].dtype
                    assert tensor.device.type == "cpu" and tensor.is_contiguous()
                    torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
                if patterns is None:
                    assert "confidence_head.weight" in actual
                    assert "markov_head.weight" in actual
                    assert "floating_buffer" in actual
                    assert "alias_fc.weight" in actual
                    old_consumer, new_consumer = _model(None), _model(None)
                    old_load = old_consumer.load_state_dict(reference, strict=False)
                    new_load = new_consumer.load_state_dict(actual, strict=False)
                    assert old_load == new_load
                    results = []
                    for consumer in (old_consumer, new_consumer):
                        optimizer = torch.optim.SGD(consumer.parameters(), lr=0.01)
                        hidden = consumer.fc(
                            torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11
                        )
                        loss = (
                            consumer.markov_head(hidden).square().mean()
                            + consumer.confidence_head(hidden).square().mean()
                        )
                        loss.backward()
                        gradient = consumer.fc.weight.grad.clone()
                        optimizer.step()
                        results.append(
                            (
                                loss.detach(),
                                gradient,
                                consumer.fc.weight.detach().clone(),
                            )
                        )
                    for old, new in zip(*results, strict=True):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                assert parameter.grad is None and parameter.requires_grad
            # Update the model only after both snapshots: neither may alias it.
            with torch.no_grad():
                model.fc.weight.to_local().add_(1)
                model.floating_buffer.add_(1)
            if actual is not None:
                for name in actual:
                    torch.testing.assert_close(
                        actual[name], reference[name], rtol=0, atol=0
                    )
        # Exercise the real versioned publication lifecycle with the new path.
        config.update(
            publish_param_name_patterns=None, publish_trainable_only_export=False
        )
        reference = trainer.get_model_state_dict()
        config["publish_trainable_only_export"] = True
        assert trainer.prepare_model_state_dict_for_publish(42) == (rank == 0)
        ready, snapshot = trainer.pop_model_state_dict_for_publish(43)
        assert not ready and snapshot is None
        assert trainer.prepare_model_state_dict_for_publish(42) == (rank == 0)
        ready, snapshot = trainer.pop_model_state_dict_for_publish(42)
        assert ready
        if rank == 0:
            assert set(snapshot) == set(reference)
            for name in snapshot:
                torch.testing.assert_close(
                    snapshot[name], reference[name], rtol=0, atol=0
                )
        else:
            assert snapshot is None
        ready, snapshot = trainer.pop_model_state_dict_for_publish(42)
        assert not ready and snapshot is None
        checkpoint = trainer._get_full_model_state_dict()
        if rank == 0:
            assert {
                "embed_tokens.weight",
                "lm_head.weight",
                "frozen.weight",
                "t2d",
            } <= set(checkpoint)
        original = DTensor.full_tensor

        def fail_gather(*args, **kwargs):
            raise RuntimeError("injected publish gather failure")

        DTensor.full_tensor = fail_gather
        try:
            with pytest.raises(RuntimeError, match="injected publish gather failure"):
                trainer.get_model_state_dict()
        finally:
            DTensor.full_tensor = original
        # Unsupported/plain layouts decline before starting collectives.
        assert (
            export_fsdp2_publish_state(
                torch.nn.Linear(4, 2),
                is_frozen=lambda name: False,
                trains_lm_head=True,
                patterns=None,
                dtype=None,
            )
            is None
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_selective_publish_matches_legacy_state_on_two_ranks(tmp_path, backend):
    if backend == "nccl" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < 2
    ):
        pytest.skip("requires two CUDA GPUs and NCCL")
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("requires Gloo")
    torch.multiprocessing.spawn(
        _distributed_check,
        args=((tmp_path / "rendezvous").as_uri(), backend),
        nprocs=2,
        join=True,
    )
