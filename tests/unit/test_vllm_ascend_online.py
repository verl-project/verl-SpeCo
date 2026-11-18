# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Exercise online control flow with real CPU tensors and injected execution."""

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.backends.vllm_ascend_engine import VllmAscendDSparkEngine
from verl_speco.integration.deepseek_v4_online import (
    TARGET_OUTPUT_NAMES,
    EmbeddedDraftLoader,
    apply_shared_token_weights,
    apply_target_output,
    prepare_online_resume,
    runtime_weight_name,
    select_target_output,
)
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract
from verl_speco.trainer.vllm_ascend_online import OnlineMetrics, VllmAscendOnlineMixin


@pytest.fixture
def contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 2,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 3,
                "dspark_target_layer_ids": [0, 1],
                "dspark_block_size": 2,
            }
        )
    )
    return DeepSeekV4Contract.from_model(tmp_path)


def teacher():
    model = torch.nn.Module()
    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale", "norm_weight"):
        model.register_parameter(name, torch.nn.Parameter(torch.zeros(1)))
    model.lm_head = torch.nn.Linear(2, 3, bias=False)
    return model


def shared_wrapper():
    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Module()
    wrapper.model.model = torch.nn.Module()
    wrapper.model.model.embed_tokens = torch.nn.Embedding(3, 2)
    wrapper.model.lm_head = torch.nn.Linear(2, 3, bias=False)
    wrapper.requires_grad_(False)
    wrapper.runtime_contract = SimpleNamespace(vocab_size=5, hidden_size=2)
    return wrapper


def target_state(with_shared=False):
    result = {name: torch.ones(1) for name in TARGET_OUTPUT_NAMES}
    result["lm_head.weight"] = torch.arange(10).float().reshape(5, 2)
    if with_shared:
        result["embed_tokens.weight"] = torch.arange(10).float().reshape(5, 2) + 20
    return result


def test_teacher_sync_refreshes_hc_norm_and_padded_vocab_without_partial_mutation():
    model = teacher()
    state = target_state()
    apply_target_output(model, state, tp_rank=1, tp_size=2)
    assert torch.equal(
        model.lm_head.weight, torch.tensor([[6.0, 7.0], [8.0, 9.0], [0.0, 0.0]])
    )
    assert model.hc_head_fn.item() == model.norm_weight.item() == 1
    before = {name: value.clone() for name, value in model.state_dict().items()}
    state["norm_weight"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="norm_weight"):
        apply_target_output(model, state, tp_rank=1, tp_size=2)
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_actor_output_export_preserves_fp32_hc_and_owns_cpu_storage():
    state = target_state()
    raw = [(TARGET_OUTPUT_NAMES[name], value) for name, value in state.items()]
    exported = select_target_output(iter(raw))
    state["hc_head_fn"].zero_()
    assert exported["hc_head_fn"].item() == 1
    assert exported["hc_head_fn"].dtype == torch.float32
    with pytest.raises(ValueError, match="Incomplete"):
        select_target_output(raw[:-1])


@pytest.mark.parametrize("native_indices", [False, True])
def test_frozen_shared_tokens_track_actor_versions_in_place(native_indices):
    model = shared_wrapper()
    if native_indices:
        for module in [model.model.model.embed_tokens, model.model.lm_head]:
            module.weight = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
            module.shard_indices = SimpleNamespace(
                org_vocab_start_index=3, org_vocab_end_index=5
            )
    params = [model.model.model.embed_tokens.weight, model.model.lm_head.weight]
    pointers = [p.data_ptr() for p in params]
    for version in [1, 2]:
        state = target_state(with_shared=True)
        for key in ["embed_tokens.weight", "lm_head.weight"]:
            state[key] *= version
        apply_shared_token_weights(model, state, tp_rank=1, tp_size=2)
        for parameter, name in zip(
            params, ["embed_tokens.weight", "lm_head.weight"], strict=True
        ):
            assert torch.equal(parameter[:2], state[name][3:5])
            assert parameter[2:].eq(0).all()
        assert [p.data_ptr() for p in params] == pointers
        assert all(not p.requires_grad for p in params)
    before = [p.clone() for p in params]
    state["lm_head.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="lm_head.weight"):
        apply_shared_token_weights(model, state, tp_rank=1, tp_size=2)
    assert all(torch.equal(p, old) for p, old in zip(params, before, strict=True))


def test_online_export_requires_embedding_and_returns_owned_cpu_tensor():
    state = target_state(with_shared=True)
    raw = [
        (TARGET_OUTPUT_NAMES.get(key, "model.embed_tokens.weight"), value)
        for key, value in state.items()
    ]
    exported = select_target_output(iter(raw), include_shared_tokens=True)
    state["embed_tokens.weight"].zero_()
    assert exported["embed_tokens.weight"].eq(0).logical_not().all()
    with pytest.raises(ValueError, match="embed_tokens.weight"):
        select_target_output(iter(raw[:-1]), include_shared_tokens=True)


def test_online_sync_rejects_teacher_without_shared_embedding(contract):
    session = make_session(contract)
    with pytest.raises(ValueError, match="shared embedding"):
        session.sync_target_output({"ds4_target_output": target_state()}, 1)


def test_fsdp2_actor_sync_materializes_only_selected_tensors(monkeypatch):
    from verl_speco.integration import deepseek_v4_online as online
    from verl_speco.integration import rollout_publish

    monkeypatch.setattr(
        rollout_publish, "actor_training_backend_name", lambda _: "fsdp2"
    )
    state = target_state(with_shared=True)
    materialized = []

    class Lazy:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def full_tensor(self):
            assert self.name in state, "unrelated decoder weight was gathered"
            materialized.append(self.name)
            return self.value

    tensors = [
        (TARGET_OUTPUT_NAMES.get(key, "model.embed_tokens.weight"), Lazy(key, value))
        for key, value in state.items()
    ]
    tensors.insert(1, ("model.layers.0.mlp.experts.weight", Lazy("unrelated", None)))
    engine = SimpleNamespace(
        module=SimpleNamespace(named_parameters=lambda: iter(tensors)),
        _is_offload_param=False,
    )
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine), config={})
    result = online.export_target_output(worker)
    assert set(materialized) == set(state) == set(result)
    assert all(torch.equal(result[key], value) for key, value in state.items())


@pytest.mark.parametrize("layout", ["hf", "checkpoint"])
def test_actor_teacher_export_supports_actual_hf_and_converted_checkpoint_names(layout):
    from verl_speco.integration.deepseek_v4_online import (
        TARGET_OUTPUT_CHECKPOINT_NAMES,
        TARGET_OUTPUT_HF_NAMES,
    )

    state = target_state()
    names = dict(TARGET_OUTPUT_NAMES)
    names.update(
        TARGET_OUTPUT_HF_NAMES if layout == "hf" else TARGET_OUTPUT_CHECKPOINT_NAMES
    )
    raw = [(names[name], value) for name, value in state.items()]
    if layout == "hf":
        raw = [("_fsdp_wrapped_module." + name, value) for name, value in raw]
    # These are not aliases for the target's final output head/norm.
    raw.extend(
        [
            ("model.layers.42.hc_head_fn", torch.zeros(2)),
            ("model.layers.41.self_attn.compressor.norm.weight", torch.zeros(2)),
            ("mtp.2.hc_head_fn", torch.zeros(2)),
        ]
    )
    result = select_target_output(iter(raw))
    assert set(result) == set(state)
    assert all(torch.equal(result[name], value) for name, value in state.items())
    assert result["hc_head_fn"].dtype == state["hc_head_fn"].dtype
    with pytest.raises(ValueError, match="Ambiguous"):
        select_target_output(raw + [("model.hc_head_fn", torch.ones(1))])


class Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        self.embed_tokens = torch.nn.Embedding(4, 2)

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            if name == "mtp.0.proj.weight":
                with torch.no_grad():
                    self.proj.weight.copy_(value)
                loaded.add("proj.weight")
        return loaded


def test_embedded_update_is_in_place_and_rejects_incomplete_revision():
    model = Draft()
    pointer = model.proj.weight.data_ptr()
    session = EmbeddedDraftLoader(model)
    session.receive([(runtime_weight_name(0, 1, "proj.weight"), torch.ones(2, 2))])
    assert session.finish() == 1
    assert model.proj.weight.data_ptr() == pointer
    assert torch.equal(model.proj.weight, torch.ones(2, 2))
    session = EmbeddedDraftLoader(model)
    with pytest.raises(ValueError, match="Unexpected"):
        session.receive([(runtime_weight_name(0, 1, "unknown"), torch.ones(1))])
    with pytest.raises(RuntimeError, match="Incomplete"):
        session.finish()
    session = EmbeddedDraftLoader(model)
    with pytest.raises(ValueError, match="runtime TP shards"):
        session.receive([("model.norm.weight", torch.ones(2))])


def test_resume_requires_matching_actor_boundary(tmp_path):
    cfg = OmegaConf.create({"vllm_ascend": {"resume_from": None}})
    path = tmp_path / "draft_step_4"
    path.mkdir()
    (path / "speco_online.json").write_text(json.dumps({"global_step": 4}))
    prepare_online_resume(cfg, str(tmp_path), 4)
    assert cfg.vllm_ascend.resume_from == str(path)
    with pytest.raises(ValueError, match="actor RL step"):
        prepare_online_resume(cfg, str(tmp_path), 5)


def make_session(contract):
    session = VllmAscendOnlineMixin.__new__(VllmAscendOnlineMixin)
    session.contract = contract
    session.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"max_seq_len": 8192}}}}
    )
    session.collected_data = deque(maxlen=20)
    session.current_rl_step = 1
    session._mark_buffer_changed = lambda: None
    session.batch_size = 1
    session.optimizer_steps_total = 0
    session.training_steps = 0
    session._online_cursor_step = None
    session._online_cursor = 0
    session._online_failed = False
    session.execution = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"drafter": {"training": {"max_steps": 10}}}}}
    )
    session.cfg = OmegaConf.create(
        {
            "deepspeed": {
                "gradient_accumulation_steps": 1,
                "train_micro_batch_size_per_gpu": 1,
            },
            "train": {},
        }
    )
    session.ascend_engine = None
    session.online_metrics = OnlineMetrics()
    return session


def collect(session, step):
    session.current_rl_step = step
    session.collect_online_data(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
            "prompts": torch.tensor([[10, 11]]),
            "hidden_positions": torch.arange(4),
            "global_step": step,
            "target_head_hidden_states": torch.full((4, 4), 2.0),
        },
        torch.ones(1, 4, 4),
    )


def test_two_rl_triggers_reuse_engine_optimizer_and_positive_masks(contract):
    session = make_session(contract)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    runtime = SimpleNamespace(global_steps=0, module=shared_wrapper())
    calls = []

    def numerical(
        *,
        model_engine,
        target_lm_head,
        dataloader,
        sampler,
        initial_global_step,
        initial_global_examples,
        save_checkpoints,
        **kwargs,
    ):
        batch = next(iter(dataloader))
        calls.append((initial_global_step, batch["loss_mask"].tolist()))
        assert torch.equal(batch["loss_mask"], batch["raw_mask"])
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
        model_engine.global_steps += 1
        return {
            "global_step": initial_global_step + 1,
            "global_examples": 1,
            "observed_nonzero_optimizer_lr": True,
            "observed_update": True,
        }

    engine = VllmAscendDSparkEngine(
        model_engine=runtime,
        teacher=teacher(),
        cfg=session.cfg,
        topology=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=2),
        device="cpu",
        tracker=None,
        logger=None,
        train_function=numerical,
        checkpoint_dir="unused",
    )
    session.ascend_engine = engine

    async def activate():
        return True

    session.activate_training_model = activate
    for step in (1, 2):
        session.collected_data.clear()
        collect(session, step)
        session.sync_target_output(
            {"ds4_target_output": target_state(with_shared=True)}, step
        )
        assert asyncio.run(session.training_step(step))
        assert not asyncio.run(
            session.training_step(step)
        )  # no accidental duplicate batch
    assert calls == [(0, [[0.0, 0.0, 1.0, 1.0]]), (1, [[0.0, 0.0, 1.0, 1.0]])]
    assert session.optimizer_steps_total == 2
    assert optimizer.state[parameter]["step"].item() == 2
    assert parameter.item() < 1.0
    with pytest.raises(ValueError, match="versions differ"):
        asyncio.run(session.training_step(3))


def test_runtime_tp_lanes_keep_shards_distinct_and_reject_wrong_topology():
    models = [Draft(), Draft()]
    sessions = [
        EmbeddedDraftLoader(model, tp_rank=rank, tp_size=2)
        for rank, model in enumerate(models)
    ]
    weights = [
        (
            runtime_weight_name(rank, 2, "proj.weight"),
            torch.full((2, 2), float(rank + 1)),
        )
        for rank in range(2)
    ]
    for session in sessions:
        session.receive(weights)
        assert session.finish() == 1
    assert models[0].proj.weight[0, 0].item() == 1
    assert models[1].proj.weight[0, 0].item() == 2
    with pytest.raises(ValueError, match="TP degrees differ"):
        EmbeddedDraftLoader(Draft()).receive(weights)
    with pytest.raises(ValueError, match="layout differs"):
        EmbeddedDraftLoader(Draft()).receive(
            [(runtime_weight_name(0, 1, "proj.weight"), torch.ones(2, 3))]
        )


def test_online_publish_uses_compat_transport():
    from verl_speco.integration.native_draft_update import resolve_weight_update_mode

    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {"training": {"backend": "vllm_ascend", "mode": "online"}}
            },
            "speco": {"runtime": {"weight_update_mode": "auto"}},
        }
    )
    assert resolve_weight_update_mode(cfg) == "compat"
    cfg.speco.runtime.weight_update_mode = "native"
    with pytest.raises(ValueError, match="compat"):
        resolve_weight_update_mode(cfg)


def _lifecycle_class(path, class_name, methods, namespace):
    """Execute real lifecycle methods without importing unavailable Ray/verl."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(path).read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in methods
    ]
    for method in selected:
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name)
            and decorator.id in {"staticmethod", "classmethod"}
        ]
    cls = ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)  # noqa: S102 -- execute trusted repository methods
    return namespace[class_name]


def test_actual_worker_collect_preflight_train_and_publish_lifecycle(contract):
    import copy
    import logging
    import time
    from contextlib import nullcontext

    from verl_speco.trainer.data_buffer import DataBuffer

    common = {
        "time": time,
        "logger": logging.getLogger(__name__),
        "torch": torch,
        "_preserve_process_rng_state": lambda device: nullcontext(),
        "_resolve_ray_object_ref": lambda value: value,
        "ray": SimpleNamespace(put=copy.deepcopy),
        "release_draft_weights_payload": lambda weights: {
            "num_weights": len(weights),
            "payload_cleared": 0,
            "allocator": "test",
            "action": "none",
            "heap_trimmed": 0,
            "elapsed_sec": 0,
            "memory_before": "",
            "memory_after": "",
        },
    }
    base = _lifecycle_class(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "clear_pending_publish_state_dict",
            "prepare_model_state_dict_for_publish",
            "pop_model_state_dict_for_publish",
            "get_training_data_status",
            "_filter_training_data_by_step",
        },
        common.copy(),
    )
    worker_class = _lifecycle_class(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        {
            "_commit_rollout_features",
            "_store_rollout_sample",
            "_drafter_training_mode",
            "preflight_drafter_training",
            "train_drafter",
            "maybe_publish",
        },
        common.copy(),
    )

    class Session(VllmAscendOnlineMixin, base):
        pass

    session = Session.__new__(Session)
    session.__dict__.update(make_session(contract).__dict__)
    session.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {"training": {"backend": "vllm_ascend", "mode": "online"}}
            }
        }
    )
    session.backend = SimpleNamespace(model_type="dspark")
    session.use_data_buffer = False
    session.data_buffer = DataBuffer(max_size=20)
    session.buffer_version = 0

    def changed():
        session.buffer_version += 1

    session._mark_buffer_changed = changed
    session.model = SimpleNamespace(zero_grad=lambda: None)
    published_parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([published_parameter], lr=0.01)
    runtime = SimpleNamespace(global_steps=0, module=shared_wrapper())

    def numerical(
        *,
        model_engine,
        target_lm_head,
        dataloader,
        sampler,
        initial_global_step,
        initial_global_examples,
        save_checkpoints,
        **kwargs,
    ):
        assert next(iter(dataloader))["loss_mask"].tolist() == [[0.0, 0.0, 1.0, 1.0]]
        optimizer.zero_grad()
        published_parameter.square().backward()
        optimizer.step()
        model_engine.global_steps += 1
        return {"global_step": initial_global_step + 1, "global_examples": 1}

    session.ascend_engine = VllmAscendDSparkEngine(
        model_engine=runtime,
        teacher=teacher(),
        cfg=session.cfg,
        topology=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=2),
        device="cpu",
        tracker=None,
        logger=None,
        train_function=numerical,
        checkpoint_dir="unused",
    )

    async def activate():
        return True

    session.activate_training_model = activate
    session.get_model_state_dict = lambda: {
        runtime_weight_name(0, 1, "proj.weight"): published_parameter.detach().clone()
    }
    session.clear_pending_publish_state_dict()
    worker = worker_class()
    worker.trainer = session
    worker.enable_drafter = worker.in_drafter_train_group = (
        worker.is_global_publish_leader
    ) = True
    worker.rank = 0
    worker.replica_rank = 0
    worker.worker_incarnation = "test-incarnation"
    worker.device_name = "cpu"
    worker.last_trained_step = None
    worker.config = session.config
    worker._collection_worker_result = lambda **kwargs: {
        **kwargs,
        "accepted_samples": 0,
        "rejected_samples": 0,
    }
    payload_values = []
    for step in (1, 2):
        session.current_rl_step = worker.last_global_step = step
        sample = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]]),
            "prompts": torch.tensor([[10, 11]]),
            "responses": torch.tensor([[12, 13, 14]]),
            "hidden_states": torch.ones(1, 4, 4),
            "hidden_positions": torch.arange(4),
            "target_head_hidden_states": torch.full((4, 4), 2.0),
            "global_step": step,
        }
        assert (
            worker._commit_rollout_features(f"collection-{step}", [sample])[
                "accepted_samples"
            ]
            == 1
        )
        session.sync_target_output(
            {"ds4_target_output": target_state(with_shared=True)}, step
        )
        plan = {
            "launch": True,
            "plan_id": str(step),
            "execution_strategy": "sync",
            "source_global_step": step,
            "required_target_version": step,
            "data_version": step,
            "max_batches": 1,
            "publish_after_success": True,
            "worker_snapshots": {
                "0": {
                    "worker_incarnation": worker.worker_incarnation,
                    "buffer_version": session.buffer_version,
                    "data_version": step,
                    "trainable_samples": 1,
                }
            },
        }
        assert asyncio.run(worker.preflight_drafter_training(plan))["ready"]
        result = asyncio.run(worker.train_drafter(plan))
        assert result["trained"] and result["optimizer_step"] == step
        payload = worker.maybe_publish()
        payload_values.append(
            payload["weights_ref"][runtime_weight_name(0, 1, "proj.weight")].item()
        )
        assert worker.maybe_publish() is None
    assert payload_values[1] < payload_values[0] < 1.0
    assert optimizer.state[published_parameter]["step"].item() == 2


@pytest.mark.parametrize("source", ["rollout", "actor"])
def test_online_constructor_uses_packaged_config_and_worker_topology(
    contract, tmp_path, source
):
    class Base:
        def __init__(self, config, backend, **kwargs):
            self.config = config
            self.backend = backend
            self.training_group_world_size = 2
            self.dp_group_world_size = 3

    class Trainer(VllmAscendOnlineMixin, Base):
        pass

    cfg = OmegaConf.create(
        {
            "model": {"path": str(tmp_path)},
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DSPARK",
                    "training": {
                        "collect_hidden_states_from_rollout": True,
                        "batch_size_per_gpu": 1,
                        "vllm_ascend": {
                            "work_dir": str(tmp_path / "work"),
                            "total_optimizer_steps": 10,
                            "overrides": {
                                "train": {"segment_length": None, "max_len": None},
                                "deepspeed": {
                                    "train_micro_batch_size_per_gpu": 1,
                                    "gradient_accumulation_steps": 1,
                                },
                            },
                        },
                    },
                }
            },
        }
    )
    if source == "actor":
        cfg.rollout.drafter.training.collect_hidden_states_from_rollout = False
        cfg.rollout.drafter.training.collect_hidden_states_from_old_logprob = True
    if source == "rollout":
        with pytest.raises(ValueError, match="removed"):
            Trainer(cfg)
        return
    trainer = Trainer(cfg)
    assert trainer._feature_source == source
    assert trainer.cfg.model.vllm_tensor_parallel_size == 2
    assert trainer.cfg.model.vllm_data_parallel_size == 3
    assert trainer.cfg.model.checkpoint_format == "bf16"
    assert trainer.cfg.model.dequantize_mtp_for_training is False
    assert trainer.cfg.deepspeed.zero_optimization.stage == 1
    assert trainer.identity["optimizer_horizon"] == 10
    cfg.rollout.drafter.training.batch_size_per_gpu = 2
    with pytest.raises(ValueError, match="microbatch"):
        Trainer(cfg)
    cfg.rollout.drafter.training.batch_size_per_gpu = 1
    cfg.rollout.drafter.training.vllm_ascend.overrides.train.dspark_enable_confidence_head = True
    with pytest.raises(ValueError, match="confidence-head export"):
        Trainer(cfg)


def test_training_rng_continues_when_worker_restores_ambient_rng(contract):
    from verl_speco.trainer.vllm_ascend_checkpoint import rank_rng_state

    session = make_session(contract)
    observed = []
    runtime = SimpleNamespace(global_steps=0, module=shared_wrapper())

    def numerical(
        *,
        model_engine,
        target_lm_head,
        dataloader,
        sampler,
        initial_global_step,
        initial_global_examples,
        save_checkpoints,
        **kwargs,
    ):
        observed.append(torch.rand(()).item())
        model_engine.global_steps += 1
        return {"global_step": initial_global_step + 1, "global_examples": 1}

    session.ascend_engine = VllmAscendDSparkEngine(
        model_engine=runtime,
        teacher=teacher(),
        cfg=session.cfg,
        topology=SimpleNamespace(tensor_parallel_rank=0, tensor_parallel_size=2),
        device="cpu",
        tracker=None,
        logger=None,
        train_function=numerical,
        checkpoint_dir="unused",
    )

    async def activate():
        return True

    session.activate_training_model = activate
    with torch.random.fork_rng():
        torch.manual_seed(123)
        expected = torch.rand(2).tolist()
        torch.manual_seed(123)
        session._numerical_rng = rank_rng_state()
        for step in (1, 2):
            collect(session, step)
            session.sync_target_output(
                {"ds4_target_output": target_state(with_shared=True)}, step
            )
            torch.manual_seed(9999)  # unrelated worker ambient state
            assert asyncio.run(session.training_step(step))
        assert observed == expected


def test_online_checkpoint_carries_progress_and_rank_rng(
    contract, tmp_path, monkeypatch
):
    import verl_speco.trainer.vllm_ascend_online as online
    from verl_speco.backends.vllm_ascend_engine import Progress
    from verl_speco.trainer.vllm_ascend_checkpoint import rank_rng_state

    monkeypatch.setattr(online.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(online.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        online.dist,
        "all_gather_object",
        lambda values, value: values.__setitem__(0, value),
    )
    session = make_session(contract)
    session.config.rollout.drafter.checkpoint_path = str(tmp_path)
    session.identity = {"contract": contract.identity()}
    session._numerical_rng = rank_rng_state()
    session.ascend_engine = SimpleNamespace(progress=Progress(global_step=3))
    saved = {}

    def save(root, tag, client_state):
        from pathlib import Path

        (Path(root) / tag).mkdir()
        saved.update(client_state)
        return True

    from contextlib import nullcontext

    monkeypatch.setattr(
        online, "bounded_checkpoint_writes", lambda *args: nullcontext()
    )
    session.model = SimpleNamespace(save_checkpoint=save)
    result = session.save_checkpoint(7)
    assert result["saved"]
    assert saved["speco_online"]["progress"]["global_step"] == 3
    assert len(saved["speco_online"]["rng_by_rank"]) == 1
    metadata = json.loads((tmp_path / "draft_step_7/speco_online.json").read_text())
    assert metadata["global_step"] == 7
    session._online_failed = True
    with pytest.raises(RuntimeError, match="failed"):
        session.save_checkpoint(8)


def test_embedded_validation_failure_never_mutates_live_weights():
    model = Draft()
    before = model.proj.weight.detach().clone()
    session = EmbeddedDraftLoader(model, tp_rank=0, tp_size=2)
    session.receive([(runtime_weight_name(0, 2, "proj.weight"), torch.ones(2, 2))])
    assert torch.equal(model.proj.weight, before)
    with pytest.raises(ValueError, match="Non-finite"):
        session.receive(
            [
                (
                    runtime_weight_name(1, 2, "proj.weight"),
                    torch.full((2, 2), float("nan")),
                )
            ]
        )
    with pytest.raises(RuntimeError, match="Incomplete"):
        session.finish()
    assert torch.equal(model.proj.weight, before)


def test_embedded_stages_own_copy_and_requires_every_tp_tensor():
    model = Draft()
    value = torch.ones(2, 2)
    session = EmbeddedDraftLoader(model, tp_rank=0, tp_size=2)
    session.receive([(runtime_weight_name(0, 2, "proj.weight"), value)])
    value.zero_()
    with pytest.raises(RuntimeError, match="Incomplete"):
        session.finish()
    session.receive([(runtime_weight_name(1, 2, "proj.weight"), torch.ones(2, 2))])
    session.finish()
    assert torch.equal(model.proj.weight, torch.ones(2, 2))


class CPUStateParking:
    """Inject the hardware boundary; real CPU Adam tensors retain object identity."""

    def __init__(self):
        self.state = "new"
        self.coverage_inventory = {"unsupported": []}
        self.probe = None
        self.saved = []
        self.events = []

    def synchronize(self):
        self.events.append("quiesce")

    def allocation_scope(self):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            assert self.state in {"new", "active"}
            self.state = "active"
            self.events.append("allocate")
            yield

        return scope()

    def check_coverage(self, engine):
        from verl_speco.trainer.ascend_state_parking import trainer_tensors

        return sum(
            value.numel() * value.element_size() for _, value in trainer_tensors(engine)
        )

    def park(self, engine):
        from verl_speco.trainer.ascend_state_parking import state_probe, trainer_tensors

        self.probe = state_probe(engine)
        self.saved = [
            (value, value.detach().clone()) for _, value in trainer_tensors(engine)
        ]
        with torch.no_grad():
            for value, _ in self.saved:
                value.zero_()
        self.state = "parked"
        self.events.append("park")
        return self.check_coverage(engine)

    def wake(self, engine):
        from verl_speco.trainer.ascend_state_parking import state_probe

        with torch.no_grad():
            for value, saved in self.saved:
                value.copy_(saved)
        assert state_probe(engine) == self.probe
        self.state = "active"
        self.events.append("wake")
        return self.probe


def managed_session(contract):
    import logging

    from verl_speco.trainer.vllm_ascend_lifecycle import AscendTrainingLifecycle

    cache_base = _lifecycle_class(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "clear_pending_publish_state_dict",
            "prepare_model_state_dict_for_publish",
            "pop_model_state_dict_for_publish",
        },
        {"torch": torch, "logger": logging.getLogger(__name__)},
    )

    class Session(VllmAscendOnlineMixin, cache_base):
        pass

    session = Session.__new__(Session)
    session.__dict__.update(make_session(contract).__dict__)
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    runtime = SimpleNamespace(
        module=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        global_steps=0,
        zero_grad=lambda: optimizer.zero_grad(set_to_none=True),
    )
    engine = SimpleNamespace(
        model_engine=runtime,
        teacher=teacher(),
        progress=SimpleNamespace(global_step=0),
    )
    parking = CPUStateParking()
    lifecycle = AscendTrainingLifecycle(
        parking=parking,
        gather=lambda item: [item],
        host_available=lambda: 10**9,
        device_free=lambda: 10**9,
        host="test-host",
        host_reserve_bytes=0,
    )
    session.data_buffer = deque(maxlen=20)
    session._state_parking_mode = "camem"
    session._state_lifecycle = lifecycle
    session.model = None

    async def initialize():
        assert lifecycle.state == "active"
        session.ascend_engine, session.model = engine, runtime
        return True

    session._initialize_training_model = initialize

    def numerical(step):
        assert lifecycle.state == "active"
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        scheduler.step()
        runtime.global_steps += 1
        engine.progress.global_step = runtime.global_steps
        session.optimizer_steps_total += 1
        return True

    session._training_step_active = numerical

    def read_weights():
        assert lifecycle.state == "active"
        return {"proj.weight": model.weight.detach().cpu().clone()}

    session.get_model_state_dict = read_weights
    return session, lifecycle, engine


def test_production_hooks_park_after_snapshot_and_reuse_real_adam_across_steps(
    contract,
):
    session, lifecycle, engine = managed_session(contract)
    assert asyncio.run(session.activate_training_model())
    initial_ids = tuple(
        id(item)
        for item in (
            engine,
            engine.model_engine.optimizer,
            engine.model_engine.lr_scheduler,
        )
    )
    asyncio.run(session.release_training_memory_after_activation())
    assert lifecycle.state == "parked"
    saved_weights = []
    for step in (1, 2):
        session.current_rl_step = step
        session.sync_target_output(
            {"ds4_target_output": target_state(with_shared=True)}, step
        )
        assert asyncio.run(session.training_step(step))
        assert session.prepare_model_state_dict_for_publish(step)
        asyncio.run(session.cleanup_training(clear_data=False))
        ready, payload = session.pop_model_state_dict_for_publish(step)
        assert ready
        snapshot = payload["proj.weight"]
        saved_weights.append(snapshot)
        assert lifecycle.state == "parked"
        assert (
            not engine.model_engine.module.weight.any()
        )  # stand-in for unmapped NPU memory
        assert snapshot.any()  # CPU publication remains readable after parking
        assert lifecycle.parking.probe["optimizer_steps"] == [float(step)]
    assert not torch.equal(*saved_weights)
    assert initial_ids == tuple(
        id(item)
        for item in (
            engine,
            engine.model_engine.optimizer,
            engine.model_engine.lr_scheduler,
        )
    )
    assert lifecycle.parking.probe["scheduler_state"]["last_epoch"] == 2
    assert lifecycle.parking.events.count("wake") == 2


def test_checkpoint_and_export_read_scope_wakes_and_reparks_without_rebuilding(
    contract,
):
    session, lifecycle, _engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    asyncio.run(session.release_training_memory_after_activation())
    calls = []

    def save(step, wait, *, defer_completion):
        assert lifecycle.state == "active"
        calls.append((step, wait, defer_completion))
        return {"saved": True}

    session._save_checkpoint_active = save
    assert session.save_checkpoint(7)["saved"]
    assert lifecycle.state == "parked"
    assert calls == [(7, True, False)]
    session._write_runtime_publish_shards_active = lambda path: (
        path if lifecycle.state == "active" else pytest.fail("Unmapped export read")
    )
    assert session.write_runtime_publish_shards("snapshot") == "snapshot"
    assert lifecycle.state == "parked"


def test_read_failure_restores_parked_state_and_preserves_original_error(contract):
    session, lifecycle, _engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    asyncio.run(session.release_training_memory_after_activation())

    def fail(*args):
        raise OSError("storage unavailable")

    session._write_runtime_publish_shards_active = fail
    with pytest.raises(OSError, match="storage unavailable"):
        session.write_runtime_publish_shards("snapshot")
    assert lifecycle.state == "parked"


def test_wake_requires_device_capacity_before_reading_unmapped_state(contract):
    session, lifecycle, _engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    asyncio.run(session.release_training_memory_after_activation())
    lifecycle.device_free = lambda: 0
    with pytest.raises(RuntimeError, match="wake_capacity"):
        asyncio.run(session.activate_training_model())
    assert lifecycle.state == "parked" and lifecycle.failed
    assert "wake" not in lifecycle.parking.events


def test_host_budget_rejects_before_any_rank_parks(contract):
    session, lifecycle, _engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    lifecycle.host_available = lambda: 0
    with pytest.raises(RuntimeError, match="Insufficient host memory"):
        asyncio.run(session.release_training_memory_after_activation())
    assert lifecycle.state == "active" and lifecycle.failed
    assert "park" not in lifecycle.parking.events


def test_repeated_cleanup_does_not_touch_unmapped_gradients(contract):
    session, lifecycle, engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    asyncio.run(session.cleanup_training())
    engine.model_engine.zero_grad = lambda: pytest.fail("Accessed parked runtime")
    asyncio.run(session.cleanup_training())
    assert lifecycle.state == "parked"


def test_remote_coverage_failure_prevents_local_parking(contract):
    session, lifecycle, _engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    phases = []

    def gather(value):
        phases.append(value)
        return [value, "Other rank lost state" if len(phases) == 2 else None]

    lifecycle.gather = gather
    with pytest.raises(RuntimeError, match="coverage failed"):
        asyncio.run(session.cleanup_training())
    assert lifecycle.state == "active" and lifecycle.failed
    assert "park" not in lifecycle.parking.events


@pytest.mark.parametrize("same_host", [False, True])
def test_parking_host_budget_sums_only_workers_sharing_the_host(contract, same_host):
    session, lifecycle, engine = managed_session(contract)
    asyncio.run(session.activate_training_model())
    size = lifecycle.parking.check_coverage(engine)
    lifecycle.host_available = lambda: size + 1

    def gather(value):
        if isinstance(value, dict) and "available" in value:
            return [
                value,
                {**value, "host": value["host"] if same_host else "other-host"},
            ]
        return [value, value]

    lifecycle.gather = gather
    if same_host:
        with pytest.raises(RuntimeError, match="Insufficient host memory"):
            asyncio.run(session.release_training_memory_after_activation())
        assert lifecycle.state == "active"
    else:
        asyncio.run(session.release_training_memory_after_activation())
        assert lifecycle.state == "parked"


def test_lifecycle_control_uses_reusable_cpu_group(monkeypatch, contract):
    import verl_speco.trainer.vllm_ascend_lifecycle as module

    parking = CPUStateParking()
    group = object()
    created, messages = [], []
    monkeypatch.setattr(module, "AscendStateParking", lambda: parking)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        module.dist, "new_group", lambda **kwargs: created.append(kwargs) or group
    )

    def gather(target, value, *, group):
        assert group is not None
        messages.append(value)
        target[0] = value

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    lifecycle = module.AscendTrainingLifecycle()
    assert lifecycle.gather("read metadata") == ["read metadata"]
    assert created == [{"backend": "gloo"}]
    assert messages == [None, "read metadata"]  # allocator setup, then control


@pytest.mark.parametrize("sleep_enabled", [False, True])
def test_camem_constructor_creates_control_at_worker_init_and_requires_rollout_sleep(
    contract, tmp_path, monkeypatch, sleep_enabled
):
    import verl_speco.trainer.vllm_ascend_lifecycle as lifecycle_module
    from verl_speco.trainer.deepseek_v4_resident_probe import trainer_config

    class Base:
        def __init__(self, config, backend, **kwargs):
            self.config = config
            self.training_group_world_size = 8
            self.dp_group_world_size = 2

    class Trainer(VllmAscendOnlineMixin, Base):
        pass

    cfg = trainer_config(
        str(tmp_path), str(tmp_path / "work"), 8, 1, state_parking="camem"
    )
    cfg.rollout.enable_sleep_mode = sleep_enabled
    created = []
    monkeypatch.setattr(
        lifecycle_module,
        "AscendTrainingLifecycle",
        lambda: created.append(1) or object(),
    )
    monkeypatch.setattr(lifecycle_module.dist, "get_world_size", lambda: 16)
    if not sleep_enabled:
        with pytest.raises(ValueError, match="enable_sleep_mode=true"):
            Trainer(cfg)
        assert created == []
    else:
        trainer = Trainer(cfg)
        assert created == [1]
        assert trainer._get_state_lifecycle() is trainer._state_lifecycle
        assert created == [1]
