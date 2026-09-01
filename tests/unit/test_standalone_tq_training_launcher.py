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

from pathlib import Path
import json
import threading

from omegaconf import OmegaConf
import pytest

from verl_speco.standalone_tq_training_launcher import (
    _preflight_input_file,
    _producer_max_samples,
    _target_final_layer_id,
    build_pipeline_commands,
    resolve_pipeline_config,
    run_pipeline,
    start_ray_session,
)
import verl_speco.tq_owner as tq_owner


def _training_args() -> list[str]:
    return [
        "data.train_files=/data/train.jsonl",
        "actor_rollout_ref.model.path=/models/Qwen3-8B",
        "actor_rollout_ref.rollout.drafter.model_path=/models/dspark",
        "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK",
        "actor_rollout_ref.rollout.drafter.training.max_steps=10",
    ]


def test_pipeline_config_derives_transport_identity_from_training_args() -> None:
    config = resolve_pipeline_config(_training_args(), environ={})

    assert config.input_path == "/data/train.jsonl"
    assert config.model_path == "/models/Qwen3-8B"
    assert config.tokenizer_path == "/models/Qwen3-8B"
    assert config.algorithm == "DSPARK"
    assert config.target_layer_ids == (1, 9, 17, 25, 33)
    assert config.vllm_endpoints == ("http://127.0.0.1:8000/v1",)
    assert config.run_id.startswith("dspark-")


def test_producer_max_samples_uses_remaining_total_steps() -> None:
    args = [
        *_training_args(),
        "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2",
        "speco.draft_training.nproc_per_node=4",
        "speco.draft_training.nnodes=1",
    ]

    assert _producer_max_samples(args, resumed_optimizer_step=6) == 32


def test_pipeline_config_reads_non_dspark_algorithm_from_training_args() -> None:
    args = [
        item.replace("speculative_algorithm=DSPARK", "speculative_algorithm=DFLASH")
        for item in _training_args()
    ]
    args.append(
        "actor_rollout_ref.rollout.drafter.training.dflash_target_layer_ids=[2,10,20]"
    )

    config = resolve_pipeline_config(args, environ={})

    assert config.algorithm == "DFLASH"
    assert config.target_layer_ids == (2, 10, 20)
    assert config.run_id.startswith("dflash-")


def test_pipeline_config_prefers_generic_producer_layer_ids() -> None:
    args = [
        *_training_args(),
        "speco.standalone_tq_producer.target_layer_ids=[3,11,21]",
        "actor_rollout_ref.rollout.drafter.training.dspark_target_layer_ids=[1,9,17]",
    ]

    config = resolve_pipeline_config(args, environ={})

    assert config.target_layer_ids == (3, 11, 21)


def test_pipeline_config_accepts_one_hydra_list_train_file() -> None:
    args = _training_args()
    args[0] = "data.train_files=['/data/train.jsonl']"

    config = resolve_pipeline_config(args, environ={})

    assert config.input_path == "/data/train.jsonl"


def test_pipeline_config_allows_missing_drafter_path_for_fresh_training() -> None:
    args = [
        item
        for item in _training_args()
        if not item.startswith("actor_rollout_ref.rollout.drafter.model_path=")
    ]

    config = resolve_pipeline_config(args, environ={})

    assert config.model_path == "/models/Qwen3-8B"


def test_pipeline_config_accepts_multiple_vllm_endpoints() -> None:
    config = resolve_pipeline_config(
        _training_args(),
        environ={
            "SPECO_VLLM_ENDPOINTS": (
                "[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]"
            )
        },
    )

    assert config.vllm_endpoints == (
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8001/v1",
    )


def test_target_final_layer_id_uses_local_model_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 48}}),
        encoding="utf-8",
    )

    assert _target_final_layer_id(str(tmp_path), (2, 10, 20)) == 48


def test_pipeline_config_rejects_multiple_train_files() -> None:
    args = _training_args()
    args[0] = "data.train_files=[a.jsonl,b.jsonl]"

    with pytest.raises(ValueError, match="exactly one train file"):
        resolve_pipeline_config(args, environ={})


def test_preflight_accepts_verl_prompt_parquet(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "train.parquet"
    input_path.write_bytes(b"PAR1-test-fixture")

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == input_path

        def iter_batches(self):
            class Batch:
                @staticmethod
                def to_pylist():
                    return [
                        {
                            "prompt": [{"role": "user", "content": "Solve Q"}],
                            "reward_model": {"ground_truth": "42"},
                        }
                    ]

            yield Batch()

    from verl_speco.producer import input_reader

    monkeypatch.setattr(
        input_reader.importlib,
        "import_module",
        lambda name: type("ParquetModule", (), {"ParquetFile": FakeParquetFile}),
    )

    _preflight_input_file(str(input_path))


def test_pipeline_commands_hide_and_replace_tq_overrides() -> None:
    args = [
        *_training_args(),
        "actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard",
        "actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id=user-value",
    ]
    config = resolve_pipeline_config(args, environ={})

    commands = build_pipeline_commands(
        config,
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )

    assert commands.owner[:3] == ["python", "-m", "verl_speco.tq_owner"]
    assert commands.vllm is not None
    assert commands.vllm[:3] == ["vllm", "serve", "/models/Qwen3-8B"]
    assert "ExampleHiddenStatesConnector" in " ".join(commands.vllm)
    assert commands.vllm_endpoints == ("http://127.0.0.1:8000/v1",)
    assert commands.producer[:3] == [
        "python",
        "-m",
        "verl_speco.standalone_tq_producer",
    ]
    assert commands.consumer[:3] == [
        "python",
        "-m",
        "verl_speco.draft_train_launcher",
    ]
    assert any(item.endswith("feature_store.type=tq") for item in commands.consumer)
    assert not any("run_id=user-value" in item for item in commands.consumer)
    assert any(f"run_id={config.run_id}" in item for item in commands.consumer)
    assert any(
        "vllm_endpoints=[http://127.0.0.1:8000/v1]" in item
        for item in commands.producer
    )
    assert any("max_samples=40" in item for item in commands.producer)


def test_pipeline_commands_pass_all_external_vllm_endpoints_to_producer() -> None:
    endpoints = "[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]"
    config = resolve_pipeline_config(
        _training_args(), environ={"SPECO_VLLM_ENDPOINTS": endpoints}
    )

    commands = build_pipeline_commands(
        config,
        _training_args(),
        ray_address="127.0.0.1:6379",
        python_executable="python",
    )

    # Multiple services are started by the dedicated shell script. The unified
    # launcher only verifies them and passes both URLs into the Producer pool.
    assert commands.vllm is None
    assert any(f"vllm_endpoints={endpoints}" in item for item in commands.producer)


def test_pipeline_commands_forward_producer_tuning_overrides() -> None:
    args = [
        *_training_args(),
        "speco.standalone_tq_producer.max_inflight_requests=32",
        "speco.standalone_tq_producer.per_endpoint_concurrency=8",
        "speco.standalone_tq_producer.max_feature_length=384",
    ]
    config = resolve_pipeline_config(args, environ={})

    commands = build_pipeline_commands(
        config,
        args,
        ray_address="127.0.0.1:6379",
        python_executable="python",
    )

    assert (
        "speco.standalone_tq_producer.max_inflight_requests=32"
        in commands.producer
    )
    assert (
        "speco.standalone_tq_producer.per_endpoint_concurrency=8"
        in commands.producer
    )
    assert "speco.standalone_tq_producer.max_feature_length=384" in commands.producer


class _FakeRuntimeContext:
    gcs_address = "127.0.0.1:61234"


class _FakeRay:
    def __init__(self) -> None:
        self.init_kwargs = None
        self.shutdown_called = False

    def init(self, **kwargs) -> None:
        self.init_kwargs = kwargs

    def get_runtime_context(self) -> _FakeRuntimeContext:
        return _FakeRuntimeContext()

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_ray_session_starts_local_control_plane_without_exposed_address() -> None:
    ray = _FakeRay()

    session = start_ray_session(
        environ={"RAY_ADDRESS": "172.51.9.253:35195"}, ray_module=ray
    )

    assert session.address == "127.0.0.1:61234"
    assert ray.init_kwargs == {
        "address": "local",
        "namespace": "speco-drafter",
        "include_dashboard": False,
    }
    session.close()
    assert ray.shutdown_called


class _FakeProcess:
    def __init__(self, role: str) -> None:
        self.role = role
        self.returncode = 0 if role == "consumer" else None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def test_pipeline_starts_owner_then_consumer_then_producer() -> None:
    started: list[str] = []
    processes: list[_FakeProcess] = []
    child_environments: list[dict[str, str]] = []
    vllm_started = False

    def fake_popen(command, *, env):
        nonlocal vllm_started
        if command[0] == "vllm":
            role = "vllm"
            vllm_started = True
        else:
            module = command[2]
            role = {
                "verl_speco.tq_owner": "owner",
                "verl_speco.draft_train_launcher": "consumer",
                "verl_speco.standalone_tq_producer": "producer",
            }[module]
        process = _FakeProcess(role)
        started.append(role)
        processes.append(process)
        child_environments.append(dict(env))
        if role == "owner":
            Path(env["SPECO_TQ_OWNER_READY_FILE"]).touch()
        return process

    config = resolve_pipeline_config(_training_args(), environ={})
    commands = build_pipeline_commands(
        config,
        _training_args(),
        ray_address="127.0.0.1:61234",
    )

    assert (
        run_pipeline(
            commands,
            ray_address="127.0.0.1:61234",
            environ={"RAY_ADDRESS": "172.51.9.253:35195"},
            popen=fake_popen,
            endpoint_ready=lambda _: vllm_started,
        )
        == 0
    )
    assert started == ["vllm", "owner", "consumer", "producer"]
    assert all(env["RAY_ADDRESS"] == "127.0.0.1:61234" for env in child_environments)
    assert all(process.poll() is not None for process in processes)


def test_owner_writes_internal_ready_file(monkeypatch, tmp_path) -> None:
    ready_file = tmp_path / "owner.ready"
    monkeypatch.setenv("SPECO_TQ_OWNER_READY_FILE", str(ready_file))
    monkeypatch.setattr(tq_owner, "configure_transfer_queue", lambda config: True)
    monkeypatch.setattr(tq_owner, "connect_ray_cluster", lambda *args: None)
    monkeypatch.setattr(tq_owner, "start_transfer_queue_owner", lambda config: None)
    monkeypatch.setattr(tq_owner, "close_transfer_queue_owner", lambda: None)
    monkeypatch.setattr(tq_owner, "publish_owner_ready", lambda *args: "ready-key")
    stop_event = threading.Event()
    stop_event.set()
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "training": {
                            "transfer_queue": {
                                "enable": True,
                                "run_id": "test-run",
                                "schema_version": 1,
                                "ray": {
                                    "address": "127.0.0.1:6379",
                                    "namespace": "speco-drafter",
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    assert tq_owner.run_owner(config, stop_event=stop_event) == 0
    assert ready_file.is_file()
