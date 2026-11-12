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
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

pytest.importorskip("torch")

from verl_speco.standalone_tq_training_launcher import (
    _hidden_states_store_overrides,
    _preflight_input_file,
    _producer_max_samples,
    _resolve_hidden_states_store,
    _target_final_layer_id,
    _tq_backend_overrides,
    _vllm_capture_layer_ids,
    _vllm_kv_transfer_config,
    build_pipeline_commands,
    resolve_pipeline_config,
    run_pipeline,
    start_ray_session,
    validate_hidden_states_store,
    validate_tq_backend,
)
import verl_speco.tq_owner as tq_owner


@pytest.mark.parametrize("owned_group", [False, True])
def test_process_stop_limits_wait_and_signals_only_owned_group(monkeypatch, owned_group):
    import signal
    from verl_speco import standalone_tq_training_launcher as launcher

    monkeypatch.setattr(signal, "SIGKILL", getattr(signal, "SIGKILL", 9), raising=False)

    calls = []
    waits = []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise launcher.subprocess.TimeoutExpired("owned-child", timeout)
        return 0
    process = SimpleNamespace(pid=42, poll=lambda: None, wait=wait,
                              terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"))
    if owned_group:
        process._speco_process_group = 42
    def killpg(group, sig):
        if sig == 0:
            raise ProcessLookupError()
        calls.append((group, sig))
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="posix", getpgid=lambda pid: 42, killpg=killpg))
    launcher._stop_process(process)
    assert waits == [10, 10]
    assert calls == ([(42, signal.SIGTERM), (42, signal.SIGKILL)] if owned_group else ["terminate", "kill"])


@pytest.mark.parametrize("already_exited", [False, True])
def test_process_stop_retains_resources_when_descendants_may_survive(monkeypatch, already_exited):
    import signal
    from verl_speco import standalone_tq_training_launcher as launcher

    signals = []
    process = SimpleNamespace(pid=42, _speco_process_group=42,
                              poll=lambda: 0 if already_exited else None, wait=lambda timeout: 0)
    monkeypatch.setattr(launcher, "os", SimpleNamespace(
        name="posix", getpgid=lambda pid: 42, killpg=lambda group, sig: signals.append(sig)))
    with pytest.raises(RuntimeError, match="exit is unconfirmed"):
        launcher._stop_process(process)
    assert signals == ([0] if already_exited else [signal.SIGTERM, 0])


@pytest.mark.parametrize("hidden_store,runtime,is_reference", [
    ("file", "ray", True), ("mooncake", "ray", True), ("file", "subprocess", False),
])
def test_reference_mode_uses_existing_ray_prefetch(hidden_store, runtime, is_reference):
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": hidden_store}
    args = [*_training_args(), f"speco.draft_training.runtime_backend={runtime}"]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ=env), args, ray_address="10.0.0.1:6379", env=env,
    )
    kind = "actor_rollout_ref.rollout.drafter.training.transfer_queue.expected_feature.payload_kind=vllm_reference"
    assert (kind in commands.producer_overrides) is is_reference
    assert (kind in commands.consumer_overrides) is is_reference
    if is_reference:
        assert "actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_prefetch_batches=4" in commands.consumer_overrides
        assert "actor_rollout_ref.rollout.drafter.training.feature_store.type=ray_reference" in commands.consumer_overrides
        assert "actor_rollout_ref.rollout.drafter.training.feature_store.type=ray_reference" in commands.producer_overrides
        assert "actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=false" in commands.consumer_overrides
        assert "actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=false" in commands.producer_overrides


def test_launcher_keeps_producer_data_recipe_equal_on_both_sides():
    from verl_speco.standalone_ray_runtime import compose_runtime_config
    from verl_speco.trainer.draft_training_loop import _standalone_producer_shuffle
    from verl_speco.trainer.standalone_resume import producer_data_recipe
    args = [*_training_args(),
            "speco.standalone_tq_producer.stable_source_ids=true",
            "speco.standalone_tq_producer.max_sequence_length=2048",
            "speco.standalone_tq_producer.max_feature_length=256",
            "speco.standalone_tq_producer.on_truncated_supervision=drop",
            "speco.standalone_tq_producer.render_boundary.enabled=true"]
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                       ray_address="10.0.0.1:6379", env={})
    producer = compose_runtime_config(commands.producer_overrides)
    consumer = compose_runtime_config(commands.consumer_overrides)
    assert _standalone_producer_shuffle(consumer)["data_recipe"] == producer_data_recipe(
        producer.speco.standalone_tq_producer)


def test_reference_mode_rejects_disabled_prefetch():
    args = [*_training_args(), "actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_prefetch_batches=0"]
    with pytest.raises(ValueError, match="prefetch"):
        build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                ray_address="10.0.0.1:6379", env={})


def test_native_launcher_builds_bounded_reader_and_scheduler_settings():
    from verl_speco.standalone_ray_runtime import compose_runtime_config
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native",
           "SPECO_NATIVE_SOURCE_ENDPOINT": "10.0.0.1:1234",
           "SPECO_NATIVE_READER_ENDPOINTS": '["10.0.0.2:1234"]'}
    config = resolve_pipeline_config(_training_args(), environ=env)
    commands = build_pipeline_commands(config, _training_args(), ray_address="10.0.0.1:6379", env=env)
    producer = compose_runtime_config(commands.producer_overrides)
    consumer = compose_runtime_config(commands.consumer_overrides)
    assert producer.speco.standalone_tq_producer.hidden_states_store.backend == "native"
    assert producer.speco.standalone_tq_producer.max_reference_execution_slots == 4
    assert consumer.speco.draft_training.scheduler.per_rank_reference_ack
    training = consumer.actor_rollout_ref.rollout.drafter.training
    assert training.transfer_queue.consumer_batch_workers == 0
    assert list(training.target_feature_replay.hidden_states_store.native.endpoints) == ["10.0.0.2:1234"]
    assert commands.native_settings["run_id"] == config.run_id
    transfer = json.loads(commands.vllm[commands.vllm.index("--kv-transfer-config") + 1])
    assert transfer["kv_connector"] == "SpecoNativeHiddenStatesConnector"


@pytest.mark.parametrize("changes,args,error", [
    ({"SPECO_NATIVE_READER_ENDPOINTS": "[]"}, [], "one distinct endpoint"),
    ({"SPECO_NATIVE_SOURCE_MAX_SLOTS": "1"}, [], "SOURCE_MAX_SLOTS"),
    ({"SPECO_NATIVE_MAX_RECORDS": "1"}, [], "MAX_RECORDS"),
    ({}, ["speco.draft_training.scheduler.per_rank_reference_ack=false"], "per_rank_reference_ack"),
    ({}, ["actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_batch_workers=1"], "consumer_batch_workers"),
    ({}, ["speco.draft_training.runtime_backend=subprocess"], "Ray runtime"),
])
def test_native_launcher_rejects_unsafe_settings(changes, args, error):
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_NATIVE_SOURCE_ENDPOINT": "source:1",
           "SPECO_NATIVE_READER_ENDPOINTS": '["reader:1"]', **changes}
    args = [*_training_args(), *args]
    with pytest.raises(ValueError, match=error):
        build_pipeline_commands(resolve_pipeline_config(args, environ=env), args, ray_address="10.0.0.1:6379", env=env)


def test_external_native_launcher_uses_shared_run_identity_and_rank_endpoints():
    from verl_speco.standalone_tq_training_launcher import _validate_runtime_backend_topology
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379",
           "SPECO_VLLM_ENDPOINTS": "[http://10.0.0.3:8000/v1,http://10.0.0.4:8000/v1]",
           "SPECO_NATIVE_DIRECTORY": "run-directory", "SPECO_NATIVE_RUN_ID": "run-id",
           "SPECO_NATIVE_READER_ENDPOINTS": '["10.0.0.1:1234","10.0.0.2:1234"]'}
    args = [*_training_args(), "speco.draft_training.nnodes=2",
            "speco.draft_training.trainer_node_ips=[10.0.0.1,10.0.0.2]"]
    _validate_runtime_backend_topology("ray", args, environ=env)
    config = resolve_pipeline_config(args, environ=env)
    commands = build_pipeline_commands(config, args, ray_address="10.0.0.1:6379", env=env)
    assert config.run_id == "run-id" and commands.vllm is None
    assert commands.native_settings["directory"] == "run-directory"
    assert commands.native_settings["max_sources"] == 2
    with pytest.raises(ValueError, match="RUN_ID"):
        missing_id = {key: value for key, value in env.items() if key != "SPECO_NATIVE_RUN_ID"}
        build_pipeline_commands(resolve_pipeline_config(args, environ=missing_id), args,
                                ray_address="10.0.0.1:6379", env=missing_id)


def test_native_launcher_counts_prefixed_topology_overrides():
    from verl_speco.standalone_ray_runtime import compose_runtime_config
    args = [*_training_args(), "++speco.draft_training.nproc_per_node=2",
            "++actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=3"]
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_NATIVE_SOURCE_ENDPOINT": "source:1",
           "SPECO_NATIVE_READER_ENDPOINTS": '["reader:1","reader:2"]'}
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ=env), args,
                                       ray_address="ray:6379", env=env)
    producer = compose_runtime_config(commands.producer_overrides)
    consumer = compose_runtime_config(commands.consumer_overrides)
    assert producer.speco.standalone_tq_producer.max_reference_execution_slots == 6
    assert consumer.speco.draft_training.nproc_per_node == 2
    assert consumer.actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu == 3


def test_prefixed_producer_credits_reach_both_roles_and_internal_values_cannot_override():
    from verl_speco.standalone_ray_runtime import compose_runtime_config
    args = [*_training_args(), "++speco.standalone_tq_producer.max_published_references=8",
            "++speco.standalone_tq_producer.max_reference_execution_slots=8",
            "++actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id=wrong"]
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_NATIVE_SOURCE_ENDPOINT": "source:1",
           "SPECO_NATIVE_READER_ENDPOINTS": '["reader:1"]'}
    config = resolve_pipeline_config(args, environ=env)
    commands = build_pipeline_commands(config, args, ray_address="ray:6379", env=env)
    for overrides in (commands.producer_overrides, commands.consumer_overrides):
        composed = compose_runtime_config(overrides)
        assert composed.speco.standalone_tq_producer.max_reference_execution_slots == 8
        assert composed.speco.standalone_tq_producer.max_published_references == 8
        assert composed.actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id == config.run_id


def test_prefixed_runtime_backend_selects_subprocess_path():
    args = [*_training_args(), "++speco.draft_training.runtime_backend=subprocess"]
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                       ray_address="ray:6379", env={})
    assert commands.owner and not any(item.endswith("feature_store.type=ray_reference")
                                      for item in commands.consumer_overrides)
    assert not any("runtime_backend=" in item for item in commands.consumer_overrides)


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
    assert config.vllm_aux_hidden_state_layer_ids == (1, 9, 17, 25, 33)
    assert config.target_layer_ids == (0, 8, 16, 24, 32)
    assert config.vllm_endpoints == ("http://127.0.0.1:8000/v1",)
    assert config.run_id.startswith("dspark-")


@pytest.mark.parametrize("prefix", ["", "++"])
def test_pipeline_forwards_dflash_block_size_to_producer(prefix):
    key = "actor_rollout_ref.rollout.drafter.training.dflash_block_size"
    args = _training_args() + [f"{prefix}{key}=4"]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )
    assert f"{key}=4" in commands.producer_overrides


@pytest.mark.parametrize("override", [None, "false", "true"])
@pytest.mark.parametrize("prefix", ["", "++"])
def test_pipeline_forwards_prompt_hidden_state_option_to_producer(override, prefix):
    key = "actor_rollout_ref.rollout.drafter.training.preserve_prompt_hidden_states"
    args = _training_args()
    if override is not None:
        args.append(f"{prefix}{key}={override}")
    config = resolve_pipeline_config(args, environ={})
    commands = build_pipeline_commands(
        config, args, ray_address="10.0.0.1:6379", python_executable="python"
    )
    expected = override if override is not None else "true"
    assert f"{key}={expected}" in commands.producer_overrides


@pytest.mark.parametrize("first_prefix,last_prefix", [("", "++"), ("++", "")])
@pytest.mark.parametrize("last_value", ["false", "true"])
def test_pipeline_prompt_hidden_option_last_override_wins(
    first_prefix, last_prefix, last_value
):
    key = "actor_rollout_ref.rollout.drafter.training.preserve_prompt_hidden_states"
    first_value = "true" if last_value == "false" else "false"
    args = _training_args() + [
        f"{first_prefix}{key}={first_value}",
        f"{last_prefix}{key}={last_value}",
    ]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )
    assert f"{key}={last_value}" in commands.producer_overrides


@pytest.mark.parametrize("prefix", ["", "+", "++"])
@pytest.mark.parametrize("source", ["target_argmax", "TARGET_ARGMAX"])
def test_ce_only_target_argmax_expected_layout(prefix, source):
    key = "actor_rollout_ref.rollout.drafter.training.ce_target_source"
    args = _training_args() + [
        "actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0",
        f"{prefix}{key}={source}",
    ]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )
    assert any(
        item.endswith("expected_feature.hidden_states_layout=dflash_aux_plus_last")
        for item in commands.consumer_overrides
    )


@pytest.mark.parametrize(
    "l1_prefix,confidence_prefix",
    [("+", "+"), ("", "++"), ("++", "")],
)
def test_confidence_only_expected_layout_matches_producer(l1_prefix, confidence_prefix):
    # A confidence-only configuration (l1=0, confidence>0, ce_target_source=
    # labels) still needs the target's final hidden state. The Consumer must
    # therefore expect dflash_aux_plus_last AND the Producer must be told the
    # same loss values; otherwise the two layouts disagree and the feature
    # contract rejects the samples. The loss overrides carry arbitrary Hydra
    # prefixes (``+``/``++``/none), which must all resolve.
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
    )

    l1_key = "actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha"
    confidence_key = (
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha"
    )
    ce_key = "actor_rollout_ref.rollout.drafter.training.ce_target_source"
    args = _training_args() + [
        f"{l1_prefix}{l1_key}=0",
        f"{confidence_prefix}{confidence_key}=1.0",
        f"+{ce_key}=labels",
    ]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )

    # The requested loss values must reach the Producer verbatim; the previous
    # fallback (l1=0.9) would otherwise produce dflash_aux_plus_last by luck.
    assert f"{l1_key}=0" in commands.producer_overrides
    assert f"{confidence_key}=1.0" in commands.producer_overrides
    assert any(
        item.endswith("expected_feature.hidden_states_layout=dflash_aux_plus_last")
        for item in commands.consumer_overrides
    )

    producer_cfg: dict[str, str] = {}
    for item in commands.producer_overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            producer_cfg[key.lstrip("+")] = value
    producer_layout = resolve_drafter_hidden_states_layout(
        "DSPARK",
        {
            "dspark_l1_loss_alpha": float(producer_cfg[l1_key]),
            "dspark_confidence_loss_alpha": float(producer_cfg[confidence_key]),
            "ce_target_source": producer_cfg[ce_key],
        },
    )
    assert producer_layout == "dflash_aux_plus_last"


@pytest.mark.parametrize("prefix", ["", "+", "++"])
@pytest.mark.parametrize(
    "algorithm,lk_key,extra",
    [
        ("DFLASH", "dflash_lk_loss_alpha", []),
        (
            "DSPARK",
            "dspark_lk_loss_alpha",
            ["actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0"],
        ),
    ],
)
def test_lk_loss_forwarded_to_producer_and_consumer_layout(
    prefix, algorithm, lk_key, extra
):
    key = f"actor_rollout_ref.rollout.drafter.training.{lk_key}"
    args = [
        item
        for item in _training_args()
        if "speculative_algorithm" not in item
    ] + [
        f"actor_rollout_ref.rollout.drafter.speculative_algorithm={algorithm}",
        *extra,
        f"{prefix}{key}=0.5",
    ]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )
    assert f"{key}=0.5" in commands.producer_overrides
    assert any(
        item.endswith("expected_feature.hidden_states_layout=dflash_aux_plus_last")
        for item in commands.consumer_overrides
    )


@pytest.mark.parametrize("key", ["dflash_lk_loss_alpha", "dspark_lk_loss_alpha"])
def test_domino_launcher_rejects_lk_loss(key) -> None:
    args = [item for item in _training_args() if "speculative_algorithm" not in item]
    args.extend(
        [
            "actor_rollout_ref.rollout.drafter.speculative_algorithm=DOMINO",
            f"actor_rollout_ref.rollout.drafter.training.{key}=0.5",
        ]
    )
    with pytest.raises(ValueError, match=f"Domino does not support LK loss.*{key}"):
        build_pipeline_commands(
            resolve_pipeline_config(args, environ={}),
            args,
            ray_address="10.0.0.1:6379",
            python_executable="python",
        )


def test_dflash_without_lk_expects_dflash_aux_layout() -> None:
    args = [
        item
        for item in _training_args()
        if "speculative_algorithm" not in item
    ] + ["actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH"]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="10.0.0.1:6379",
        python_executable="python",
    )
    assert any(
        item.endswith("expected_feature.hidden_states_layout=dflash_aux")
        for item in commands.consumer_overrides
    )


def test_producer_max_samples_uses_remaining_total_steps() -> None:
    args = [
        *_training_args(),
        "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2",
        "speco.draft_training.nproc_per_node=4",
        "speco.draft_training.nnodes=1",
    ]

    assert _producer_max_samples(args, resumed_optimizer_step=6) == 32
    assert _producer_max_samples([*args, "++actor_rollout_ref.rollout.drafter.training.max_steps=8"],
                                 resumed_optimizer_step=6) == 16


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
    assert config.vllm_aux_hidden_state_layer_ids == (3, 11, 21)
    assert config.run_id.startswith("dflash-")


def test_pipeline_config_validates_vllm_ids_against_training_ids() -> None:
    args = [
        *_training_args(),
        "speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids=[2,10,18]",
        "actor_rollout_ref.rollout.drafter.training.dspark_target_layer_ids=[1,9,17]",
    ]

    config = resolve_pipeline_config(args, environ={})

    assert config.vllm_aux_hidden_state_layer_ids == (2, 10, 18)
    assert config.target_layer_ids == (1, 9, 17)


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


def test_vllm_capture_layer_ids_appends_missing_final_output() -> None:
    assert _vllm_capture_layer_ids((1, 9, 17, 25, 33), 36) == [
        1,
        9,
        17,
        25,
        33,
        36,
    ]


def test_vllm_capture_layer_ids_does_not_duplicate_selected_final_output() -> None:
    assert _vllm_capture_layer_ids((1, 9, 17, 25, 36), 36) == [
        1,
        9,
        17,
        25,
        36,
    ]


def test_pipeline_commands_do_not_request_selected_final_output_twice(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 36}), encoding="utf-8"
    )
    args = [
        item.replace("/models/Qwen3-8B", str(tmp_path)) for item in _training_args()
    ]
    args.append(
        "speco.standalone_tq_producer."
        "vllm_aux_hidden_state_layer_ids=[1,9,17,25,36]"
    )
    config = resolve_pipeline_config(args, environ={})

    commands = build_pipeline_commands(
        config,
        args,
        ray_address="127.0.0.1:6379",
        python_executable="python",
    )

    assert commands.vllm is not None
    spec_index = commands.vllm.index("--speculative-config") + 1
    speculative_config = json.loads(commands.vllm[spec_index])
    assert speculative_config["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ] == [1, 9, 17, 25, 36]


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


def test_preflight_respects_jsonl_line_limit(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"prompt":"question","response":"answer"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="max_jsonl_line_bytes"):
        _preflight_input_file(str(path), max_jsonl_line_bytes=10)
    _preflight_input_file(str(path), max_jsonl_line_bytes=0)


def test_pipeline_commands_hide_and_replace_tq_overrides() -> None:
    args = [
        *_training_args(),
        "speco.draft_training.runtime_backend=ray",
        "speco.draft_training.num_gpus_per_node=8",
        "speco.draft_training.num_nodes=1",
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

    assert commands.owner == []
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
    assert any(item.endswith("feature_store.type=ray_reference") for item in commands.consumer)
    assert not any("run_id=user-value" in item for item in commands.consumer)
    assert not any("runtime_backend" in item for item in commands.consumer_overrides)
    assert not any("num_gpus_per_node" in item for item in commands.consumer_overrides)
    assert not any("num_nodes" in item for item in commands.consumer_overrides)
    assert "speco.draft_training.nproc_per_node=8" in commands.consumer_overrides
    assert "speco.draft_training.nnodes=1" in commands.consumer_overrides
    assert any(f"run_id={config.run_id}" in item for item in commands.consumer)
    expected_contract = " ".join(commands.consumer)
    assert "expected_feature.algorithm=DSPARK" in expected_contract
    assert 'expected_feature.target_model_path="/models/Qwen3-8B"' in expected_contract
    assert "expected_feature.target_revision=target-path-sha256-" in expected_contract
    assert (
        "expected_feature.tokenizer_fingerprint=tokenizer-path-sha256-"
        in expected_contract
    )
    assert "expected_feature.target_layer_ids=[0,8,16,24,32]" in expected_contract
    assert (
        "expected_feature.hidden_states_layout=dflash_aux_plus_last"
        in expected_contract
    )
    assert "expected_feature.hidden_dtype=bfloat16" in expected_contract
    assert any(
        "vllm_endpoints=[http://127.0.0.1:8000/v1]" in item
        for item in commands.producer
    )
    # 10 steps * default batch_size_per_gpu=4 * 8 GPUs * 1 node.
    assert any("max_samples=320" in item for item in commands.producer)


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
        "speco.standalone_tq_producer.max_consecutive_errors=0",
        "speco.standalone_tq_producer.max_consecutive_feature_drops=5",
        "speco.standalone_tq_producer.on_error=raise",
        "speco.standalone_tq_producer.parser_strict_roles=true",
        "speco.standalone_tq_producer.min_supervised_tokens=3",
        "speco.standalone_tq_producer.on_truncated_supervision=drop",
        "speco.standalone_tq_producer.trust_remote_code=true",
        "speco.standalone_tq_producer.render_boundary.enabled=true",
        "speco.standalone_tq_producer.render_boundary.timeout=45",
        "speco.standalone_tq_producer.source_proportional_field=source",
        "speco.standalone_tq_producer.max_jsonl_line_bytes=0",
        "speco.standalone_tq_producer.max_indexed_records=42",
        "speco.standalone_tq_producer.verify_indexed_input=true",
    ]
    config = resolve_pipeline_config(args, environ={})

    commands = build_pipeline_commands(
        config,
        args,
        ray_address="127.0.0.1:6379",
        python_executable="python",
    )

    for override in (
        "speco.standalone_tq_producer.max_inflight_requests=32",
        "speco.standalone_tq_producer.per_endpoint_concurrency=8",
        "speco.standalone_tq_producer.max_feature_length=384",
        "speco.standalone_tq_producer.max_consecutive_errors=0",
        "speco.standalone_tq_producer.max_consecutive_feature_drops=5",
        "speco.standalone_tq_producer.on_error=raise",
        "speco.standalone_tq_producer.parser_strict_roles=true",
        "speco.standalone_tq_producer.min_supervised_tokens=3",
        "speco.standalone_tq_producer.on_truncated_supervision=drop",
        "speco.standalone_tq_producer.trust_remote_code=true",
        "speco.standalone_tq_producer.render_boundary.enabled=true",
        "speco.standalone_tq_producer.render_boundary.timeout=45",
        "speco.standalone_tq_producer.source_proportional_field=source",
        "speco.standalone_tq_producer.max_jsonl_line_bytes=0",
        "speco.standalone_tq_producer.max_indexed_records=42",
        "speco.standalone_tq_producer.verify_indexed_input=true",
    ):
        assert override in commands.producer
    assert "speco.standalone_tq_producer.source_proportional_field=source" in commands.consumer_overrides


def test_pipeline_commands_ignore_launcher_owned_producer_overrides() -> None:
    args = [
        *_training_args(),
        "speco.standalone_tq_producer.input_path=/evil.jsonl",
        "speco.standalone_tq_producer.max_samples=999",
    ]
    config = resolve_pipeline_config(args, environ={})

    commands = build_pipeline_commands(
        config,
        args,
        ray_address="127.0.0.1:6379",
        python_executable="python",
    )

    assert "speco.standalone_tq_producer.input_path=/data/train.jsonl" in commands.producer
    assert "speco.standalone_tq_producer.input_path=/evil.jsonl" not in commands.producer
    assert "speco.standalone_tq_producer.max_samples=999" not in commands.producer


def test_pipeline_commands_reject_unknown_producer_override() -> None:
    args = [
        *_training_args(),
        "speco.standalone_tq_producer.on_eror=skip",
    ]
    config = resolve_pipeline_config(args, environ={})

    with pytest.raises(ValueError, match="Unknown Producer override"):
        build_pipeline_commands(
            config,
            args,
            ray_address="127.0.0.1:6379",
            python_executable="python",
        )


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


def test_rollback_honors_yaml_default_and_cli_override(tmp_path, monkeypatch):
    from verl_speco import standalone_tq_training_launcher as launcher
    from verl_speco.trainer import checkpoint

    config_path = tmp_path / "draft_trainer.yaml"
    config_path.write_text("actor_rollout_ref:\n  rollout:\n    drafter:\n      training:\n        checkpoint_auto_rollback: true\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_DRAFT_TRAINER_CONFIG_PATH", config_path)
    calls = []
    monkeypatch.setattr(checkpoint, "resolve_verified_standalone_checkpoint",
                        lambda path, **kw: calls.append((path, kw)) or "/checkpoints/draft_step_2")
    args = ["data.train_files=/data/input.jsonl",
            "actor_rollout_ref.rollout.drafter.model_path=/checkpoints/draft_step_3"]
    resolved = launcher._resolve_checkpoint_rollback(args)
    assert calls == [("/checkpoints/draft_step_3", {"input_path": "/data/input.jsonl"})]
    assert any("draft_step_2" in value for value in resolved)
    calls.clear()
    disabled = args + ["actor_rollout_ref.rollout.drafter.training.checkpoint_auto_rollback=false"]
    assert launcher._resolve_checkpoint_rollback(disabled) == disabled and not calls


def test_external_ray_session_does_not_pass_local_runtime_options():
    ray = _FakeRay()
    session = start_ray_session(
        environ={"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379", "SPECO_RAY_NUM_CPUS": "8"},
        ray_module=ray,
    )
    assert ray.init_kwargs == {"address": "10.0.0.1:6379", "namespace": "speco-drafter"}
    session.close()
    assert ray.shutdown_called


def test_external_mooncake_resolves_local_ip_on_each_worker(monkeypatch):
    from verl_speco import standalone_tq_training_launcher as launcher

    monkeypatch.setattr(launcher, "_local_ip", lambda: "10.0.0.9")
    assert launcher._hidden_states_mooncake_settings({})["local_hostname"] == "10.0.0.9"
    env = {"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379"}
    assert launcher._hidden_states_mooncake_settings(env)["local_hostname"] == ""


@pytest.mark.parametrize("address", ["auto", "local", "ray://head:10001", "head", "head:0", "head:65536", "user:secret@head:6379", "head:6379/path", "head:6379?x=1"])
def test_external_ray_rejects_ambiguous_addresses_before_initializing(address):
    ray = _FakeRay()
    with pytest.raises(ValueError, match="explicit GCS"):
        start_ray_session(environ={"SPECO_EXTERNAL_RAY_ADDRESS": address}, ray_module=ray)
    assert ray.init_kwargs is None
    assert not ray.shutdown_called


@pytest.mark.parametrize("namespace,address,selected", [
    ("speco-drafter", "10.0.0.1:6379", "10.0.0.1:6379"),
    ("another-job", "10.0.0.1:6379", "10.0.0.1:6379"),
    ("speco-drafter", "10.0.0.2:6379", "10.0.0.1:6379"),
    ("speco-drafter", "10.0.0.1:6379", ""),
])
def test_reused_ray_connection_remains_owned_by_caller(namespace, address, selected):
    ray = _FakeRay()
    ray.is_initialized = lambda: True
    ray.get_runtime_context = lambda: SimpleNamespace(gcs_address=address, namespace=namespace)
    env = {"SPECO_EXTERNAL_RAY_ADDRESS": selected}
    if namespace == "speco-drafter" and address == selected:
        session = start_ray_session(environ=env, ray_module=ray)
        session.close()
    else:
        with pytest.raises(ValueError, match="already initialized"):
            start_ray_session(environ=env, ray_module=ray)
    assert ray.init_kwargs is None
    assert not ray.shutdown_called


def test_ray_context_failure_disconnects_owned_driver():
    ray = _FakeRay()
    def fail():
        raise RuntimeError("context unavailable")
    ray.get_runtime_context = fail
    with pytest.raises(RuntimeError, match="context unavailable"):
        start_ray_session(environ={}, ray_module=ray)
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


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("shutdown_unconfirmed", [False, True])
def test_pipeline_starts_owner_then_consumer_then_producer(tmp_path, use_cache, shutdown_unconfirmed, monkeypatch) -> None:
    from contextlib import nullcontext
    from verl_speco import standalone_tq_training_launcher as launcher
    import gc

    directories = []
    mkdtemp = launcher.tempfile.mkdtemp
    def make_directory(*args, **kwargs):
        directory = mkdtemp(*args, **kwargs)
        if kwargs.get("prefix") == "speco-vllm-hidden-states-":
            directories.append(Path(directory))
        return directory
    monkeypatch.setattr(launcher.tempfile, "mkdtemp", make_directory)
    stop_process = launcher._stop_process
    def stop(process):
        stop_process(process)
        if shutdown_unconfirmed and process is not None and process.role == "vllm":
            raise RuntimeError("descendants remain")
    monkeypatch.setattr(launcher, "_stop_process", stop)
    started: list[str] = []
    processes: list[_FakeProcess] = []
    child_environments: list[dict[str, str]] = []
    vllm_started = False

    def fake_popen(command, *, env):
        nonlocal vllm_started
        if command[0] == "vllm":
            if use_cache:
                assert command[2:5] == [cache.as_posix(), "--served-model-name", source.as_posix()]
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

    args = [*_training_args(), "speco.draft_training.runtime_backend=subprocess"]
    if use_cache:
        source, cache = tmp_path / "source", tmp_path / "cache"
        source.mkdir()
        cache.mkdir()
        (source / "weights.bin").write_bytes(b"weights")
        (cache / "weights.bin").write_bytes(b"weights")
        args += [f"actor_rollout_ref.model.path={source.as_posix()}",
                 f"speco.draft_training.local_target_cache_path={cache.as_posix()}"]
    config = resolve_pipeline_config(args, environ={})
    commands = build_pipeline_commands(
        config,
        args,
        ray_address="127.0.0.1:61234",
    )

    with (pytest.raises(RuntimeError, match="shutdown was not confirmed") if shutdown_unconfirmed else nullcontext()):
        assert run_pipeline(
            commands,
            ray_address="127.0.0.1:61234",
            environ={"RAY_ADDRESS": "172.51.9.253:35195"},
            popen=fake_popen,
            endpoint_ready=lambda _: vllm_started,
        ) == 0
    assert started == ["vllm", "owner", "consumer", "producer"]
    assert all(env["RAY_ADDRESS"] == "127.0.0.1:61234" for env in child_environments)
    assert all(process.poll() is not None for process in processes)
    gc.collect()
    assert len(directories) == 1
    assert directories[0].exists() == shutdown_unconfirmed
    if shutdown_unconfirmed:
        launcher._cleanup_hidden_states_directory(directories[0])
    with pytest.raises(RuntimeError, match="unowned"):
        launcher._cleanup_hidden_states_directory(tmp_path)
    assert tmp_path.exists()


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


def test_validate_tq_backend_ignores_non_mooncake_backend() -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("SimpleStorage must not probe a Mooncake master")

    validate_tq_backend({}, connect=_unexpected)
    validate_tq_backend({"SPECO_TQ_STORAGE_BACKEND": "SimpleStorage"}, connect=_unexpected)


def test_validate_tq_backend_skips_when_auto_init_enabled() -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("auto_init=true lets TransferQueue start the master")

    validate_tq_backend(
        {
            "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
            "SPECO_TQ_MOONCAKE_AUTO_INIT": "true",
        },
        connect=_unexpected,
    )


def test_validate_tq_backend_accepts_reachable_master() -> None:
    probed: list[tuple[str, int]] = []

    class _Connection:
        def close(self) -> None:
            probed.append(("closed", 0))

    def connect(address, *, timeout):
        probed.append(address)
        return _Connection()

    validate_tq_backend(
        {
            "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
            "SPECO_TQ_MOONCAKE_MASTER": "mooncake-host:50051",
        },
        connect=connect,
    )
    assert probed[0] == ("mooncake-host", 50051)


def test_validate_tq_backend_fails_fast_when_master_unreachable() -> None:
    def connect(address, *, timeout):
        raise OSError("connection refused")

    with pytest.raises(RuntimeError, match="no mooncake_master is reachable"):
        validate_tq_backend(
            {"SPECO_TQ_STORAGE_BACKEND": "MooncakeStore"},
            connect=connect,
        )


def test_validate_tq_backend_rejects_malformed_master_address() -> None:
    with pytest.raises(RuntimeError, match="host:port"):
        validate_tq_backend(
            {
                "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
                "SPECO_TQ_MOONCAKE_MASTER": "no-port",
            },
            connect=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize("backend", ["Mooncake", "mooncakestore", "MoonCakeStore", ""])
def test_validate_tq_backend_rejects_unknown_backend(backend) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("unknown backends must not be probed")

    with pytest.raises(RuntimeError, match="Unsupported SPECO_TQ_STORAGE_BACKEND"):
        validate_tq_backend(
            {"SPECO_TQ_STORAGE_BACKEND": backend}, connect=_unexpected
        )


@pytest.mark.parametrize("backend", ["Mooncake", "mooncakestore", "MoonCakeStore", ""])
def test_tq_backend_overrides_reject_unknown_backend(backend) -> None:
    with pytest.raises(RuntimeError, match="Unsupported SPECO_TQ_STORAGE_BACKEND"):
        _tq_backend_overrides({"SPECO_TQ_STORAGE_BACKEND": backend})


def test_tq_backend_overrides_accept_explicit_simple_storage() -> None:
    overrides = _tq_backend_overrides({"SPECO_TQ_STORAGE_BACKEND": "SimpleStorage"})
    assert any("backend.storage_backend=SimpleStorage" in item for item in overrides)


def test_tq_backend_overrides_accept_mooncake_store() -> None:
    overrides = _tq_backend_overrides(
        {"SPECO_TQ_STORAGE_BACKEND": "MooncakeStore"}
    )
    assert any("backend.storage_backend=MooncakeStore" in item for item in overrides)


def test_validate_tq_backend_skips_probe_when_disabled() -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("the precheck must be skipped when disabled")

    validate_tq_backend(
        {
            "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
            "SPECO_TQ_MOONCAKE_SKIP_PRECHECK": "true",
        },
        connect=_unexpected,
    )


def test_pipeline_commands_use_provided_env_for_backend() -> None:
    config = resolve_pipeline_config(_training_args(), environ={})

    commands = build_pipeline_commands(
        config,
        _training_args(),
        ray_address="127.0.0.1:6379",
        python_executable="python",
        env={"SPECO_TQ_STORAGE_BACKEND": "MooncakeStore"},
    )

    assert any("storage_backend=MooncakeStore" in item for item in commands.consumer)
    assert not any("storage_backend=SimpleStorage" in item for item in commands.consumer)


def test_resolve_hidden_states_store_defaults_to_file() -> None:
    assert _resolve_hidden_states_store({}) == "file"


@pytest.mark.parametrize("backend", ["grpc", "", "moon"])
def test_resolve_hidden_states_store_rejects_unknown(backend) -> None:
    with pytest.raises(RuntimeError, match="Unsupported SPECO_VLLM_HIDDEN_STATES_STORE"):
        _resolve_hidden_states_store({"SPECO_VLLM_HIDDEN_STATES_STORE": backend})


def test_vllm_kv_transfer_config_defaults_to_file_connector() -> None:
    config = _vllm_kv_transfer_config({})

    assert config["kv_connector"] == "ExampleHiddenStatesConnector"
    assert "kv_connector_module_path" not in config
    assert config["kv_connector_extra_config"]["use_synchronization_lock"] is True


def test_vllm_kv_transfer_config_selects_mooncake_connector() -> None:
    config = _vllm_kv_transfer_config(
        {
            "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.9:50051",
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL": "rdma",
        }
    )

    assert config["kv_connector"] == "SpecoMooncakeHiddenStatesConnector"
    assert (
        config["kv_connector_module_path"]
        == "verl_speco.mooncake_hidden_states_connector"
    )
    mooncake = config["kv_connector_extra_config"]["mooncake"]
    assert mooncake["master_server_address"] == "10.0.0.9:50051"
    assert mooncake["protocol"] == "rdma"
    assert mooncake["local_hostname"]


def test_hidden_states_store_overrides_file_is_empty() -> None:
    assert _hidden_states_store_overrides({}) == []


def test_hidden_states_store_overrides_mooncake() -> None:
    overrides = _hidden_states_store_overrides(
        {
            "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
            "SPECO_TQ_MOONCAKE_MASTER": "1.2.3.4:50051",
        }
    )

    assert any(item.endswith("hidden_states_store.backend=mooncake") for item in overrides)
    assert any("mooncake.master_server_address=1.2.3.4:50051" in item for item in overrides)


def test_validate_hidden_states_store_skips_file_backend() -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("file backend must not probe Mooncake")

    validate_hidden_states_store({}, connect=_unexpected)


def test_validate_hidden_states_store_probes_master() -> None:
    calls = []

    def _connect(address, timeout):
        calls.append((address, timeout))
        return SimpleNamespace(close=lambda: None)

    validate_hidden_states_store(
        {
            "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.1:50051",
        },
        connect=_connect,
    )

    assert calls == [(("10.0.0.1", 50051), 2.0)]


def test_validate_hidden_states_store_fails_fast_when_unreachable() -> None:
    def _connect(address, timeout):
        raise OSError("connection refused")

    with pytest.raises(RuntimeError, match="no mooncake_master is reachable"):
        validate_hidden_states_store(
            {
                "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
                "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.1:50051",
            },
            connect=_connect,
        )


@pytest.mark.parametrize(
    "environ",
    [
        {
            "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
            "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.1:50051",
        },
        {
            "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
            "SPECO_TQ_MOONCAKE_MASTER": "10.0.0.1:50051",
        },
    ],
    ids=["hidden-states-store", "tq-backend"],
)
def test_ray_pipeline_validates_mooncake_master_before_spawning(
    monkeypatch, environ
) -> None:
    import verl_speco.standalone_tq_training_launcher as launcher
    from verl_speco import standalone_ray_runtime as ray_runtime

    def _connect(address, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(launcher.socket, "create_connection", _connect)
    commands = SimpleNamespace(vllm_endpoints=["http://ready:8000/v1"], vllm=None, consumer_overrides=())

    with pytest.raises(RuntimeError, match="no mooncake_master is reachable"):
        ray_runtime.run_ray_pipeline(
            commands,
            ray_module=object(),
            ray_address="127.0.0.1:6379",
            environ=environ,
            endpoint_ready=lambda endpoint: True,
            popen=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("validation must fail before spawning processes")
            ),
        )


def test_pipeline_commands_select_mooncake_hidden_states_store() -> None:
    env = {
        "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
        "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.7:50051",
    }
    config = resolve_pipeline_config(_training_args(), environ=env)

    commands = build_pipeline_commands(
        config,
        _training_args(),
        ray_address="127.0.0.1:6379",
        python_executable="python",
        env=env,
    )

    assert commands.vllm is not None
    joined_vllm = " ".join(commands.vllm)
    assert commands.owner == []
    prefix = "actor_rollout_ref.rollout.drafter.training"
    assert f"{prefix}.feature_store.type=ray_reference" in commands.consumer_overrides
    assert f"{prefix}.transfer_queue.enable=false" in commands.consumer_overrides
    assert f"{prefix}.target_feature_replay.hidden_states_store.backend=mooncake" in commands.consumer_overrides
    assert f"{prefix}.target_feature_replay.hidden_states_store.mooncake.master_server_address=10.0.0.7:50051" in commands.consumer_overrides
    assert "SpecoMooncakeHiddenStatesConnector" in joined_vllm
    assert "verl_speco.mooncake_hidden_states_connector" in joined_vllm
    assert any(
        "hidden_states_store.backend=mooncake" in item for item in commands.producer
    )
    assert any(
        "mooncake.master_server_address=10.0.0.7:50051" in item
        for item in commands.producer
    )


def test_pipeline_commands_forward_shared_file_root_to_both_roles(tmp_path):
    from omegaconf import OmegaConf
    env = {"SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT": str(tmp_path)}
    config = resolve_pipeline_config(_training_args(), environ=env)
    commands = build_pipeline_commands(config, _training_args(), ray_address="10.0.0.1:6379",
                                       python_executable="python", env=env)
    producer = OmegaConf.from_dotlist(list(commands.producer_overrides))
    consumer = OmegaConf.from_dotlist(list(commands.consumer_overrides))
    assert producer.speco.standalone_tq_producer.hidden_states_store.file_root == str(tmp_path)
    assert consumer.actor_rollout_ref.rollout.drafter.training.target_feature_replay.hidden_states_store.file_root == str(tmp_path)
    connector = commands.vllm[commands.vllm.index("--kv-transfer-config") + 1]
    assert json.loads(connector)["kv_connector_extra_config"]["shared_storage_path"] == str(tmp_path)




def test_trainer_topology_plan_is_deterministic_and_maps_ranks():
    from verl_speco.standalone_tq_training_launcher import _trainer_topology_plan

    args = ["speco.draft_training.nnodes=2", "speco.draft_training.nproc_per_node=2",
            "speco.draft_training.trainer_node_ips=[10.0.0.2,10.0.0.1]"]
    plan = _trainer_topology_plan("ray", args)
    assert plan["nodes"][0] == {"node_rank": 0, "address": "10.0.0.1", "ranks": [
        {"local_rank": 0, "global_rank": 0}, {"local_rank": 1, "global_rank": 1}]}
    assert plan["nodes"][1]["ranks"][-1]["global_rank"] == 3
    swapped = [*args[:-1], "speco.draft_training.trainer_node_ips=[10.0.0.1,10.0.0.2]"]
    assert _trainer_topology_plan("ray", swapped)["topology_sha256"] == plan["topology_sha256"]
    changed = [*args, "speco.draft_training.nproc_per_node=3"]
    assert _trainer_topology_plan("ray", changed)["topology_sha256"] != plan["topology_sha256"]


def test_topology_dry_run_does_not_import_compute_or_open_sockets(tmp_path):
    import os
    import subprocess
    import sys

    code = '''
import builtins, socket
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "torch_npu", "ray", "transformers"}:
        raise AssertionError("unexpected compute import " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
def no_connect(*args, **kwargs):
    raise AssertionError("unexpected network access")
socket.socket.connect = no_connect
socket.socket.bind = no_connect
socket.create_connection = no_connect
from verl_speco.standalone_tq_training_launcher import main
raise SystemExit(main(["--dry-run-topology", "--runtime-backend", "ray",
    "speco.draft_training.nnodes=2", "speco.draft_training.nproc_per_node=2",
    "speco.draft_training.trainer_node_ips=[10.0.0.2,10.0.0.1]"]))
'''
    env = dict(os.environ, SPECO_VLLM_ENDPOINTS="[http://10.0.0.3:8000/v1]",
               SPECO_EXTERNAL_RAY_ADDRESS="10.0.0.1:6379", SPECO_VLLM_HIDDEN_STATES_STORE="mooncake",
               SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER="10.0.0.3:50051")
    env.pop("SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_HOSTNAME", None)
    env.pop("SPECO_TQ_MOONCAKE_LOCAL_HOSTNAME", None)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["world_size"] == 4
    assert [node["address"] for node in plan["nodes"]] == ["10.0.0.1", "10.0.0.2"]
    assert plan["vllm_endpoints"] == ["http://10.0.0.3:8000/v1"]
    assert len(plan["topology_sha256"]) == 64


def test_topology_dry_run_fingerprint_tracks_placement_and_services(monkeypatch, capsys):
    import verl_speco.standalone_tq_training_launcher as launcher

    monkeypatch.delenv("SPECO_EXTERNAL_RAY_ADDRESS", raising=False)
    monkeypatch.delenv("SPECO_VLLM_ENDPOINTS", raising=False)
    monkeypatch.setenv("SPECO_VLLM_ENDPOINT", "http://localhost:8000/v1")
    args = ["--dry-run-topology", "--runtime-backend", "subprocess", "speco.draft_training.nproc_per_node=2"]
    assert launcher.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert launcher.main(args) == 0
    assert json.loads(capsys.readouterr().out)["topology_sha256"] == first["topology_sha256"]
    monkeypatch.setenv("SPECO_VLLM_ENDPOINT", "http://localhost:8001/v1")
    assert launcher.main(args) == 0
    assert json.loads(capsys.readouterr().out)["topology_sha256"] != first["topology_sha256"]


def test_topology_dry_run_rejects_multinode_without_external_ray(monkeypatch):
    import verl_speco.standalone_tq_training_launcher as launcher

    monkeypatch.delenv("SPECO_EXTERNAL_RAY_ADDRESS", raising=False)
    assert launcher.main(["--dry-run-topology", "--runtime-backend", "ray",
                          "speco.draft_training.nnodes=2",
                          "speco.draft_training.trainer_node_ips=[10.0.0.1,10.0.0.2]"]) == 2


def test_verified_local_model_cache_checks_all_files_and_contents(tmp_path):
    from verl_speco.standalone_tq_training_launcher import _verify_local_model_cache

    original, cache = tmp_path / "source", tmp_path / "cache"
    for directory in (original, cache):
        (directory / "nested").mkdir(parents=True)
        (directory / "config.json").write_text('{"layers": 1}', encoding="utf-8")
        (directory / "nested" / "weights.bin").write_bytes(b"abcd")
    _verify_local_model_cache(str(original), str(cache))
    (cache / "nested" / "weights.bin").write_bytes(b"abce")
    with pytest.raises(ValueError, match="content differs"):
        _verify_local_model_cache(str(original), str(cache))
    (cache / "nested" / "weights.bin").write_bytes(b"abcd")
    (cache / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        _verify_local_model_cache(str(original), str(cache))


def test_local_cache_command_preserves_logical_model_identity(tmp_path):
    original, cache = tmp_path / "source", tmp_path / "cache"
    original.mkdir()
    cache.mkdir()
    args = [*_training_args(), f"actor_rollout_ref.model.path={original.as_posix()}",
            f"speco.draft_training.local_target_cache_path={cache.as_posix()}"]
    config = resolve_pipeline_config(args, environ={})
    commands = build_pipeline_commands(config, args, ray_address="127.0.0.1:6379", env={})
    assert commands.local_model_cache == (original.as_posix(), cache.as_posix())
    assert commands.vllm[2:5] == [cache.as_posix(), "--served-model-name", original.as_posix()]
    assert f"speco.standalone_tq_producer.vllm_model={original.as_posix()}" in commands.producer_overrides
    assert any(item.endswith(json.dumps(original.as_posix())) and "expected_feature.target_model_path=" in item
               for item in commands.consumer_overrides)


@pytest.mark.parametrize("backend", ["subprocess", "ray"])
@pytest.mark.parametrize("endpoint_alive", [False, True])
def test_cache_verification_occurs_before_local_service_start(monkeypatch, tmp_path, backend, endpoint_alive):
    from dataclasses import replace
    import verl_speco.standalone_tq_training_launcher as launcher
    import verl_speco.standalone_ray_runtime as ray_runtime

    args = _training_args()
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                       ray_address="127.0.0.1:6379", env={})
    commands = replace(commands, local_model_cache=("source", "cache"))
    calls = []
    module = launcher if backend == "subprocess" else ray_runtime
    monkeypatch.setattr(module, "validate_tq_backend", lambda *_: None)
    monkeypatch.setattr(module, "validate_hidden_states_store", lambda *_: None)
    monkeypatch.setattr(module, "_verify_local_model_cache", lambda *paths: calls.append(paths))
    def popen(*args, **kwargs):
        assert calls == [("source", "cache")]
        raise RuntimeError("stopped after verification")
    options = dict(commands=commands, ray_address="127.0.0.1:6379", environ={},
                   endpoint_ready=lambda _: endpoint_alive, popen=popen)
    if backend == "ray":
        options["ray_module"] = object()
        run = ray_runtime.run_ray_pipeline
    else:
        run = launcher.run_pipeline
    with pytest.raises(RuntimeError, match="requires a new" if endpoint_alive else "stopped after verification"):
        run(**options)
    assert calls == ([] if endpoint_alive else [("source", "cache")])


@pytest.mark.parametrize("backend", ["subprocess", "ray"])
def test_corrupted_model_cache_never_starts_service(monkeypatch, tmp_path, backend):
    import verl_speco.standalone_tq_training_launcher as launcher
    import verl_speco.standalone_ray_runtime as ray_runtime

    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (source / "weights.bin").write_bytes(b"good")
    (cache / "weights.bin").write_bytes(b"evil")
    args = [*_training_args(), f"actor_rollout_ref.model.path={source.as_posix()}",
            f"speco.draft_training.local_target_cache_path={cache.as_posix()}"]
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                       ray_address="127.0.0.1:6379", env={})
    module = launcher if backend == "subprocess" else ray_runtime
    monkeypatch.setattr(module, "validate_tq_backend", lambda *_: None)
    monkeypatch.setattr(module, "validate_hidden_states_store", lambda *_: None)
    options = dict(commands=commands, ray_address="127.0.0.1:6379", environ={},
                   endpoint_ready=lambda _: False,
                   popen=lambda *args, **kwargs: pytest.fail("must not start an unverified model"))
    if backend == "ray":
        options["ray_module"] = object()
        run = ray_runtime.run_ray_pipeline
    else:
        run = launcher.run_pipeline
    with pytest.raises(ValueError, match="content differs"):
        run(**options)


def test_ray_pipeline_starts_vllm_from_verified_local_cache(monkeypatch, tmp_path):
    import verl_speco.standalone_ray_runtime as runtime

    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    (cache / "weights.bin").write_bytes(b"weights")
    args = [*_training_args(), f"actor_rollout_ref.model.path={source.as_posix()}",
            f"speco.draft_training.local_target_cache_path={cache.as_posix()}"]
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ={}), args,
                                       ray_address="127.0.0.1:6379", env={})
    monkeypatch.setattr(runtime, "validate_tq_backend", lambda *_: None)
    monkeypatch.setattr(runtime, "validate_hidden_states_store", lambda *_: None)
    monkeypatch.setattr(runtime, "_wait_for_owner_ready", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "_wait_for_vllm_ready", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "run_ray_actors", lambda *a, **kw: 0)
    spawned = []
    def popen(command, *, env):
        spawned.append(command)
        return _FakeProcess("vllm" if command[0] == "vllm" else "owner")
    assert runtime.run_ray_pipeline(commands, ray_module=object(), ray_address="127.0.0.1:6379",
                                    environ={}, endpoint_ready=lambda _: False, popen=popen) == 0
    assert spawned[0][2:5] == [cache.as_posix(), "--served-model-name", source.as_posix()]


def test_model_cache_rejects_source_change_during_verification(tmp_path, monkeypatch):
    from verl_speco.standalone_tq_training_launcher import _verify_local_model_cache

    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    source_file, cache_file = source / "weights.bin", cache / "weights.bin"
    source_file.write_bytes(b"good")
    cache_file.write_bytes(b"good")
    original_open = Path.open
    def mutate_before_cache_read(path, *args, **kwargs):
        if path == cache_file:
            source_file.write_bytes(b"evil")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", mutate_before_cache_read)
    with pytest.raises(ValueError, match="changed during cache verification"):
        _verify_local_model_cache(str(source), str(cache))
