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

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.feature_store import DraftFeatureSample, DraftReplaySample  # noqa: E402
from verl_speco.trainer.target_feature_replay import (  # noqa: E402
    BoundedReplayCache,
    TargetFeatureReplayer,
    _VllmEndpointState,
    _hidden_capture_target,
    _normalize_vllm_endpoints,
)


def _feature_sample() -> DraftFeatureSample:
    return DraftFeatureSample(
        input_ids=torch.arange(8),
        loss_mask=torch.ones(8),
        hidden_states=torch.zeros(8, 16),
        position_ids=torch.arange(1, 9),
    )


def test_hidden_capture_target_matches_transformers_hidden_state_indices():
    assert _hidden_capture_target(0, 36) == ("layer", 0)
    assert _hidden_capture_target(34, 36) == ("layer", 34)
    assert _hidden_capture_target(35, 36) == ("final", None)


def test_normalize_vllm_endpoints_prefers_pool_and_deduplicates():
    assert _normalize_vllm_endpoints(
        {
            "vllm_endpoint": "http://legacy:8000/v1",
            "vllm_endpoints": [
                "http://host1:8000/v1/",
                "http://host2:8000/v1",
                "http://host1:8000/v1",
            ],
        }
    ) == ["http://host1:8000/v1", "http://host2:8000/v1"]


def test_vllm_request_fails_over_to_another_endpoint(monkeypatch):
    class _Completions:
        def __init__(self, error=None):
            self.error = error
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.error is not None:
                raise self.error
            return SimpleNamespace(choices=[])

    failed = _Completions(RuntimeError("endpoint down"))
    healthy = _Completions()
    replayer = TargetFeatureReplayer.__new__(TargetFeatureReplayer)
    replayer.rank = 0
    replayer.vllm_timeout = 1
    replayer.vllm_max_retries = 1
    replayer.vllm_endpoint_cooldown = 5
    replayer.vllm_requests = 0
    replayer.vllm_request_seconds = 0.0
    replayer._metrics_lock = threading.Lock()
    replayer._endpoint_lock = threading.Lock()
    replayer._vllm_clients_initialized = True
    replayer._vllm_endpoint_states = [
        _VllmEndpointState(
            index=0,
            url="http://host1:8000/v1",
            client=SimpleNamespace(completions=failed),
            model="target",
        ),
        _VllmEndpointState(
            index=1,
            url="http://host2:8000/v1",
            client=SimpleNamespace(completions=healthy),
            model="target",
        ),
    ]
    monkeypatch.setattr("verl_speco.trainer.target_feature_replay.time.sleep", lambda _: None)

    response = replayer._request_vllm_response([1, 2, 3])

    assert response.choices == []
    assert failed.calls == 1
    assert healthy.calls == 1
    assert replayer._vllm_endpoint_states[0].failures == 1
    assert replayer._vllm_endpoint_states[1].requests == 1


def test_bounded_replay_cache_roundtrip(tmp_path):
    cache = BoundedReplayCache(
        tmp_path,
        max_size_gb=0.01,
        rank=1,
        world_size=2,
    )

    assert cache.put("sample", _feature_sample()) is True
    loaded = cache.get("sample")

    assert loaded is not None
    assert torch.equal(loaded.input_ids, torch.arange(8))
    assert cache.metrics()["replay/cache_size_gb"] > 0
    assert (tmp_path / "rank00001" / "sample.pt").exists()


def test_bounded_replay_cache_disables_zero_budget(tmp_path):
    cache = BoundedReplayCache(
        tmp_path,
        max_size_gb=0,
        rank=0,
        world_size=1,
    )

    assert cache.put("sample", _feature_sample()) is False
    assert cache.get("sample") is None


def test_token_replay_algorithm_mismatch_is_warning_not_error(caplog):
    replayer = TargetFeatureReplayer.__new__(TargetFeatureReplayer)
    replayer.rank = 0
    replayer.algorithm = "DFLASH"
    replayer.target_layer_ids = [1, 3]
    replayer.hidden_layout = "dflash_aux"
    replayer.strict_target_model_path = False
    replayer._warned_replay_algorithm_mismatch = False
    replayer._warned_replay_layer_mismatch = False
    replayer._warned_replay_layout_mismatch = False
    sample = DraftReplaySample(
        algorithm="DSPARK",
        input_ids=torch.arange(8),
        loss_mask=torch.ones(8),
        attention_mask=torch.ones(8, dtype=torch.bool),
        position_ids=torch.arange(8),
        feature_positions=torch.arange(2, 6),
        draft_position_ids=torch.arange(3, 7),
        metadata={
            "target_layer_ids": [2, 4],
            "hidden_states_layout": "dflash_aux_plus_last",
        },
    )

    replayer._validate_target_path(sample)

    assert "token replay algorithm differs" in caplog.text
    assert "token replay target layers differ" in caplog.text
    assert "token replay hidden layout differs" in caplog.text


def test_vllm_payload_maps_suffix_hidden_rows_to_absolute_positions():
    replayer = TargetFeatureReplayer.__new__(TargetFeatureReplayer)
    replayer.rank = 0
    replayer.target_layer_ids = [1, 3]
    replayer.hidden_layout = "dflash_aux_plus_last"
    replayer.dtype = torch.float32
    replayer.model_path = "/target"
    replayer.target_revision = None
    replayer.target_config_fingerprint = "unit"
    replayer.use_logits = False

    sample = DraftReplaySample(
        algorithm="DSPARK",
        input_ids=torch.arange(10, dtype=torch.long),
        loss_mask=torch.ones(10, dtype=torch.float32),
        attention_mask=torch.ones(10, dtype=torch.bool),
        position_ids=torch.arange(10, dtype=torch.long),
        feature_positions=torch.arange(4, 10, dtype=torch.long),
        draft_position_ids=torch.arange(5, 11, dtype=torch.long),
        metadata={"global_step": 1},
    )
    hidden = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4)
    payload = {
        "token_ids": torch.arange(10, dtype=torch.long),
        "hidden_states": hidden,
    }

    feature = replayer._feature_from_vllm_payload(
        sample,
        payload,
        prompt_ids=list(range(10)),
        source="token_replay_vllm_file",
    )

    assert torch.equal(feature.input_ids, torch.arange(5, 10))
    assert torch.equal(feature.position_ids, torch.arange(6, 11))
    assert feature.hidden_states.shape == (5, 12)
    assert feature.metadata["feature_start"] == 5
    assert feature.metadata["feature_end"] == 10
    assert feature.metadata["vllm_hidden_position_offset"] == 5
