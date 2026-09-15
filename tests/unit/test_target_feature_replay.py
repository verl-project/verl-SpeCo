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
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.feature_store import DraftFeatureSample, DraftReplaySample  # noqa: E402
from verl_speco.trainer.target_feature_replay import (  # noqa: E402
    BoundedReplayCache,
    FeatureContract,
    TargetFeatureReplayer,
    _VllmEndpointState,
    _hidden_capture_target,
    _normalize_vllm_endpoints,
    feature_from_vllm_payload,
    load_vllm_final_norm,
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
    monkeypatch.setattr(
        "verl_speco.trainer.target_feature_replay.time.sleep", lambda _: None
    )

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
    replayer.vllm_final_norm = torch.nn.RMSNorm(4, eps=1e-6)

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
    torch.testing.assert_close(feature.hidden_states[:, :8], hidden[:, :2].flatten(1))
    torch.testing.assert_close(
        feature.hidden_states[:, 8:], replayer.vllm_final_norm(hidden[:, 2])
    )


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3", "qwen3_moe"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vllm_final_norm_matches_target_forward(
    tmp_path, monkeypatch, model_type, sharded, dtype
):
    transformers = pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from verl_speco import checkpoint_tensor

    if model_type not in CONFIG_MAPPING:
        pytest.skip(f"Installed Transformers does not include {model_type}")
    config = transformers.AutoConfig.for_model(
        model_type,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        rms_norm_eps=1e-5,
        head_dim=4,
        moe_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    model = transformers.AutoModelForCausalLM.from_config(config).to(dtype).eval()
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.5, 2.0, 8))
    model.save_pretrained(tmp_path, max_shard_size="1KB" if sharded else "1GB")
    loaded_keys = []
    original_load = checkpoint_tensor._load_checkpoint_tensor

    def load_one(path, key):
        loaded_keys.append(key)
        return original_load(path, key)

    monkeypatch.setattr(checkpoint_tensor, "_load_checkpoint_tensor", load_one)
    norm = load_vllm_final_norm(str(tmp_path), dtype=dtype)
    assert loaded_keys == ["model.norm.weight"]
    assert all(
        p.device.type == "cpu" and not p.requires_grad for p in norm.parameters()
    )

    captured = {}
    handle = model.model.norm.register_forward_pre_hook(
        lambda module, args: captured.update(final_input=args[0].detach().clone())
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        output = model(ids, output_hidden_states=True)
    handle.remove()
    # A connector-style payload: auxiliary layer output + final PRE-norm output.
    raw = torch.stack([output.hidden_states[1][0], captured["final_input"][0]], dim=1)
    original_raw = raw.clone()
    request = DraftReplaySample(
        input_ids=ids[0],
        loss_mask=torch.ones(4),
        attention_mask=torch.ones(4),
        position_ids=torch.arange(4),
        feature_positions=torch.arange(4),
        draft_position_ids=torch.arange(1, 5),
    )
    for algorithm, layout in [
        ("DSPARK", "dflash_aux_plus_last"),
        ("EAGLE3", "eagle3_aux_plus_last"),
    ]:
        contract = FeatureContract(
            algorithm=algorithm,
            target_layer_ids=[0],
            hidden_states_layout=layout,
            dtype=dtype,
            target_model_id=str(tmp_path),
            target_model_revision=None,
            tokenizer_fingerprint="test",
        )
        payload = {"token_ids": ids[0], "hidden_states": raw}
        feature = feature_from_vllm_payload(payload, request, contract, final_norm=norm)
        torch.testing.assert_close(
            feature.hidden_states[:, :8], raw[:, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            feature.hidden_states[:, 8:], output.hidden_states[-1][0]
        )
        torch.testing.assert_close(raw, original_raw, rtol=0, atol=0)
        assert not feature.hidden_states.requires_grad
        # Re-reading the same raw payload must not apply norm to an already-mutated tensor.
        again = feature_from_vllm_payload(payload, request, contract, final_norm=norm)
        torch.testing.assert_close(
            again.hidden_states, feature.hidden_states, rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="require the target final norm"):
            feature_from_vllm_payload(payload, request, contract)
        aux_only = feature_from_vllm_payload(
            payload,
            request,
            replace(contract, hidden_states_layout="dflash_aux", algorithm="DFLASH"),
        )
        torch.testing.assert_close(aux_only.hidden_states, raw[:, 0], rtol=0, atol=0)


def test_final_norm_loader_requires_checkpoint_weight(tmp_path):
    transformers = pytest.importorskip("transformers")
    from safetensors import SafetensorError
    from safetensors.torch import save_file

    config = transformers.LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    config.save_pretrained(tmp_path)
    save_file({"unrelated.weight": torch.ones(8)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(SafetensorError, match="model.norm.weight"):
        load_vllm_final_norm(str(tmp_path), dtype=torch.float32)


def test_vllm_replay_initializes_norm_and_invalidates_old_cache(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import transformers

    transformers.LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    ).save_pretrained(tmp_path)
    calls = []
    norm = torch.nn.RMSNorm(8)

    def loader(*args, **kwargs):
        calls.append(args)
        return norm

    monkeypatch.setattr(
        "verl_speco.trainer.target_feature_replay.load_vllm_final_norm", loader
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": str(tmp_path)},
                "rollout": {
                    "drafter": {
                        "speculative_algorithm": "DSPARK",
                        "target_layer_ids": [0],
                        "training": {
                            "use_logits": False,
                            "dspark_l1_loss_alpha": 0.9,
                            "target_feature_replay": {
                                "backend": "vllm_file",
                                "dtype": "float32",
                            },
                        },
                    }
                },
            }
        }
    )
    replayer = TargetFeatureReplayer(
        config, rank=0, world_size=1, device=torch.device("cpu")
    )
    assert calls == [(str(tmp_path),)]
    assert replayer.model is None  # No full target model was loaded for replay.
    assert replayer.vllm_final_norm is norm
    sample = DraftReplaySample(
        input_ids=torch.arange(4),
        loss_mask=torch.ones(4),
        attention_mask=torch.ones(4),
        position_ids=torch.arange(4),
        feature_positions=torch.arange(4),
        draft_position_ids=torch.arange(1, 5),
    )
    new_key = replayer._cache_key(sample)
    replayer.backend = "torch"
    assert new_key != replayer._cache_key(sample)
    config.actor_rollout_ref.rollout.drafter.training.target_feature_replay.backend = (
        "torch"
    )
    calls.clear()
    torch_replayer = TargetFeatureReplayer(
        config, rank=0, world_size=1, device=torch.device("cpu")
    )
    assert calls == []
    assert torch_replayer.vllm_final_norm is None
