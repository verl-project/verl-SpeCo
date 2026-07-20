from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from verl_speco.integration.sglang_adapter import (
    DFLASH_RETURN_AUX_HIDDEN_PARAM,
    DRAFTER_RAW_TOP_LOGPROBS_PARAM,
    DRAFTER_RETURN_LAST_HIDDEN_PARAM,
    SGLangSpecoPatchConfig,
    bucket_drafter_samples_by_replica,
    build_hidden_state_request_params,
    install_sglang_speco_patches,
    speco_step_matches_interval,
)
from verl_speco.integration.sglang_runtime import (
    _SpecoSGLangHttpServerMixin,
    attach_update_draft_weights_to_rollout,
    speco_update_draft_weights,
)


@pytest.mark.parametrize(
    ("step", "interval", "expected"),
    [(None, 5, False), (0, 5, False), (4, 5, False), (5, 5, True), (10, 5, True), (5, 0, False)],
)
def test_interval_gate(step, interval, expected) -> None:
    assert speco_step_matches_interval(step, interval) is expected


def test_hidden_state_request_flags_are_independent() -> None:
    assert build_hidden_state_request_params(return_last_hidden=True) == {
        DRAFTER_RETURN_LAST_HIDDEN_PARAM: True
    }
    assert build_hidden_state_request_params(
        return_dflash_aux_hidden=True, raw_top_logprobs=True
    ) == {
        DFLASH_RETURN_AUX_HIDDEN_PARAM: True,
        DRAFTER_RAW_TOP_LOGPROBS_PARAM: True,
    }


def test_samples_are_routed_to_replica_owner() -> None:
    assert bucket_drafter_samples_by_replica(
        [{"replica_rank": 1, "id": "b"}, {"replica_rank": 0, "id": "a"}], 2
    ) == [[{"replica_rank": 0, "id": "a"}], [{"replica_rank": 1, "id": "b"}]]

    with pytest.raises(ValueError, match="out of range"):
        bucket_drafter_samples_by_replica([{"replica_rank": 2}], 2)


def test_sglang_draft_update_attachment_is_idempotent() -> None:
    rollout = SimpleNamespace()

    assert attach_update_draft_weights_to_rollout(rollout) is rollout
    first = rollout.update_draft_weights
    assert first.__func__ is speco_update_draft_weights
    assert attach_update_draft_weights_to_rollout(rollout).update_draft_weights == first


def test_sglang_patch_install_forwards_config_and_is_repeatable(monkeypatch) -> None:
    calls = []

    fake_patch_module = SimpleNamespace(
        enable_sglang_original_logprob_return=lambda: calls.append(("enable_original", None)),
        install_sglang_verl_patches=lambda **kwargs: calls.append(("install", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "verl_speco.integration.sglang_patch", fake_patch_module)

    config = SGLangSpecoPatchConfig(
        set_envs_and_config=lambda: None,
        target_weight_loader="target.loader",
        draft_weight_loader="draft.loader",
        patches={"hidden_states_tensor_output"},
    )

    install_sglang_speco_patches(config)
    install_sglang_speco_patches(config)

    install_calls = [payload for name, payload in calls if name == "install"]
    assert len(install_calls) == 2
    assert install_calls[0]["target_weight_loader"] == "target.loader"
    assert install_calls[0]["draft_weight_loader"] == "draft.loader"
    assert install_calls[0]["patches"] == {"hidden_states_tensor_output"}


def test_dflash_hidden_collection_requests_aux_hidden_without_raw_topk(monkeypatch) -> None:
    monkeypatch.setenv("VERL_DRAFTER_RAW_TOP_LOGPROBS", "1")
    server = _SpecoSGLangHttpServerMixin()
    server.replica_rank = 0
    server._drafter_collection_step = None
    server._drafter_collection_samples = 0
    server._drafter_collection_tokens = 0
    server._speco_drafter_config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DFLASH",
        "training": {
            "collect_hidden_states_from_sgl": True,
            "collect_interval_steps": 5,
            "use_logits": False,
            "dflash_max_window": 64,
            "hidden_state_window_mode": "front",
        },
    }

    sampling_params: dict = {}
    should_collect, _, custom_params = server._speco_request_hidden_state_params(
        sampling_params,
        prompt_len=8,
        request_id="req-dflash",
        collection_global_steps=5,
        max_new_tokens=16,
    )

    assert should_collect is True
    assert custom_params[DFLASH_RETURN_AUX_HIDDEN_PARAM] is True
    assert DRAFTER_RETURN_LAST_HIDDEN_PARAM not in custom_params
    assert DRAFTER_RAW_TOP_LOGPROBS_PARAM not in custom_params
    assert sampling_params["custom_params"] == custom_params


def test_eagle3_last_hidden_collection_does_not_request_raw_topk_without_logits(monkeypatch) -> None:
    monkeypatch.setenv("VERL_DRAFTER_RAW_TOP_LOGPROBS", "1")
    server = _SpecoSGLangHttpServerMixin()
    server.replica_rank = 0
    server._drafter_collection_step = None
    server._drafter_collection_samples = 0
    server._drafter_collection_tokens = 0
    server._speco_drafter_config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "training": {
            "collect_hidden_states_from_sgl": True,
            "collect_interval_steps": 5,
            "use_logits": False,
            "hidden_state_window_mode": "front",
        },
    }

    sampling_params: dict = {}
    should_collect, _, custom_params = server._speco_request_hidden_state_params(
        sampling_params,
        prompt_len=8,
        request_id="req-eagle3",
        collection_global_steps=5,
        max_new_tokens=16,
    )

    assert should_collect is True
    assert custom_params[DRAFTER_RETURN_LAST_HIDDEN_PARAM] is True
    assert DFLASH_RETURN_AUX_HIDDEN_PARAM not in custom_params
    assert DRAFTER_RAW_TOP_LOGPROBS_PARAM not in custom_params
    assert sampling_params["custom_params"] == custom_params
