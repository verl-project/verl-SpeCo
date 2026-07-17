"""Tests for the fail-closed lossless speculative-decoding guard.

RL rollout under speculative decoding is only unbiased when the verifier samples
exactly from the target policy (SPECO recomputes PPO old_log_probs as the target
logprob with no importance-sampling correction). These tests lock in that SPECO
refuses known-lossy acceptance settings unless the user explicitly opts in.
"""

from __future__ import annotations

import pytest

from verl_speco.integration.vllm_runtime import (
    assert_lossless_vllm_speculative_config,
    build_vllm_speculative_config_from_drafter,
)


def test_default_greedy_config_passes() -> None:
    # The config SPECO builds by default: greedy DRAFT sampling is lossless
    # (one-hot proposal -> rejection sampling), so it must not be rejected.
    assert_lossless_vllm_speculative_config(
        {"method": "eagle3", "num_speculative_tokens": 3, "draft_sample_method": "greedy"},
        allow_lossy=False,
    )
    print("tests/integration/test_lossless_sd_guard.py::test_default_greedy_config_passes", flush=True)


@pytest.mark.parametrize(
    "lossy_override",
    [
        {"acceptance_method": "typical_acceptance_sampler"},
        {"spec_decoding_acceptance_method": "typical_acceptance_sampler"},
        {"rejection_sample_method": "synthetic"},
        {"posterior_threshold": 0.3},
        {"posterior_alpha": 0.5},
    ],
)
def test_lossy_acceptance_is_rejected(lossy_override) -> None:
    config = {"method": "eagle3", "num_speculative_tokens": 3, **lossy_override}
    with pytest.raises(ValueError, match="lossy speculative-decoding config"):
        assert_lossless_vllm_speculative_config(config, allow_lossy=False)
    print("tests/integration/test_lossless_sd_guard.py::test_lossy_acceptance_is_rejected", flush=True)


def test_opt_out_allows_lossy() -> None:
    config = {"method": "eagle3", "acceptance_method": "typical_acceptance_sampler"}
    # No raise when the user knowingly opts in.
    assert_lossless_vllm_speculative_config(config, allow_lossy=True)
    print("tests/integration/test_lossless_sd_guard.py::test_opt_out_allows_lossy", flush=True)


def _drafter_cfg(**vllm_overrides):
    return {
        "enable": True,
        "speculative_algorithm": "EAGLE3",
        "model_path": "/tmp/drafter",
        "rollout": {"spec_steps": 3},
        "vllm": vllm_overrides,
    }


def test_build_rejects_lossy_override() -> None:
    cfg = _drafter_cfg(speculative_config_overrides={"rejection_sample_method": "synthetic"})
    with pytest.raises(ValueError, match="lossy speculative-decoding config"):
        build_vllm_speculative_config_from_drafter(cfg)
    print("tests/integration/test_lossless_sd_guard.py::test_build_rejects_lossy_override", flush=True)


def test_build_allows_lossy_override_with_opt_out() -> None:
    cfg = _drafter_cfg(
        speculative_config_overrides={"rejection_sample_method": "synthetic"},
        allow_lossy_speculative_sampling=True,
    )
    out = build_vllm_speculative_config_from_drafter(cfg)
    assert out["rejection_sample_method"] == "synthetic"
    print("tests/integration/test_lossless_sd_guard.py::test_build_allows_lossy_override_with_opt_out", flush=True)


def test_build_default_is_lossless_and_passes() -> None:
    out = build_vllm_speculative_config_from_drafter(_drafter_cfg())
    assert out["draft_sample_method"] == "greedy"
    assert "acceptance_method" not in out
    print("tests/integration/test_lossless_sd_guard.py::test_build_default_is_lossless_and_passes", flush=True)


def test_string_false_opt_out_does_not_bypass_guard() -> None:
    # A YAML/CLI override can deliver the flag as the string "false"; bool("false")
    # is True and would silently disable this fail-closed guard, so the opt-out must
    # be parsed as a real boolean.
    cfg = _drafter_cfg(
        speculative_config_overrides={"rejection_sample_method": "synthetic"},
        allow_lossy_speculative_sampling="false",
    )
    with pytest.raises(ValueError, match="lossy speculative-decoding config"):
        build_vllm_speculative_config_from_drafter(cfg)
    print("tests/integration/test_lossless_sd_guard.py::test_string_false_opt_out_does_not_bypass_guard", flush=True)


def test_whitespace_padded_lossy_value_is_rejected() -> None:
    config = {"method": "eagle3", "acceptance_method": " typical_acceptance_sampler "}
    with pytest.raises(ValueError, match="lossy speculative-decoding config"):
        assert_lossless_vllm_speculative_config(config, allow_lossy=False)
    print("tests/integration/test_lossless_sd_guard.py::test_whitespace_padded_lossy_value_is_rejected", flush=True)
