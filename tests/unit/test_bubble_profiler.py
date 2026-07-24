from __future__ import annotations

import math

from verl_speco.trainer.bubble_profiler import (
    BUBBLE_PREFIX,
    compute_bubble_metrics,
    inject_bubble_metrics,
)


def _base_step_metrics() -> dict[str, float]:
    # A representative post-move step: gen + old_log_prob + update_actor +
    # drafter account for most of the step, leaving a small unaccounted gap.
    return {
        "training/global_step": 3,
        "timing_s/step": 10.0,
        "timing_s/gen": 4.0,
        "timing_s/old_log_prob": 1.0,
        "timing_s/update_actor": 2.0,
        "timing_s/drafter": 2.0,
        # sub-stage breakdown that must NOT be double counted
        "timing_s/drafter_train_rpc": 1.5,
    }


def test_compute_basic_accounting_and_headroom():
    metrics = compute_bubble_metrics(_base_step_metrics())
    assert metrics["bubble/step_s"] == 10.0
    # 4 + 1 + 2 + 2 = 9 top-level; drafter_train_rpc excluded.
    assert metrics["bubble/accounted_s"] == 9.0
    assert metrics["bubble/unaccounted_s"] == 1.0
    assert metrics["bubble/unaccounted_ratio"] == 0.1
    assert metrics["bubble/gen_s"] == 4.0
    assert metrics["bubble/gen_ratio"] == 0.4
    assert metrics["bubble/drafter_s"] == 2.0
    assert metrics["bubble/drafter_ratio"] == 0.2
    # min(gen=4, drafter=2) = 2 recoverable by overlap.
    assert metrics["bubble/overlap_headroom_s"] == 2.0
    assert metrics["bubble/overlap_headroom_ratio"] == 0.2


def test_all_bubble_keys_use_prefix():
    metrics = compute_bubble_metrics(_base_step_metrics())
    assert metrics
    assert all(key.startswith(BUBBLE_PREFIX) for key in metrics)


def test_missing_step_returns_empty():
    assert compute_bubble_metrics({"timing_s/gen": 4.0}) == {}


def test_non_positive_step_returns_empty():
    assert compute_bubble_metrics({"timing_s/step": 0.0}) == {}
    assert compute_bubble_metrics({"timing_s/step": -1.0}) == {}


def test_non_mapping_returns_empty():
    assert compute_bubble_metrics(None) == {}
    assert compute_bubble_metrics([("timing_s/step", 1.0)]) == {}


def test_accounted_clamped_to_step_on_timer_skew():
    # Stage timers overshoot the wall clock: accounted is clamped and the
    # unaccounted remainder never goes negative.
    metrics = compute_bubble_metrics(
        {"timing_s/step": 5.0, "timing_s/gen": 4.0, "timing_s/update_actor": 4.0}
    )
    assert metrics["bubble/accounted_s"] == 5.0
    assert metrics["bubble/unaccounted_s"] == 0.0
    assert metrics["bubble/unaccounted_ratio"] == 0.0


def test_gen_and_drafter_clamped_to_step():
    metrics = compute_bubble_metrics(
        {"timing_s/step": 3.0, "timing_s/gen": 9.0, "timing_s/drafter": 9.0}
    )
    assert metrics["bubble/gen_s"] == 3.0
    assert metrics["bubble/gen_ratio"] == 1.0
    # drafter_s reports the raw value, drafter_ratio uses the capped value.
    assert metrics["bubble/drafter_s"] == 9.0
    assert metrics["bubble/drafter_ratio"] == 1.0
    assert metrics["bubble/overlap_headroom_s"] == 3.0


def test_alternate_generation_key_is_recognized():
    metrics = compute_bubble_metrics(
        {
            "timing_s/step": 10.0,
            "timing_s/generate_sequences": 6.0,
            "timing_s/drafter": 3.0,
        }
    )
    assert metrics["bubble/gen_s"] == 6.0
    assert metrics["bubble/overlap_headroom_s"] == 3.0


def test_rollout_only_step_has_zero_drafter_headroom():
    metrics = compute_bubble_metrics(
        {"timing_s/step": 8.0, "timing_s/gen": 5.0, "timing_s/old_log_prob": 2.0}
    )
    assert metrics["bubble/drafter_s"] == 0.0
    assert metrics["bubble/overlap_headroom_s"] == 0.0
    assert metrics["bubble/unaccounted_s"] == 1.0


def test_negative_and_nonfinite_stage_values_ignored():
    metrics = compute_bubble_metrics(
        {
            "timing_s/step": 10.0,
            "timing_s/gen": -2.0,
            "timing_s/old_log_prob": float("nan"),
            "timing_s/update_actor": float("inf"),
            "timing_s/drafter": 3.0,
        }
    )
    # only drafter=3 counts toward accounted; gen<0 and nan/inf dropped.
    assert metrics["bubble/accounted_s"] == 3.0
    assert metrics["bubble/gen_s"] == 0.0
    assert metrics["bubble/drafter_s"] == 3.0


def test_bool_values_are_not_treated_as_numbers():
    # A stray bool must not be coerced to 1.0 and counted as a duration.
    metrics = compute_bubble_metrics(
        {"timing_s/step": 10.0, "timing_s/gen": True, "timing_s/drafter": 2.0}
    )
    assert metrics["bubble/gen_s"] == 0.0
    assert metrics["bubble/accounted_s"] == 2.0


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_tensor_like_scalars_are_coerced():
    metrics = compute_bubble_metrics(
        {"timing_s/step": _Scalar(10.0), "timing_s/gen": _Scalar(4.0)}
    )
    assert metrics["bubble/step_s"] == 10.0
    assert metrics["bubble/gen_s"] == 4.0


class _BadScalar:
    def item(self):
        raise RuntimeError("no scalar")


def test_unconvertible_scalar_treated_as_missing():
    assert compute_bubble_metrics({"timing_s/step": _BadScalar()}) == {}


def test_non_numeric_string_stage_value_ignored():
    metrics = compute_bubble_metrics(
        {"timing_s/step": 10.0, "timing_s/gen": "n/a", "timing_s/drafter": 2.0}
    )
    assert metrics["bubble/gen_s"] == 0.0
    assert metrics["bubble/accounted_s"] == 2.0


def test_custom_prefix_and_stage_keys():
    metrics = compute_bubble_metrics(
        {"timing_s/step": 10.0, "timing_s/gen": 4.0, "timing_s/custom": 3.0},
        stage_keys=("timing_s/gen", "timing_s/custom"),
        prefix="pipe/",
    )
    assert metrics["pipe/accounted_s"] == 7.0
    assert metrics["pipe/unaccounted_s"] == 3.0
    assert all(key.startswith("pipe/") for key in metrics)


def test_inject_adds_metrics_without_mutating_input():
    data = _base_step_metrics()
    original = dict(data)
    result = inject_bubble_metrics(data)
    assert data == original  # input untouched
    assert result is not data
    assert result["bubble/unaccounted_s"] == 1.0
    # original keys are preserved
    assert result["timing_s/step"] == 10.0


def test_inject_is_noop_without_step():
    data = {"training/global_step": 1, "timing_s/gen": 4.0}
    result = inject_bubble_metrics(data)
    assert result is data


def test_inject_non_dict_passthrough():
    assert inject_bubble_metrics(None) is None
    sentinel = ["not", "a", "dict"]
    assert inject_bubble_metrics(sentinel) is sentinel


def test_inject_does_not_overwrite_existing_bubble_key():
    data = _base_step_metrics()
    data["bubble/unaccounted_s"] = 999.0
    result = inject_bubble_metrics(data)
    assert result["bubble/unaccounted_s"] == 999.0


def test_ratios_are_finite_and_bounded():
    metrics = compute_bubble_metrics(_base_step_metrics())
    for key, value in metrics.items():
        assert math.isfinite(value), key
        if key.endswith("_ratio"):
            assert 0.0 <= value <= 1.0, key
