# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
import pytest

from verl_speco.trainer.hard_sample_selection import partition_hard_candidates


def _rows(count: int):
    return [
        {"batch_idx": index, "mean_accept_len": float(index), "hash": index / count}
        for index in range(count)
    ]


def test_hard_candidates_use_lowest_acceptance_prefix():
    hard, candidate_count = partition_hard_candidates(
        _rows(20), candidate_ratio=0.4
    )
    assert [row["batch_idx"] for row in hard] == list(range(8))
    assert candidate_count == 8


@pytest.mark.parametrize(
    "candidate_ratio,expected_count",
    [(0.0, 0), (0.4, 128), (1.0, 320)],
)
def test_production_batch_ratios_keep_exact_hard_band(
    candidate_ratio, expected_count
):
    hard, candidate_count = partition_hard_candidates(
        _rows(320), candidate_ratio=candidate_ratio
    )
    assert candidate_count == len(hard) == expected_count
    assert [row["batch_idx"] for row in hard] == list(range(expected_count))


@pytest.mark.parametrize("candidate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_partition_ratios(candidate):
    with pytest.raises(ValueError):
        partition_hard_candidates(
            _rows(20), candidate_ratio=candidate
        )
