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

from verl_speco.trainer.v1.feature_adapter import from_transfer_queue_batch


def test_rollout_finite_scan_matches_torch_and_preserves_input():
    import torch
    from verl_speco.trainer.v1.feature_adapter import _rollout_hidden_isfinite

    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        value = torch.randn(2, 70000).to(dtype)
        original = value.clone()
        assert _rollout_hidden_isfinite(value)
        assert torch.equal(value, original)
        for bad in (float("nan"), float("inf"), -float("inf")):
            for index in (0, 65535, 65536, value.numel() - 1):
                value.reshape(-1)[index] = bad
                assert _rollout_hidden_isfinite(value) == bool(
                    torch.isfinite(value).all()
                )
                value.reshape(-1)[index] = 0
        assert _rollout_hidden_isfinite(value.t())


def test_feature_adapter_keeps_optional_fields_explicit():
    batch = {"input_ids": [1, 2], "attention_mask": [1, 1], "response_mask": [0, 1]}
    view = from_transfer_queue_batch(batch, global_step=7)

    assert view.input_ids == [1, 2]
    assert view.attention_mask == [1, 1]
    assert view.response_mask == [0, 1]
    assert view.hidden_states is None
    assert view.target_logprobs is None
    assert view.global_step == 7


def test_rollout_sample_validation_rejects_wrong_tq_row():
    import pytest

    torch = pytest.importorskip("torch")
    from verl_speco.trainer.v1.feature_adapter import validate_rollout_samples

    class Batch:
        batch = {"responses": torch.tensor([[4, 5, 0], [8, 9, 0]])}

        def __len__(self):
            return 2

    sample = {
        "hidden_source": "vllm_rollout",
        "global_step": 2,
        "_speco_sample_id": "a",
        "responses": torch.tensor([[4, 5]]),
        "hidden_states": torch.ones(1, 2, 3),
        "hidden_positions": torch.tensor([[2, 3]]),
    }
    batch = Batch()
    batch.non_tensor_batch = {"drafter_sample": [sample, None]}
    validate_rollout_samples(batch, 2)
    with pytest.raises(ValueError, match="step"):
        validate_rollout_samples(batch, 3)
    batch.non_tensor_batch["drafter_sample"] = [None, sample]
    with pytest.raises(ValueError, match="response"):
        validate_rollout_samples(batch, 2)


def test_rollout_validation_checks_prompt_and_position_range():
    import pytest

    torch = pytest.importorskip("torch")
    from verl_speco.trainer.v1.feature_adapter import validate_rollout_samples

    class Batch:
        batch = {
            "prompts": torch.tensor([[0, 1, 2]]),
            "responses": torch.tensor([[3, 4]]),
        }

        def __len__(self):
            return 1

    sample = {
        "hidden_source": "vllm_rollout",
        "global_step": 2,
        "_speco_sample_id": "s",
        "prompts": torch.tensor([[1, 2]]),
        "responses": torch.tensor([[3, 4]]),
        "hidden_states": torch.ones(1, 2, 3),
        "hidden_positions": torch.tensor([[1, 2]]),
    }
    batch = Batch()
    batch.non_tensor_batch = {"drafter_sample": [sample]}
    validate_rollout_samples(batch, 2)
    sample["prompts"] = torch.tensor([[9, 2]])
    with pytest.raises(ValueError, match="prompt"):
        validate_rollout_samples(batch, 2)
    sample["prompts"] = torch.tensor([[1, 2]])
    sample["hidden_positions"] = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="prediction range"):
        validate_rollout_samples(batch, 2)
    sample["hidden_positions"] = torch.tensor([[1, 3]])
    with pytest.raises(ValueError, match="contiguous"):
        validate_rollout_samples(batch, 2)


def test_async_prefetched_versions_reject_missing_mixed_and_future_heads(tmp_path):
    import pytest
    from verl_speco.trainer.v1.feature_adapter import validate_async_rollout_versions

    head = tmp_path / "0.pt"
    head.write_bytes(b"snapshot")
    assert (
        validate_async_rollout_versions([{"target_version": 0}, None], {0: head}, 2)
        == 0
    )
    for samples in (
        [{}],
        [{"target_version": 0}, {"target_version": 1}],
        [{"target_version": 3}],
        [{"target_version": True}],
    ):
        with pytest.raises(RuntimeError):
            validate_async_rollout_versions(samples, {0: head}, 2)
    with pytest.raises(RuntimeError, match="missing"):
        validate_async_rollout_versions([{"target_version": 0}], {}, 2)
    head.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        validate_async_rollout_versions([{"target_version": 0}], {0: head}, 2)
