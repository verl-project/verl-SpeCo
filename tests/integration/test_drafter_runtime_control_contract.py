from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


_speco_ray_trainer = pytest.importorskip(
    "verl_speco.trainer.speco_ray_trainer",
    reason="drafter runtime control contract needs the trainer dependency stack",
)
SpecoRayPPOTrainer = _speco_ray_trainer.SpecoRayPPOTrainer
from verl_speco.trainer.base_trainer import DrafterBaseTrainer


class _FakeOldLogProbBatch:
    non_tensor_batch = {}

    def __init__(self) -> None:
        self.selected_non_tensor_keys = None

    def select(self, *, non_tensor_batch_keys=None, **kwargs):
        self.selected_non_tensor_keys = non_tensor_batch_keys
        return self

    def to_tensordict(self):
        raise AssertionError("non-collect old-logprob steps should not enter the collection compute path")


class _FakeRolloutWorkerGroup:
    def __init__(self) -> None:
        self.compute_log_prob_calls = 0

    def generate_sequences(self, *args, **kwargs):
        return SimpleNamespace(meta_info={"metrics": {}})

    def compute_log_prob(self, batch):
        self.compute_log_prob_calls += 1
        raise AssertionError("non-collect old-logprob steps should use the original compute path")


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def _trainer(training_cfg: dict, *, step: int = 1) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(calculate_entropy=False),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=True,
                    enable_drafter_training=True,
                    training=training_cfg,
                )
            )
        )
    )
    trainer._pending_drafter_publish_refs = None
    trainer._speco_last_collected_samples = 0
    trainer._ray_get_if_needed = lambda value: value
    return trainer


def _no_drafter_trainer(*, calculate_entropy=Ellipsis) -> SpecoRayPPOTrainer:
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    actor = SimpleNamespace()
    if calculate_entropy is not Ellipsis:
        actor.calculate_entropy = calculate_entropy
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=actor,
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(
                    enable=False,
                    enable_drafter_training=False,
                    training={},
                )
            ),
        )
    )
    return trainer


def test_drafter_collect_train_and_publish_intervals() -> None:
    trainer = _trainer(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
        },
        step=6,
    )

    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is True
    assert trainer._speco_should_publish_drafter_weights(True) is False

    trainer.global_steps = 8
    assert trainer._speco_should_collect_drafter_this_step() is True
    assert trainer._speco_should_train_drafter_this_step() is False
    assert trainer._speco_should_publish_drafter_weights(True) is True
    assert trainer._speco_should_publish_drafter_weights(False) is False


def test_drafter_training_attempt_requires_interval_and_samples() -> None:
    trainer = _trainer({"training_interval_steps": 5}, step=4)
    trainer._speco_last_collected_samples = 10
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer.global_steps = 5
    trainer._speco_last_collected_samples = 0
    trainer._speco_oldlogprob_collection_requested = lambda: True
    assert trainer._speco_should_attempt_drafter_train_this_step() is False

    trainer._speco_last_collected_samples = 1
    assert trainer._speco_should_attempt_drafter_train_this_step() is True


def test_oldlogprob_entropy_wrapper_respects_no_drafter_entropy_config() -> None:
    assert _no_drafter_trainer(calculate_entropy=False)._speco_oldlogprob_entropy_hook_enabled() is True
    assert _no_drafter_trainer(calculate_entropy=True)._speco_oldlogprob_entropy_hook_enabled() is False
    assert _no_drafter_trainer()._speco_oldlogprob_entropy_hook_enabled() is False


def test_oldlogprob_non_collect_step_uses_original_compute_path() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 2,
            "training_interval_steps": 1,
        },
        step=1,
    )
    trainer.config.actor_rollout_ref.actor.calculate_entropy = True
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp"
    trainer.actor_rollout_wg = _FakeRolloutWorkerGroup()
    trainer._update_actor = lambda *args, **kwargs: SimpleNamespace(meta_info={"metrics": {}})
    original_calls = []

    def original_compute_old_log_prob(batch):
        original_calls.append(batch)
        return "old-log-prob", 0.5

    trainer._compute_old_log_prob = original_compute_old_log_prob
    batch = _FakeOldLogProbBatch()

    with trainer._speco_online_fit_hooks():
        result = trainer._compute_old_log_prob(batch)

    assert result == ("old-log-prob", 0.5)
    assert original_calls == [batch]
    assert batch.selected_non_tensor_keys is None
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 0
    assert trainer._speco_last_collect_interval_matched == 0


def test_dspark_l1_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({"dspark_l1_loss_alpha": 0.9}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"


def test_dspark_default_oldlogprob_layout_collects_final_hidden() -> None:
    trainer = _trainer({}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux_plus_last"
    assert trainer._speco_get_drafter_target_lm_head_row_selection() is None


def test_dspark_ce_only_oldlogprob_layout_keeps_aux_only_hidden() -> None:
    trainer = _trainer({"dspark_l1_loss_alpha": 0.0}, step=1)
    trainer.config.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"

    assert trainer._speco_oldlogprob_hidden_layout() == "dflash_aux"


def test_dspark_oldlogprob_collect_plan_uses_request_hard_quota() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 16,
            "max_collect_tokens_per_step_per_replica": 16384,
            "batch_size_per_gpu": 8,
            "dspark_hard_candidate_ratio": 0.20,
            "dspark_hard_sample_ratio": 0.375,
            "hidden_state_window_mode": "random",
        },
        step=20,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 5

    batch_size = 100
    prompt_width = 4
    response_width = 520
    prompts = torch.ones(batch_size, prompt_width, dtype=torch.long)
    responses = torch.ones(batch_size, response_width, dtype=torch.long)
    attention_mask = torch.ones(batch_size, prompt_width + response_width, dtype=torch.long)
    response_mask = torch.ones(batch_size, response_width, dtype=torch.long)
    batch = SimpleNamespace(
        batch={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0 + index / 1000.0 for index in range(batch_size)],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
            "_speco_vllm_request_completion_index": list(reversed(range(batch_size))),
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["selected_count"] == 80
    assert int(plan["request_is_hard"].logical_and(plan["collect_mask"]).sum().item()) == 16
    assert plan["request_is_hard"][:16].all()
    assert not plan["request_is_hard"][16:].any()
    assert [int((plan["request_is_hard"] & (plan["owner_rank"] == owner)).sum().item()) for owner in range(5)] == [
        4,
        3,
        3,
        3,
        3,
    ]
    records = trainer._speco_last_request_accept_len_records
    assert records[0]["request_id"] == "req-99"
    assert records[-1]["request_id"] == "req-0"
    assert plan["request_completion_indices"][0] == 99


def test_oldlogprob_collect_plan_accepts_array_like_request_stats() -> None:
    trainer = _trainer(
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_interval_steps": 1,
            "training_interval_steps": 1,
            "hidden_state_window_tokens_per_sample": 512,
            "max_collect_samples_per_step_per_replica": 8,
            "max_collect_tokens_per_step_per_replica": 8192,
            "batch_size_per_gpu": 4,
            "dspark_hard_candidate_ratio": 0.5,
            "dspark_hard_sample_ratio": 0.5,
            "hidden_state_window_mode": "front",
        },
        step=21,
    )
    trainer.config.actor_rollout_ref.actor.strategy = "fsdp2"
    trainer._speco_online_enabled = lambda: True
    trainer._speco_owner_bucket_count = lambda: 2

    batch_size = 12
    prompt_width = 4
    response_width = 520
    batch = SimpleNamespace(
        batch={
            "prompts": torch.ones(batch_size, prompt_width, dtype=torch.long),
            "responses": torch.ones(batch_size, response_width, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, prompt_width + response_width, dtype=torch.long),
            "response_mask": torch.ones(batch_size, response_width, dtype=torch.long),
        },
        non_tensor_batch={
            "_verl_request_mean_accept_len": _ArrayLike([1.0 + index for index in range(batch_size)]),
            "_speco_vllm_request_id": _ArrayLike([f"req-{index}" for index in range(batch_size)]),
            "_speco_vllm_request_completion_index": _ArrayLike(range(batch_size)),
            "_speco_vllm_request_elapsed_sec": _ArrayLike([0.1 * index for index in range(batch_size)]),
        },
    )

    plan = trainer._speco_build_oldlogprob_collect_plan(batch)

    assert plan["selected_count"] == 12
    assert int(plan["request_is_hard"].logical_and(plan["collect_mask"]).sum().item()) == 6
    assert plan["request_is_hard"][:6].all()
    assert not plan["request_is_hard"][6:].any()
    records = trainer._speco_last_request_accept_len_records
    assert len(records) == batch_size
    assert records[0]["request_id"] == "req-0"
    assert records[-1]["request_id"] == "req-11"
    assert trainer._speco_last_request_accept_len_var > 0.0


def test_request_accept_len_variance_logs_without_hard_sampling(capsys) -> None:
    trainer = _trainer(
        {
            "request_accept_len_variance_interval_steps": 2,
            "dspark_hard_candidate_ratio": 0.0,
            "dspark_hard_sample_ratio": 0.0,
        },
        step=4,
    )

    batch_size = 4
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 2.0, 3.0, 4.0],
            "_speco_vllm_request_id": [f"req-{index}" for index in range(batch_size)],
        },
    )

    logged = trainer._speco_log_request_accept_len_variance_from_batch(batch, source="test")

    captured = capsys.readouterr()
    assert logged is True
    assert "source=test" in captured.out
    assert "request_accept_len_var=1.250" in captured.out
    assert trainer._speco_last_request_accept_len_var == 1.25


def test_request_accept_len_variance_respects_log_interval(capsys) -> None:
    trainer = _trainer(
        {
            "request_accept_len_variance_interval_steps": 3,
        },
        step=4,
    )

    batch_size = 2
    batch = SimpleNamespace(
        batch={},
        non_tensor_batch={
            "_verl_request_mean_accept_len": [1.0, 3.0],
            "_speco_vllm_request_id": ["req-0", "req-1"],
        },
    )

    logged = trainer._speco_log_request_accept_len_variance_from_batch(batch, source="test")

    assert logged is True
    assert "[speco request accept len]" not in capsys.readouterr().out
    assert trainer._speco_last_request_accept_len_var == 1.0


def test_block_drafter_training_sampler_honors_explicit_hard_labels() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.rank = 0
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "dspark_hard_sample_ratio": 0.375,
                }
            )
        )
    )
    items = [{"_verl_is_hard": True, "id": f"hard-{index}"} for index in range(16)]
    items.extend({"_verl_is_hard": False, "id": f"normal-{index}"} for index in range(64))

    selected = trainer._sample_training_items(items, 8, __import__("random").Random(1))

    assert sum(1 for item in selected if item["_verl_is_hard"]) == 3


def test_block_drafter_training_sampler_normal_fill_excludes_explicit_hard() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.rank = 0
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={
                    "dspark_hard_sample_ratio": 0.125,
                }
            )
        )
    )
    items = [{"_verl_is_hard": True, "id": f"hard-{index}"} for index in range(32)]
    items.extend({"_verl_is_hard": False, "id": f"normal-{index}"} for index in range(48))

    selected = trainer._sample_training_items(items, 8, __import__("random").Random(1))

    assert len(selected) == 8
    assert sum(1 for item in selected if item["_verl_is_hard"]) == 1


def test_async_publish_sets_pending_ref_and_waits_before_next_publish() -> None:
    calls: list[tuple[str, object, int]] = []
    waited: list[object] = []
    trainer = _trainer({"publish_interval_steps": 1, "publish_async": True}, step=10)
    trainer._pending_drafter_publish_refs = ["old-ref"]
    trainer._ray_get_if_needed = lambda value: waited.append(value) or value
    trainer._speco_get_published_drafter_weights = lambda: {"weights": 1}
    trainer._speco_actor_rollout_method = lambda name: (
        lambda payload, global_steps=None: calls.append((name, payload, global_steps)) or ["new-ref"]
    )

    metrics = trainer._speco_publish_drafter_weights(True)

    assert waited == [["old-ref"]]
    assert calls == [("update_draft_weights_async", {"weights": 1}, 10)]
    assert trainer._pending_drafter_publish_refs == ["new-ref"]
    assert metrics == {"drafter/publish_attempted": 1, "drafter/published": 1}


def test_disabled_or_untrained_drafter_does_not_publish() -> None:
    trainer = _trainer({"publish_interval_steps": 1}, step=1)
    assert trainer._speco_publish_drafter_weights(False) == {
        "drafter/publish_attempted": 0,
        "drafter/published": 0,
    }


def test_drafter_checkpoint_results_require_a_successful_training_replica() -> None:
    SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
        [
            {"saved": True, "reason": "saved"},
            {"saved": False, "reason": "not_checkpoint_replica"},
            {"saved": False, "reason": "not_in_training_group"},
        ],
        require_saved=True,
    )

    with pytest.raises(RuntimeError, match="produced no saved state"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "not_checkpoint_replica"}],
            require_saved=True,
        )


def test_drafter_checkpoint_results_propagate_save_failure() -> None:
    with pytest.raises(RuntimeError, match="missing_checkpoint_dir"):
        SpecoRayPPOTrainer._speco_validate_drafter_checkpoint_results(
            [{"saved": False, "reason": "missing_checkpoint_dir"}],
            require_saved=True,
        )
