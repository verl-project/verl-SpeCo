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

"""V1 request-aware sampling and fixed-target freeze lifecycle contracts."""

from types import SimpleNamespace

import pytest

from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def test_v1_request_evidence_excludes_tagged_padding_without_hiding_duplicates(
    monkeypatch,
):
    import sys
    import torch
    from omegaconf import OmegaConf
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    data = {
        "prompts": torch.tensor([[1], [2], [1]]),
        "responses": torch.tensor([[3], [4], [3]]),
        "input_ids": torch.tensor([[1, 3], [2, 4], [1, 3]]),
        "response_mask": torch.tensor([[1], [1], [0]]),
        "position_ids": torch.tensor([[0, 1]] * 3),
    }
    cells = [
        {"_speco_vllm_request_id": request, "_verl_request_mean_accept_len": 3.0}
        for request in ("req-a", "req-b", "req-a")
    ]

    def get(**kwargs):
        return (
            {"extra_fields": cells}
            if kwargs["select_fields"] == ["extra_fields"]
            else data
        )

    monkeypatch.setitem(
        sys.modules, "transfer_queue", SimpleNamespace(kv_batch_get=get)
    )
    trainer = SpecoV1Mixin()
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {}}})
    batch = SimpleNamespace(
        keys=["a", "b", "pad"],
        partition_id="train",
        tags=[{}, {}, {"is_padding": True}],
    )
    _, evidence = trainer._speco_v1_batch_data(batch)
    assert len(evidence) == 3  # Preserve the actor's divisible batch.
    assert evidence.meta_info["speco_v1_padding_rows"] == 1
    assert evidence.non_tensor_batch["_speco_vllm_request_id"].tolist() == [
        "req-a",
        "req-b",
        None,
    ]
    assert evidence.non_tensor_batch["_verl_request_mean_accept_len"].tolist() == [
        3.0,
        3.0,
        None,
    ]
    assert SpecoRayPPOTrainer._speco_request_ids(
        trainer, evidence, 3, require=True, required_mask=[True, True, False]
    )[:2] == ["req-a", "req-b"]
    batch.tags[-1] = {}  # A genuine duplicate must still fail closed.
    _, evidence = trainer._speco_v1_batch_data(batch)
    with pytest.raises(RuntimeError, match="duplicate request ids"):
        SpecoRayPPOTrainer._speco_request_ids(trainer, evidence, 3, require=True)


def test_publish_pairs_probes_after_target_sync_and_waits_before_after_arm():
    calls = []

    class Harness(SpecoV1Mixin):
        _speco_v1_pending_training = True
        _speco_v1_training_plan = object()
        _speco_freeze_policy = SimpleNamespace(
            drafter_version=2, probe_due=lambda version: version == 3
        )
        _speco_v1_train_cost_seconds = 1.5

        def _speco_run_freeze_probe_arm(self):
            calls.append("probe")
            return [], 0.25

        def _speco_publish_drafter_weights(self, trained, plan, *, after_weight_update):
            assert trained and after_weight_update
            calls.append("publish")
            return {"drafter/published": 1}

        def _speco_wait_pending_drafter_publish(self):
            calls.append("wait")
            return 0

        def _speco_build_freeze_probe_comparison(
            self, before, after, *, version_before
        ):
            assert version_before == 2
            return "paired"

        def _speco_freeze_publish_cost_seconds(self, metrics):
            return 0.5

        def _speco_freeze_observe_publish(self, metrics, *, update_cost_seconds, probe):
            assert update_cost_seconds == 2.0 and probe == "paired"
            calls.append("observe")

        def _speco_freeze_log_probe(self, probe, error):
            assert probe == "paired" and error is None

        def _speco_freeze_metrics(self):
            return {}

    trainer = Harness()
    assert trainer._speco_finalize_v1_drafter_publish()["drafter/published"] == 1
    assert calls == ["probe", "publish", "wait", "probe", "observe"]
    assert not trainer._speco_v1_pending_training


def test_probe_uses_fixed_prompts_servers_and_never_tq(monkeypatch):
    torch = pytest.importorskip("torch")
    ray = pytest.importorskip("ray")
    calls = []

    def submit(server_id, **kwargs):
        calls.append((server_id, kwargs))
        return SimpleNamespace(
            token_ids=[5, 6],
            extra_fields={
                "_verl_request_mean_accept_len": 3.0,
                "_speco_vllm_request_elapsed_sec": 0.5,
            },
        )

    monkeypatch.setattr(ray, "get", lambda refs: refs)

    class Harness(SpecoV1Mixin):
        global_steps = 2
        llm_server_manager = SimpleNamespace(
            server_handles=[
                SimpleNamespace(
                    generate=SimpleNamespace(remote=lambda **kw: submit(0, **kw))
                ),
                SimpleNamespace(
                    generate=SimpleNamespace(remote=lambda **kw: submit(1, **kw))
                ),
            ]
        )

        def _speco_freeze_probe_config(self):
            return {"max_new_tokens": 64}

        def _speco_freeze_probe_pool_data(self):
            return [
                {
                    "input_ids": torch.tensor([[0, 1, 2], [0, 3, 4]]),
                    "attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
                }
            ]

    trainer = Harness()
    before, _ = trainer._speco_run_freeze_probe_arm()
    after, _ = trainer._speco_run_freeze_probe_arm()
    assert before == after
    assert [server for server, _ in calls] == [0, 1, 0, 1]
    assert [kwargs["prompt_ids"] for _, kwargs in calls] == [
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
    ]
    assert len({kwargs["request_id"] for _, kwargs in calls}) == 4
    assert all(
        not any(key.startswith("_verl_") for key in kwargs["sampling_params"])
        for _, kwargs in calls
    )
    assert all(kwargs["sampling_params"]["temperature"] == 0 for _, kwargs in calls)


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_v1_freeze_allows_only_modes_with_fixed_target_probe_boundary(mode):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": mode}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "collect_hidden_states_from_old_logprob": True,
                            "drafter_convergence_freeze": {"enabled": True},
                        },
                    },
                }
            },
            "algorithm": {},
        }
    )
    if mode == "separate_async":
        with pytest.raises(ValueError, match="sync or colocate_async"):
            SpecoV1Mixin._speco_validate_v1_training_config(config)
    else:
        SpecoV1Mixin._speco_validate_v1_training_config(config)


def test_async_freeze_projects_actual_revisions_without_changing_ppo_or_hard_batch():
    import numpy as np
    from omegaconf import OmegaConf
    from verl.protocol import DataProto
    import json

    evidence = DataProto(
        non_tensor_batch={
            "_speco_vllm_request_id": np.array(["pure", "stale", "mixed", "missing"]),
            "_verl_request_mean_accept_len": np.array([3.0, 4.0, 2.0, 3.0]),
            "_speco_drafter_revision_min": np.array([2, 1, 1, None], dtype=object),
            "_speco_drafter_revision_max": np.array([2, 1, 2, None], dtype=object),
            "_speco_drafter_revision_stats": np.array(
                [
                    None,
                    None,
                    json.dumps(
                        [
                            {
                                "revision": 1,
                                "verify_rounds": 1,
                                "accepted_tokens": 1,
                                "response_tokens": 4,
                            },
                            {
                                "revision": 2,
                                "verify_rounds": 3,
                                "accepted_tokens": 6,
                                "response_tokens": 8,
                            },
                        ]
                    ),
                    None,
                ],
                dtype=object,
            ),
        },
        meta_info={"speco_v1_padding_rows": 0},
    )
    observed = []

    class Parent:
        def _compute_old_log_prob(self, batch, metrics):
            return batch

    class Harness(SpecoV1Mixin, Parent):
        config = OmegaConf.create(
            {
                "trainer": {"v1": {"trainer_mode": "colocate_async"}},
                "actor_rollout_ref": {
                    "rollout": {
                        "drafter": {
                            "training": {
                                "drafter_convergence_freeze": {"enabled": True}
                            }
                        }
                    }
                },
            }
        )
        global_steps = 3
        _speco_freeze_policy = SimpleNamespace(drafter_version=2)

        def _speco_v1_batch_data(self, batch):
            return None, evidence

        def _speco_populate_request_accept_len_records(self, data):
            self.requests = data.non_tensor_batch["_speco_vllm_request_id"].tolist()

        def _speco_observe_rollout_evidence(self, data, version, **kwargs):
            observed.append(
                (self.requests[:], version, kwargs["opportunity_this_step"])
            )

        def _speco_should_train_drafter_this_step(self):
            return True

        def _speco_freeze_metrics(self):
            return {}

        def _speco_oldlogprob_collection_enabled(self):
            return False

    trainer, batch, metrics = Harness(), object(), {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert observed == [(["stale", "mixed"], 1, False), (["pure", "mixed"], 2, True)]
    assert trainer.requests == ["pure", "stale", "mixed", "missing"]
    assert metrics["drafter/freeze_revision_matched_rows"] == 3
    assert metrics["drafter/freeze_revision_excluded_rows"] == 1


def test_v1_raw_prompt_probe_uses_continuous_tokens_once_and_caps_left(monkeypatch):
    import sys
    from omegaconf import OmegaConf
    from transformers import AutoConfig

    calls = []

    def create(tokenizer, **kwargs):
        assert tokenizer == "tokenizer"
        assert kwargs["hf_model_type"] == "qwen3"
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": True}
        assert kwargs["processor"] is None

        def build(messages):
            calls.append(messages)
            return [10, 20, 30, 40]

        return SimpleNamespace(build_initial_tokens=build)

    monkeypatch.setitem(
        sys.modules,
        "verl.utils.tokenizer.continuous_token_wiring",
        SimpleNamespace(create_continuous_token_builder=create),
    )
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(model_type="qwen3"),
    )
    trainer = SpecoV1Mixin()
    trainer.tokenizer, trainer.processor = "tokenizer", None
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/model"},
                "rollout": {"prompt_length": 2},
            },
            "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
        }
    )
    chunk = {
        "raw_prompt": [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
    }
    before = trainer._speco_freeze_probe_prompt_ids(chunk)
    after = trainer._speco_freeze_probe_prompt_ids(chunk)
    assert before is after and before == [[30, 40], [30, 40]]
    assert len(calls) == 2


def test_freeze_transition_skips_collection_and_requests_branch_checkpoint():
    from omegaconf import OmegaConf
    from verl.protocol import DataProto
    import numpy as np

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "training": {
                            "training_interval_steps": 1,
                            "drafter_convergence_freeze": {
                                "enabled": True,
                                "mode": "active",
                                "branch_checkpoint": {"enabled": True},
                                "observation": {
                                    "min_requests_per_version": 4,
                                    "min_rollout_steps_per_version": 1,
                                },
                                "gain": {
                                    "window_updates": 1,
                                    "patience_updates": 1,
                                    "practical_gain_all": 1.0,
                                    "practical_gain_hard": 1.0,
                                },
                            },
                        }
                    }
                }
            },
            "trainer": {"resume_mode": "disable"},
        }
    )

    class Harness(SpecoV1Mixin):
        pass

    trainer = Harness()
    trainer.config = config
    trainer._speco_init_state()
    trainer._speco_freeze_policy = trainer._speco_init_freeze_policy()
    output = DataProto(
        non_tensor_batch={"_verl_request_mean_accept_len": np.array([3.0] * 4)}
    )
    for version in range(3):
        trainer.global_steps = version + 1
        trainer._speco_populate_request_accept_len_records(output)
        trainer._speco_observe_rollout_evidence(output, version)
        if version < 2:
            trainer._speco_freeze_observe_publish(
                {"drafter/published": 1}, update_cost_seconds=1
            )
    assert trainer._speco_drafter_frozen
    assert trainer._speco_freeze_branch_save_pending["trigger_step"] == 3
    assert trainer._speco_freeze_branch_save_pending["drafter_version"] == 2
