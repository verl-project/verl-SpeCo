"""One-use SpecOPD lifecycle contracts for V1 sync, independent of hardware."""

from types import SimpleNamespace

from verl_speco.trainer.v1 import specopd_adapter
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def test_v1_sync_bypasses_legacy_replay_and_publishes_after_actor(monkeypatch):
    events = []
    value = SimpleNamespace(
        metrics={"specopd/trained": 1},
        before_actor_update=lambda: events.append("draft-update"),
        publish=lambda: events.append("draft-publish-ack"),
    )
    monkeypatch.setattr(specopd_adapter, "enabled", lambda trainer: True)
    monkeypatch.setattr(specopd_adapter, "driver", lambda trainer: value)
    monkeypatch.setattr(specopd_adapter, "prepare_rollout", lambda trainer, batch: events.append("bind-rollout") or batch)
    monkeypatch.setattr(specopd_adapter, "old_log_prob", lambda trainer, batch, metrics: events.append("joint-tq-oldlogprob") or batch)

    class Base:
        def _submit_batch_to_rollout(self, batch):
            events.append("rollout")
            return batch

        def _compute_old_log_prob(self, batch, metrics):
            raise AssertionError("normal oldlogprob must be replaced")

        def _update_actor(self, batch, metrics):
            events.append("actor-update")
            return batch

        def on_step_end(self):
            events.append("actor-publish")

    class Trainer(SpecoV1Mixin, Base):
        pass

    trainer = Trainer()
    batch, metrics = object(), {}
    assert trainer._submit_batch_to_rollout(batch) is batch
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer._update_actor(batch, metrics) is batch
    trainer.on_step_end()
    assert trainer._pending_sync_metrics["specopd/trained"] == 1
    assert events == ["bind-rollout", "rollout", "joint-tq-oldlogprob", "draft-update", "actor-update", "actor-publish", "draft-publish-ack"]


def test_v1_empty_config_keeps_baseline_path():
    assert not specopd_adapter.enabled(SimpleNamespace())
    assert not specopd_adapter.enabled(SimpleNamespace(config={}))


def test_joint_oldlogprob_writes_response_aligned_tq_fields(monkeypatch):
    import sys
    from types import ModuleType
    import torch

    calls = []

    class Fields(dict):
        def select(self, *keys):
            return Fields({key: self[key] for key in keys})

    nested = Fields(response_mask=[torch.ones(2), torch.ones(3)])
    padded = SimpleNamespace(batch={"response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])})
    log_probs = torch.tensor([[-1., -2., 0.], [-3., -4., -5.]])
    output = SimpleNamespace(batch={"old_log_probs": log_probs, "entropys": torch.ones(2, 3)})

    def scored(batch):
        calls.append(("joint-forward", batch))
        return output, 0.0

    value = SimpleNamespace(old_log_prob=scored, metrics={"specopd/model_forward_calls": 2})
    monkeypatch.setattr(specopd_adapter, "driver", lambda trainer: value)
    tq = ModuleType("transfer_queue")

    def put(**kwargs):
        calls.append(("tq-put", kwargs))
        return "updated-meta"

    tq.kv_batch_put = put
    monkeypatch.setitem(sys.modules, "transfer_queue", tq)
    padding = ModuleType("verl.workers.utils.padding")
    padding.response_to_nested = lambda values, mask: [row[:len(m)] for row, m in zip(values, mask, strict=True)]
    monkeypatch.setitem(sys.modules, "verl.workers.utils.padding", padding)
    algos = ModuleType("verl.trainer.ppo.core_algos")
    algos.agg_loss = lambda **kwargs: (kwargs["loss_mat"] * kwargs["loss_mask"]).sum() / kwargs["loss_mask"].sum()
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", algos)
    trainer = SimpleNamespace(
        _speco_v1_batch_data=lambda batch: (nested, padded),
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None), rollout={}
        )),
    )
    batch = SimpleNamespace(keys=["a", "b"], partition_id="train")
    metrics = {}
    assert specopd_adapter.old_log_prob(trainer, batch, metrics) == "updated-meta"
    assert [name for name, _ in calls] == ["joint-forward", "tq-put"]
    written = calls[-1][1]
    assert written["keys"] == batch.keys and written["partition_id"] == "train"
    assert set(written["fields"]) == {"old_log_probs", "entropy"}
    assert [row.tolist() for row in written["fields"]["old_log_probs"]] == [[-1., -2.], [-3., -4., -5.]]
    assert metrics["actor/entropy"] == 1.0
