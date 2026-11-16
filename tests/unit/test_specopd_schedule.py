from types import SimpleNamespace

from verl_speco.integration.specopd_driver import SpecOPDDriver


def test_five_step_twenty_update_schedule_with_one_publication():
    events = []
    training = dict(step=20, collect_interval_steps=5, training_interval_steps=5,
                    dspark_block_size=7, drafter_opd=dict(enable=True, method="spec_opd"))
    cfg = dict(trainer=dict(use_v1=True, v1=dict(trainer_mode="sync")), actor_rollout_ref=dict(
        actor=dict(strategy="fsdp2", ulysses_sequence_parallel_size=4),
        model=dict(use_remove_padding=True), rollout=dict(name="vllm", temperature=1.0,
            drafter=dict(enable=True, enable_drafter_training=True, speculative_algorithm="DSPARK",
                         vllm=dict(draft_sample_method="probabilistic"), training=training))))
    trainer = SimpleNamespace(config=cfg, global_steps=1, actor_revision=0)
    worker = SimpleNamespace(train_specopd_group=lambda samples, step, revision, indices: events.append(("train", step, 20)) or [dict(trained=True, revision=revision+1, optimizer_updates=20)])
    trainer._ray_get_if_needed = lambda x: x
    trainer._require_speco_worker_group = lambda: worker
    trainer._speco_set_drafter_global_step = lambda: None
    trainer._speco_sync_target_lm_head_weight = lambda: {}
    trainer._speco_publish_drafter_weights = lambda *a, **kw: events.append(("publish", trainer.global_steps)) or {"drafter/published": 1}
    ack = SimpleNamespace(speco_set_serving_drafter_revision=SimpleNamespace(remote=lambda rev: dict(actor_revision=trainer.actor_revision, drafter_revision=rev)))
    trainer.llm_server_manager = SimpleNamespace(server_handles=[ack])
    driver = SpecOPDDriver(trainer)
    driver.bootstrapped = True
    for step in range(1, 11):
        trainer.global_steps = step
        due = driver.begin_step(step)
        assert due == (step in (5, 10))
        if due:
            driver.cycle.actor_revision = trainer.actor_revision
            request = SimpleNamespace(branches=[SimpleNamespace(token_ids=(1, 2))])
            driver.cycle.scored(((request, None),), trainer.actor_revision, driver.cycle.drafter_revision)
        driver.before_actor_update()
        trainer.actor_revision = step
        driver.finish_step()
        assert driver.metrics["specopd/ack_actor_revision"] == step
        assert driver.metrics["specopd/ack_drafter_revision"] == step // 5
        assert driver.metrics["specopd/optimizer_updates"] == (20 if due else 0)
        assert driver.metrics["specopd/published"] == int(due)
    assert events == [("train", 5, 20), ("publish", 5), ("train", 10, 20), ("publish", 10)]
