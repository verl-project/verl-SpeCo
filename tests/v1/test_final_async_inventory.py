from types import SimpleNamespace as NS

import pytest
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

rb_module = pytest.importorskip(
    "verl.trainer.ppo.v1.replay_buffer", exc_type=ImportError
)


class Upstream:
    def prepare_step(self):
        self.prefetch_calls += 1
        return {'prefetched': 1}


class Harness(SpecoV1Mixin, Upstream):
    def __init__(self, replay, *, required=1, mode='colocate_async', step=5, total=5):
        self.config = NS(trainer=NS(v1={'trainer_mode': mode}), data=NS(train_batch_size=required))
        self.replay_buffer = replay
        self.global_steps, self.total_training_steps = step, total
        self.parameter_sync_step = 2 if required == 4 else 1
        self.prefetch_calls = 0


def actual_buffer(monkeypatch, metadata, *, refill=None, threshold=8):
    if not hasattr(rb_module.ReplayBufferAsync, "get_sampleable_count"):
        pytest.skip("requires the uni-agent ReplayBufferAsync inventory API")
    monkeypatch.setattr(rb_module.tq, 'kv_list', lambda: metadata)
    def clear(*, partition_id, keys):
        for key in keys:
            metadata.get(partition_id, {}).pop(key, None)
    monkeypatch.setattr(rb_module.tq, 'kv_clear', clear)
    return rb_module.ReplayBufferAsync(
        trainer_mode='colocate_async', trainer_config={},
        max_off_policy_threshold=threshold, max_off_policy_strategy='drop',
        sampler_kwargs={}, refill_fn=refill,
    )


def groups(statuses, step=4):
    return {'train': {f'p{i}': {'is_prompt': True, 'status': status, 'global_steps': step}
                      for i, status in enumerate(statuses)}}


@pytest.mark.parametrize('statuses,required,should_submit', [
    (['pending'], 1, False), (['running'], 1, False),
    (['finished'], 1, False), (['failure'], 1, False),
    ([], 1, True),  # zero warmup / empty inventory still seeds the final batch
    (['pending', 'running'], 4, True),  # count the full parameter-sync update
    (['pending'] * 4, 4, False),
    (['finished', 'pending', 'running', 'failure'], 4, False),
])
def test_final_admission_uses_unconsumed_prompt_inventory(monkeypatch, statuses, required, should_submit):
    replay = actual_buffer(monkeypatch, groups(statuses))
    trainer = Harness(replay, required=required)
    result = trainer.prepare_step()
    assert trainer.prefetch_calls == int(should_submit)
    assert result == ({'prefetched': 1} if should_submit else {})


def test_existing_pending_prompt_finishes_without_a_new_batch(monkeypatch):
    metadata = groups(['running'])
    refills = []
    replay = actual_buffer(monkeypatch, metadata, refill=refills.append)
    trainer = Harness(replay)
    assert trainer.prepare_step() == {}
    def complete_pending(partition_id, last_debug_time):
        metadata['train']['p0']['status'] = 'finished'
        return last_debug_time
    monkeypatch.setattr(replay, '_wait_for_next_poll', complete_pending)
    sampleable, metrics = replay.wait_for_sampleable(5, 'train', 1)
    assert sampleable == {'p0'} and not metrics
    assert refills == [] and trainer.prefetch_calls == 0


@pytest.mark.parametrize('kind', ['failure', 'stale'])
def test_real_wait_path_evicts_and_refills_terminal_inventory(monkeypatch, kind):
    metadata = groups(['failure' if kind == 'failure' else 'finished'], step=0 if kind == 'stale' else 4)
    refills = []
    def refill(count):
        refills.append(count)
        metadata['train']['replacement'] = {'is_prompt': True, 'status': 'finished', 'global_steps': 5}
    replay = actual_buffer(monkeypatch, metadata, refill=refill, threshold=1 if kind == 'stale' else 8)
    trainer = Harness(replay)
    assert trainer.prepare_step() == {}  # no redundant normal prefetch
    sampleable, metrics = replay.wait_for_sampleable(5, 'train', 1)
    assert sampleable == {'replacement'}
    assert refills == [1] and trainer.prefetch_calls == 0
    assert 'p0' not in metadata['train']
    key = 'training/rollout_failure/evicted_samples' if kind == 'failure' else 'training/off_policy/evicted_samples'
    assert metrics[key] == 1


def test_consumed_prompt_is_not_counted_again(monkeypatch):
    metadata = groups(['finished'])
    metadata['train']['p0_0_0'] = {'is_prompt': False, 'status': 'finished'}
    replay = actual_buffer(monkeypatch, metadata)
    assert replay.get_sampleable_count(5, 'train') == 1
    replay._materialize_batch('train', ['p0'], dict(replay.partitions['train']))
    trainer = Harness(replay)
    assert trainer.prepare_step() == {'prefetched': 1}
    assert trainer.prefetch_calls == 1


@pytest.mark.parametrize('mode,step', [
    ('sync', 5), ('separate_async', 5), ('colocate_async', 4), ('colocate_async', 5),
])
def test_replay_without_inventory_keeps_upstream_behavior(mode, step):
    trainer = Harness(None, mode=mode, step=step)
    assert trainer.prepare_step() == {'prefetched': 1}


@pytest.mark.parametrize('ready,should_submit', [(0, True), (1, False)])
def test_custom_sampler_without_inventory_preserves_ready_only_fallback(ready, should_submit):
    trainer = Harness(NS(get_sampleable_count=lambda *args: ready))
    trainer.prepare_step()
    assert trainer.prefetch_calls == int(should_submit)


def test_uniagent_adapter_without_agent_loop_workers_has_no_drain_refs():
    # Actual AgentFrameworkRolloutAdapter exposes framework_worker, not
    # agent_loop_workers. Do not claim the stock worker drain covers this path.
    adapter = NS(framework_worker=object(), gateway_manager=object())
    assert SpecoV1Mixin._speco_v1_drain_agent_loop(adapter) == 0
