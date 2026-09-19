"""Reject unsupported engines before model initialization or optional imports."""

from types import SimpleNamespace

import pytest

from verl_speco.trainer.base_trainer import DrafterBaseTrainer


@pytest.mark.parametrize(
    ("engine", "algorithm", "mesh", "sp", "message"),
    [
        ("unknown", "peagle", None, False, "Unknown drafter training engine"),
        ("veomni", "eagle3", None, False, "dense P-EAGLE"),
        ("veomni", "peagle", None, False, "requires FSDP2"),
        ("veomni", "peagle", object(), True, "without sequence parallelism"),
        ("veomni", "peagle", object(), False, "dedicated standalone processes"),
    ],
)
def test_invalid_engine_configuration(engine, algorithm, mesh, sp, message):
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.rollout_dp_rank = 0
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={"engine": engine}))
    )
    trainer.backend = SimpleNamespace(model_type=algorithm)
    trainer.fsdp_device_mesh = mesh
    trainer.use_native_dp_sp = False
    trainer.use_ulysses_sp = sp
    with pytest.raises(ValueError, match=message):
        trainer._build_draft_model()
