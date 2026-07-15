from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra", reason="config overlay tests need hydra-core")
omegaconf = pytest.importorskip("omegaconf", reason="config overlay tests need omegaconf")

compose = hydra.compose
initialize_config_dir = hydra.initialize_config_dir
OmegaConf = omegaconf.OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "verl_speco" / "config"


def test_overlay_has_expected_default_drafter_shape() -> None:
    raw = OmegaConf.load(CONFIG_DIR / "speco_trainer.yaml")
    drafter = raw.actor_rollout_ref.rollout.drafter

    assert raw.speco.verl_base.version == "0.8.0"
    assert raw.speco.verl_base.commit == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    assert drafter.enable is False
    assert drafter.enable_drafter_training is False
    assert drafter.training.collect_hidden_states_from_sgl is False
    assert drafter.training.collect_hidden_states_from_old_logprob is False
    assert drafter.training.lr == pytest.approx(1e-5)
    assert drafter.training.lr_scheduler_type == "global_cosine"
    assert drafter.training.lr_decay_steps == 100
    assert drafter.training.min_lr_ratio == pytest.approx(0.1)
    assert drafter.training.warmup_style is None
    assert drafter.training.resume_trainer_state_from_checkpoint is True


def test_overlay_composes_with_pinned_upstream_verl(tmp_path: Path) -> None:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip("set VERL_SPECO_UPSTREAM_ROOT to check compose against pinned upstream verl")
    upstream_config = Path(upstream_root) / "verl" / "trainer" / "config"
    assert upstream_config.is_dir()

    # Hydra's pkg:// provider imports verl.__init__, which pulls the full
    # training dependency stack. Point the unchanged overlay at the exact
    # checked-out config directory so this contract remains CPU-light.
    composed_config_dir = tmp_path / "config"
    shutil.copytree(upstream_config, composed_config_dir)
    overlay_source = (CONFIG_DIR / "speco_trainer.yaml").read_text(encoding="utf-8")
    overlay_source = overlay_source.replace(
        "pkg://verl.trainer.config",
        composed_config_dir.resolve().as_uri(),
    )
    (composed_config_dir / "speco_trainer.yaml").write_text(overlay_source, encoding="utf-8")

    with initialize_config_dir(config_dir=str(composed_config_dir), version_base=None):
        config = compose(config_name="speco_trainer")

    assert config.speco.verl_base.version == "0.8.0"
    assert config.actor_rollout_ref.rollout.drafter.enable is False
    assert "trainer" in config
    assert "algorithm" in config
