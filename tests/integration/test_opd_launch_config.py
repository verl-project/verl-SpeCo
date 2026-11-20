import os
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl_speco.specopd_config import specopd_loss_weights, validate_specopd_config


def test_dspark_specopd_launch_defaults_to_pure_tv():
    root = Path(__file__).resolve().parents[2]
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("bash is unavailable")
    env = {**os.environ, **dict.fromkeys(("MODEL_PATH", "DRAFTER_PATH", "TRAIN_FILE", "TEST_FILE"), "/unused/test-path")}
    harness = 'python3() { printf "%s\\0" "$@"; }; source "$1"'
    script = "examples/run_qwen3-8b-specopd_drafter_dspark_vllm_npu.sh"
    result = subprocess.run([bash, "-c", harness, "specopd-config-test", script], cwd=root,
                            env=env, capture_output=True, check=True)
    argv = result.stdout.decode().rstrip("\0").split("\0")
    assert argv[:3] == ["-m", "verl_speco.main", "--config-name=speco_v1_trainer"]
    config = OmegaConf.merge(OmegaConf.load(root / "verl_speco/config/speco_base.yaml"),
                             OmegaConf.from_dotlist([arg.lstrip("+") for arg in argv[3:]]))
    training = config.actor_rollout_ref.rollout.drafter.training
    spec = validate_specopd_config(config)
    assert spec.enable and spec.objective == "tv"
    assert not training.drafter_opd.enable
    assert specopd_loss_weights(training, "DSPARK") == (1.0, 0.0)
