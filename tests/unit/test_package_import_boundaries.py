import subprocess
import sys


def _run_import_probe(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_trainer_package_does_not_eagerly_import_ray_trainer() -> None:
    _run_import_probe(
        "import sys; import verl_speco.trainer; "
        "assert 'verl_speco.trainer.speco_ray_trainer' not in sys.modules"
    )


def test_workers_package_does_not_eagerly_import_speco_worker() -> None:
    _run_import_probe(
        "import sys; import verl_speco.workers; "
        "assert 'verl_speco.workers.speco_worker' not in sys.modules"
    )


def test_resolving_feature_store_does_not_load_ray_trainer() -> None:
    _run_import_probe(
        "import importlib.util, sys; "
        "assert importlib.util.find_spec('verl_speco.trainer.feature_store') is not None; "
        "assert 'verl_speco.trainer.speco_ray_trainer' not in sys.modules"
    )
