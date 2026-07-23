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
