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
"""Optional original-source oracle for development comparisons only."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

_REQUIRED = (
    "offline_training/train_mtp.py",
    "training_common/modeling/ds4_dspark.py",
    "training_common/modeling/ds4_runtime.py",
    "training_common/modeling/lm_head.py",
    "training_common/utils/mtp_trainer.py",
    "training_common/utils/data.py",
    "training_common/utils/parallel.py",
    "loading_trained_mtp/loading_ds4.py",
)


def source_fingerprint(root: str | Path) -> str:
    """Identify the external Python implementation, without reading weights."""
    root = Path(root).expanduser().resolve(strict=True)
    for name in _REQUIRED:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Incomplete AIRDrafter checkout: {root / name}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] not in {
            "training_common",
            "offline_training",
            "co_training",
            "loading_trained_mtp",
            "tuner",
        }:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def activate_source(root: str | Path) -> Path:
    """Expose AIR's mixed absolute imports in this opt-in worker process only.

    AIR currently imports both ``AIRDrafter.training_common`` and
    ``training_common``. Never replace an already imported, unrelated package.
    No files, symlinks, installed packages or environment variables are changed.
    """
    root = Path(root).expanduser().resolve(strict=True)
    source_fingerprint(root)
    for name in (
        "AIRDrafter",
        "training_common",
        "offline_training",
        "co_training",
        "tuner",
        "loading_trained_mtp",
    ):
        module = sys.modules.get(name)
        if module is None:
            continue
        paths = list(getattr(module, "__path__", []))
        expected = root if name == "AIRDrafter" else root / name
        if not paths or any(Path(path).resolve() != expected for path in paths):
            raise RuntimeError(
                f"Package {name!r} is already loaded from another checkout"
            )
    if "AIRDrafter" not in sys.modules:
        package = types.ModuleType("AIRDrafter")
        package.__path__ = [str(root)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            "AIRDrafter", loader=None, is_package=True
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["AIRDrafter"] = package
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def engine_module(name: str):
    """Import from the activated checkout; runtime dependencies stay lazy."""
    if "AIRDrafter" not in sys.modules:
        raise RuntimeError("Call activate_source() before importing AIRDrafter")
    return importlib.import_module(f"AIRDrafter.{name}")
