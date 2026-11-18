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
"""Fingerprint the maintained in-package implementation, never another tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# This exact original implementation was used by the previous SpeCo adapter.
# Weight names and numerical options are preserved. Recognize it only to keep
# already-created SpeCo recovery/export artifacts usable; arbitrary sources
# never become trusted just because a source_root argument was supplied.
LEGACY_IMPLEMENTATION_FINGERPRINT = (
    "8c499de6be1a1b4915706bb94201535969ed49d6f786ddb1bab547a8651aab3c"
)


def implementation_paths():
    root = Path(__file__).parents[1]
    provenance = json.loads(
        (root / "config/deepseek_v4_backend_sources.json").read_text()
    )
    names = {
        name.removeprefix("verl_speco.").replace(".", "/") + ".py"
        for name in provenance["files"]
    }
    names.update(
        {
            "backends/vllm_ascend_api.py",
            "backends/vllm_ascend_compat.py",
            "ops/ascend_moe_training.py",
            "ops/ascend_moe_combine.py",
            "ops/ascend_dsa_compat.py",
            "ops/ascend_weight_cache.py",
            "integration/vllm_ascend_implementation.py",
            "config/deepseek_v4_dspark_backend.yaml",
            "config/deepseek_v4_backend_sources.json",
        }
    )
    return [root / name for name in sorted(names)]


def implementation_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in implementation_paths():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compatible_implementation_fingerprint(value):
    return value in {implementation_fingerprint(), LEGACY_IMPLEMENTATION_FINGERPRINT}


def implementation_identity_matches(stored, current):
    # Artifact/schema compatibility is not numerical recovery compatibility.
    # In particular, the legacy implementation used a different BF16 MoE scale
    # order; do not alias its optimizer state to this implementation's identity.
    return isinstance(stored, dict) and isinstance(current, dict) and stored == current
