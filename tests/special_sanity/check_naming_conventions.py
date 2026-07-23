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

"""Check project spelling conventions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}
EXCLUDED_FILES = {
    ".pre-commit-config.yaml",
    "ascend_sglang_best_practices.rst",
    "check_naming_conventions.py",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SGLANG_MISSPELLING = re.compile(r"Sglang|sgLang|sglAng|sglaNg|sglanG")


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def main() -> int:
    failures: list[str] = []
    for path in _iter_text_files(Path(".")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "veRL" in text:
            failures.append(f"{path}: please use verl instead of veRL")
        match = SGLANG_MISSPELLING.search(text)
        if match is not None:
            failures.append(
                f"{path}: please use SGLang or sglang instead of {match.group(0)}"
            )

    if failures:
        print("[FAIL] Naming convention violations:", file=sys.stderr)
        for failure in failures:
            print("  - " + failure, file=sys.stderr)
        return 1
    print("[OK] Naming conventions look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
