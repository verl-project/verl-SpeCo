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

"""Enforce SpeCo example run-script naming.

SpeCo examples intentionally expose both the drafter backend and rollout
backend in the filename because those combinations are the product surface:

    run_<model>_drafter_<drafter-backend>_<rollout-backend>[_npu].sh

The standalone/offline draft-training entry point uses:

    run_<model>_drafter_separate_training.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DRAFTER_BACKENDS = ("dflash", "dspark", "eagle3")
ROLLOUT_BACKENDS = ("vllm", "sglang")
OPTIONAL_SUFFIXES = ("npu",)
STANDALONE_SUFFIX = ("separate", "training")

DEFAULT_IGNORE_DIRS: tuple[str, ...] = ()
DEFAULT_IGNORE_FILES: tuple[str, ...] = ()


def _split_tokens(stem: str) -> list[str]:
    parts = stem.split("_")
    if parts and parts[0] == "run":
        parts = parts[1:]
    return parts


def _is_ignored(
    path: Path,
    repo_root: Path,
    ignore_dirs: tuple[str, ...],
    ignore_files: tuple[str, ...],
) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    if rel in ignore_files:
        return True
    return any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in ignore_dirs)


def _format_expected() -> str:
    return (
        "expected run_<model>_drafter_<drafter-backend>_<rollout-backend>[_npu].sh "
        f"with drafter-backend in {list(DRAFTER_BACKENDS)} and rollout-backend "
        f"in {list(ROLLOUT_BACKENDS)}, or run_<model>_drafter_separate_training.sh"
    )


def check_filename(path: Path, display: str | None = None) -> list[str]:
    errors: list[str] = []
    name = path.name
    shown = display if display is not None else str(path)

    if not name.startswith("run_"):
        return [f"{shown}: example script must start with 'run_'"]
    if not name.endswith(".sh"):
        return [f"{shown}: example script must end with '.sh'"]

    tokens = _split_tokens(path.stem)
    if "drafter" not in tokens:
        return [f"{shown}: missing '_drafter_' marker; {_format_expected()}"]

    drafter_index = tokens.index("drafter")
    model_tokens = tokens[:drafter_index]
    spec_tokens = tokens[drafter_index + 1 :]
    if not model_tokens:
        errors.append(f"{shown}: model name is missing before '_drafter_'")

    if tuple(spec_tokens) == STANDALONE_SUFFIX:
        return errors

    if len(spec_tokens) not in (2, 3):
        errors.append(f"{shown}: invalid backend suffix; {_format_expected()}")
        return errors

    drafter_backend, rollout_backend = spec_tokens[0], spec_tokens[1]
    suffix = spec_tokens[2] if len(spec_tokens) == 3 else None
    if drafter_backend not in DRAFTER_BACKENDS:
        errors.append(
            f"{shown}: unknown drafter backend '{drafter_backend}', expected one of {list(DRAFTER_BACKENDS)}"
        )
    if rollout_backend not in ROLLOUT_BACKENDS:
        errors.append(
            f"{shown}: unknown rollout backend '{rollout_backend}', expected one of {list(ROLLOUT_BACKENDS)}"
        )
    if suffix is not None and suffix not in OPTIONAL_SUFFIXES:
        errors.append(
            f"{shown}: unknown optional suffix '{suffix}', expected one of {list(OPTIONAL_SUFFIXES)}"
        )
    return errors


def collect_scripts(
    root: Path,
    repo_root: Path,
    ignore_dirs: tuple[str, ...],
    ignore_files: tuple[str, ...],
) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.sh")
        if p.is_file() and not _is_ignored(p, repo_root, ignore_dirs, ignore_files)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("examples"),
        help="Directory to scan (default: examples)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used for relative-path matching against ignores",
    )
    parser.add_argument("--ignore-dirs", nargs="*", default=list(DEFAULT_IGNORE_DIRS))
    parser.add_argument("--ignore-files", nargs="*", default=list(DEFAULT_IGNORE_FILES))
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    root = args.root.resolve()
    if not root.is_dir():
        print(
            f"[FAIL] --root '{args.root}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        return 2

    scripts = collect_scripts(
        root,
        repo_root,
        tuple(args.ignore_dirs),
        tuple(args.ignore_files),
    )

    all_errors: list[str] = []
    for script in scripts:
        try:
            display = script.relative_to(repo_root).as_posix()
        except ValueError:
            display = str(script)
        all_errors.extend(check_filename(script, display=display))

    if all_errors:
        print("[FAIL] Example script naming violations:\n", file=sys.stderr)
        for err in all_errors:
            print("  - " + err, file=sys.stderr)
        print("\nNaming convention:\n  " + _format_expected() + "\n", file=sys.stderr)
        return 1

    print(
        f"[OK] {len(scripts)} example scripts under '{args.root}' follow the SpeCo naming convention."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
