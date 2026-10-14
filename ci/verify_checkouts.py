# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ensure tests use the checked-out source and requested upstream release."""

import argparse
from pathlib import Path

from packaging.version import Version


def verify(workspace, verl_path, speco_path, version, expected_ref):
    workspace = Path(workspace).resolve()
    if workspace / "upstream-verl" not in Path(verl_path).resolve().parents:
        raise RuntimeError(f"verl imported outside upstream checkout: {verl_path}")
    if workspace / "verl_speco" not in Path(speco_path).resolve().parents:
        raise RuntimeError(
            f"verl_speco imported outside current checkout: {speco_path}"
        )
    expected = expected_ref.removeprefix("release/v")
    if Version(version).base_version != expected:
        raise RuntimeError(f"Expected verl {expected}, imported {version}")


def main():
    import verl
    import verl_speco

    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-ref", default="release/v0.8.0")
    args = parser.parse_args()
    verify(
        Path(__file__).resolve().parents[1],
        verl.__file__,
        verl_speco.__file__,
        verl.__version__,
        args.expected_ref,
    )
    print(f"Verified verl {verl.__version__}: {verl.__file__}")
    print(f"Verified verl_speco: {verl_speco.__file__}")


if __name__ == "__main__":
    main()
