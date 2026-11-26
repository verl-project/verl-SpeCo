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

"""Compile Python files with warnings promoted to errors."""

from __future__ import annotations

import py_compile
import subprocess
import tempfile
import warnings
from pathlib import Path


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    failed = False
    with tempfile.TemporaryDirectory(prefix="speco-compile-") as directory, warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for index, filename in enumerate(sorted(set(result.stdout.split("\0")) - {""})):
            if not Path(filename).is_file():
                continue
            try:
                py_compile.compile(filename, cfile=str(Path(directory) / f"{index}.pyc"), doraise=True)
            except (py_compile.PyCompileError, OSError) as exc:
                print(exc)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
