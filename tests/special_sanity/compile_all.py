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

import compileall
import os
import re

EXCLUDE = re.compile(r"(^|[\\/])(\.venv|venv|\.git|__pycache__)([\\/]|$)")


def main() -> int:
    os.environ["PYTHONWARNINGS"] = "error"
    return 0 if compileall.compile_dir(".", quiet=1, rx=EXCLUDE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
