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
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script,filename,invalid",
    [
        ("compile_all.py", "new.py", "def broken(:\n"),
        ("check_naming_conventions.py", "notes.txt", "Sg" + "lang"),
        ("check_docs_time_info.py", "docs/new.md", "Missing timestamp"),
    ],
)
def test_repository_checks_respect_git_visibility(tmp_path, script, filename, invalid):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("Last updated: 11/25/2026\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".cache/\ndocs/local.md\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/bad.py").write_text("Sg" + "lang; def broken(:\n", encoding="utf-8")
    (tmp_path / "docs/local.md").write_text("Missing timestamp", encoding="utf-8")
    entry = Path(__file__).with_name(script)

    def run_check():
        return subprocess.run([sys.executable, str(entry)], cwd=tmp_path, capture_output=True)

    assert run_check().returncode == 0
    bad = tmp_path / filename
    bad.write_text(invalid, encoding="utf-8")
    assert run_check().returncode != 0
    subprocess.run(["git", "-C", str(tmp_path), "add", filename], check=True)
    bad.unlink()
    assert run_check().returncode == 0
    assert not (tmp_path / "__pycache__").exists()
