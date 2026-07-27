# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Check that CUDA/NCCL literals stay inside SpeCo backend and runtime boundaries."""

import os
from argparse import ArgumentParser
from pathlib import Path

# Files that are allowed to mention CUDA because they select backend/runtime
# behavior at the verl/SpeCo integration boundary.
CUDA_KEYWORD_CHECK_WHITELIST = [
    "verl_speco/backends/dflash_trainer_backend.py",
    "verl_speco/backends/domino_trainer_backend.py",
    "verl_speco/backends/dspark_trainer_backend.py",
    "verl_speco/integration/sglang_patch.py",
    "verl_speco/integration/sglang_runtime.py",
    "verl_speco/trainer/base_trainer.py",
    "verl_speco/trainer/draft_training_loop.py",
    "verl_speco/workers/speco_worker.py",
]

# Files that are allowed to select NCCL for distributed training.
NCCL_KEYWORD_CHECK_WHITELIST = [
    "verl_speco/trainer/draft_training_loop.py",
    "verl_speco/workers/speco_worker.py",
]

SEARCH_WHITELIST = CUDA_KEYWORD_CHECK_WHITELIST + NCCL_KEYWORD_CHECK_WHITELIST

SEARCH_KEYWORDS = [".cuda", '"cuda"', '"nccl"']


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        path_in_str = str(path.absolute())

        # judge whether current path is in pre-defined search whitelist or not.
        path_in_whitelist = False

        for sw in SEARCH_WHITELIST:
            # for easy debugging in non-linux system
            sw = sw.replace("/", os.sep)
            if sw in path_in_str:
                print(
                    f"[SKIP] File {path_in_str} is in device API usage check "
                    "whitelist, checking is skipped."
                )
                path_in_whitelist = True
                break

        if path_in_whitelist:
            continue

        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            find_invalid_device_management = False

            for sk in SEARCH_KEYWORDS:
                if sk in file_content:
                    find_invalid_device_management = True
                    break

            print(
                f"[CHECK] File {path_in_str} is detected for device api usage check, check result: "
                f"{'success' if not find_invalid_device_management else f'failed, because detect {sk}'}."
            )

            assert not find_invalid_device_management, (
                f'file {path_in_str} contains .cuda/"cuda"/"nccl" usage outside '
                "the SpeCo backend/runtime boundary. Use device abstractions or "
                "add a documented boundary whitelist entry."
            )
