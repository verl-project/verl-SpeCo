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
"""Fail closed on absent, invalid, or unsuccessful CI evidence."""

import argparse
import json
from pathlib import Path

from verl_speco.integration.ci_result import validate_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--training", choices=("true", "false"), required=True)
    args = parser.parse_args()
    validate_result(
        json.loads(Path(args.path).read_text(encoding="utf-8")),
        training=args.training == "true",
    )


if __name__ == "__main__":
    main()
