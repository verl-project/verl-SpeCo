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
"""Run an allowlisted example with explicit manual configuration and logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

from verl_speco.integration.ci_result import validate_result


ROOT = Path(__file__).resolve().parents[1]


def prepare(example, settings, hydra_args, output):
    manifest = json.loads(
        (ROOT / "ci/manual_examples.json").read_text(encoding="utf-8")
    )
    if example not in manifest:
        raise ValueError("Select an example listed in manual_examples.json")
    if not isinstance(settings, dict) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(value, str)
        for key, value in settings.items()
    ):
        raise ValueError(
            "environment_json must contain environment variable names and string values"
        )
    env = {**os.environ, **settings}
    env.setdefault("TRAIN_LOGGERS", '["console"]')
    online = "separate_training" not in example
    if online:
        defaults = {"TRAIN_FILE": "TRAIN_FILE", "TEST_FILE": "TEST_FILE"}
        # A 4B target and DFlash2 drafter require their own compatible assets.
        if "qwen3-8b" in example:
            defaults["MODEL_PATH"] = "TARGET_MODEL"
        if "qwen3-8b" in example and "dflash2" not in example:
            algorithm = (
                "EAGLE3"
                if "eagle3" in example
                else "DSPARK"
                if "dspark" in example
                else "DFLASH"
            )
            defaults["DRAFTER_PATH"] = algorithm + "_DRAFT_MODEL"
        for key, default_key in defaults.items():
            value = env.get("SPECO_DEFAULT_" + default_key)
            if value:
                env.setdefault(key, value)
    source = (ROOT / example).read_text(encoding="utf-8")
    missing = []
    for key in re.findall(r"^\w+=\$\{(\w+):-/path/", source, re.MULTILINE):
        if key.endswith("CKPTS_DIR") or key in {"CKPTS_DIR", "FEATURE_STORE_DIR"}:
            env.setdefault(key, str(output / key.lower()))
        if not env.get(key) or env[key].startswith("/path/"):
            missing.append(key)
    if missing:
        raise ValueError(
            f"Set these paths in environment_json: {', '.join(sorted(set(missing)))}"
        )
    args = [arg.strip() for arg in hydra_args.splitlines() if arg.strip()]
    if online:
        profile = json.loads(
            (ROOT / "ci/online_smoke_profile.json").read_text(encoding="utf-8")
        )
        variables = json.loads(env.get("SPECO_CI_VARIABLES_JSON", "{}"))
        defaults = []
        for key, spec in profile.items():
            value = env.get(key) or variables.get(key) or spec["default"]
            env[key] = value
            defaults.append(spec["key"] + "=" + value)
        # Explicit Hydra arguments are last and retain priority.
        args = defaults + args
        if manifest[example] == "npu" and "eagle3" in example:
            env.setdefault("SPECO_EAGLE3_DISABLE_TORCH_COMPILE", "true")
    online = "separate_training" not in example
    if online:
        env["SPECO_CI_RESULT_PATH"] = str(output / "result.json")
    return env, args, online


def main():
    output = Path(os.environ["RUNNER_TEMP"]) / "manual-e2e"
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {"exit_code": 1, "completed": False}
    try:
        example = os.environ["EXAMPLE"]
        env, args, online = prepare(
            example,
            json.loads(os.environ.get("EXAMPLE_ENVIRONMENT_JSON", "{}")),
            os.environ.get("EXAMPLE_HYDRA_ARGS", ""),
            output,
        )
        result_path = output / "result.json"
        result_path.unlink(missing_ok=True)
        with (output / "example.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", example, *args],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        status["exit_code"] = code
        if code:
            raise RuntimeError(f"Example exited with code {code}")
        if online:
            training = True
            key = "actor_rollout_ref.rollout.drafter.enable_drafter_training="
            for arg in args:
                if arg.startswith(key):
                    training = arg[len(key) :].lower() == "true"
            validate_result(
                json.loads(result_path.read_text(encoding="utf-8")), training=training
            )
        status.update(exit_code=0, completed=True)
    except Exception as error:
        status.update(
            exit_code=status["exit_code"] or 1,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
