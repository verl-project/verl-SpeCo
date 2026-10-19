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

ROOT = Path(__file__).resolve().parents[1]
AUTOMATIC_NPU_CASES = {
    f"examples/run_qwen3-8b_drafter_{algorithm}_vllm_npu.sh": algorithm
    for algorithm in ("eagle3", "dflash", "dspark")
}



def configured_environment(environment):
    """Merge trusted GitHub CI variables; explicit process values take priority."""
    env = dict(environment)
    variables = json.loads(env.get("SPECO_CI_VARIABLES_JSON", "{}"))
    for key, value in variables.items():
        if re.fullmatch(r"SPECO_[A-Z0-9_]+", key) and isinstance(value, str) and value:
            if not env.get(key):
                env[key] = value
    return env


def command_for(example, env, args):
    if example not in AUTOMATIC_NPU_CASES:
        return ["bash", example, *args]
    for source, target in {
        "MODEL_PATH": "SPECO_TARGET_MODEL",
        "DRAFTER_PATH": "SPECO_" + AUTOMATIC_NPU_CASES[example].upper() + "_DRAFT_MODEL",
        "TRAIN_FILE": "SPECO_TRAIN_FILE",
        "TEST_FILE": "SPECO_TEST_FILE",
        "CKPTS_DIR": "SPECO_CKPT_DIR",
    }.items():
        if env.get(source):
            env[target] = env[source]
    env.setdefault("SPECO_ACCELERATOR_COUNT", "8")
    env.setdefault("SPECO_TENSOR_PARALLEL_SIZE", "1")
    env.setdefault("SPECO_SEQUENCE_PARALLEL_SIZE", "1")
    env["SPECO_EXTRA_HYDRA_ARGS"] = "\n".join(args)
    return ["bash", "ci/run_example_test.sh", "npu", "vllm", AUTOMATIC_NPU_CASES[example]]


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
    env = configured_environment(os.environ)
    env.update(settings)
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
            value = env.get("SPECO_" + default_key) or env.get("SPECO_DEFAULT_" + default_key)
            if value:
                env.setdefault(key, value)
    source = (ROOT / example).read_text(encoding="utf-8")
    missing = []
    for key in re.findall(r"^[ \t]*\w+=\$\{(\w+):-/path/", source, re.MULTILINE):
        if key.endswith("CKPTS_DIR") or key in {"CKPTS_DIR", "FEATURE_STORE_DIR"}:
            env.setdefault(key, env.get("SPECO_CKPT_DIR", str(output / key.lower())))
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
        if example in AUTOMATIC_NPU_CASES:
            extra = env.get("SPECO_EXTRA_HYDRA_ARGS", "")
            args = defaults + [line.strip() for line in extra.splitlines() if line.strip()] + args
        else:
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
                command_for(example, env, args),
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
            key = "actor_rollout_ref.rollout.drafter.enable_drafter_training="
            check_args = (
                env.get("SPECO_EXTRA_HYDRA_ARGS", "").splitlines()
                if example in AUTOMATIC_NPU_CASES else args
            )
            training = env.get("SPECO_ENABLE_TRAINING", "true").lower() == "true"
            for arg in check_args:
                if arg.startswith(key):
                    training = arg[len(key) :].lower() == "true"
            from verl_speco.integration.ci_result import validate_result

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
