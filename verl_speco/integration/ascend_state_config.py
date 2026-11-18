# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Configuration for Ascend trainer communication and resident-state parking."""

import os

from verl_speco.backends.training_backend import _get, resolve_training_backend

DETERMINISTIC_TRAINING_ENV = {
    "HCCL_DETERMINISTIC": "strict",
    "LCCL_DETERMINISTIC": "1",
}


def require_deterministic_training_environment():
    for name, expected in DETERMINISTIC_TRAINING_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                f"Ascend TP training requires {name}={expected} before worker startup "
                f"(got {os.environ.get(name)!r})"
            )


def standalone_launch_environment(arguments, environment=None):
    """Override inherited machine defaults only for an explicit Ascend launch."""
    values = {}
    arguments = list(arguments)
    config_name = None
    backend_key = "actor_rollout_ref.rollout.drafter.training.backend"
    for index, argument in enumerate(arguments):
        if argument == "~" + backend_key or argument.startswith(
            "~" + backend_key + "="
        ):
            values[backend_key] = "pytorch"
        elif argument == "--config-name" and index + 1 < len(arguments):
            config_name = arguments[index + 1]
        elif argument.startswith("--config-name="):
            config_name = argument.split("=", 1)[1]
        elif "=" in argument and not argument.startswith("--"):
            key, value = argument.split("=", 1)
            values[key.lstrip("+")] = value.strip().strip("\"'")
    backend = values.get(backend_key)
    if backend is None and str(config_name).removesuffix(".yaml") in {
        "deepseek_v4_0731_draft_trainer",
        "deepseek_v4_mtp_draft_trainer",
    }:
        backend = "vllm_ascend"
    if str(backend).strip().lower() != "vllm_ascend":
        return None
    for name, expected in DETERMINISTIC_TRAINING_ENV.items():
        supplied = values.get(f"draft_engine.environment.{name}", expected)
        if supplied != expected:
            raise ValueError(f"Ascend TP training requires {name}={expected}")
    child = dict(os.environ if environment is None else environment)
    child.update(DETERMINISTIC_TRAINING_ENV)
    return child


def training_worker_environment(config):
    """Set communication policy before Ray starts the Ascend drafter process."""
    if resolve_training_backend(config) != "vllm_ascend":
        return {}
    section = _get(config, "actor_rollout_ref", config)
    training = _get(_get(_get(section, "rollout"), "drafter"), "training", {})
    if _get(training, "mode", "online") != "online":
        return {}
    configured = _get(_get(training, "vllm_ascend", {}), "environment", {})
    for name, expected in DETERMINISTIC_TRAINING_ENV.items():
        if str(_get(configured, name, expected)) != expected:
            raise ValueError(f"Ascend TP training requires {name}={expected}")
    parking, _ = parking_worker_environments(config)
    return {**DETERMINISTIC_TRAINING_ENV, **parking}


def state_parking_mode(config):
    if resolve_training_backend(config) != "vllm_ascend":
        return "none"
    section = _get(config, "actor_rollout_ref", config)
    training = _get(_get(_get(section, "rollout"), "drafter"), "training", {})
    if _get(training, "mode", "online") != "online":
        return "none"
    mode = str(_get(_get(training, "vllm_ascend"), "state_parking", "none")).lower()
    if mode not in {"none", "camem"}:
        raise ValueError("training.vllm_ascend.state_parking must be none or camem")
    return mode


def hccl_environments(trainer_base=61000, serving_base=62000):
    """Reserve separate host/device listeners for the training and serving roles."""
    for base in (trainer_base, serving_base):
        if not 1024 <= base <= 65336:
            raise ValueError("HCCL port base must leave 200 ports within [1024, 65535]")
    if abs(trainer_base - serving_base) < 200:
        raise ValueError("Trainer and serving HCCL port ranges overlap")

    def environment(base):
        return {
            "HCCL_NPU_SOCKET_PORT_RANGE": f"{base}-{base + 99}",
            "HCCL_HOST_SOCKET_PORT_RANGE": f"{base + 100}-{base + 199}",
        }

    return environment(trainer_base), environment(serving_base)


def parking_worker_environments(config):
    if state_parking_mode(config) != "camem":
        return {}, {}
    section = _get(config, "actor_rollout_ref", config)
    settings = _get(
        _get(_get(_get(section, "rollout"), "drafter"), "training"), "vllm_ascend"
    )
    trainer, serving = hccl_environments(
        int(_get(settings, "training_hccl_port_base", 61000)),
        int(_get(settings, "serving_hccl_port_base", 62000)),
    )
    trainer["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    serving["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    return trainer, serving
