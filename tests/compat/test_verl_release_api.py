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
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest

COMMON_REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    "verl.trainer.ppo.ray_trainer": ("RayPPOTrainer",),
    "verl.trainer.ppo.utils": (
        "Role",
        "need_critic",
        "need_reference_policy",
    ),
    "verl.utils.device": (
        "auto_set_device",
        "get_device_id",
        "get_device_name",
        "get_torch_device",
    ),
    "verl.utils.fsdp_utils": (
        "get_fsdp_full_state_dict",
        "get_fsdp_wrap_policy",
        "apply_fsdp2",
        "fsdp2_load_full_state_dict",
        "load_fsdp_model_to_gpu",
        "load_fsdp_optimizer",
        "offload_fsdp_model_to_cpu",
        "offload_fsdp_optimizer",
    ),
    "verl.utils.dataset.dataset_utils": ("DatasetPadMode",),
    "verl.utils.dataset.rl_dataset": ("collate_fn",),
    "verl.utils.ulysses": (
        "get_ulysses_sequence_parallel_group",
        "set_ulysses_sequence_parallel_group",
        "slice_input_tensor",
        "gather_outputs_and_unpad",
    ),
    "verl.utils.tensordict_utils": (
        "assign_non_tensor",
        "assign_non_tensor_data",
        "get",
        "get_non_tensor_data",
        "get_tensordict",
    ),
    "verl.utils.checkpoint.checkpoint_manager": ("find_latest_ckpt_path",),
    "verl.utils.tracking": ("Tracking",),
    "verl.utils.ray_utils": ("auto_await", "parallel_put"),
    "verl.utils.distributed": (
        "initialize_global_process_group_ray",
        "set_numa_affinity",
    ),
    "verl.workers.engine_workers": ("ActorRolloutRefWorker", "TrainingWorker"),
    "verl.workers.engine.veomni.transformer_impl": (
        "VeOmniEngineWithLMHead",
        "postprocess_batch_func",
    ),
    "verl.workers.engine.veomni.utils": (
        "load_veomni_model_to_gpu",
        "offload_veomni_model_to_cpu",
    ),
    "verl.workers.rollout.replica": ("RolloutReplica", "TokenOutput"),
    "verl.workers.rollout.llm_server": ("LLMServerClient",),
    "verl.workers.rollout.vllm_rollout.vllm_async_server": (
        "vLLMHttpServer",
        "vLLMReplica",
    ),
    "verl.workers.rollout.vllm_rollout.utils": (
        "vLLMColocateWorkerExtension",
        "build_cli_args_from_config",
    ),
    "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": (
        "BucketedWeightSender",
        "BucketedWeightReceiver",
    ),
    "verl.workers.rollout.sglang_rollout.utils": (
        "get_named_tensor_buckets",
        "SGLANG_LORA_NAME",
    ),
    "verl.utils.sglang.sglang_fp8_utils": ("SGLangFP8QuantizerHelper",),
    "verl.workers.utils.padding": ("left_right_2_no_padding", "no_padding_2_padding"),
    "verl.single_controller.base": ("Worker",),
    "verl.single_controller.base.decorator": ("Dispatch", "register"),
    "verl.single_controller.base.worker_group": ("WorkerGroup",),
    "verl.single_controller.ray": ("RayClassWithInitArgs",),
    "verl.experimental.agent_loop.agent_loop": ("AgentLoopManager",),
}

REQUIRED_MODULES_BY_RELEASE: dict[str, dict[str, tuple[str, ...]]] = {
    "0.8.0": {
        **COMMON_REQUIRED_MODULES,
        "verl.trainer.main_ppo": (
            "TaskRunner",
            "create_rl_dataset",
            "create_rl_sampler",
            "run_ppo",
            "migrate_legacy_reward_impl",
        ),
        "verl.utils.config": ("validate_config",),
    },
    "0.9.0": {
        **COMMON_REQUIRED_MODULES,
        "verl.trainer.main_ppo": ("run_ppo",),
        "verl.trainer.main_ppo_v0": ("BaseTaskRunner",),
        "verl.trainer.ppo.utils": (
            "Role",
            "create_rl_dataset",
            "create_rl_sampler",
            "need_critic",
            "need_reference_policy",
        ),
        "verl.utils.config": ("omega_conf_to_dataclass", "validate_config"),
        "verl.workers.config.model": ("HFModelConfig",),
    },
}

REQUIRED_CLASS_METHODS_V090: dict[tuple[str, str], tuple[str, ...]] = {
    ("verl.trainer.main_ppo_v0", "BaseTaskRunner"): (
        "add_actor_rollout_worker",
        "add_critic_worker",
        "add_ref_policy_worker",
        "add_reward_model_resource_pool",
        "add_teacher_model_resource_pool",
        "init_resource_pool_mgr",
    ),
    ("verl.trainer.ppo.ray_trainer", "RayPPOTrainer"): (
        "_compute_old_log_prob",
        "_save_checkpoint",
        "_update_actor",
        "fit",
        "init_workers",
    ),
}


def _module_file(root: Path, module_name: str) -> Path:
    module_path = root.joinpath(*module_name.split("."))
    file_path = module_path.with_suffix(".py")
    if file_path.is_file():
        return file_path
    package_init = module_path / "__init__.py"
    if package_init.is_file():
        return package_init
    raise AssertionError(f"missing upstream module: {module_name}")


def _upstream_repo_root(upstream_root: str) -> Path:
    base = Path(upstream_root)
    for candidate in (base, base / "verl"):
        if (candidate / "verl").is_dir():
            return candidate
    raise AssertionError(
        "VERL_SPECO_UPSTREAM_ROOT must point to the upstream verl checkout "
        "or to a directory containing it"
    )


def _defined_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _class_method_names(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _release_for_upstream_root(root: Path) -> str:
    if (root / "verl" / "trainer" / "main_ppo_v0.py").is_file():
        return "0.9.0"
    return "0.8.0"


def test_supported_release_modules_and_symbols_are_present() -> None:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip("set VERL_SPECO_UPSTREAM_ROOT to check a supported verl API")

    root = _upstream_repo_root(upstream_root)
    release = _release_for_upstream_root(root)
    missing: list[str] = []
    for module_name, symbols in REQUIRED_MODULES_BY_RELEASE[release].items():
        try:
            names = _defined_names(
                _module_file(root, module_name).read_text(encoding="utf-8")
            )
        except (AssertionError, OSError, SyntaxError) as exc:
            missing.append(f"{module_name}: {exc}")
            continue
        for symbol in symbols:
            if symbol not in names:
                missing.append(f"{module_name}.{symbol}")

    class_methods = REQUIRED_CLASS_METHODS_V090 if release == "0.9.0" else {}
    for (module_name, class_name), methods in class_methods.items():
        try:
            source = _module_file(root, module_name).read_text(encoding="utf-8")
            actual_methods = _class_method_names(source, class_name)
        except (AssertionError, OSError, SyntaxError) as exc:
            missing.append(f"{module_name}.{class_name}: {exc}")
            continue
        for method in methods:
            if method not in actual_methods:
                missing.append(f"{module_name}.{class_name}.{method}")

    assert not missing, f"release/v{release} API drift: " + ", ".join(missing)


@pytest.mark.parametrize(
    ("version", "release"),
    (
        ("0.8.0", "0.8.0"),
        ("0.8.0.dev0", "0.8.0"),
        ("0.9.0", "0.9.0"),
        ("0.9.0.dev", "0.9.0"),
        ("0.9.0.post1", "0.9.0"),
    ),
)
def test_supported_release_version_variants_are_accepted(
    version: str, release: str
) -> None:
    from verl_speco.integration.compat import _version_matches_release

    assert _version_matches_release(version, release)


@pytest.mark.parametrize("version", (None, "0.7.0", "0.10.0", "0.9.1.dev0"))
def test_other_verl_versions_are_rejected(version: str | None) -> None:
    from verl_speco.integration.compat import _version_matches_release

    assert not any(
        _version_matches_release(version, release) for release in ("0.8.0", "0.9.0")
    )


def test_imported_verl_version_wins_over_stale_distribution_metadata(
    monkeypatch,
) -> None:
    from verl_speco.integration import compat

    imported_verl = types.ModuleType("verl")
    imported_verl.__version__ = "0.9.0.dev"
    monkeypatch.setitem(sys.modules, "verl", imported_verl)
    monkeypatch.setattr(compat.metadata, "version", lambda _: "0.8.0")

    assert compat._read_imported_verl_version() == "0.9.0.dev"


def test_speco_task_runner_selects_release_specific_legacy_extension_points() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "integration"
        / "task_runner.py"
    ).read_text(encoding="utf-8")

    assert "from verl.trainer.main_ppo_v0 import BaseTaskRunner as _TaskRunnerBase" in source
    assert "from verl.trainer.main_ppo import TaskRunner as _TaskRunnerBase" in source
    assert "class SpecoTaskRunner(_TaskRunnerBase):" in source
    assert '_VERL_TASK_RUNNER_API = "0.8"' in source
    assert '_VERL_TASK_RUNNER_API = "0.9"' in source
    assert "omega_conf_to_dataclass(" in source
    assert "copy_to_local(" in source
    assert "model_config.tokenizer" in source
    assert "model_config.processor" in source
    assert 'config.trainer.get("use_v1", False)' in source

    main_source = (
        Path(__file__).resolve().parents[2] / "verl_speco" / "main.py"
    ).read_text(encoding="utf-8")
    assert "migrate_legacy_reward_impl = getattr(" in main_source
    assert 'main_ppo, "migrate_legacy_reward_impl", None' in main_source
    assert "main_ppo.run_ppo(" in main_source
