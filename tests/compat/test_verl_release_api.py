from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    "verl.trainer.main_ppo": (
        "TaskRunner",
        "create_rl_dataset",
        "create_rl_sampler",
        "run_ppo",
        "migrate_legacy_reward_impl",
    ),
    "verl.trainer.ppo.ray_trainer": ("RayPPOTrainer",),
    "verl.trainer.ppo.utils": ("Role", "need_critic", "need_reference_policy"),
    "verl.utils.config": ("validate_config",),
    "verl.utils.device": ("auto_set_device", "get_device_id", "get_device_name", "get_torch_device"),
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
    "verl.utils.distributed": ("initialize_global_process_group_ray", "set_numa_affinity"),
    "verl.workers.engine_workers": ("ActorRolloutRefWorker", "TrainingWorker"),
    "verl.workers.rollout.replica": ("RolloutReplica", "TokenOutput"),
    "verl.workers.rollout.llm_server": ("LLMServerClient",),
    "verl.workers.rollout.vllm_rollout.vllm_async_server": ("vLLMHttpServer", "vLLMReplica"),
    "verl.workers.rollout.vllm_rollout.utils": (
        "vLLMColocateWorkerExtension",
        "build_cli_args_from_config",
    ),
    "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": (
        "BucketedWeightSender",
        "BucketedWeightReceiver",
    ),
    "verl.workers.rollout.sglang_rollout.utils": ("get_named_tensor_buckets", "SGLANG_LORA_NAME"),
    "verl.utils.sglang.sglang_fp8_utils": ("SGLangFP8QuantizerHelper",),
    "verl.workers.utils.padding": ("left_right_2_no_padding", "no_padding_2_padding"),
    "verl.single_controller.base": ("Worker",),
    "verl.single_controller.base.decorator": ("Dispatch", "register"),
    "verl.single_controller.base.worker_group": ("WorkerGroup",),
    "verl.single_controller.ray": ("RayClassWithInitArgs",),
    "verl.experimental.agent_loop.agent_loop": ("AgentLoopManager",),
}


def _module_file(root: Path, module_name: str) -> Path:
    module_path = root.joinpath(*module_name.split("."))
    file_path = module_path.with_suffix(".py")
    if file_path.is_file():
        return file_path
    package_init = module_path / "__init__.py"
    if package_init.is_file():
        return package_init
    raise AssertionError(f"missing release/v0.8.0 module: {module_name}")


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


def test_release_v080_modules_and_symbols_are_present() -> None:
    upstream_root = os.getenv("VERL_SPECO_UPSTREAM_ROOT")
    if not upstream_root:
        pytest.skip("set VERL_SPECO_UPSTREAM_ROOT to check the release/v0.8.0 API")

    # REQUIRED_MODULES keys already start with the ``verl`` package segment
    # (e.g. ``verl.trainer.main_ppo``), so resolve them from the checkout root,
    # not from ``<root>/verl`` which would double the segment into
    # ``<root>/verl/verl/...`` and report every module as missing.
    root = Path(upstream_root)
    missing: list[str] = []
    for module_name, symbols in REQUIRED_MODULES.items():
        try:
            names = _defined_names(_module_file(root, module_name).read_text(encoding="utf-8"))
        except (AssertionError, OSError, SyntaxError) as exc:
            missing.append(f"{module_name}: {exc}")
            continue
        for symbol in symbols:
            if symbol not in names:
                missing.append(f"{module_name}.{symbol}")

    assert not missing, "release/v0.8.0 API drift: " + ", ".join(missing)
