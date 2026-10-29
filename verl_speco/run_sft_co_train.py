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

"""SFT Co-Train 独立运行入口。

使用方式：
    cd /model/sft_cotrain/verl-SpeCo
    python run_sft_co_train.py

实验说明：
    - 单卡对比实验：纯 SFT vs SFT + co-train
    - 数据集：GSM8K + DAPO 合并数据集
    - 训练集：7473 条
    - 验证集：1319 条
"""

import os
import sys

import hydra
from omegaconf import OmegaConf, open_dict

# ========== 路径配置 ==========
# 本脚本不维护独立的训练参数常量，所有参数由 shell 脚本通过 CLI 传入，
# 此处仅设置路径和兜底默认值。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.environ.get("VERL_ROOT", "/model/sft_cotrain/verl")
SPECO_ROOT = os.path.dirname(SCRIPT_DIR)

if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)
if SPECO_ROOT not in sys.path:
    sys.path.insert(0, SPECO_ROOT)

_sft_debug = os.environ.get("SFT_DEBUG", "").lower() in ("1", "true", "yes")
if _sft_debug:
    print(f"[DEBUG] VERL_ROOT = {VERL_ROOT}")
    print(f"[DEBUG] SPECO_ROOT = {SPECO_ROOT}")
    print(f"[DEBUG] sys.path[:3] = {sys.path[:3]}")

# verl config 目录
VERL_CONFIG_DIR = os.path.join(VERL_ROOT, "verl", "trainer", "config")
# speco config 目录
SPECO_CONFIG_DIR = os.path.join(SPECO_ROOT, "verl_speco", "config")
if _sft_debug:
    print(f"[DEBUG] VERL_CONFIG_DIR = {VERL_CONFIG_DIR}")
    print(f"[DEBUG] exists: {os.path.exists(VERL_CONFIG_DIR)}")


@hydra.main(
    config_path=SPECO_CONFIG_DIR, config_name="speco_sft_trainer", version_base=None
)
def main(config):
    """SFT Co-Train 主入口。"""
    import logging

    import ray

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    from verl.trainer.sft_trainer_ray import create_sft_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.device import auto_set_device

    from verl_speco.integration.compat import check_compatible_verl
    from verl_speco.trainer.speco_sft_trainer import SpecoRaySFTRayTrainer

    check_compatible_verl()
    auto_set_device(config)

    if not ray.is_initialized():
        ray.init()

    # ========== 关键：关闭 OmegaConf struct 模式 ==========
    OmegaConf.set_struct(config, False)

    # ========== 路径映射：speco_base.yaml 的 drafter 配置在 actor_rollout_ref.rollout.drafter，
    # SFT 代码读取 config.rollout.drafter，在此桥接 ==========
    if hasattr(config, "actor_rollout_ref") and hasattr(
        config.actor_rollout_ref, "rollout"
    ):
        with open_dict(config):
            if not hasattr(config, "rollout"):
                config.rollout = OmegaConf.create({})
            config.rollout.drafter = config.actor_rollout_ref.rollout.drafter
            print(
                "[SFT-CONFIG] Bridged actor_rollout_ref.rollout.drafter -> rollout.drafter"
            )
            print(
                f"[SFT-CONFIG] speculative_algorithm = {config.rollout.drafter.speculative_algorithm}"
            )

    # ========== 基础配置 ==========
    config.model.path = os.path.expanduser(config.model.get("path") or "")
    config.data.train_files = os.path.expanduser(config.data.get("train_files") or "")
    config.data.val_files = os.path.expanduser(config.data.get("val_files") or "")
    config.data.train_max_samples = config.data.get("train_max_samples") or -1
    _ignore_mismatch = config.data.get("ignore_input_ids_mismatch", True)
    config.data.ignore_input_ids_mismatch = (
        True if _ignore_mismatch is None else bool(_ignore_mismatch)
    )
    config.data.num_workers = config.data.get("num_workers") or 0

    # ========== 训练配置 ==========
    config.data.train_batch_size = config.data.get("train_batch_size") or 16
    config.data.micro_batch_size_per_gpu = (
        config.data.get("micro_batch_size_per_gpu") or 1
    )
    config.data.max_token_len_per_gpu = config.data.get("max_token_len_per_gpu") or 4096
    config.trainer.total_training_steps = (
        config.trainer.get("total_training_steps") or 0
    )
    config.trainer.total_epochs = config.trainer.get("total_epochs") or 1
    config.trainer.n_gpus_per_node = config.trainer.get("n_gpus_per_node") or 1
    config.trainer.nnodes = config.trainer.get("nnodes") or 1
    config.trainer.balance_batch = config.trainer.get("balance_batch", False)
    config.trainer.resume_mode = config.trainer.get("resume_mode") or "disable"

    # ========== 保存/测试配置 ==========
    config.trainer.save_freq = config.trainer.get("save_freq") or 0
    config.trainer.test_freq = config.trainer.get("test_freq") or 0
    config.trainer.logger = config.trainer.get("logger") or ["console", "wandb"]
    config.trainer.project_name = config.trainer.get("project_name") or "sft_cotrain"
    config.trainer.experiment_name = (
        config.trainer.get("experiment_name") or "sft_cotrain"
    )
    config.trainer.default_local_dir = (
        config.trainer.get("default_local_dir") or "checkpoints/sft_cotrain"
    )

    # ========== 数据格式配置 ==========
    config.model.use_remove_padding = config.model.get("use_remove_padding", True)
    config.data.pad_mode = config.data.get("pad_mode", "no_padding")

    # ========== 显存优化配置（腾出 GPU 空间给 Drafter）==========
    config.engine.param_offload = config.engine.get("param_offload", True)
    config.engine.optimizer_offload = config.engine.get("optimizer_offload", True)
    config.engine.offload_policy = config.engine.get("offload_policy", True)
    config.model.enable_activation_offload = config.model.get(
        "enable_activation_offload", True
    )

    # ========== 实验类型 & 算法 ==========
    experiment_type = config.get("experiment_type", "cotrain")
    speculative_algorithm = str(
        config.rollout.drafter.get("speculative_algorithm", "DSPARK")
    ).upper()
    enable_co_train = experiment_type == "cotrain"

    # ========== speco.sft_specific 配置（drafter 开关的唯一数据源）==========
    if not hasattr(config, "speco"):
        with open_dict(config):
            config.speco = OmegaConf.create({})
    with open_dict(config.speco):
        config.speco.mode = "sft"
        if (
            not hasattr(config.speco, "sft_specific")
            or config.speco.sft_specific is None
        ):
            config.speco.sft_specific = OmegaConf.create({})
        sft_specific = config.speco.sft_specific
        # 唯一的 drafter 训练开关，其他位置自动同步于此
        if getattr(sft_specific, "enable_drafter_training", None) is None:
            sft_specific.enable_drafter_training = enable_co_train
        if getattr(sft_specific, "drafter_train_interval", None) is None:
            sft_specific.drafter_train_interval = 5
        if getattr(sft_specific, "max_samples_per_step", None) is None:
            sft_specific.max_samples_per_step = 200
        if getattr(sft_specific, "hidden_layer_id", None) is None:
            sft_specific.hidden_layer_id = -1
        if getattr(sft_specific, "publish_during_sft", None) is None:
            sft_specific.publish_during_sft = False
        if getattr(sft_specific, "checkpoint_save_dir", None) in (None, ""):
            sft_specific.checkpoint_save_dir = "checkpoints/sft_cotrain/drafter"

    print("=" * 60)
    print(f"SPECO SFT - Experiment: {experiment_type}")
    print(f"Co-train enabled: {enable_co_train}")
    print("=" * 60)

    # ========== rollout.drafter 配置（从 speco.sft_specific 同步）==========
    if not hasattr(config, "rollout"):
        with open_dict(config):
            config.rollout = OmegaConf.create({})

    with open_dict(config.rollout):
        if not hasattr(config.rollout, "tensor_model_parallel_size"):
            config.rollout.tensor_model_parallel_size = 1
        if not hasattr(config.rollout, "data_parallel_size"):
            config.rollout.data_parallel_size = 1
        if not hasattr(config.rollout, "pipeline_model_parallel_size"):
            config.rollout.pipeline_model_parallel_size = 1

    if not hasattr(config.rollout, "drafter"):
        with open_dict(config.rollout):
            config.rollout.drafter = OmegaConf.create({})

    # 从 speco.sft_specific 读取唯一数据源
    _enable_drafter = bool(config.speco.sft_specific.enable_drafter_training)
    _drafter_ckpt_dir = str(config.speco.sft_specific.checkpoint_save_dir)
    _drafter_train_interval = int(config.speco.sft_specific.drafter_train_interval)

    with open_dict(config.rollout.drafter):
        config.rollout.drafter.enable = _enable_drafter
        config.rollout.drafter.enable_drafter_training = _enable_drafter
        config.rollout.drafter.speculative_algorithm = speculative_algorithm
        config.rollout.drafter.checkpoint_path = _drafter_ckpt_dir
        # Treat empty string and /path/to/* placeholder as unconfigured
        _raw_model_path = str(config.rollout.drafter.get("model_path") or "")
        if not _raw_model_path or _raw_model_path.startswith("/path/to/"):
            config.rollout.drafter.model_path = ""

    # Validate drafter model path when drafter training is enabled
    if _enable_drafter and not config.rollout.drafter.get("model_path"):
        raise ValueError(
            "Drafter training is enabled (experiment_type=cotrain) but "
            "rollout.drafter.model_path is empty or a /path/to/ placeholder. "
            "Please set DRAFTER_MODEL_PATH in the shell script."
        )

    # ========== drafter.training 配置 ==========
    with open_dict(config.rollout.drafter.training):
        config.rollout.drafter.training.enable = _enable_drafter
        config.rollout.drafter.training.enable_drafter_training = _enable_drafter
        config.rollout.drafter.training.collect_hidden_states_from_old_logprob = True
        config.rollout.drafter.training.collect_hidden_states_from_sgl = False

        if speculative_algorithm == "DSPARK":
            if (
                getattr(config.rollout.drafter.training, "dspark_l1_loss_alpha", None)
                is None
            ):
                config.rollout.drafter.training.dspark_l1_loss_alpha = 0.9
            if (
                getattr(
                    config.rollout.drafter.training, "dspark_num_target_layers", None
                )
                is None
            ):
                config.rollout.drafter.training.dspark_num_target_layers = 5
            if not hasattr(config.rollout.drafter.training, "dspark_target_layer_ids"):
                config.rollout.drafter.training.dspark_target_layer_ids = None

        config.rollout.drafter.training.fsdp_config = OmegaConf.create(
            {
                "param_offload": True,
                "optimizer_offload": True,
                "use_orig_params": True,
                "forward_prefetch": False,
                "wrap_policy": OmegaConf.create({"min_num_params": 0}),
            }
        )

        config.rollout.drafter.training.training_batch_size = 8
        config.rollout.drafter.training.micro_batch_size = 8
        config.rollout.drafter.training.batch_size_per_gpu = 8
        config.rollout.drafter.training.max_grad_norm = 1.0
        config.rollout.drafter.training.weight_decay = 0.0
        config.rollout.drafter.training.use_kl_loss = False
        config.rollout.drafter.training.use_adv_loss = False
        config.rollout.drafter.training.hidden_state_clip_value = 1e8
        config.rollout.drafter.training.is_offload_param = True
        config.rollout.drafter.training.is_offload_optimizer = True

    # ========== sft_drafter 配置（从 sft_specific 同步）==========
    if not hasattr(config, "sft_drafter"):
        with open_dict(config):
            config.sft_drafter = OmegaConf.create({})
    with open_dict(config.sft_drafter):
        config.sft_drafter.enable_training = _enable_drafter
        config.sft_drafter.train_interval = _drafter_train_interval
        config.sft_drafter.max_samples_per_step = int(
            config.speco.sft_specific.max_samples_per_step
        )

    # ========== 打印配置摘要 ==========
    print("=" * 60)
    print("SPECO SFT Co-Train Configuration")
    print("=" * 60)
    print(f"Experiment: {experiment_type}")
    print(f"Train data: {config.data.train_files}")
    print(f"Val data: {config.data.val_files}")
    print(f"Model: {config.model.path}")
    print(f"GPUs: {config.trainer.n_gpus_per_node}")
    print(f"Total steps: {config.trainer.total_training_steps}")
    print(f"Epochs: {config.trainer.total_epochs}")
    print(f"Co-train enabled: {enable_co_train}")
    print(f"Save freq: {config.trainer.save_freq}")
    print(f"Test freq: {config.trainer.test_freq}")
    # drafter 关键参数验证
    _drafter_cfg = getattr(config, "rollout", None)
    _drafter_cfg = (
        getattr(_drafter_cfg, "drafter", None) if _drafter_cfg is not None else None
    )
    _drafter_ckpt = (
        getattr(_drafter_cfg, "checkpoint_path", None)
        if _drafter_cfg is not None
        else None
    )
    _drafter_enable = (
        getattr(_drafter_cfg, "enable_drafter_training", None)
        if _drafter_cfg is not None
        else None
    )
    _sft_specific = (
        getattr(getattr(config, "speco", None), "sft_specific", None)
        if getattr(config, "speco", None) is not None
        else None
    )
    _sft_enable_drafter = (
        getattr(_sft_specific, "enable_drafter_training", None)
        if _sft_specific is not None
        else None
    )
    _sft_ckpt_save_dir = (
        getattr(_sft_specific, "checkpoint_save_dir", None)
        if _sft_specific is not None
        else None
    )
    print(f"[Drafter] rollout.drafter.checkpoint_path = {_drafter_ckpt!r}")
    print(f"[Drafter] rollout.drafter.enable_drafter_training = {_drafter_enable!r}")
    print(
        f"[Drafter] speco.sft_specific.enable_drafter_training = {_sft_enable_drafter!r}"
    )
    print(f"[Drafter] speco.sft_specific.checkpoint_save_dir = {_sft_ckpt_save_dir!r}")
    if not _drafter_ckpt:
        print(
            "[WARNING] rollout.drafter.checkpoint_path is EMPTY/None! Drafter will NOT be saved!"
        )
    if not _sft_ckpt_save_dir:
        print(
            "[WARNING] speco.sft_specific.checkpoint_save_dir is EMPTY/None! Final drafter save will be skipped!"
        )
    print("=" * 60)

    model_path = config.model.path
    print(f"Model path: {model_path}")

    trust_remote_code = config.data.get("trust_remote_code", False)
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    print(f"Loading processor from {model_path}...")
    processor = hf_processor(
        model_path, trust_remote_code=trust_remote_code, use_fast=True
    )

    print(f"Loading training data from {config.data.train_files}...")
    train_dataset = create_sft_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("train_max_samples", -1),
    )
    print(f"Train dataset size: {len(train_dataset)}")

    speco_worker_cls = None
    if hasattr(config, "speco") and hasattr(config.speco, "sft_specific"):
        sft_specific = config.speco.sft_specific
        if getattr(sft_specific, "enable_drafter_training", False):
            try:
                from verl_speco.workers import SpecoWorker

                speco_worker_cls = ray.remote(SpecoWorker)
                logger.info("SpecoWorker created")
            except ImportError as e:
                logger.warning(f"SpecoWorker import failed: {e}")

            print("\nCreating SpecoRaySFTRayTrainer...")
    trainer = SpecoRaySFTRayTrainer(
        config=config,
        speco_worker_cls=speco_worker_cls,
    )

    print("\n" + "=" * 60)
    print("Starting SFT co-train...")
    print("=" * 60)

    try:
        trainer.fit()
        print("\n" + "=" * 60)
        print("✅ SFT co-train completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
