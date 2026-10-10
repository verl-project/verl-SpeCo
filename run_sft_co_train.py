#!/usr/bin/env python
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

# ==================== 实验配置（修改此处即可）====================
EXPERIMENT_TYPE = "cotrain"  # "sft_only" 或 "cotrain"
SPECULATIVE_ALGORITHM = "EAGLE3"  # "EAGLE3" / "DSPARK" / "DFLASH"
EXPERIMENT_TIMESTAMP = "20260831"

# 数据集路径
TRAIN_DATA_PATH = (
    "/model/sft_cotrain/verl-SpeCo/data/gsm8k_dapo_merged_train_sft.parquet"
)
VAL_DATA_PATH = "/model/sft_cotrain/verl-SpeCo/data/gsm8k_dapo_merged_test_sft.parquet"

# 模型路径
MODEL_PATH = "/nas/disk1/Qwen3-0.6B"

# 训练参数
TOTAL_TRAINING_STEPS = 100
TOTAL_EPOCHS = 2
TRAIN_BATCH_SIZE = 16
MICRO_BATCH_SIZE_PER_GPU = 2
MAX_TOKEN_LEN_PER_GPU = 4096

# 保存/测试频率
SAVE_FREQ = 50  # 每50步保存1次
TEST_FREQ = 25  # 每25步测试1次

# GPU 配置
N_GPUS_PER_NODE = 1  # 单卡验证
N_NODES = 1
# ================================================================

# os.environ["WANDB_DISABLED"] = "true"
# ========== 关键：添加所有需要的路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERL_ROOT = os.environ.get("VERL_ROOT", "/model/sft_cotrain/verl")
SPECO_ROOT = SCRIPT_DIR

if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)
if SPECO_ROOT not in sys.path:
    sys.path.insert(0, SPECO_ROOT)

print(f"[DEBUG] VERL_ROOT = {VERL_ROOT}")
print(f"[DEBUG] SPECO_ROOT = {SPECO_ROOT}")
print(f"[DEBUG] sys.path[:3] = {sys.path[:3]}")

# verl config 目录
VERL_CONFIG_DIR = os.path.join(VERL_ROOT, "verl", "trainer", "config")
# speco config 目录
SPECO_CONFIG_DIR = os.path.join(SPECO_ROOT, "verl_speco", "config")
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
    from omegaconf import OmegaConf, open_dict

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

    # ========== 基础配置（CLI 优先，常量作为 fallback）==========
    # speco_sft_trainer.yaml 中已将这些字段置 null，确保脚本常量生效
    # expanduser 处理 ~ 路径（transformers 不会自动展开 ~）
    config.model.path = os.path.expanduser(config.model.get("path", None) or MODEL_PATH)
    config.data.train_files = os.path.expanduser(
        config.data.get("train_files", None) or TRAIN_DATA_PATH
    )
    config.data.val_files = os.path.expanduser(
        config.data.get("val_files", None) or VAL_DATA_PATH
    )
    config.data.train_max_samples = config.data.get("train_max_samples", None) or -1
    config.data.ignore_input_ids_mismatch = True  # Qwen Thinking 模板需要忽略拼接差异
    config.data.num_workers = config.data.get("num_workers", None) or 0

    # ========== 训练配置（CLI 优先，常量作为 fallback）==========
    config.data.train_batch_size = (
        config.data.get("train_batch_size", None) or TRAIN_BATCH_SIZE
    )
    config.data.micro_batch_size_per_gpu = (
        config.data.get("micro_batch_size_per_gpu", None) or MICRO_BATCH_SIZE_PER_GPU
    )
    config.data.max_token_len_per_gpu = (
        config.data.get("max_token_len_per_gpu", None) or MAX_TOKEN_LEN_PER_GPU
    )
    config.trainer.total_training_steps = (
        config.trainer.get("total_training_steps", None) or TOTAL_TRAINING_STEPS
    )
    config.trainer.total_epochs = (
        config.trainer.get("total_epochs", None) or TOTAL_EPOCHS
    )
    config.trainer.n_gpus_per_node = (
        config.trainer.get("n_gpus_per_node", None) or N_GPUS_PER_NODE
    )
    config.trainer.nnodes = config.trainer.get("nnodes", None) or N_NODES
    config.trainer.balance_batch = False  # SFT co-train 不需要 batch balancing
    config.trainer.resume_mode = config.trainer.get("resume_mode", None) or "disable"

    # ========== 保存/测试配置（CLI 优先，常量作为 fallback）==========
    config.trainer.save_freq = config.trainer.get("save_freq", None) or SAVE_FREQ
    config.trainer.test_freq = config.trainer.get("test_freq", None) or TEST_FREQ
    config.trainer.logger = config.trainer.get("logger", None) or ["console", "wandb"]
    config.trainer.project_name = (
        config.trainer.get("project_name", None) or "sft_cotrain_comparison"
    )
    config.trainer.experiment_name = (
        config.trainer.get("experiment_name", None)
        or f"sft_{EXPERIMENT_TYPE}_{EXPERIMENT_TIMESTAMP}"
    )
    config.trainer.default_local_dir = (
        config.trainer.get("default_local_dir", None)
        or f"checkpoints/sft_{EXPERIMENT_TYPE}_{EXPERIMENT_TIMESTAMP}"
    )

    # ========== 数据格式配置 ==========
    config.model.use_remove_padding = True
    config.data.pad_mode = "no_padding"

    # ========== 显存优化配置（腾出 GPU 空间给 Drafter）==========
    config.engine.param_offload = True
    config.engine.optimizer_offload = True
    config.engine.offload_policy = True
    config.model.enable_activation_offload = True

    # ========== 实验类型 & 算法（CLI 优先，常量作为 fallback）==========
    experiment_type = config.get("experiment_type", EXPERIMENT_TYPE)
    speculative_algorithm = str(
        config.rollout.drafter.get("speculative_algorithm", SPECULATIVE_ALGORITHM)
    ).upper()
    enable_co_train = experiment_type == "cotrain"

    if not hasattr(config, "speco"):
        from omegaconf import open_dict

        with open_dict(config):
            config.speco = OmegaConf.create({})

    # 必须使用 open_dict 包裹，才能添加不存在的字段
    from omegaconf import open_dict

    with open_dict(config.speco):
        config.speco.mode = "sft"
        # 保留用户已传入的 sft_specific 字段，仅对缺失字段设置默认值
        if (
            not hasattr(config.speco, "sft_specific")
            or config.speco.sft_specific is None
        ):
            config.speco.sft_specific = OmegaConf.create({})
        sft_specific = config.speco.sft_specific
        defaults = {
            "enable_drafter_training": enable_co_train,
            "drafter_train_interval": 5,
            "max_samples_per_step": 200,
            "hidden_layer_id": -1,
            "publish_during_sft": False,
            "checkpoint_save_dir": f"checkpoints/sft_{experiment_type}_{EXPERIMENT_TIMESTAMP}/drafter",
        }
        for key, value in defaults.items():
            if not hasattr(sft_specific, key) or getattr(sft_specific, key) is None:
                setattr(sft_specific, key, value)

    print("=" * 60)
    print(f"SPECO SFT - Experiment: {experiment_type}")
    print(f"Co-train enabled: {enable_co_train}")
    print("=" * 60)

    # ========== rollout.drafter 配置 ==========
    if not hasattr(config, "rollout"):
        from omegaconf import open_dict

        with open_dict(config):
            config.rollout = OmegaConf.create({})

    # 注入 rollout 基础配置
    if not hasattr(config.rollout, "tensor_model_parallel_size"):
        from omegaconf import open_dict

        with open_dict(config.rollout):
            config.rollout.tensor_model_parallel_size = 1
    if not hasattr(config.rollout, "data_parallel_size"):
        from omegaconf import open_dict

        with open_dict(config.rollout):
            config.rollout.data_parallel_size = 1
    if not hasattr(config.rollout, "pipeline_model_parallel_size"):
        from omegaconf import open_dict

        with open_dict(config.rollout):
            config.rollout.pipeline_model_parallel_size = 1

    # 注入 drafter 基础配置
    if not hasattr(config.rollout, "drafter"):
        from omegaconf import open_dict

        with open_dict(config.rollout):
            config.rollout.drafter = OmegaConf.create({})

    # 设置 drafter 基础配置（必须用 open_dict 包裹）
    from omegaconf import open_dict

    with open_dict(config.rollout.drafter):
        config.rollout.drafter.enable = enable_co_train
        config.rollout.drafter.enable_drafter_training = enable_co_train
        config.rollout.drafter.model_path = ""
        config.rollout.drafter.speculative_algorithm = speculative_algorithm
        config.rollout.drafter.checkpoint_path = (
            f"checkpoints/sft_{experiment_type}_{EXPERIMENT_TIMESTAMP}/drafter"
        )

    # ========== drafter training 配置（实验级覆盖，其余走 speco_base.yaml + speco_sft_trainer.yaml 默认值）==========
    from omegaconf import open_dict

    with open_dict(config.rollout.drafter.training):
        config.rollout.drafter.training.enable = enable_co_train
        config.rollout.drafter.training.enable_drafter_training = enable_co_train
        config.rollout.drafter.training.collect_hidden_states_from_old_logprob = True
        config.rollout.drafter.training.collect_hidden_states_from_sgl = False

        # DSpark 专用参数（仅 DSPARK 算法时覆盖，其余走 speco_base.yaml 默认值）
        if speculative_algorithm == "DSPARK":
            config.rollout.drafter.training.dspark_l1_loss_alpha = (
                0.9  # >0 → layout=dflash_aux_plus_last
            )
            config.rollout.drafter.training.dspark_num_target_layers = 5  # context 层数
            config.rollout.drafter.training.dspark_target_layer_ids = (
                None  # 让框架自动推导
            )

        # FSDP config（SFT 单卡验证用）
        config.rollout.drafter.training.fsdp_config = OmegaConf.create(
            {
                "param_offload": True,
                "optimizer_offload": True,
                "use_orig_params": True,
                "forward_prefetch": False,
                "wrap_policy": OmegaConf.create({"min_num_params": 0}),
            }
        )

        # 训练 batch 参数
        config.rollout.drafter.training.training_batch_size = 8
        config.rollout.drafter.training.micro_batch_size = 8
        config.rollout.drafter.training.batch_size_per_gpu = 8
        config.rollout.drafter.training.max_grad_norm = 1.0
        config.rollout.drafter.training.weight_decay = 0.0
        config.rollout.drafter.training.use_kl_loss = False
        config.rollout.drafter.training.use_adv_loss = False
        config.rollout.drafter.training.hidden_state_clip_value = 1e8
        # 其他必要配置
        config.rollout.drafter.training.is_offload_param = True
        config.rollout.drafter.training.is_offload_optimizer = True

    # ========== sft_drafter 配置（SpecoWorker 需要）==========
    if not hasattr(config, "sft_drafter"):
        from omegaconf import open_dict

        with open_dict(config):
            config.sft_drafter = OmegaConf.create({})

    with open_dict(config.sft_drafter):
        config.sft_drafter.enable_training = enable_co_train
        config.sft_drafter.train_interval = 5
        config.sft_drafter.max_samples_per_step = 200

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

    print("\nVerifying methods:")
    methods = ["fit", "_speco_fit_with_hooks", "_speco_build_sft_collect_plan"]
    for m in methods:
        print(f"  - {m}: {'✅' if hasattr(trainer, m) else '❌'}")

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

# export PYTHONPATH="/model/sft_cotrain/verl:/model/sft_cotrain/verl-SpeCo:$PYTHONPATH"
