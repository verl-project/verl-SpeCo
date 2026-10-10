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
"""SPECO SFT Trainer - SFT 版协同训练器。

继承 verl 原生 SFTTrainer，在 SFT 训练过程中通过 forward hook 采集特征，
按间隔触发 drafter 训练，最终产出成对的 Actor + Drafter 权重。
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any
from uuid import uuid4

import ray
import torch
from omegaconf import OmegaConf
from verl.trainer.sft_trainer_ray import SFTTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device
from verl.utils.logger import log_with_rank

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_AUX_LAYER_IDS_KEY,
    OLD_LOGPROB_COLLECT_MASK_KEY,
    OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
    OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
    OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY,
    OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
    OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
    OLD_LOGPROB_HIDDEN_REF_META_KEY,
    OLD_LOGPROB_HIDDEN_REFS_KEY,
    OLD_LOGPROB_HIDDEN_STATES_KEY,
    OLD_LOGPROB_OWNER_RANK_KEY,
    OLD_LOGPROB_SAMPLE_INDICES_KEY,
)
from verl_speco.workers.speco_worker import SpecoWorker

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_SPECO_LOGGING_LEVEL", "WARN"))


class SpecoRaySFTRayTrainer(SFTTrainer):
    """SPECO adapter for verl SFT trainer.

    与 PPO 版本的区别：
    1. 不 hook generate_sequences（SFT 无 rollout）
    2. hook 的是 forward_step（SFT 的前向），而不是 compute_old_log_prob
    3. batch 结构不同：用 loss_mask 区分"响应"位置
    4. 默认不发布 drafter 权重（无 rollout 引擎），改为 save checkpoint
    5. 显存更紧张（有 optimizer state），需额外的显存防护
    """

    def __init__(self, *args, **kwargs):
        """初始化 SFT Trainer。

        从 kwargs 中提取 speco_worker_cls 参数，其余参数传递给父类。
        """
        self.speco_worker_cls = kwargs.pop("speco_worker_cls", None)

        # 多卡资源池共享：_build_engine（在 super().__init__ 中调用）会设置此属性
        # 必须在 super().__init__ 之前初始化，避免被覆盖
        self._speco_actor_resource_pool = None

        super().__init__(*args, **kwargs)

        # SPECO 状态变量（复用 SpecoRayPPOTrainer 的命名）
        self.drafter_wg = None
        self._pending_drafter_checkpoint_refs = []
        # 缓存真实的 owner_count（从 drafter_wg 的 dispatch mesh 查询）
        self._speco_owner_count_cache = None

        # SFT 专属统计变量
        self._speco_last_sft_collected_samples = 0
        self._speco_last_sft_payload_mib = 0.0

        # 从配置中读取 SFT 专属参数
        self._speco_sft_config = self._parse_sft_config()

        logger.info("SpecoRaySFTRayTrainer initialized with SFT mode")

    def _parse_sft_config(self) -> dict[str, Any]:
        """解析 SFT 专属配置。

        Returns:
            dict: SFT 专属配置项
        """
        speco_cfg = getattr(self.config, "speco", None)
        if speco_cfg is None:
            return {}

        sft_specific = getattr(speco_cfg, "sft_specific", None)
        if sft_specific is None:
            return {}

        # 转换为字典
        if OmegaConf.is_config(sft_specific):
            return OmegaConf.to_container(sft_specific, resolve=True)
        elif isinstance(sft_specific, dict):
            return sft_specific
        else:
            return {}

    def attach_speco_worker_group(self, worker_group):
        """挂载 SpecoWorker group。

        Args:
            worker_group: SpecoWorker 的 Ray worker group
        """
        self.drafter_wg = worker_group
        logger.info("SpecoWorker group attached")

    def _speco_init_worker_group(self):
        """初始化 SpecoWorker group。

        与 PPO co-train 流程保持一致：共享 Actor 的 resource_pool，
        通过 max_colocate_count=2 让 Actor 和 Drafter WorkerGroup 共存于同一组 GPU 上。
        避免创建独立 resource_pool 导致 GPU 资源需求翻倍。
        """
        import ray
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

        if self.speco_worker_cls is None:
            self.speco_worker_cls = ray.remote(SpecoWorker)

        logger.info("Initializing SpecoWorker group...")

        device_name = self.device_name
        logger.info(f"Using device: {device_name}")

        # ===== 关键：共享 Actor 的 resource_pool =====
        # 与 PPO 的 _init_speco_drafter_workers 行为一致：
        # resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rp = self._speco_actor_resource_pool
        if actor_rp is None:
            raise RuntimeError(
                "Actor resource_pool is not initialized. "
                "_build_engine() must be called before _speco_init_worker_group()."
            )

        logger.info(
            f"Sharing Actor resource_pool with Drafter: "
            f"world_size={actor_rp.world_size}, "
            f"max_colocate_count={actor_rp.max_colocate_count}, "
            f"store={actor_rp.store}"
        )

        drafter_resource_pool = actor_rp

        # 创建 SpecoWorker group，传递 device_name 参数
        ray_cls_with_init = RayClassWithInitArgs(
            cls=self.speco_worker_cls,
            config=self.config,
            role="drafter",
            device_name=device_name,
        )

        self.drafter_wg = RayWorkerGroup(
            resource_pool=drafter_resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix="speco_drafter",
            device_name=device_name,
        )

        logger.info(
            f"SpecoWorker group created: world_size={self.drafter_wg.world_size}"
        )

        # 初始化模型（注册 dispatch info 在此期间发生）
        self.drafter_wg.init_model()
        logger.info("SpecoWorker group initialized successfully")

        # 查询真实的 owner_count（必须在 init_model() 之后，因为 dispatch info
        # 在 _ensure_training_group_initialized() 中注册，该方法由 init_model() 调用）
        try:
            from verl_speco.workers.speco_worker import DRAFTER_OWNER_ROUTE_MESH

            mapping = self.drafter_wg._query_dispatch_info(DRAFTER_OWNER_ROUTE_MESH)
            if mapping:
                self._speco_owner_count_cache = max(int(r) for r in mapping) + 1
                logger.info(
                    f"Drafter owner_route mesh dp_size={self._speco_owner_count_cache} (mapping={mapping})"
                )
            else:
                self._speco_owner_count_cache = 1
        except Exception as e:
            logger.warning(
                f"Failed to query owner_count from drafter_wg: {e}, defaulting to 1"
            )
            self._speco_owner_count_cache = 1

    def _require_speco_worker_group(self):
        """获取 SpecoWorker group，若未初始化则抛出异常。

        Returns:
            SpecoWorker group

        Raises:
            RuntimeError: 若 worker group 未初始化
        """
        if self.drafter_wg is None:
            raise RuntimeError("SpecoWorker group has not been initialized yet.")
        return self.drafter_wg

    def _is_drafter_training_enabled(self) -> bool:
        """检查是否启用了 drafter 训练。

        Returns:
            bool: 是否启用
        """
        return self._speco_sft_config.get("enable_drafter_training", False)

    def _speco_drafter_train_interval(self) -> int:
        """获取 drafter 训练的 optimizer step 间隔。

        Returns:
            int: 训练间隔
        """
        return int(self._speco_sft_config.get("drafter_train_interval", 5))

    def _ray_get_if_needed(self, value):
        """如果 value 是 Ray ObjectRef，先 ray.get() 再返回。和 PPO trainer 保持一致。"""
        if value is None:
            return None
        try:
            import ray
        except Exception:
            return value
        object_ref_type = getattr(ray, "ObjectRef", ())
        if object_ref_type and isinstance(value, object_ref_type):
            return ray.get(value)
        return value

    def speco_sync_target_lm_head_weight(self, payload, global_step=None):
        """代理到 SpecoWorker group 的 sync_target_lm_head_weight。"""
        return self._require_speco_worker_group().sync_target_lm_head_weight(
            payload, global_step=global_step
        )

    def speco_get_drafter_target_lm_head_row_indices(self):
        """代理到 SpecoWorker group 的 get_drafter_target_lm_head_row_indices。"""
        return (
            self._require_speco_worker_group().get_drafter_target_lm_head_row_indices()
        )

    def _speco_max_samples_per_step(self) -> int:
        """获取每个 step 最多收集的样本数。

        Returns:
            int: 最大样本数
        """
        return int(self._speco_sft_config.get("max_samples_per_step", 16))

    def _speco_get_num_hidden_layers(self) -> int | None:
        """获取目标模型（actor）的 num_hidden_layers。

        优先从 self.model_config 获取，其次从 self.config.model 获取。
        """
        num_hidden_layers = None
        if hasattr(self, "model_config"):
            if hasattr(self.model_config, "hf_config") and hasattr(
                self.model_config.hf_config, "num_hidden_layers"
            ):
                num_hidden_layers = self.model_config.hf_config.num_hidden_layers
            elif hasattr(self.model_config, "num_hidden_layers"):
                num_hidden_layers = self.model_config.num_hidden_layers
            elif hasattr(self.model_config, "text_config") and hasattr(
                self.model_config.text_config, "num_hidden_layers"
            ):
                num_hidden_layers = self.model_config.text_config.num_hidden_layers
        if num_hidden_layers is None:
            model_cfg = getattr(self.config, "model", None)
            if model_cfg is not None:
                hf_cfg = getattr(model_cfg, "hf_config", None)
                if hf_cfg is not None and hasattr(hf_cfg, "num_hidden_layers"):
                    num_hidden_layers = hf_cfg.num_hidden_layers
                elif hasattr(model_cfg, "num_hidden_layers"):
                    num_hidden_layers = model_cfg.num_hidden_layers
        return num_hidden_layers

    def _speco_oldlogprob_hidden_layout(self) -> str:
        """根据 speculative_algorithm 动态解析 hidden states layout。"""
        drafter_cfg = self.config.rollout.drafter
        algorithm = str(getattr(drafter_cfg, "speculative_algorithm", "")).upper()
        return resolve_drafter_hidden_states_layout(algorithm, drafter_cfg.training)

    def _speco_oldlogprob_aux_layer_ids(self) -> list[int]:
        """根据 speculative_algorithm 动态解析 target_layer_ids。

        resolve_oldlogprob_aux_layer_ids 内部已处理所有算法分支：
        - DFlash/DSpark → 按 num_context_layers 均匀采样
        - EAGLE3 → [2, mid, -3]
        - EAGLE1/2 → [last]
        仅当 num_hidden_layers 无法确定时才返回 None，此时用保守兜底。
        """
        drafter_cfg = self.config.rollout.drafter
        num_hidden_layers = self._speco_get_num_hidden_layers()
        layer_ids = resolve_oldlogprob_aux_layer_ids(
            drafter_cfg,
            target_num_hidden_layers=num_hidden_layers,
        )
        if layer_ids is not None:
            return layer_ids
        logger.warning("Cannot resolve aux layer ids, using default [2, 16, -3]")
        return [2, 16, -3]

    def _speco_get_aux_layer_ids(self) -> list[int]:
        """获取 aux layer ids（用于 EAGLE3 风格 hidden states 采集）。

        优先从配置读取，否则从模型 config 推导默认值：
        [2, num_hidden_layers // 2, num_hidden_layers - 3]

        Returns:
            list[int]: aux layer ids
        """
        # 1. 尝试从 speco 配置中读取
        config_layer_ids = self._speco_sft_config.get("aux_layer_ids", None)
        if config_layer_ids is not None:
            if isinstance(config_layer_ids, int):
                return [int(config_layer_ids)]
            return [int(x) for x in config_layer_ids]

        # 2. 从模型 config 推导默认值
        num_hidden_layers = self._speco_get_num_hidden_layers()

        if num_hidden_layers is None:
            logger.warning(
                "Cannot determine num_hidden_layers, using default [2, 16, -3]"
            )
            return [2, 16, -3]

        # EAGLE3 默认公式：[2, mid, -3]
        mid = max(1, num_hidden_layers // 2)
        last_minus_3 = max(0, num_hidden_layers - 3)
        aux_layer_ids = sorted(set([2, mid, last_minus_3]))
        logger.info(
            f"Derived aux_layer_ids={aux_layer_ids} from num_hidden_layers={num_hidden_layers}"
        )
        return aux_layer_ids

    def _speco_hidden_layer_id(self) -> int:
        """获取抓取的 hidden 层位置。

        Returns:
            int: 层 ID（-1 表示最后一层）
        """
        return int(self._speco_sft_config.get("hidden_layer_id", -1))

    def _speco_publish_during_sft(self) -> bool:
        """获取是否在 SFT 过程中热发布 drafter 权重。

        Returns:
            bool: 是否发布
        """
        return bool(self._speco_sft_config.get("publish_during_sft", False))

    def _speco_checkpoint_save_dir(self) -> str | None:
        """获取 drafter checkpoint 保存路径。

        Returns:
            str or None: 保存路径
        """
        return self._speco_sft_config.get("checkpoint_save_dir", None)

    def speco_set_global_step(self, global_step: int):
        """设置全局 step 到 SpecoWorker。

        Args:
            global_step: 当前全局 step
        """
        return self._require_speco_worker_group().set_global_step(global_step)

    def speco_collect_rollout_features(self, samples: list[list[dict]]):
        """收集 rollout 特征到 SpecoWorker。

        Args:
            samples: 特征样本列表

        Returns:
            SpecoWorker 的收集结果
        """
        return self._require_speco_worker_group().collect_rollout_features(samples)

    def speco_train_drafter(self):
        """训练 drafter（适配社区版两阶段流程）。

        社区版 worker 的 train_drafter 需要先通过 preflight_drafter_training 设置
        _prepared_training_plan_id，否则会返回 preflight_not_ready。

        Returns:
            训练结果
        """
        import ray as _ray

        wg = self._require_speco_worker_group()

        # 1. 读取 drafter 训练配置
        drafter_training_cfg = self.config.rollout.drafter.training
        sample_last_n_steps = int(drafter_training_cfg.get("sample_last_n_steps", 20))
        require_full_batch = bool(drafter_training_cfg.get("require_full_batch", False))
        max_batches = int(drafter_training_cfg.get("step", 10))
        min_batches = int(drafter_training_cfg.get("min_trainable_batches", 1))
        publish_after_success = bool(self._speco_publish_during_sft())

        global_step = int(getattr(self, "global_steps", 0) or 0)

        # 2. 获取各 worker 的数据状态（用于构建 worker_snapshots）
        logger.warning(
            "[speco_train_drafter] Fetching training data status from workers..."
        )
        status_ref = wg.get_drafter_training_data_status(
            sample_last_n_steps, require_full_batch
        )
        status_list = self._ray_get_if_needed(status_ref)
        if isinstance(status_list, dict):
            status_list = [status_list]
        if not isinstance(status_list, list):
            status_list = []

        # 过滤出可用的 worker 状态
        available_statuses = [
            s for s in status_list if isinstance(s, dict) and s.get("available", False)
        ]
        if not available_statuses:
            logger.warning(
                "[speco_train_drafter] No workers with available training data, skipping"
            )
            return [
                {
                    "trained": False,
                    "triggered": False,
                    "reason": "no_available_workers",
                    "successful_steps": 0,
                    "attempted_steps": 0,
                    "elapsed_sec": 0.0,
                }
            ]

        # 3. 构建 worker_snapshots（每个 rank 的 incarnation/buffer_version/data_version）
        worker_snapshots = {}
        data_versions = []
        for s in available_statuses:
            rank_str = str(s.get("rank", s.get("worker_id", "0")))
            worker_snapshots[rank_str] = {
                "worker_incarnation": s.get("worker_incarnation"),
                "buffer_version": s.get("buffer_version"),
                "data_version": s.get("data_version"),
                "trainable_samples": s.get("trainable_samples", 0),
            }
            dv = s.get("data_version")
            if dv is not None:
                data_versions.append(int(dv))

        # 取所有 worker 的最大 data_version 作为 plan 的 data_version
        plan_data_version = max(data_versions) if data_versions else None

        # 4. 构建 training plan
        plan_id = uuid4().hex
        training_plan = {
            "launch": True,
            "execution_strategy": "sync",
            "source_global_step": global_step,
            "max_batches": max_batches,
            "publish_after_success": publish_after_success,
            "min_batches": min_batches,
            "require_full_batch": require_full_batch,
            "sample_last_n_steps": sample_last_n_steps,
            "data_version": plan_data_version,
            "required_target_version": None,
            "min_sample_step": None,
            "max_sample_step": None,
            "plan_id": plan_id,
            "worker_snapshots": worker_snapshots,
            "data_source": "local_buffer",
        }

        logger.warning(
            f"[speco_train_drafter] Built plan: plan_id={plan_id[:8]}, "
            f"global_step={global_step}, max_batches={max_batches}, "
            f"data_version={plan_data_version}, workers={len(worker_snapshots)}"
        )

        # 5. 调用 preflight_drafter_training（设置 _prepared_training_plan_id）
        logger.warning("[speco_train_drafter] Calling preflight_drafter_training...")
        preflight_ref = wg.preflight_drafter_training(training_plan)
        preflight_results = self._ray_get_if_needed(preflight_ref)
        if isinstance(preflight_results, dict):
            preflight_results = [preflight_results]
        if not isinstance(preflight_results, list):
            preflight_results = []

        not_ready = [
            r
            for r in preflight_results
            if isinstance(r, dict) and not r.get("ready", False)
        ]
        if not_ready:
            reasons = {r.get("reason", "unknown") for r in not_ready}
            logger.warning(
                f"[speco_train_drafter] Preflight not ready, reasons={reasons}, "
                f"skipping train_drafter. Results: {not_ready}"
            )
            return [
                {
                    "trained": False,
                    "triggered": False,
                    "reason": f"preflight_failed:{reasons}",
                    "successful_steps": 0,
                    "attempted_steps": 0,
                    "elapsed_sec": 0.0,
                }
            ]

        logger.warning(
            "[speco_train_drafter] All workers ready, calling train_drafter()..."
        )

        # 6. 执行训练
        try:
            result_ref = wg.train_drafter(training_plan)
            logger.warning(
                f"[speco_train_drafter] Worker group returned: {type(result_ref)}"
            )

            if isinstance(result_ref, list):
                logger.warning(
                    f"[speco_train_drafter] Waiting for {len(result_ref)} refs..."
                )
                result = _ray.get(result_ref)
            else:
                result = _ray.get(result_ref)

            logger.warning(f"[speco_train_drafter] ray.get() returned: {type(result)}")

            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float, str, bool)):
                        logger.warning(f"[speco_train_drafter] {key}: {value}")
            elif isinstance(result, list) and len(result) > 0:
                logger.warning(
                    f"[speco_train_drafter] List result length: {len(result)}"
                )
                if isinstance(result[0], dict):
                    for key, value in result[0].items():
                        if isinstance(value, (int, float, str, bool)):
                            logger.warning(
                                f"[speco_train_drafter] result[0].{key}: {value}"
                            )

            return result
        except Exception as e:
            logger.error(f"[speco_train_drafter] ERROR: {e}")
            import traceback

            traceback.print_exc()
            raise

    def speco_activate_drafter_training_model(self):
        """激活 drafter 训练模型。

        Returns:
            激活结果
        """
        return self._require_speco_worker_group().activate_drafter_training_model()

    def speco_save_checkpoint(self, global_step: int, wait: bool = True):
        """保存 drafter checkpoint。

        Args:
            global_step: 当前全局 step
            wait: 是否等待保存完成

        Returns:
            保存结果
        """
        return self._require_speco_worker_group().save_checkpoint(
            global_step, wait=wait
        )

    def speco_wait_checkpoint(self):
        """等待 checkpoint 保存完成。

        Returns:
            等待结果
        """
        return self._require_speco_worker_group().wait_checkpoint()

    def speco_maybe_publish(self):
        """可能发布 drafter 权重（用于 rollout 引擎）。

        Returns:
            发布结果
        """
        return self._require_speco_worker_group().maybe_publish()

    def _speco_should_train_drafter_this_step(self, global_step: int) -> bool:
        """判断当前 step 是否应该训练 drafter。

        Args:
            global_step: 当前全局 step

        Returns:
            bool: 是否应该训练
        """
        interval = self._speco_drafter_train_interval()
        if interval <= 0:
            if global_step <= 5:
                logger.warning(
                    f"[Drafter Trigger] Step {global_step}: interval={interval} <= 0, SKIP drafter training"
                )
            return False

        should_train = global_step % interval == 0
        if global_step <= 5 or should_train:
            logger.warning(
                f"[Drafter Trigger] Step {global_step}: interval={interval}, global_step % interval = {global_step % interval}, should_train={should_train}"
            )
        return should_train

    def _speco_get_current_rank(self) -> int:
        """获取当前训练进程的 rank。

        注意：SFT 模式下不需要查询 dispatch mesh（会导致 Ray worker crash）。
        主进程用轮转分配 owner_rank，所以这里直接返回 0 即可。
        """
        return 0

    def _speco_get_owner_count(self) -> int:
        """获取 owner bucket 数量（用于多卡分发）。

        优先使用从 drafter_wg dispatch mesh 查询的真实值（与 PPO 一致），
        仅在 drafter_wg 未初始化时 fallback 到配置推断。
        """
        if self._speco_owner_count_cache is not None:
            return self._speco_owner_count_cache
        # Fallback: 当 drafter_wg 尚未初始化时（不常见）
        try:
            n_gpus = int(self.config.trainer.get("n_gpus_per_node", 1))
            nnodes = int(self.config.trainer.get("nnodes", 1))
            return max(n_gpus * nnodes, 1)
        except Exception:
            return 1

    def _speco_owner_route_mapping(self):
        """获取 drafter_wg 的 owner route mapping。

        SFT 模式下不需要 route mapping（用 config 推断 GPU 数），
        直接返回 None。
        """
        return

    def _speco_build_sft_collect_plan(self, batch) -> dict[str, Any] | None:
        """构建 SFT 版的 collect plan。

        根据 SFT batch 的 input_ids 和 loss_mask，计算 hidden states 的抓取位置。
        返回格式与 PPO 版本兼容的 tensor。

        Args:
            batch: SFT batch（TensorDict 格式）

        Returns:
            dict or None: collect plan，包含 collect_mask、hidden_positions 等 tensor
        """
        try:
            # 获取 input_ids 和 loss_mask（TensorDict 访问方式）
            input_ids = tu.get(batch, key="input_ids")
            loss_mask = tu.get(batch, key="loss_mask")

            if input_ids is None or loss_mask is None:
                logger.warning("input_ids or loss_mask not found in batch")
                return None

            # === 新增诊断 ===
            if not input_ids.is_nested and input_ids.size(0) <= 4:  # 只打印小 batch
                for di in range(min(2, input_ids.size(0))):
                    logger.warning(
                        f"[LOSS-MASK-RAW] sample {di}: loss_mask={loss_mask[di][:30].tolist()}... sum={loss_mask[di].sum().item()}"
                    )

            # 获取配置
            max_samples = self._speco_max_samples_per_step()

            # 判断是否为 NestedTensor（use_remove_padding）
            is_nested = (
                input_ids.is_nested if hasattr(input_ids, "is_nested") else False
            )

            batch_size = input_ids.size(0)

            # 计算每个样本的响应长度和需要收集的位置数
            # 先处理每个样本，确定 hidden_rows（最大需要收集的长度）
            sample_info_list = []  # [(collect_positions,), ...]
            max_hidden_rows = 0
            valid_count = 0

            for idx in range(batch_size):
                # 获取该样本的 loss_mask
                if is_nested:
                    # NestedTensor 不能直接切片，需要用 .values() 获取扁平存储
                    input_offsets = input_ids.offsets()
                    if idx >= len(input_offsets) - 1:
                        break
                    start = int(input_offsets[idx])
                    end = int(input_offsets[idx + 1])

                    # 使用 .values() 获取扁平存储后再切片
                    loss_mask_flat = (
                        loss_mask.values()
                        if hasattr(loss_mask, "values")
                        else loss_mask
                    )

                    if hasattr(loss_mask, "offsets"):
                        lm_offsets = loss_mask.offsets()
                        lm_start = int(lm_offsets[idx])
                        lm_end = int(lm_offsets[idx + 1])
                        sample_loss_mask = loss_mask_flat[lm_start:lm_end].to(torch.int)
                    else:
                        sample_loss_mask = loss_mask_flat[start:end].to(torch.int)
                else:
                    sample_loss_mask = loss_mask[idx].to(torch.int)

                # 左移 loss_mask：hidden i 对应 token i+1
                shifted_mask = torch.zeros_like(sample_loss_mask)
                shifted_mask[:-1] = sample_loss_mask[1:]
                # 第一个位置置零（避免污染）
                shifted_mask[0] = 0

                # 找到需要收集的位置
                collect_positions = torch.where(shifted_mask > 0)[0]

                if len(collect_positions) > 0 and valid_count < max_samples:
                    sample_info_list.append(collect_positions)
                    max_hidden_rows = max(max_hidden_rows, len(collect_positions))
                    valid_count += 1
                else:
                    sample_info_list.append(None)

            if valid_count == 0:
                logger.debug("No valid samples with response tokens found")
                return None

            # 构建与 PPO 兼容的 tensor 格式
            # collect_mask: (batch_size,) bool, 标记哪些样本需要收集
            collect_mask = torch.zeros(batch_size, dtype=torch.bool)
            # hidden_positions: (batch_size, max_hidden_rows) long, 每个样本的收集位置
            hidden_positions = torch.zeros(
                batch_size, max_hidden_rows, dtype=torch.long
            )
            # hidden_position_mask: (batch_size, max_hidden_rows) bool, 标记有效位置
            hidden_position_mask = torch.zeros(
                batch_size, max_hidden_rows, dtype=torch.bool
            )
            # owner_rank: (batch_size,) long
            # 用轮转方式分配，确保样本均匀分布到各 drafter worker
            owner_count = self._speco_get_owner_count()
            owner_rank = torch.zeros(batch_size, dtype=torch.long)
            for i in range(batch_size):
                owner_rank[i] = i % owner_count

            # 填充数据
            valid_idx = 0
            for idx, positions in enumerate(sample_info_list):
                if positions is not None:
                    collect_mask[idx] = True
                    n = min(len(positions), max_hidden_rows)
                    hidden_positions[idx, :n] = positions[:n]
                    hidden_position_mask[idx, :n] = True
                    # owner_rank 已经在上面通过轮转分配好了
                    valid_idx += 1

            # 动态获取 owner_count（用于多卡分发）
            owner_count = self._speco_get_owner_count()

            logger.debug(
                f"SFT collect plan: {valid_count}/{batch_size} samples, "
                f"max_hidden_rows={max_hidden_rows}, owner_count={owner_count}"
            )

            return {
                "collect_mask": collect_mask,
                "hidden_positions": hidden_positions,
                "hidden_position_mask": hidden_position_mask,
                "owner_rank": owner_rank,
                "owner_count": owner_count,
                "total_samples": valid_count,
            }

        except Exception as e:
            logger.error(f"Error building SFT collect plan: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _speco_collect_sft_features(self, batch, collect_plan, output) -> int:
        """从 SFT forward output 中收集 hidden states。

        通过 oldlogprob_runtime patch 注入的 hidden refs 来获取 hidden states，
        与 PPO 版本的 _speco_collect_oldlogprob_features 兼容。

        Args:
            batch: SFT batch
            collect_plan: 由 _speco_build_sft_collect_plan 生成的 collect plan
            output: training_client.train_batch 的输出

        Returns:
            int: 收集到的样本数量
        """
        try:
            import ray

            # 从 output 中获取 hidden refs（Ray ObjectRef）
            hidden_refs = tu.get(output, OLD_LOGPROB_HIDDEN_REFS_KEY)
            hidden_ref_meta = tu.get(output, OLD_LOGPROB_HIDDEN_REF_META_KEY)
            chunk_refs = tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY)
            _chunk_meta = tu.get(output, OLD_LOGPROB_HIDDEN_CHUNK_META_KEY)

            # 也尝试直接获取 hidden_states
            hidden_states = tu.get(output, OLD_LOGPROB_HIDDEN_STATES_KEY)

            if hidden_states is None and hidden_refs is None and chunk_refs is None:
                logger.debug("No hidden states or refs found in output")
                return 0

            # 获取原始 input_ids 和 loss_mask（TensorDict 访问方式）
            input_ids = tu.get(batch, key="input_ids")
            loss_mask = tu.get(batch, key="loss_mask")
            attention_mask = tu.get(batch, key="attention_mask")

            # [TRACE Step A] 检查 batch 级别的 loss_mask 是否正确
            try:
                if loss_mask.ndim >= 2:
                    per_sample_sums = [
                        loss_mask[i].sum().item()
                        for i in range(min(4, loss_mask.size(0)))
                    ]
                    logger.debug(
                        f"[TRACE] Step A: batch loss_mask shape={loss_mask.shape}, "
                        f"per_sample_sums={per_sample_sums}"
                    )
                else:
                    logger.debug(
                        f"[TRACE] Step A: batch loss_mask shape={loss_mask.shape}, sum={loss_mask.sum().item()}"
                    )
            except Exception as e:
                logger.debug(
                    f"[TRACE] Step A: batch loss_mask shape={loss_mask.shape}, error={e}"
                )

            batch_size = collect_plan["collect_mask"].size(0)
            collect_mask = collect_plan["collect_mask"]
            hidden_positions = collect_plan["hidden_positions"]
            hidden_position_mask = collect_plan["hidden_position_mask"]
            owner_rank = collect_plan["owner_rank"]

            # [DEBUG] 验证 hidden_refs 长度与 collect_mask 的关系
            hidden_refs_len = len(hidden_refs) if hidden_refs else 0
            hidden_ref_meta_len = len(hidden_ref_meta) if hidden_ref_meta else 0
            collect_mask_sum = int(collect_mask.sum().item())
            logger.debug(
                f"[VERIFY-A] hidden_refs_len={hidden_refs_len}, hidden_ref_meta_len={hidden_ref_meta_len}, batch_size={batch_size}, collect_mask_sum={collect_mask_sum}"
            )
            if hidden_refs_len > 0 and hidden_refs_len != collect_mask_sum:
                logger.debug(
                    f"[VERIFY-A] MISMATCH! hidden_refs_len ({hidden_refs_len}) != collect_mask_sum ({collect_mask_sum}), 索引可能不对齐"
                )

            # 构建 bucket（按 owner_count 分发到多卡）
            owner_count = int(collect_plan.get("owner_count", 1))
            owner_count = max(owner_count, 1)
            buckets: list[list[dict[str, Any]]] = [[] for _ in range(owner_count)]
            collected_count = 0

            # 每个 step 只计算一次 hidden layout 和 target layer ids
            sft_hidden_layout = self._speco_oldlogprob_hidden_layout()
            sft_target_layer_ids = self._speco_oldlogprob_aux_layer_ids()

            # ===== 修复 refs/metas 索引错位问题 =====
            # oldlogprob_runtime 返回的 refs/metas 可能被 compact（None 被过滤），
            # list index 不等于 batch_idx。需要根据 meta.batch_idx 建立正确映射。
            ref_meta_map: dict[int, tuple] = {}  # batch_idx -> (hidden_ref, ref_meta)
            if hidden_refs is not None and hidden_ref_meta is not None:
                for i, meta in enumerate(hidden_ref_meta):
                    if meta is None:
                        continue
                    meta_bidx = meta.get("batch_idx", i)  # fallback: 用 list index
                    ref = hidden_refs[i] if i < len(hidden_refs) else None
                    ref_meta_map[int(meta_bidx)] = (ref, meta)
                if self.global_steps <= 3:
                    logger.debug(
                        f"[REF-MAP] 构建 ref_meta_map: {len(ref_meta_map)} 个映射, "
                        f"keys={sorted(ref_meta_map.keys())[:16]}..."
                    )

            # [DEBUG] 收集有效索引映射
            valid_indices = []
            for batch_idx in range(batch_size):
                if not bool(collect_mask[batch_idx].item()):
                    continue

                # 获取该样本的有效位置数
                valid_positions = hidden_position_mask[batch_idx]
                n_valid = int(valid_positions.sum().item())
                if n_valid <= 0:
                    # [DEBUG] VERIFY-B: 样本被收集但无有效 token
                    logger.debug(
                        f"[VERIFY-B] batch_idx={batch_idx}: collect_mask=True but n_valid=0, hidden_position_mask sum={int(valid_positions.sum().item())}"
                    )
                    continue

                # 记录有效索引
                valid_indices.append(batch_idx)

                # 获取该样本的 hidden ref 和 meta（用 ref_meta_map 正确映射）
                mapped = ref_meta_map.get(batch_idx)
                if mapped is not None:
                    hidden_ref, ref_meta = mapped
                else:
                    # fallback: 按 list index 取（旧版本兼容）
                    hidden_ref = self._speco_get_sequence_item(hidden_refs, batch_idx)
                    ref_meta = self._speco_get_sequence_item(hidden_ref_meta, batch_idx)
                    if self.global_steps <= 3:
                        logger.debug(
                            f"[REF-MAP-FALLBACK] batch_idx={batch_idx}: ref_meta_map miss, fallback to list index. "
                            f"hidden_ref={'OK' if hidden_ref else 'None'}, ref_meta={'OK' if ref_meta else 'None'}"
                        )

                # 获取 hidden states（通过 ray.get 解析 ObjectRef）
                # ===== Bug #1 修复：按 ref_meta 正确切分 chunk =====
                if hidden_ref is not None:
                    try:
                        hidden_tensor = ray.get(hidden_ref)
                    except Exception as e:
                        logger.warning(
                            f"[VERIFY-C] Failed to get hidden ref for sample {batch_idx}: {e}"
                        )
                        continue

                    # oldlogprob_runtime 把同 owner 的多个样本 cat 成一个大 chunk，
                    # ref_meta 里记录了该样本在 chunk 中的精确行范围。
                    # 必须用 meta 切分，不能用 positions（SEQ 绝对位置）索引 chunk！
                    if ref_meta is None:
                        # ref_meta 缺失 = 无法正确切分 chunk，硬兜底：跳过防止错位
                        logger.warning(
                            f"[VERIFY-REF] batch_idx={batch_idx}: ref_meta is None, "
                            f"无法切分 chunk，跳过样本以防止跨样本 hidden 错位"
                        )
                        continue
                    chunk_start = int(ref_meta.get("chunk_start", 0) or 0)
                    chunk_length = int(ref_meta.get("chunk_length", 0) or 0)
                    if (
                        chunk_length <= 0
                        or chunk_start + chunk_length > hidden_tensor.size(0)
                    ):
                        # chunk_length 无效或切分越界，硬兜底：跳过
                        logger.warning(
                            f"[VERIFY-REF] batch_idx={batch_idx}: chunk 切分参数无效 "
                            f"start={chunk_start} length={chunk_length} "
                            f"tensor_rows={hidden_tensor.size(0)}，跳过样本"
                        )
                        continue
                    hidden_tensor = hidden_tensor[
                        chunk_start : chunk_start + chunk_length
                    ]
                    # chunk_row_indices: SP 稀疏情况下的行重排列
                    row_idx_payload = ref_meta.get("chunk_row_indices")
                    if row_idx_payload is not None:
                        if isinstance(row_idx_payload, (list, tuple)):
                            row_idx_tensor = torch.tensor(
                                [int(x) for x in row_idx_payload], dtype=torch.long
                            )
                        elif torch.is_tensor(row_idx_payload):
                            row_idx_tensor = (
                                row_idx_payload.detach().cpu().long().reshape(-1)
                            )
                        else:
                            row_idx_tensor = None
                        if (
                            row_idx_tensor is not None
                            and row_idx_tensor.numel() == hidden_tensor.size(0)
                        ):
                            hidden_tensor = hidden_tensor[row_idx_tensor]
                elif hidden_states is not None:
                    # fallback 路径：hidden_states 也是压缩后的 tensor，
                    # 直接取前 n_valid 行，不用 positions 索引
                    raw_h = (
                        hidden_states[batch_idx]
                        if batch_idx < len(hidden_states)
                        else None
                    )
                    if raw_h is None:
                        logger.warning(
                            f"[VERIFY-C] batch_idx={batch_idx}: raw hidden_states is None"
                        )
                        continue
                    hidden_tensor = raw_h[:n_valid]
                else:
                    logger.warning(
                        f"[VERIFY-C] batch_idx={batch_idx}: both hidden_ref and hidden_states are None"
                    )
                    continue

                if hidden_tensor is None:
                    logger.warning(
                        f"[VERIFY-C] batch_idx={batch_idx}: hidden_tensor is None after retrieval"
                    )
                    continue

                # 切完 chunk 后 hidden_tensor 已正确指向本样本行段。
                # 修复 VERIFY-D: 当 hidden_tensor 行数 < n_valid 时（如 SP/owner_mask 过滤），
                # 容错使用实际行数，同步截断 positions，不跳过样本。
                actual_rows = hidden_tensor.size(0)
                if actual_rows < n_valid:
                    # SP/owner_mask 过滤导致部分行不在当前 rank → 正常现象，容错处理
                    logger.warning(
                        f"[VERIFY-D-ADAPT] batch_idx={batch_idx}: hidden_tensor rows ({actual_rows}) < n_valid ({n_valid}), "
                        f"SP/owner_mask 过滤，使用实际行数"
                    )
                    # 更新 n_valid 为实际行数，后续 positions 和其他引用自动对齐
                    n_valid = actual_rows
                hidden_tensor = hidden_tensor[:n_valid]

                # ===== [DIAG-1] 打印 hidden_tensor 统计值 =====
                if batch_idx < 2:
                    with torch.no_grad():
                        ht = hidden_tensor.float()
                        logger.warning(
                            f"[DIAG-HIDDEN] batch={batch_idx} shape={tuple(ht.shape)} "
                            f"mean={ht.mean().item():.6f} std={ht.std().item():.6f} "
                            f"min={ht.min().item():.4f} max={ht.max().item():.4f} "
                            f"nan_count={ht.isnan().sum().item()} "
                            f"zero_rows={(ht.abs().sum(dim=-1) < 1e-6).sum().item()}/{ht.size(0)}"
                        )
                        # 打印 hidden 的前 3 行前 10 维
                        first_rows = ht[:3, :10].cpu().tolist()
                        logger.warning(
                            f"[DIAG-HIDDEN] batch={batch_idx} first_3_rows_first_10={first_rows}"
                        )

                # ===== SANITY CHECK: 验证 Bug #1 修复生效 =====
                # 每步只对前 3 个样本打，避免日志刷屏
                if batch_idx < 3 and hidden_ref is not None and ref_meta is not None:
                    positions_sanity = hidden_positions[batch_idx][:n_valid]
                    first_seq_pos = (
                        int(positions_sanity[0].item())
                        if len(positions_sanity) > 0
                        else -1
                    )
                    last_seq_pos = (
                        int(positions_sanity[-1].item())
                        if len(positions_sanity) > 0
                        else -1
                    )
                    chunk_start_val = int(ref_meta.get("chunk_start", 0) or 0)
                    chunk_length_val = int(ref_meta.get("chunk_length", 0) or 0)

                    # 关键断言：chunk_length 应该 >= n_valid
                    # 如果不等，说明 meta 信息和实际 valid_rows 不一致
                    if chunk_length_val != n_valid:
                        logger.warning(
                            f"[SANITY-MISMATCH] batch={batch_idx}: "
                            f"chunk_length={chunk_length_val} != n_valid={n_valid}! "
                            f"chunk_start={chunk_start_val}"
                        )

                    logger.info(
                        f"[SANITY-OK] batch={batch_idx} "
                        f"chunk_start={chunk_start_val} chunk_len={chunk_length_val} "
                        f"n_valid={n_valid} hidden_shape={tuple(hidden_tensor.shape)} "
                        f"seq_pos_range=[{first_seq_pos}, {last_seq_pos}]"
                    )
                    # 对比：修复前用 positions 索引 chunk 会导致错位
                    # 如果 chunk_start=0，positions[0]=99（seq 位置），用 99 索引 chunk 是错的
                    # 修复后直接 [:n_valid] 取连续行，不依赖 seq 位置
                # ====================================================

                # 获取该样本的 token 范围
                is_nested = (
                    input_ids.is_nested if hasattr(input_ids, "is_nested") else False
                )
                if is_nested:
                    offsets = input_ids.offsets()
                    start = int(offsets[batch_idx])
                    end = int(offsets[batch_idx + 1])
                    # NestedTensor 不能直接切片，使用 .values() 获取扁平存储
                    input_ids_flat = (
                        input_ids.values()
                        if hasattr(input_ids, "values")
                        else input_ids
                    )
                    sample_input_ids = input_ids_flat[start:end]
                else:
                    sample_input_ids = input_ids[batch_idx]
                    start = 0
                    end = sample_input_ids.size(0)

                # positions 仅保留用于 sample dict 的 meta 信息（hidden_position_start/end）
                positions = hidden_positions[batch_idx][:n_valid]

                if len(positions) == 0:
                    continue  # 没有有效位置，跳过

                # 获取该样本的原始 loss_mask（与 input_ids 对齐，不再左移）
                if is_nested:
                    lm_offsets = (
                        loss_mask.offsets() if hasattr(loss_mask, "offsets") else None
                    )
                    loss_mask_flat = (
                        loss_mask.values()
                        if hasattr(loss_mask, "values")
                        else loss_mask
                    )
                    if lm_offsets is not None:
                        lm_start = int(lm_offsets[batch_idx])
                        lm_end = int(lm_offsets[batch_idx + 1])
                        sample_loss_mask = (
                            loss_mask_flat[lm_start:lm_end].detach().cpu()
                        )
                    else:
                        sample_loss_mask = loss_mask_flat[start:end].detach().cpu()
                else:
                    sample_loss_mask = loss_mask[batch_idx].detach().cpu()

                # [TRACE Step B] 检查 sample_loss_mask 是否正确
                if sample_loss_mask is not None:
                    try:
                        logger.debug(
                            f"[TRACE] Step B: batch_idx={batch_idx}, "
                            f"sample_loss_mask shape={sample_loss_mask.shape}, "
                            f"sum={sample_loss_mask.sum().item()}, "
                            f"first_40={sample_loss_mask[:40].tolist()}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"[TRACE] Step B: batch_idx={batch_idx}, error={e}"
                        )

                # 构建样本（注意：collect_online_data 期望 2D tensor）
                sample_input_ids_2d = (
                    sample_input_ids.unsqueeze(0).detach().cpu()
                )  # (1, seq_len)
                hidden_tensor_2d = (
                    hidden_tensor.unsqueeze(0).detach().cpu()
                )  # (1, hidden_dim)

                # 在构建 sample dict 之前添加调试日志
                logger.debug(
                    f"[DEBUG] Before sample dict: hidden_tensor.shape={hidden_tensor.shape}, positions.shape={positions.shape}"
                )

                sample = {
                    "input_ids": sample_input_ids_2d,
                    "hidden_positions": positions.detach().cpu().unsqueeze(0),  # 2D
                    "hidden_states": hidden_tensor_2d,
                    "hidden_position_start": int(positions[0].item())
                    if len(positions) > 0
                    else 0,
                    "hidden_position_end": int(positions[-1].item()) + 1
                    if len(positions) > 0
                    else 0,
                    "global_step": self.global_steps
                    if hasattr(self, "global_steps")
                    else 0,
                    "replica_rank": int(owner_rank[batch_idx].item()),
                    "loss_mask": sample_loss_mask.unsqueeze(
                        0
                    ),  # (1, seq_len) 原始 loss_mask
                    "hidden_states_layout": sft_hidden_layout,
                    "target_layer_ids": sft_target_layer_ids,
                }

                # [TRACE Step C] 检查 sample dict 中的 loss_mask
                try:
                    final_loss_mask = sample["loss_mask"]
                    logger.debug(
                        f"[TRACE] Step C: batch_idx={batch_idx}, "
                        f"final_loss_mask shape={final_loss_mask.shape}, "
                        f"sum={final_loss_mask.sum().item()}, "
                        f"first_40={final_loss_mask[0, :40].tolist()}"
                    )
                except Exception as e:
                    logger.debug(f"[TRACE] Step C: batch_idx={batch_idx}, error={e}")

                # 添加调试日志
                logger.debug(
                    f"[DEBUG] Sample dict created: hidden_states.shape={sample['hidden_states'].shape}, hidden_positions.shape={sample['hidden_positions'].shape}, loss_mask_sum={sample_loss_mask.sum().item()}"
                )

                if attention_mask is not None:
                    if hasattr(attention_mask, "offsets") and is_nested:
                        am_offsets = attention_mask.offsets()
                        am_start = int(am_offsets[batch_idx])
                        am_end = int(am_offsets[batch_idx + 1])
                        # NestedTensor 不能直接切片
                        attention_mask_flat = (
                            attention_mask.values()
                            if hasattr(attention_mask, "values")
                            else attention_mask
                        )
                        sample["attention_mask"] = (
                            attention_mask_flat[am_start:am_end].detach().cpu()
                        )
                    else:
                        sample["attention_mask"] = (
                            attention_mask[batch_idx].detach().cpu()
                        )

                # 按 owner_rank 分发到对应 bucket（多卡支持）
                owner = int(owner_rank[batch_idx].item())
                owner = min(owner, owner_count - 1)  # 确保不越界
                buckets[owner].append(sample)
                collected_count += 1

            if collected_count > 0:
                # 发送到 SpecoWorker
                # buckets 已经是 list[list[dict]] 格式，与 PPO 模式一致
                logger.warning(
                    f"[_speco_collect_sft_features] Sending {len(buckets)} buckets with {collected_count} samples"
                )
                self.speco_collect_rollout_features(buckets)

                # 更新统计
                self._speco_last_sft_collected_samples = collected_count
                total_elements = sum(
                    s["hidden_states"].numel() for bucket in buckets for s in bucket
                )
                self._speco_last_sft_payload_mib = total_elements * 2 / (1024 * 1024)

                logger.debug(
                    f"Collected {collected_count} samples, "
                    f"{self._speco_last_sft_payload_mib:.2f} MiB"
                )

            return collected_count

        except Exception as e:
            logger.error(f"Error collecting SFT features: {e}")
            import traceback

            traceback.print_exc()
            return 0

    @staticmethod
    def _speco_get_sequence_item(sequence, index):
        """从序列中安全获取第 index 项。"""
        if sequence is None:
            return None
        if isinstance(sequence, (list, tuple)):
            return sequence[index] if index < len(sequence) else None
        return None

    def _build_engine(self):
        """Override _build_engine to use SpecoTrainingWorker with oldlogprob patch.

        The patch MUST be installed inside the worker process (not the driver),
        because FSDPEngineWithLMHead runs in separate Ray worker processes.
        Using SpecoTrainingWorker ensures the patch is installed in each worker.
        """
        from verl.single_controller.ray import (
            RayClassWithInitArgs,
            RayResourcePool,
            RayWorkerGroup,
        )
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        from verl_speco.workers.speco_training_worker import SpecoTrainingWorker

        logger.info("[_build_engine] Using SpecoTrainingWorker with oldlogprob patch")

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        wg_kwargs = {}
        if self.start_profile_step != -1:
            wg_kwargs["profile_steps"] = list(
                range(self.start_profile_step, self.end_profile_step + 1)
            )
            if OmegaConf.select(self.config.profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )

        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        nnodes = self.config.trainer.nnodes
        # max_colocate_count=2: 允许 Drafter WorkerGroup 与 Actor WorkerGroup 共存
        # 与 PPO 模式中 resource_pool_manager 的行为一致
        self.resource_pool = RayResourcePool(
            process_on_nodes=[n_gpus_per_node] * nnodes,
            max_colocate_count=2,
        )
        # 保存供 Drafter WorkerGroup 共享使用（与 PPO 的 get_resource_pool 行为一致）
        self._speco_actor_resource_pool = self.resource_pool

        # Use SpecoTrainingWorker instead of TrainingWorker
        # This ensures oldlogprob patch is installed in each worker process
        ray_cls_with_init = RayClassWithInitArgs(
            ray.remote(SpecoTrainingWorker), config=config
        )
        self.training_client = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.config.trainer.device,
            **wg_kwargs,
        )
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.training_client.reset()

        logger.info("[_build_engine] SpecoTrainingWorker engine built successfully")

    def fit(self):
        """SFT fit 入口，挂上 SPECO hooks。

        在原生 SFT fit 的基础上，添加：
        1. Forward hook 抓取 hidden states
        2. 按间隔触发 drafter 训练
        3. 训练结束保存 drafter checkpoint

        注意：oldlogprob patch 已在 SpecoTrainingWorker.__init__ 中安装（worker 进程内），
        无需在 driver 进程中再次调用 install_oldlogprob_hidden_runtime_patch()。
        """
        logger.info("=" * 60)
        logger.info("SpecoRaySFTRayTrainer.fit() started")
        logger.info(f"SPECO SFT config: {self._speco_sft_config}")
        logger.info("=" * 60)

        if self._is_drafter_training_enabled():
            logger.info("Drafter training enabled, initializing worker group...")

            # 0. 初始化 SpecoWorker group（如果尚未初始化）
            if self.drafter_wg is None:
                self._speco_init_worker_group()

            # 1. 激活 drafter 训练模型
            self.speco_activate_drafter_training_model()

            # 2. oldlogprob patch 已在 SpecoTrainingWorker.__init__ 中安装
            #    这里不再需要在 driver 中调用
            logger.info(
                "Oldlogprob patch installed in worker processes via SpecoTrainingWorker"
            )

            # 3. 执行带 SPECO hooks 的训练
            self._speco_fit_with_hooks()
        else:
            logger.info("Drafter training disabled, running vanilla SFT")
            super().fit()

        # 训练结束后保存 drafter checkpoint
        if self._is_drafter_training_enabled():
            logger.info("SFT training finished, saving drafter checkpoint...")
            self._speco_save_final_checkpoint()

        logger.info("SpecoRaySFTRayTrainer.fit() completed")

    def _speco_fit_with_hooks(self):
        """带 SPECO hooks 的训练循环。

        在原生 SFT 训练循环的每个 step 中：
        1. 构建 collect plan
        2. 设置 collect plan 到 batch
        3. 执行 train_batch
        4. 收集 features
        5. 按间隔触发 drafter 训练
        """
        from tensordict.tensorclass import NonTensorData
        from tqdm import tqdm
        from verl.utils.tracking import Tracking

        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_step = self.resume_global_step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        total_collected = 0
        total_drafter_train_steps = 0

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch
                    if epoch == start_epoch
                    else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                )
            ):
                global_step += 1

                # === 同步 global_steps 到实例属性（关键！供其他方法引用）===
                self.global_steps = global_step

                # 设置全局 step 到 SpecoWorker
                try:
                    self.speco_set_global_step(global_step)
                except Exception:
                    pass

                # 构建 tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data).tolist()
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(
                    data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd
                )

                # === SPECO: 构建 collect plan ===
                collect_plan = self._speco_build_sft_collect_plan(data)

                # 调试日志
                if global_step <= 3 or collect_plan is not None:
                    logger.debug(
                        f"[DIAG] Step {global_step}: collect_plan={'None' if collect_plan is None else 'valid'}"
                    )
                    if collect_plan is not None:
                        logger.debug(
                            f"[DIAG] Step {global_step}: collect_mask sum={collect_plan['collect_mask'].sum().item()}/{collect_plan['collect_mask'].size(0)}"
                        )

                # === SPECO: 设置 collect plan 到 batch（用于 forward hook）===
                if collect_plan is not None:
                    # 直接设置 tensor key 到 batch（与 PPO 版本一致）
                    data[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
                    data[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan[
                        "hidden_positions"
                    ]
                    data[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan[
                        "hidden_position_mask"
                    ]
                    data[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
                    # Track original batch indices so the worker can restore
                    # sample order after dynamic micro-batching reorders them.
                    _input_ids = tu.get(data, key="input_ids")
                    if _input_ids is not None:
                        data[OLD_LOGPROB_SAMPLE_INDICES_KEY] = torch.arange(
                            _input_ids.size(0), dtype=torch.long
                        )
                    # 配置 ObjectRef 传递和 capture impl
                    tu.assign_non_tensor_data(
                        data, OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY, "forward_hook"
                    )
                    tu.assign_non_tensor_data(
                        data, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True
                    )
                    # 设置 hidden layout 和 aux layer ids（根据算法动态选择）
                    hidden_layout = self._speco_oldlogprob_hidden_layout()
                    aux_layer_ids = self._speco_oldlogprob_aux_layer_ids()
                    tu.assign_non_tensor_data(
                        data, OLD_LOGPROB_HIDDEN_LAYOUT_KEY, hidden_layout
                    )
                    tu.assign_non_tensor_data(
                        data, OLD_LOGPROB_AUX_LAYER_IDS_KEY, aux_layer_ids
                    )
                    if global_step <= 3:
                        logger.debug(
                            f"[DIAG] Step {global_step}: aux_layer_ids={aux_layer_ids}, layout={hidden_layout}"
                        )
                else:
                    # 清空所有 key
                    if OLD_LOGPROB_COLLECT_MASK_KEY in data.keys():
                        del data[OLD_LOGPROB_COLLECT_MASK_KEY]

                if self.config.trainer.balance_batch:
                    try:
                        from verl.utils.seqlen_balancing import (
                            calculate_workload,
                            get_seqlen_balanced_partitions,
                        )
                    except ImportError:
                        calculate_workload = None
                        get_seqlen_balanced_partitions = None
                    if (
                        calculate_workload is not None
                        and get_seqlen_balanced_partitions is not None
                    ):
                        global_seqlen_lst = torch.Tensor(
                            [item.size()[0] for item in data["input_ids"]]
                        )
                        global_seqlen_lst = calculate_workload(global_seqlen_lst)
                        dp_size = (
                            max(self.training_client._query_dispatch_info("train")) + 1
                        )
                        global_partition_lst = get_seqlen_balanced_partitions(
                            global_seqlen_lst, k_partitions=dp_size, equal_size=True
                        )
                    for idx, partition in enumerate(global_partition_lst):
                        partition.sort(key=lambda x: (global_seqlen_lst[x], x))
                        ordered_partition = partition[::2] + partition[1::2][::-1]
                        global_partition_lst[idx] = ordered_partition

                    global_idx = torch.tensor(
                        [j for partition in global_partition_lst for j in partition]
                    )
                    data = tu.index_select_tensor_dict(data, global_idx)
                    # Reorder collect_plan tensors to match the reordered data
                    # so token/hidden-position/owner mapping stays consistent.
                    if collect_plan is not None:
                        for _cp_key in (
                            "collect_mask",
                            "hidden_positions",
                            "hidden_position_mask",
                            "owner_rank",
                        ):
                            _cp_val = collect_plan.get(_cp_key)
                            if torch.is_tensor(_cp_val):
                                collect_plan[_cp_key] = _cp_val[global_idx]

                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                # 诊断：发送给 worker 前检查 data keys
                if global_step <= 3 and collect_plan is not None:
                    try:
                        data_keys = list(data.keys())
                        has_collect = OLD_LOGPROB_COLLECT_MASK_KEY in data_keys
                        has_positions = OLD_LOGPROB_HIDDEN_POSITIONS_KEY in data_keys
                        logger.debug(
                            f"[DIAG] Step {global_step}: Before train_batch - has_collect_mask={has_collect}, has_positions={has_positions}, keys_count={len(data_keys)}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"[DIAG] Step {global_step}: Before train_batch - error checking keys: {e}"
                        )

                # === SPECO: 判断是否需要 drafter 训练 + 【关键】在 train_batch 之前同步 lm_head ===
                # 时序：先同步 lm_head_V_a → train_batch forward 收集 hidden_V_a → drafter train 用 lm_head_V_a × hidden_V_a ✅
                # 之前是 after-sync: hidden_V_a × lm_head_V_b ← 错了！
                should_train_drafter = self._speco_should_train_drafter_this_step(
                    global_step
                )
                if should_train_drafter:
                    import time as _time

                    _sync_start = _time.perf_counter()
                    _synced = False
                    logger.warning(
                        f"[LM-HEAD-SYNC] Step {global_step}: SYNC BEFORE train_batch (correct timing)"
                    )

                    try:
                        # 1. 获取 row_indices (drafter vocab subset)
                        row_indices = None
                        try:
                            row_infos_raw = self._ray_get_if_needed(
                                self._require_speco_worker_group().get_drafter_target_lm_head_row_indices()
                            )
                            if isinstance(row_infos_raw, dict):
                                row_infos = [row_infos_raw]
                            elif isinstance(row_infos_raw, list):
                                row_infos = row_infos_raw
                            else:
                                row_infos = []
                            non_null_infos = [
                                info
                                for info in row_infos
                                if isinstance(info, dict)
                                and info.get("row_indices") is not None
                            ]
                            if non_null_infos:
                                first_info = non_null_infos[0]
                                row_indices = first_info.get("row_indices")
                                if hasattr(row_indices, "detach"):
                                    row_indices = (
                                        row_indices.detach().cpu().long().reshape(-1)
                                    )
                                elif isinstance(row_indices, (list, tuple)):
                                    import torch as _t

                                    row_indices = _t.tensor(
                                        [int(i) for i in row_indices], dtype=_t.long
                                    )
                        except Exception as e_ri:
                            logger.warning(
                                f"[LM-HEAD-SYNC] Failed to get row_indices: {e_ri}"
                            )

                        # 2. 从 SFT worker 导出当前 lm_head 权重
                        try:
                            _lm_head_export_ref = (
                                self.training_client.export_lm_head_weight_for_drafter(
                                    row_indices
                                )
                            )
                            payloads = self._ray_get_if_needed(_lm_head_export_ref)
                            if isinstance(payloads, list):
                                payload = next(
                                    (p for p in payloads if p is not None), None
                                )
                            else:
                                payload = payloads

                            if payload is not None:
                                w = payload.get("weight")
                                logger.warning(
                                    f"[LM-HEAD-SYNC] Exported lm_head: shape={tuple(w.shape) if hasattr(w, 'shape') else 'unknown'}"
                                )
                                _sync_result = self._ray_get_if_needed(
                                    self.speco_sync_target_lm_head_weight(
                                        payload, global_step=global_step
                                    )
                                )
                                _synced = True
                                logger.warning(
                                    f"[LM-HEAD-SYNC] Sync OK: {_sync_result}"
                                )
                            else:
                                logger.warning(
                                    "[LM-HEAD-SYNC] Export returned None (OK on non-rank-0)"
                                )
                        except Exception as e_export:
                            logger.warning(
                                f"[LM-HEAD-SYNC] Export/sync failed: {e_export}"
                            )

                    except Exception as e_sync:
                        logger.warning(f"[LM-HEAD-SYNC] Sync block error: {e_sync}")

                    _sync_elapsed = _time.perf_counter() - _sync_start
                    logger.warning(
                        f"[LM-HEAD-SYNC] Step {global_step}: synced={_synced}, elapsed={_sync_elapsed:.3f}s"
                    )

                # === 执行训练 ===
                output = self.training_client.train_batch(data)
                output = output.get()

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                # NOTE: Hidden refs are now injected directly into output by
                # patched engine.train_batch in SpecoTrainingWorker.
                # No need for separate get_speco_hidden_refs() call.

                # === SPECO: 收集 features ===
                if collect_plan is not None:
                    # 诊断：检查 output 里是否有 hidden refs
                    if global_step <= 3:
                        out_keys = (
                            list(output.keys())
                            if hasattr(output, "keys")
                            else "no keys attr"
                        )
                        has_hidden_refs = (
                            OLD_LOGPROB_HIDDEN_REFS_KEY in output.keys()
                            if hasattr(output, "keys")
                            else False
                        )
                        has_hidden_states = (
                            OLD_LOGPROB_HIDDEN_STATES_KEY in output.keys()
                            if hasattr(output, "keys")
                            else False
                        )
                        logger.debug(
                            f"[DIAG] Step {global_step}: output has hidden_refs={has_hidden_refs}, has_hidden_states={has_hidden_states}, output keys={out_keys[:10]}"
                        )

                    collected = self._speco_collect_sft_features(
                        data, collect_plan, output
                    )
                    total_collected += collected
                    if global_step <= 3 or collected > 0:
                        logger.debug(
                            f"[DIAG] Step {global_step}: collected {collected} samples"
                        )
                    # 调试日志
                    if global_step <= 3 or collected > 0:
                        logger.info(
                            f"Step {global_step}: collected {collected} samples"
                        )

                # === SPECO: 按间隔触发 drafter 训练 ===
                # lm_head 已在 train_batch 之前同步 ✅
                if should_train_drafter:
                    logger.warning(f"{'=' * 60}")
                    logger.warning(
                        f"[Drafter Training] Step {global_step}: STARTING (lm_head synced before train_batch)"
                    )
                    logger.warning(f"{'=' * 60}")
                    try:
                        import time

                        start_time = time.time()
                        train_result = self.speco_train_drafter()
                        elapsed = time.time() - start_time
                        total_drafter_train_steps += 1
                        logger.warning(
                            f"[Drafter Training] Step {global_step}: COMPLETED in {elapsed:.2f}s, Result: {train_result}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[Drafter Training] Step {global_step}: FAILED: {e}"
                        )
                        import traceback

                        traceback.print_exc()

                metrics = tu.get(output, "metrics")
                if should_train_drafter:
                    metrics["drafter/target_lm_head_synced"] = int(
                        _synced if "_synced" in dir() else 0
                    )
                metrics["train/loss"] = metrics.pop("loss")
                metrics["train/grad_norm"] = metrics.pop("grad_norm")
                metrics["train/lr"] = metrics.pop("lr")
                metrics["train/mfu"] = metrics.pop("mfu")
                metrics["train/global_tokens"] = torch.sum(
                    torch.tensor(batch_seqlens, device=self.device_name)
                ).item()
                total_tokens += metrics["train/global_tokens"]
                metrics["train/total_tokens(B)"] = total_tokens / 1e9

                # 添加 SPECO 相关指标
                metrics["speco/collected_samples"] = total_collected
                metrics["speco/drafter_train_steps"] = total_drafter_train_steps
                metrics["speco/drafter_should_train"] = int(should_train_drafter)
                if hasattr(self, "_speco_last_sft_payload_mib"):
                    metrics["speco/payload_mib"] = self._speco_last_sft_payload_mib

                tracking.log(data=metrics, step=global_step)

                # === 新增：显存清理（关键！）===
                import gc

                gc.collect()
                device_module = get_torch_device()
                if (
                    hasattr(device_module, "is_available")
                    and device_module.is_available()
                ):
                    device_module.empty_cache()
                # ==================================

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                if (
                    is_last_step
                    and self.val_dataloader is not None
                    or (self.test_freq > 0 and is_valid_step)
                ):
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(
                            tensor_dict=val_data, non_tensor_dict=meta_info
                        )
                        val_output = self.training_client.infer_batch(val_data)
                        val_output = val_output.get()
                        val_metrics = tu.get(val_output, "metrics")
                        val_losses.append(val_metrics["loss"])

                    val_loss = torch.mean(
                        torch.tensor(val_losses, device=self.device_name)
                    )
                    metric = {"val/loss": val_loss.detach().item()}
                    tracking.log(data=metric, step=global_step)
                    last_valid_metric = metric

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)
                    if self._is_drafter_training_enabled():
                        self.speco_save_checkpoint(global_step=global_step, wait=True)
                        logger.info(f"Drafter checkpoint saved at step {global_step}")

                if is_last_step:
                    logger.info(f"Total collected samples: {total_collected}")
                    logger.info(
                        f"Total drafter train steps: {total_drafter_train_steps}"
                    )
                    print(f"Total time for train steps: {train_time:.2f}s")
                    print(f"Final validation metrics: {last_valid_metric}")
                    return

    def _speco_save_final_checkpoint(self):
        """保存最终的 drafter checkpoint。"""
        save_dir = self._speco_checkpoint_save_dir()
        if save_dir is None:
            logger.warning("No checkpoint_save_dir specified, skipping final save")
            return

        # 使用 self.global_steps（训练循环中已同步）；若训练循环提前 return 则用 resume_global_step 兜底
        final_step = getattr(self, "global_steps", None) or getattr(
            self, "resume_global_step", 0
        )
        self.speco_save_checkpoint(global_step=final_step, wait=True)
        logger.info(f"Drafter checkpoint saved to: {save_dir}")
