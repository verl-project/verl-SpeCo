import logging
import os
import glob
from copy import deepcopy
import safetensors

import torch
from torch.nn import SmoothL1Loss
from torch.nn import functional as F

from .model.auto import AutoDraftModelConfig, AutoEagleDraftModel

from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _masked_soft_cross_entropy(
    logits: torch.Tensor,
    target_p: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    target_p = target_p.float()
    finite_logits = torch.isfinite(logits).all(dim=-1)
    finite_target = torch.isfinite(target_p).all(dim=-1) & (target_p.sum(dim=-1) > 0)
    valid_position = (loss_mask > 0) & finite_logits & finite_target

    safe_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    safe_target = torch.where(torch.isfinite(target_p), target_p, torch.zeros_like(target_p))
    safe_target = torch.where(valid_position.unsqueeze(-1), safe_target, torch.zeros_like(safe_target))

    log_probs = F.log_softmax(safe_logits, dim=-1)
    per_token_ploss = -(safe_target * log_probs).sum(dim=-1)
    per_token_ploss = torch.where(valid_position, per_token_ploss, torch.zeros_like(per_token_ploss))
    return per_token_ploss, valid_position


class EagleTrainerBackend:
    def __init__(
        self,
        config,
        target_model_config
    ):
        self.config = config
        self.target_model_config = target_model_config

        self.criterion = SmoothL1Loss(reduction="none")

    @property
    def model_type(self):
        return "eagle"

    def build_model(self):
        """build draft model"""
        logger.info(f"Initializing Eagle model with type: {self.target_model_config.model_type}")
        spec_model_path = self.config.rollout.drafter.model_path
        config_path = os.path.join(spec_model_path, "config.json")

        # 1、加载 Config
        if os.path.exists(config_path):
            drafter_config = AutoDraftModelConfig.from_file(config_path)
        else:
            drafter_config = deepcopy(self.target_model_config)
            drafter_config.num_hidden_layers = 1
            drafter_config.torch_dtype = torch.bfloat16
            drafter_config.tie_word_embeddings = False
            drafter_config.architectures = ["LlamaForCausalLMEagle"]

        factory_cls = AutoEagleDraftModel
        
        drafter_module = factory_cls.from_config(drafter_config)

        # Initialize model
        if spec_model_path and os.path.exists(spec_model_path):
            drafter_module = factory_cls.from_pretrained(spec_model_path, ignore_mismatched_sizes = True)

        
        # 复用主模型的Embedding和LM_Head
        target_model_path = self.config.model.path
        logger.info("Start load lm_head for eagle")
        drafter_module.load_lm_head(target_model_path)
        drafter_module.freeze_lm_head()
        
        drafter_module.load_embedding(target_model_path)
        drafter_module.freeze_embedding()

        return drafter_module, drafter_config
    
    def setup_optimizer(self, drafter_model, drafter_train_config):
        trainable_params = [p for p in drafter_model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=drafter_train_config.lr,
            betas=(0.9, 0.95),
            weight_decay=drafter_train_config.get("weight_decay", 1e-2),
        )

        return optimizer


    def setup_scheduler(self, optimizer, train_cfg):
        total_steps = train_cfg.get("step", 0)
        num_warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))
        warmup_style = train_cfg.get("warmup_style", "constant")

        if warmup_style == "constant":
            return get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=train_cfg.get("min_lr_ratio", 0.0),
                num_cycles=train_cfg.get("num_cycles", 0.5),
            )
        # elif warmup_style == "linear":
        #     return get_linear_schedule_with_warmup(
        #         optimizer=optimizer,
        #         num_warmup_steps=num_warmup_steps,
        #         num_training_steps=total_steps,
        #     )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")
        
    def preprocess_individual_items(self, items, device, model_config):
        """
        针对单条数据：裁剪窗口、生成Mask、确保维度对齐
        """
        res = {'ids':[], 'h_states':[], 'masks': [], 'position_ids': []}
        pad_id = int(getattr(model_config, "pad_token_id", 0) or 0)

        for item in items:
            # 1. 搬运到GPU
            ids = item["input_ids"].to(device, non_blocking=True)
            raw_h = item["hidden_states"]

            if isinstance(raw_h, (list, tuple)):
                # 将hidden_states进行拼接
                h_states = torch.cat(raw_h, dim=-1).to(device, dtype=torch.bfloat16)
            else:
                h_states = raw_h.to(device, dtype=torch.bfloat16)

            # Compute loss_mask if not present (for DataBuffer items)
            full_len = ids.size(0)
            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                if "prompts" in item and "responses" in item:
                    prompt_len = item["prompts"].size(0)
                    response_len = item["responses"].size(0)
                    for j in range(response_len):
                        token_idx = prompt_len + j
                        if token_idx < full_len and item["responses"][j] != pad_id:
                            item_loss_mask[token_idx] = 1.0
                elif "responses" in item:
                    response_start = full_len - item["responses"].size(0)
                    response_mask = (item["responses"] != pad_id).float()
                    item_loss_mask[response_start:] = response_mask
                else:
                    # If no response info, assume all tokens are valid
                    item_loss_mask[:] = 1.0
            else:
                item_loss_mask = item["loss_mask"].to(device, dtype=torch.float32, non_blocking=True)
            item_position_ids = item.get("position_ids")
            if item_position_ids is None:
                item_position_ids = torch.arange(full_len, device=device, dtype=torch.long)
            else:
                item_position_ids = item_position_ids.to(device, dtype=torch.long, non_blocking=True)
            
            start = 0
            end = full_len
            res['ids'].append(ids[start:end])
            res['h_states'].append(h_states[start:end])
            res['masks'].append(item_loss_mask[start:end])
            res['position_ids'].append(item_position_ids[start:end])
        
        return res
    
    def compute_loss(self, model, batch, _current_pad_size, logits=None):
        """
        计算 Eagle 特有的V-Loss 和 P-Loss
        """
        # 前向传播
        draft_model = model.module if hasattr(model, "module") else model
        outputs = model(
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden_states"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        )

        hidden_states = outputs["hidden_states"]
        logits = outputs["logits"]

        # Gather outputs if using Ulysses SP
        if getattr(self, "use_ulysses_sp", False):
            from verl.utils.ulysses import gather_outputs_and_unpad

            hidden_states = gather_outputs_and_unpad(
                hidden_states.squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)

            logits = gather_outputs_and_unpad(
                logits.squeeze(0), gather_dim=0, unpad_dim=0, padding_size=_current_pad_size
            ).unsqueeze(0)

            target = gather_outputs_and_unpad(
                batch["target"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)

            loss_mask = gather_outputs_and_unpad(
                batch["loss_mask"].squeeze(0),
                gather_dim=0,
                unpad_dim=0,
                padding_size=_current_pad_size,
            ).unsqueeze(0)
        else:
            target = batch["target"]
            loss_mask = batch["loss_mask"]

        # V-Loss：隐藏态回归损失
        safe_hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        vloss_all = self.criterion(safe_hidden_states, safe_target)  # [B,T,H]
        vloss_per_token = vloss_all.mean(dim=-1) # [B, T]
        finite_vloss = torch.isfinite(vloss_per_token)

        # P-Loss: 概率分布对齐损失
        with torch.no_grad():
            target_p = F.softmax(draft_model.lm_head(safe_target), dim=-1)

        ploss_per_token, valid_ploss = _masked_soft_cross_entropy(logits, target_p, loss_mask)
        valid_position = valid_ploss & finite_vloss
        if (loss_mask > 0).any() and not valid_position[loss_mask > 0].all():
            dropped_tokens = ((loss_mask > 0) & ~valid_position).sum()
            logger.debug(
                "Dropping %s EAGLE target positions with non-finite vloss, logits or targets",
                int(dropped_tokens.detach().cpu().item()),
            )
        vloss_per_token = torch.where(valid_position, vloss_per_token, torch.zeros_like(vloss_per_token))
        ploss_per_token = torch.where(valid_position, ploss_per_token, torch.zeros_like(ploss_per_token))
        valid_loss_mask = valid_position.float()

        # 结合 Mask
        total_local_vloss = (vloss_per_token * valid_loss_mask).sum()
        total_local_ploss = (ploss_per_token * valid_loss_mask).sum()
        local_num_tokens = valid_loss_mask.sum()

        # 读取权重并返回 Loss 字典
        train_config = getattr(self, "train_config", self.config.rollout.drafter.training)
        w_v = float(train_config.get("vloss_weight", 0.5))
        w_p = float(train_config.get("ploss_weight", 0.5))

        return {
            "total_local_vloss": total_local_vloss,
            "total_local_ploss": total_local_ploss,
            "local_num_tokens": local_num_tokens,
            "v_weight": w_v,
            "p_weight": w_p
        }
