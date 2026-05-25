import os
import sys
import torch
import torch.distributed as dist
import asyncio
import unittest
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from transformers import AutoConfig

from verl.workers.drafter.eagle_trainer_backend import EagleTrainerBackend
from verl.workers.drafter.base_trainer import DrafterBaseTrainer
from verl.workers.drafter.model.eagle import LlamaForCausalLMEagle


def setup_real_dist():
    # 1. 初始化分布式环境（FSDP2 依赖）
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
        
    return dist.get_rank()
    
def test_eagle_training_flow():
    # 1、初始化分布式环境
    rank = setup_real_dist()
    device = torch.device(f"cuda:{rank}")
    world_size = dist.get_world_size()
       
    # 2、配置加载
    config = OmegaConf.load("/nas/disk6/ls/workspace/verl-spac-2/tests/eagle_trainer/config/eagle3_trainer.yaml")

    # 3、准备模型配置
    target_config = AutoConfig.from_pretrained("/model/Llama-3.2-1B/", trust_remote_code=True)

    # 4. 实例化 Backend
    backend = EagleTrainerBackend(
        config=config,
        target_model_config=target_config,
    )

    # 5. 实例化 Trainer
    trainer = DrafterBaseTrainer(
        config=config,
        world_size=world_size,
        rollout_dp_rank=rank,
        backend=backend
    )

    # 6. 模型初始化
    print("--- 步骤 1: 构建模型 ---")
    trainer.build_draft_model()

    # 7. 数据采集测试
    print("--- 步骤 2: 采集在线数据 ---")
    # 定义 batch 大小和序列长度
    batch_size = 2  # 对应你 config 里的 batch_size_per_gpu
    seq_len = 256
    prompt_len = 128
    response_len = 128 # 确保 prompt_len + response_len = seq_len
    hidden_dim = target_config.hidden_size

    # 构造 Mock 数据，所有 Tensor 第一维必须是 batch_size
    mock_batch = {
        "input_ids": torch.randint(0, target_config.vocab_size, (batch_size, seq_len)),
        "responses": torch.randint(0, target_config.vocab_size, (batch_size, response_len)),
        "prompts": torch.randint(0, target_config.vocab_size, (batch_size, prompt_len)),
    }
    # 模拟隐藏层状态
    mock_hiddens_states = [torch.randn(batch_size, seq_len, hidden_dim)]

    trainer.collect_online_data(mock_batch, mock_hiddens_states)
    print(f"Data collect. Buffer size: {len(trainer.data_buffer)}")

    # 采集两组数据以满足 batch_size=2
    # trainer.collect_online_data(mock_batch, mock_h_states)
    # trainer.collect_online_data(mock_batch, mock_h_states)

    # self.assertEqual(len(trainer.collected_data), 2)

    # 8. 训练步执行测试
    print("--- 步骤 3: 执行训练步 ---")
    import asyncio
    async def run_step():
        success = await trainer.training_step(step=1)
        return success
    
    loop = asyncio.get_event_loop() 
    step_success = loop.run_until_complete(run_step()) 

    if step_success: 
        print("Congratulations! The training_step finished successfully.") 
        print(f"Current training steps counter: {trainer.training_steps}") 
    else: 
        print("training_step failed (check logs for 'Not enough data' or other warnings).") 
    print("--- 集成测试完成！ ---")


if __name__ == "__main__":
    test_eagle_training_flow()

    

