import asyncio
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM

from verl.workers.drafter.base_trainer import DrafterBaseTrainer
from verl.workers.drafter.eagle3_trainer_backend import Eagle3TrainerBackend, _masked_soft_cross_entropy
from verl.workers.drafter.model.eagle import LlamaForCausalLMEagle3


def _init_dist() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("EAGLE3 multi-card integration test requires CUDA")
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, for example: torchrun --standalone --nproc-per-node=2 this_file.py")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def _create_tiny_checkpoints(root: Path) -> tuple[Path, Path]:
    target_dir = root / "target"
    drafter_dir = root / "drafter"
    target_dir.mkdir(parents=True, exist_ok=True)
    drafter_dir.mkdir(parents=True, exist_ok=True)

    target_config = _tiny_llama_config()
    target_model = LlamaForCausalLM(target_config)
    target_model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="10KB")

    drafter_config = _tiny_llama_config()
    drafter_config.architectures = ["LlamaForCausalLMEagle3"]
    drafter_config.draft_vocab_size = drafter_config.vocab_size
    drafter_config.target_hidden_size = target_config.hidden_size
    drafter_config.num_hidden_layers = 1
    drafter_model = LlamaForCausalLMEagle3(drafter_config)
    drafter_model.save_pretrained(drafter_dir, safe_serialization=True, max_shard_size="10KB")

    return target_dir, drafter_dir


def _create_compressed_vocab_checkpoints(root: Path) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    target_dir = root / "target_compressed_vocab"
    drafter_dir = root / "drafter_compressed_vocab"
    target_dir.mkdir(parents=True, exist_ok=True)
    drafter_dir.mkdir(parents=True, exist_ok=True)

    target_config = _tiny_llama_config()
    target_model = LlamaForCausalLM(target_config)
    target_model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="10KB")

    drafter_config = _tiny_llama_config()
    drafter_config.architectures = ["LlamaForCausalLMEagle3"]
    drafter_config.draft_vocab_size = 8
    drafter_config.target_hidden_size = target_config.hidden_size
    drafter_config.num_hidden_layers = 1
    drafter_model = LlamaForCausalLMEagle3(drafter_config)

    selected_tokens = torch.tensor([0, 2, 5, 7, 11, 13, 17, 19], dtype=torch.long)
    t2d = torch.zeros(target_config.vocab_size, dtype=torch.bool)
    t2d[selected_tokens] = True
    d2t = selected_tokens - torch.arange(drafter_config.draft_vocab_size, dtype=torch.long)
    drafter_model.t2d.copy_(t2d)
    drafter_model.d2t.copy_(d2t)
    drafter_model.save_pretrained(drafter_dir, safe_serialization=True, max_shard_size="10KB")

    return target_dir, drafter_dir, t2d, d2t


def _build_config(target_dir: Path, drafter_dir: Path):
    return OmegaConf.create(
        {
            "model": {
                "path": str(target_dir),
                "local_hf_config_path": str(target_dir),
                "trust_remote_code": False,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "EAGLE3",
                    "model_path": str(drafter_dir),
                    "checkpoint_path": None,
                    "training": {
                        "collect_hidden_states_from_sgl": True,
                        "use_data_buffer": False,
                        "batch_size_per_gpu": 2,
                        "step": 100,
                        "lr": 1e-4,
                        "lr_warmup_steps": 0,
                        "warmup_style": "constant",
                        "use_logits": False,
                        "ttt_length": 2,
                        "current_max_samples": 16,
                        "data_buffer_max_size": 16,
                        "fsdp_config": {
                            "wrap_policy": {"min_num_params": 0},
                            "use_orig_params": True,
                            "forward_prefetch": False,
                        },
                    },
                },
            },
        }
    )


def test_eagle3_build_model_uses_checkpoint_vocab_mapping_without_mapping_path():
    root = Path(tempfile.mkdtemp(prefix="verl_eagle3_vocab_mapping_"))
    try:
        target_dir, drafter_dir, expected_t2d, expected_d2t = _create_compressed_vocab_checkpoints(root)
        config = _build_config(target_dir, drafter_dir)
        config.rollout.drafter.training.use_logits = True

        target_config = AutoConfig.from_pretrained(target_dir)
        backend = Eagle3TrainerBackend(config=config, target_model_config=target_config)
        drafter_model, _ = backend.build_model()

        assert drafter_model.draft_vocab_size == int(expected_t2d.sum().item())
        assert torch.equal(drafter_model.t2d.cpu(), expected_t2d)
        assert torch.equal(drafter_model.d2t.cpu(), expected_d2t)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_eagle3_masked_ploss_keeps_bad_logits_out_of_backward():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = torch.nan
        logits[0, 2, 1] = torch.inf

    target_p = torch.zeros_like(logits)
    target_p[0, 0, 2] = 1.0
    target_p[0, 1, 3] = 1.0
    target_p[0, 2, 4] = 1.0
    position_mask = torch.tensor([[0.0, 1.0, 1.0]])

    per_token_ploss, valid_position = _masked_soft_cross_entropy(
        logits=logits,
        target_p=target_p,
        position_mask=position_mask,
    )
    loss = per_token_ploss.sum()

    assert torch.isfinite(loss)
    assert valid_position.tolist() == [[False, True, False]]

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0, 0], torch.zeros_like(logits.grad[0, 0]))
    assert torch.equal(logits.grad[0, 2], torch.zeros_like(logits.grad[0, 2]))


def test_drafter_training_batch_sanitizer_masks_bad_hidden_rows():
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.backend = type("Backend", (), {"model_type": "eagle3"})()
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "hidden_state_clip_value": 10.0,
                        "ttt_length": 2,
                    }
                }
            }
        }
    )

    hidden_states = torch.ones(1, 4, 3)
    hidden_states[0, 1, 0] = torch.nan
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "hidden_states": hidden_states,
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "target_logprobs": torch.zeros(1, 4, 2, 2),
    }

    sanitized = trainer._sanitize_training_batch(batch)

    assert torch.isfinite(sanitized["hidden_states"]).all()
    assert sanitized["attention_mask"].tolist() == [[1, 0, 1, 1]]
    assert sanitized["loss_mask"].tolist() == [[1.0, 0.0, 0.0, 1.0]]


def test_drafter_trainable_state_dict_skips_non_floating_buffers():
    class TinyDrafter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.register_buffer("t2d", torch.ones(4, dtype=torch.bool))
            self.register_buffer("d2t", torch.arange(4, dtype=torch.long))

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = TinyDrafter()
    trainer.training_device_mesh = None
    trainer.backend = type("Backend", (), {"model_type": "eagle3"})()
    trainer._frozen_param_names = set()

    trainable_state = trainer._get_trainable_state_dict()

    assert set(trainable_state) == {"weight"}
    assert trainable_state["weight"].shape == (2, 2)


def test_drafter_prepare_training_batch_uses_current_step_collected_data_only():
    class TinyDrafter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class Backend:
        model_type = "eagle"

        def preprocess_individual_items(self, items, device, model_config):
            return {
                "ids": [item["input_ids"].to(device) for item in items],
                "h_states": [item["hidden_states"].to(device) for item in items],
                "masks": [item["loss_mask"].to(device) for item in items],
                "position_ids": [item["position_ids"].to(device) for item in items],
                "last_h_states": [],
                "target_logprobs": [],
            }

    def make_item(step: int, token_base: int):
        return {
            "step": step,
            "input_ids": torch.tensor([token_base, token_base + 1, token_base + 2], dtype=torch.long),
            "hidden_states": torch.full((3, 2), float(token_base)),
            "loss_mask": torch.ones(3),
            "position_ids": torch.arange(3, dtype=torch.long),
        }

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 0
    trainer.current_rl_step = 2
    trainer.use_data_buffer = False
    trainer.batch_size = 2
    trainer.use_ulysses_sp = False
    trainer.backend = Backend()
    trainer.model = TinyDrafter()
    trainer.model_config = type("ModelConfig", (), {"hidden_size": 2})()
    trainer.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "use_logits": False,
                        "hidden_state_clip_value": None,
                        "ttt_length": 1,
                    }
                }
            }
        }
    )
    trainer.collected_data = deque([make_item(1, 10), make_item(2, 20)])

    batch = trainer._prepare_training_batch()

    assert batch is not None
    assert batch["input_ids"].tolist() == [[20, 21]]


def _mock_rollout_batch(config: LlamaConfig, batch_size: int = 2, seq_len: int = 24, prompt_len: int = 8):
    response_len = seq_len - prompt_len
    input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len), device="cuda")
    prompts = input_ids[:, :prompt_len].contiguous()
    responses = input_ids[:, prompt_len:].contiguous()

    hidden_states = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size * 4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return {
        "input_ids": input_ids,
        "prompts": prompts,
        "responses": responses,
    }, hidden_states


def test_eagle3_multigpu_model_and_training_flow():
    _, world_size = _init_dist()
    root = Path(os.environ.get("EAGLE3_TEST_TMPDIR", tempfile.gettempdir())) / (
        f"verl_eagle3_tiny_{os.environ.get('MASTER_PORT', 'standalone')}"
    )

    if dist.get_rank() == 0:
        shutil.rmtree(root, ignore_errors=True)
        target_dir, drafter_dir = _create_tiny_checkpoints(root)
    dist.barrier()
    target_dir, drafter_dir = root / "target", root / "drafter"

    config = _build_config(target_dir, drafter_dir)
    target_config = AutoConfig.from_pretrained(target_dir)
    backend = Eagle3TrainerBackend(config=config, target_model_config=target_config)
    trainer = DrafterBaseTrainer(
        config=config,
        world_size=world_size,
        rollout_dp_rank=dist.get_rank(),
        training_device_mesh=None,
        backend=backend,
        training_process_group=dist.group.WORLD,
    )

    try:
        assert asyncio.run(trainer.activate_training_model())
        assert isinstance(backend.target_model, torch.nn.Module)

        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        assert raw_model.draft_vocab_size == target_config.vocab_size
        assert raw_model.target_hidden_size == target_config.hidden_size

        batch, hidden_states = _mock_rollout_batch(target_config)
        trainer.collect_online_data(batch, hidden_states)
        assert len(trainer.collected_data) == batch["input_ids"].size(0)

        train_batch = trainer._prepare_training_batch()
        assert train_batch is not None
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = trainer.backend.compute_loss(trainer.model, train_batch, trainer._current_pad_size)
        assert outputs["total_local_ploss"].requires_grad
        assert outputs["local_num_tokens"].item() > 0

        assert asyncio.run(trainer.training_step(step=1))
        assert trainer.training_steps == 1
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root, ignore_errors=True)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    test_eagle3_multigpu_model_and_training_flow()
