import asyncio
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig

from verl.workers.drafter.base_trainer import DrafterBaseTrainer
from verl.workers.drafter.dflash_trainer_backend import DFlashTrainerBackend, DFlashTrainingModel


TARGET_MODEL_PATH = os.environ.get("DFLASH_TARGET_MODEL_PATH", "/nas/disk1/Qwen3-8B")
DRAFT_MODEL_PATH = os.environ.get("DFLASH_DRAFT_MODEL_PATH", "/nas/disk1/Qwen3-8B-DFlash")
DRAFT_CHECKPOINT_PATH = os.environ.get("DFLASH_DRAFT_CHECKPOINT_PATH", "/nas/disk6/ls/test_spec/dflash_checkpoint-1")


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("VERL_LOGGING_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [rank=%(rank)s] %(name)s:%(lineno)d - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        return record

    logging.setLogRecordFactory(record_factory)


def _init_dist() -> tuple[int, int]:
    if not torch.cuda.is_available():
        pytest.skip("DFlash multi-card integration test requires CUDA")
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun --standalone --nproc-per-node=2 tests/eagle_trainer/trainer/test_dflash_trainer.py")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _require_model_paths() -> None:
    missing = [path for path in (TARGET_MODEL_PATH, DRAFT_MODEL_PATH) if not os.path.exists(path)]
    if missing:
        pytest.skip(f"Missing model path(s): {missing}")


def _build_config(target_model_path: str, draft_model_path: str, checkpoint_path: str):
    return OmegaConf.create(
        {
            "model": {
                "path": target_model_path,"local_hf_config_path": target_model_path,
                "trust_remote_code": True,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "DFLASH",
                    "model_path": draft_model_path,
                    "checkpoint_path": checkpoint_path,
                    "training": {
                        "collect_hidden_states_from_sgl": True,
                        "use_data_buffer": False,
                        "batch_size_per_gpu": int(os.environ.get("DFLASH_TEST_BATCH_SIZE", "2")),
                        "step": 100,
                        "lr": float(os.environ.get("DFLASH_TEST_LR", "1e-6")),
                        "lr_warmup_steps": 0,
                        "warmup_style": "constant",
                        "dflash_block_size": int(os.environ.get("DFLASH_TEST_BLOCK_SIZE", "4")),
                        "dflash_num_anchors": int(os.environ.get("DFLASH_TEST_NUM_ANCHORS", "4")),
                        "dflash_loss_decay_gamma": 7.0,
                        "dflash_max_window": int(os.environ.get("DFLASH_TEST_MAX_WINDOW", "64")),
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


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _mock_rollout_batch(target_config, dflash_model: DFlashTrainingModel):
    text_config = getattr(target_config, "text_config", target_config)
    batch_size = int(os.environ.get("DFLASH_TEST_BATCH_SIZE", "2"))
    seq_len = int(os.environ.get("DFLASH_TEST_SEQ_LEN", "32"))
    prompt_len = int(os.environ.get("DFLASH_TEST_PROMPT_LEN", "8"))
    response_len = seq_len - prompt_len
    if response_len <= dflash_model.block_size:
        raise ValueError(
            f"DFLASH_TEST_SEQ_LEN must leave response_len > block_size, got response_len={response_len}, "
            f"block_size={dflash_model.block_size}"
        )

    vocab_size = int(text_config.vocab_size)
    hidden_dim = int(dflash_model.draft_model.target_hidden_size * dflash_model.draft_model.num_target_layers)
    input_ids = torch.randint(3, vocab_size - 1, (batch_size, seq_len), device="cuda")
    prompts = input_ids[:, :prompt_len].contiguous()
    responses = input_ids[:, prompt_len:].contiguous()
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    return {"input_ids": input_ids, "prompts": prompts, "responses": responses}, hidden_states


def _assert_model_built(trainer: DrafterBaseTrainer, backend: DFlashTrainerBackend, target_config) -> DFlashTrainingModel:
    raw_model = _unwrap_model(trainer.model)
    assert isinstance(raw_model, DFlashTrainingModel), f"Expected DFlashTrainingModel, got {type(raw_model)}"
    assert raw_model.draft_model.num_target_layers > 0
    assert raw_model.draft_model.target_hidden_size == getattr(getattr(target_config, "text_config", target_config), "hidden_size")
    assert raw_model.block_size > 1
    assert raw_model.num_anchors > 0
    assert backend.target_lm_head is not None
    first_param = next(trainer.model.parameters())
    assert first_param.device.type == "cuda", f"model is not on cuda: {first_param.device}"
    return raw_model


def _assert_loss_outputs(outputs: dict, raw_model: DFlashTrainingModel) -> None:
    required = {
        "total_local_vloss",
        "total_local_ploss",
        "local_num_tokens",
        "v_weight",
        "p_weight",
        "accuracy",
        "loss_per_position",
        "acc_per_position",
        "count_per_position",
    }
    missing = required - set(outputs)
    assert not missing, f"compute_loss output missing keys: {missing}"
    assert outputs["total_local_ploss"].requires_grad
    assert torch.isfinite(outputs["total_local_ploss"]).item()
    assert torch.isfinite(outputs["local_num_tokens"]).item()
    assert outputs["local_num_tokens"].item() > 0
    assert outputs["loss_per_position"].numel() == raw_model.block_size
    assert outputs["acc_per_position"].numel() == raw_model.block_size
    assert outputs["count_per_position"].numel() == raw_model.block_size
    assert outputs["count_per_position"][1:].sum().item() > 0


def _assert_checkpoint_saved(checkpoint_root: Path, step: int, logger: logging.Logger) -> None:
    checkpoint_file = checkpoint_root / f"eagle_step_{step}" / "model.pt"
    assert checkpoint_file.exists(), f"checkpoint file was not saved: {checkpoint_file}"
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    assert checkpoint.get("step") == step, f"checkpoint step mismatch: {checkpoint.get('step')} != {step}"
    model_state = checkpoint.get("model")
    assert isinstance(model_state, dict) and model_state, "checkpoint model state is empty"
    assert all(torch.is_tensor(value) for value in model_state.values()), "checkpoint model state contains non-tensors"
    assert not any("embed_tokens.weight" in name for name in model_state), "frozen DFlash embeddings should not be saved"
    expected_fragments = ("context_proj.weight", "context_norm.weight", "final_norm.weight", "layers.")
    assert any(any(fragment in name for fragment in expected_fragments) for name in model_state), (
        f"checkpoint does not contain expected DFlash trainable keys; sample={list(model_state)[:8]}"
    )
    optimizer_state = checkpoint.get("optimizer")
    assert isinstance(optimizer_state, dict), "checkpoint optimizer state is not a dict"
    model_numel = sum(value.numel() for value in model_state.values())
    assert model_numel > 0, "checkpoint model tensors are empty"
    logger.info(
        "checkpoint save ok: path=%s model_keys=%s model_numel=%s optimizer_keys=%s",
        checkpoint_file,
        len(model_state),
        model_numel,
        list(optimizer_state.keys()),
    )


def test_dflash_multigpu_model_and_training_flow():
    _setup_logging()
    local_rank, world_size = _init_dist()
    _require_model_paths()
    logger = logging.getLogger(__name__)
    trainer = None
    backend = None
    cleanup_done = False
    checkpoint_root = Path(DRAFT_CHECKPOINT_PATH)

    try:
        logger.info(
            "Starting DFlash test with target=%s draft=%s local_rank=%s world_size=%s",
            TARGET_MODEL_PATH,
            DRAFT_MODEL_PATH,
            local_rank,
            world_size,
        )
        config = _build_config(TARGET_MODEL_PATH, DRAFT_MODEL_PATH, DRAFT_CHECKPOINT_PATH)
        target_config = AutoConfig.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
        backend = DFlashTrainerBackend(config=config, target_model_config=target_config)
        trainer = DrafterBaseTrainer(
            config=config,
            world_size=world_size,
            rollout_dp_rank=dist.get_rank(),
            training_device_mesh=None,
            backend=backend,
            training_process_group=dist.group.WORLD,
        )

        activated = asyncio.run(trainer.activate_training_model())
        assert activated, "activate_training_model returned False"
        raw_model = _assert_model_built(trainer, backend, target_config)
        logger.info(
            "Model built: block_size=%s num_anchors=%s num_target_layers=%s target_hidden_size=%s",
            raw_model.block_size,
            raw_model.num_anchors,
            raw_model.draft_model.num_target_layers,
            raw_model.draft_model.target_hidden_size,)

        rollout_batch, hidden_states = _mock_rollout_batch(target_config, raw_model)
        trainer.collect_online_data(rollout_batch, hidden_states)
        assert len(trainer.collected_data) == rollout_batch["input_ids"].size(0), (
            f"collected_data size mismatch: {len(trainer.collected_data)}"
        )

        train_batch = trainer._prepare_training_batch()
        assert train_batch is not None, "trainer._prepare_training_batch returned None"
        assert train_batch["input_ids"].dim() == 2, f"DFlash batch should be [B,S], got {train_batch['input_ids'].shape}"
        assert train_batch["hidden_states"].shape[:2] == train_batch["input_ids"].shape
        loss_tokens = float(train_batch["loss_mask"].sum().item())
        if loss_tokens <= 0:
            logger.error(
                "Prepared DFlash batch has zero loss tokens: loss_mask=%s input_shape=%s hidden_shape=%s collected=%s",
                train_batch["loss_mask"].detach().cpu().tolist(),
                tuple(train_batch["input_ids"].shape),
                tuple(train_batch["hidden_states"].shape),
                [
                    {
                        "input_len": int(item["input_ids"].size(0)),
                        "hidden_len": int(item["hidden_states"].size(0)),
                        "loss_sum": float(item.get("loss_mask", torch.tensor([])).float().sum().item())
                        if item.get("loss_mask") is not None
                        else None,
                        "prompt_len": item.get("_verl_prompt_len"),
                        "response_len": item.get("_verl_response_len"),
                        "feature_start": item.get("_verl_feature_start"),
                        "feature_end": item.get("_verl_feature_end"),
                    }
                for item in trainer.collected_data
                ],
            )
        assert loss_tokens > 0, "prepared DFlash batch has no loss tokens"
        logger.info(
            "Prepared batch: input=%s hidden=%s loss_tokens=%s",
            tuple(train_batch["input_ids"].shape),
            tuple(train_batch["hidden_states"].shape),
            loss_tokens,
        )

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = trainer.backend.compute_loss(trainer.model, train_batch, trainer._current_pad_size)
        _assert_loss_outputs(outputs, raw_model)
        logger.info(
            "compute_loss ok: ploss=%s tokens=%s acc=%s counts=%s",
            float((outputs["total_local_ploss"] / outputs["local_num_tokens"]).detach().float().item()),
            float(outputs["local_num_tokens"].detach().float().item()),
            float(outputs["accuracy"].detach().float().item()),
            outputs["count_per_position"].detach().cpu().tolist(),
        )

        step_ok = asyncio.run(trainer.training_step(step=1))
        assert step_ok, "trainer.training_step returned False"
        assert trainer.training_steps == 1, f"training_steps should be 1, got {trainer.training_steps}"
        logger.info("training_step ok: training_steps=%s", trainer.training_steps)

        expected_ckpt_step = trainer.current_rl_step if trainer.current_rl_step > 0 else trainer.training_steps
        awaitable = trainer.cleanup_training(clear_data=True)
        asyncio.run(awaitable)
        cleanup_done = True
        assert trainer.training_steps == 0, f"cleanup_training should reset training_steps, got {trainer.training_steps}"
        assert not trainer.collected_data, "cleanup_training(clear_data=True) should clear collected_data"
        assert not trainer.data_buffer, "cleanup_training(clear_data=True) should clear data_buffer"
        logger.info("cleanup_training ok: expected_ckpt_step=%s", expected_ckpt_step)
        dist.barrier()
        if dist.get_rank() == 0:
            _assert_checkpoint_saved(checkpoint_root, expected_ckpt_step, logger)
        dist.barrier()

    except Exception:
        print(f"\n[rank {dist.get_rank() if dist.is_initialized() else -1}] FULL TRACEBACK:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            print(
                f"[rank {dist.get_rank() if dist.is_initialized() else -1}] "
                f"cuda_allocated={torch.cuda.memory_allocated()} cuda_reserved={torch.cuda.memory_reserved()}",
                flush=True,
            )
        raise
    finally:
        try:
            if not cleanup_done and trainer is not None and getattr(trainer, "optimizer", None) is not None:
                trainer.optimizer.zero_grad(set_to_none=True)
            del trainer
            del backend
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
        if dist.is_initialized():
            try:
                dist.barrier()
                if dist.get_rank() == 0 and os.environ.get("DFLASH_TEST_CLEAN_CKPT", "0") == "1":
                    shutil.rmtree(checkpoint_root, ignore_errors=True)
                dist.barrier()
            except Exception:
                traceback.print_exc()
            try:
                dist.barrier()
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    test_dflash_multigpu_model_and_training_flow()