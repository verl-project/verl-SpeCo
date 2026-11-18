# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Actual autograd/Adam comparisons for the native MTP numerical adapter."""

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from verl_speco.ops.deepseek_v4_mtp_loss import vocab_parallel_eagle_loss
from verl_speco.trainer.deepseek_v4_mtp_contract import (
    DeepSeekV4MTPContract,
    from_engine_tensors,
)
from verl_speco.trainer.deepseek_v4_mtp_training import (
    run_mtp_training,
    step_inputs,
    valid_token_counts,
)


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "mtp_model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4,
                "hc_mult": 2,
                "vocab_size": 11,
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 3,
            }
        )
    )
    return path


def config(model, store):
    root = Path(__file__).resolve().parents[2] / "verl_speco/config"
    base = OmegaConf.load(root / "deepseek_v4_0731_draft_trainer.yaml")
    base.actor_rollout_ref.model.path = str(model)
    base.actor_rollout_ref.rollout.drafter.speculative_algorithm = "MTP"
    base.actor_rollout_ref.rollout.drafter.training.feature_store.path = str(store)
    base.draft_engine.variant = "ds4_mtp"
    base.draft_engine.target_revision = None
    base.draft_engine.overrides.model = {"vllm_tensor_parallel_size": 1}
    return base


def test_mtp_store_roundtrip_preflight_and_dspark_rejection(model_path, tmp_path):
    from verl_speco.integration.vllm_ascend_config import load_vllm_ascend_config
    from verl_speco.trainer.feature_store import TorchShardFeatureStore
    from verl_speco.trainer.vllm_ascend_training_loop import preflight

    contract = DeepSeekV4MTPContract.from_model(model_path)
    sample = from_engine_tensors(
        {
            "input_ids": torch.arange(8),
            "positions": torch.arange(100, 108),
            "hidden_states": torch.randn(8, 8),
        },
        [0, 0, 1, 1, 0, 1, 1, 1],
        contract,
        sample_id="real-pre-hc",
    )
    store_path = tmp_path / "features"
    store = TorchShardFeatureStore(
        store_path,
        metadata={"source": "trajectory_replay", "ds4_contract": contract.identity()},
    )
    store.write_many([sample])
    store.close()
    cfg = config(model_path, store_path)
    report, resolved, dataset = preflight(cfg)
    assert report["algorithm"] == "MTP" and report["segments"] == 1
    assert report["contract"]["layout"] == "deepseek_v4_mtp_pre_hc_v1"
    assert (
        resolved.model.type == "ds4" and report["hardware_training_validated"] is False
    )
    assert torch.equal(dataset[0]["positions"], torch.arange(100, 108))
    assert torch.equal(dataset[0]["hidden_states"], sample.hidden_states)
    dataset.store.close()
    cfg.actor_rollout_ref.rollout.drafter.speculative_algorithm = "DSPARK"
    with pytest.raises(ValueError, match="speculative_algorithm=MTP"):
        load_vllm_ascend_config(cfg)
    values = json.loads((model_path / "config.json").read_text())
    values["dspark_block_size"] = 5
    (model_path / "config.json").write_text(json.dumps(values))
    with pytest.raises(ValueError, match="DSpark checkpoints"):
        DeepSeekV4MTPContract.from_model(model_path)


def test_mtp_rejects_missing_hc_channels_and_nonbinary_mask(model_path):
    contract = DeepSeekV4MTPContract.from_model(model_path)
    values = {
        "input_ids": torch.arange(5),
        "positions": torch.arange(5),
        "hidden_states": torch.ones(5, 4),
    }
    with pytest.raises(ValueError, match="complete pre-HC"):
        from_engine_tensors(
            values, torch.ones(5), contract, sample_id="post-hc-is-insufficient"
        )
    values["hidden_states"] = torch.ones(5, 8)
    with pytest.raises(ValueError, match="1=supervised"):
        from_engine_tensors(
            values, torch.full((5,), 0.5), contract, sample_id="invalid-mask"
        )


def test_mtp_shift_keeps_absolute_positions_and_masks_padding():
    batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "positions": torch.tensor([[100, 101, 102, 103]]),
        "loss_mask": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
    }
    ids, positions, mask = step_inputs(batch, 1, 3)
    assert ids.tolist() == [[3, 4, 5, 0]]
    assert positions.tolist() == [[100, 101, 102, 103]]
    assert mask.tolist() == [[1, 0, 1, 0]]
    ids, positions, mask = step_inputs(batch, 3, 3)
    assert ids.tolist() == [[5, 0, 0, 0]]
    assert positions.tolist() == [[102, 103, 0, 0]]
    assert mask.tolist() == [[1, 0, 0, 0]]
    assert valid_token_counts(batch, 3) == [2, 1, 1]


@pytest.mark.parametrize("chunk", [0, 1, 3])
def test_eagle_forward_gradient_and_repeated_fp32_teacher_are_exact(chunk):
    torch.manual_seed(41)
    student = torch.randn(2, 4, 11, requires_grad=True)
    teacher = torch.randn_like(student).softmax(-1).detach()
    original = teacher.clone()
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]])
    logits = student.detach().clone().requires_grad_()
    expected = (
        (
            F.kl_div(logits.log_softmax(-1), teacher, reduction="none").sum(-1) * mask
        ).sum()
        / (mask.sum() + 1e-6)
        * 0.8**2
    )
    expected.backward()
    for _ in range(2):
        actual = vocab_parallel_eagle_loss(
            student, teacher, mask, 2, token_chunk_size=chunk
        )
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(student.grad, logits.grad, atol=2e-8, rtol=2e-6)
        assert torch.equal(teacher, original)
        student.grad = None


class ToyMTP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)
        self.embed = torch.nn.Embedding(11, 8)
        self.head = torch.nn.Linear(8, 11, bias=False)
        self.embed.requires_grad_(False)
        self.head.requires_grad_(False)

    def forward(self, *, input_ids, hidden_states, **kwargs):
        return torch.tanh(self.projection(hidden_states) + self.embed(input_ids))

    def compute_local_logits(self, hidden, **kwargs):
        return self.head(hidden)


class Teacher(torch.nn.Linear):
    dtype = torch.float32
    local_logits = torch.nn.Linear.forward


class CPUEngine:
    def __init__(self, model, gas):
        self.module, self.gas = model, gas
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.003
        )
        self.global_steps = 0

    def __call__(self, **kwargs):
        return self.module(**kwargs)

    def compute_local_logits(self, *args, **kwargs):
        return self.module.compute_local_logits(*args, **kwargs)

    def set_gradient_accumulation_boundary(self, boundary):
        self.boundary = boundary

    def get_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]

    def backward(self, loss):
        (loss / self.gas).backward()

    def step(self):
        if self.boundary:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_steps += 1


@pytest.mark.parametrize("gas,batch_count", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("detach", [False, True])
def test_recursive_mtp_adam_matches_independent_dense_global_token_objective(
    gas, batch_count, detach
):
    torch.manual_seed(7)
    model = ToyMTP()
    reference = copy.deepcopy(model)
    teacher = Teacher(8, 11, bias=False).requires_grad_(False)
    engine = CPUEngine(model, gas)
    optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003
    )
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "positions": torch.arange(20, 25).unsqueeze(0),
            "loss_mask": torch.tensor([[0.0, 0.0, 1.0, float(i == 0), 1.0]]),
            "hidden_states": torch.randn(1, 5, 8),
        }
        for i in range(batch_count)
    ]
    denominator = [
        sum(float(b["loss_mask"][:, k:].sum()) for b in batches) + 1e-6 for k in (1, 2)
    ]
    objective = 0
    for batch in batches:
        current = batch["hidden_states"].detach().clone()
        target = teacher(current).softmax(-1).detach()
        for k in (1, 2):
            tokens = torch.cat(
                [batch["input_ids"][:, k:], torch.zeros(1, k, dtype=torch.long)], 1
            )
            new = reference(input_ids=tokens, hidden_states=current)
            logits = reference.compute_local_logits(new)[:, : 5 - k]
            kl = F.kl_div(logits.log_softmax(-1), target[:, k:], reduction="none").sum(
                -1
            )
            objective = objective + (
                kl * batch["loss_mask"][:, k:]
            ).sum() / denominator[k - 1] * 0.8 ** (k - 1)
            current = new.detach() if detach else new
    objective.backward()
    optimizer.step()
    cfg = OmegaConf.create(
        {
            "model": {"type": "ds4"},
            "train": {
                "loss_type": "eagle",
                "ttt_length": 2,
                "max_optimizer_steps": 1,
                "detach_hs_graph": detach,
            },
            "deepspeed": {"gradient_accumulation_steps": gas},
            "optimizations": {"mtp": {"valid_token_logits_only": True}},
        }
    )
    topology = SimpleNamespace(
        data_parallel_size=1, tensor_parallel_group=None, data_parallel_group=None
    )
    metrics = []
    result = run_mtp_training(
        model_engine=engine,
        target_lm_head=teacher,
        dataloader=batches,
        sampler=None,
        cfg=cfg,
        logger=logging.getLogger(__name__),
        tracker=SimpleNamespace(log_metrics=lambda m, **kwargs: metrics.append(m)),
        device=torch.device("cpu"),
        parallel_topology=topology,
        checkpoint_dir="unused",
    )
    assert (
        result["global_step"] == engine.global_steps == 1 and result["observed_update"]
    )
    assert metrics[0]["train_loss"] == pytest.approx(
        float(objective.detach()), abs=2e-6
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter, dict(reference.named_parameters())[name], rtol=1e-5, atol=1e-7
        )
    assert all(p.grad is None for p in teacher.parameters())


def test_dspark_numerical_identity_remains_unchanged():
    from verl_speco.integration.vllm_ascend_implementation import (
        implementation_fingerprint,
    )
    from verl_speco.integration.vllm_ascend_mtp_config import (
        mtp_implementation_fingerprint,
    )

    assert (
        implementation_fingerprint()
        == "73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582"
    )
    assert mtp_implementation_fingerprint() != implementation_fingerprint()


def test_mtp_capture_uses_owned_pre_hc_without_dspark_auxiliary_taps():
    from verl_speco.integration.vllm_rollout_capture import TargetModelCapture

    head = torch.arange(40.0).reshape(5, 2, 4)
    model = SimpleNamespace(
        model=SimpleNamespace(), get_mtp_target_hidden_states=lambda: head
    )
    capture = TargetModelCapture(
        SimpleNamespace(get_model=lambda: model),
        {
            "model_family": "deepseek_v4",
            "feature_kind": "mtp",
            "hidden_size": 4,
            "hc_mult": 2,
        },
    )
    hidden, extra = capture.tensors(3)
    expected = head[:3].reshape(3, 8).clone()
    head.zero_()
    assert extra is None and torch.equal(hidden, expected)
    capture.close()


def test_mtp_replay_writes_real_store_with_unshifted_tokens_and_positive_masks(
    model_path, tmp_path, monkeypatch
):
    import verl_speco.replay_deepseek_v4_features as replay
    from verl_speco.trainer.feature_store import TorchShardFeatureStore

    ids = list(range(5))
    hidden = torch.randn(5, 8)
    observed = []

    def capture(engine, tokens, params, settings):
        observed.append(settings)
        return (
            "r",
            ids + [0],
            {
                "input_ids": torch.tensor(ids),
                "positions": torch.arange(5),
                "hidden_states": hidden,
            },
        )

    monkeypatch.setattr(replay, "capture_request", capture)
    monkeypatch.setattr(replay, "close_capture", lambda llm, store: store.close())
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    )
    records = [
        SimpleNamespace(
            input_ids=ids,
            loss_mask=[0, 0, 1, 1, 1],
            sample_id="r",
            line_number=1,
            source_file="recorded.jsonl",
        )
    ]
    contract = DeepSeekV4MTPContract.from_model(model_path)
    report = replay.replay_features(
        llm,
        records,
        contract=contract,
        output=tmp_path / "replay",
        sampling_params=None,
        algorithm="mtp",
    )
    assert report["status"] == "passed" and observed[0]["feature_kind"] == "mtp"
    store = TorchShardFeatureStore(tmp_path / "replay", read_only=True)
    sample = store.read(next(store.iter_keys()))
    assert sample.algorithm == "MTP" and torch.equal(sample.hidden_states, hidden)
    assert sample.loss_mask.tolist() == [0, 0, 1, 1, 1]
    store.close()


def test_mtp_import_never_reads_air_inverted_masks(model_path, tmp_path):
    from verl_speco.trainer.deepseek_v4_dataset import import_capture
    from verl_speco.trainer.feature_store import TorchShardFeatureStore

    directory = tmp_path / "capture" / "request_000000"
    directory.mkdir(parents=True)
    for name, value in {
        "input_ids": torch.arange(5),
        "positions": torch.arange(5),
        "hidden_states": torch.ones(5, 8),
    }.items():
        torch.save(value, directory / f"step_0_{name}.cpu.pt")
    torch.save(torch.ones(5), directory / "step_0_raw_mask.cpu.pt")
    masks = tmp_path / "masks.jsonl"
    masks.write_text(
        json.dumps({"sample_idx": directory.name, "loss_mask": [0, 1, 0, 1, 1]}) + "\n"
    )
    report = import_capture(
        model=str(model_path),
        capture=str(directory.parent),
        masks=str(masks),
        output=str(tmp_path / "import"),
        algorithm="MTP",
    )
    assert report["status"] == "passed"
    store = TorchShardFeatureStore(tmp_path / "import", read_only=True)
    sample = store.read(next(store.iter_keys()))
    assert sample.loss_mask.tolist() == [0, 1, 0, 1, 1]
    assert store.get_metadata()["source"] == "mtp_capture_import"
    store.close()
