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

import json

import pytest


def _minimal_eagle3_config(**overrides) -> dict:
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 128,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "target_hidden_size": 4,
        "vocab_size": 32,
    }
    config.update(overrides)
    return config


def test_qwen3_eagle3_config_alias_preserves_five_target_layers(tmp_path) -> None:
    pytest.importorskip("transformers")
    from verl_speco.models.auto import AutoDraftModelConfig

    config_path = tmp_path / "config.json"
    config = _minimal_eagle3_config(
        architectures=["Qwen3Eagle3Model"],
        target_layer_ids=[1, 9, 17, 25, 33],
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = AutoDraftModelConfig.from_file(str(config_path))

    assert loaded.architectures == ["LlamaForCausalLMEagle3"]
    assert loaded.target_layer_ids == [1, 9, 17, 25, 33]
    assert loaded.eagle_aux_hidden_state_layer_ids == [1, 9, 17, 25, 33]
    assert loaded.eagle_config["eagle_aux_hidden_state_layer_ids"] == [1, 9, 17, 25, 33]
    assert loaded.num_aux_hidden_states == 5


def test_eagle3_model_uses_dynamic_aux_hidden_count() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3

    config = AutoDraftModelConfig._config_mapping["LlamaForCausalLMEagle3"].from_dict(
        _minimal_eagle3_config(
            target_hidden_size=4,
            target_layer_ids=[1, 9, 17, 25, 33],
            num_aux_hidden_states=5,
        )
    )
    model = LlamaForCausalLMEagle3(config)

    assert model.num_aux_hidden_states == 5
    assert model.fc.in_features == 20
    projected = model.project_hidden_states(torch.randn(2, 3, 20))
    assert projected.shape == (2, 3, config.hidden_size)

    with pytest.raises(ValueError, match="num_aux_hidden_states=5"):
        model.project_hidden_states(torch.randn(2, 3, 12))


def test_eagle3_model_defaults_missing_pretraining_tp() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.eagle.llama_eagle import LlamaMLP

    raw_config = _minimal_eagle3_config()
    raw_config.pop("pretraining_tp")
    config = AutoDraftModelConfig._config_mapping["LlamaForCausalLMEagle3"].from_dict(
        raw_config
    )
    mlp = LlamaMLP(config)

    output = mlp(torch.randn(2, 3, config.hidden_size))

    assert output.shape == (2, 3, config.hidden_size)


def _eagle_empty_rank_worker(rank, port, strategy, ttt_length, results):
    import os
    os.environ["SPECO_EAGLE3_DISABLE_TORCH_COMPILE"] = "1"
    from copy import deepcopy
    from datetime import timedelta
    from omegaconf import OmegaConf
    import torch
    import torch.distributed as dist
    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3
    from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(27)
        config = AutoDraftModelConfig._config_mapping["LlamaForCausalLMEagle3"].from_dict(
            _minimal_eagle3_config(target_layer_ids=[0, 1], num_aux_hidden_states=2)
        )
        raw = LlamaForCausalLMEagle3(config)
        reference = deepcopy(raw)
        if strategy == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(raw, broadcast_buffers=False)
            compared = model.module
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard
            mesh = init_device_mesh("cpu", (2,))
            fully_shard(raw.midlayer, mesh=mesh)
            fully_shard(raw, mesh=mesh)
            model = compared = raw
        backend = Eagle3TrainerBackend(OmegaConf.create({
            "rollout": {"drafter": {"training": {"use_logits": False, "ttt_length": ttt_length}}},
            "model": {"path": "/unused"},
        }), OmegaConf.create({}))
        backend.target_model = torch.nn.Linear(4, 32, bias=False)
        base_batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                      "hidden_states": torch.randn(1, 4, 8),
                      "last_hidden_states": torch.randn(1, 4, 4),
                      "attention_mask": torch.ones(1, 4, dtype=torch.long),
                      "position_ids": torch.arange(4).unsqueeze(0)}
        rng = torch.get_rng_state()
        if rank == 0:
            ref = backend.compute_loss(reference, {**base_batch, "loss_mask": torch.ones(1, 4)}, 0)
            (ref["total_local_ploss"] / ref["local_num_tokens"]).backward()
            torch.set_rng_state(rng)
        result = backend.compute_loss(model, {**base_batch, "loss_mask": torch.ones(1, 4) if rank == 0 else torch.zeros(1, 4)}, 0)
        count = result["local_num_tokens"].detach().clone()
        total = count.clone()
        dist.all_reduce(total)
        assert total.item() > 0
        (result["total_local_ploss"] * 2.0 / total).backward()
        diffs = []
        for parameter, ref_parameter in zip(compared.parameters(), reference.parameters(), strict=True):
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None
            grad = parameter.grad.full_tensor() if strategy == "fsdp2" else parameter.grad
            assert torch.isfinite(grad).all()
            if rank == 0:
                diffs.append((grad - ref_parameter.grad).abs().max().item())
        if rank == 0:
            results.put(max(diffs, default=0.0))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["ddp", "fsdp2"])
@pytest.mark.parametrize("ttt_length", [1, 2])
def test_eagle3_empty_supervision_rank_matches_reference(strategy, ttt_length):
    import multiprocessing
    import socket
    import torch
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_eagle_empty_rank_worker,
                                 args=(rank, port, strategy, ttt_length, results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
        assert all(process.exitcode == 0 for process in processes)
        assert results.get(timeout=5) < 1e-5
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()
