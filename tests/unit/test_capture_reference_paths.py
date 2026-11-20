# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def test_incremental_reference_keeps_token_order_and_cache(capsys):
    path = Path(__file__).parents[1] / "heavy" / "test_vllm_rollout_capture_gpu.py"
    spec = importlib.util.spec_from_file_location("capture_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def model(tokens, past_key_values=0, **kwargs):
        calls.append(tokens.tolist())
        state = tokens.cumsum(1) + past_key_values
        return SimpleNamespace(
            hidden_states=(state.unsqueeze(-1),), past_key_values=int(state[0, -1])
        )

    tokens = torch.tensor([[2, 4, 6, 8, 10]])
    result = module.cached_reference_states(model, tokens, 2)
    assert calls == [[[2, 4]], [[6]], [[8]], [[10]]]
    assert torch.equal(result[0].squeeze(-1), tokens.cumsum(1))
    output = capsys.readouterr().out
    assert "incremental_prefill_start" in output
    assert "completed=3/3" in output
    assert "incremental_done" in output


def test_forward_audit_reports_corruption_and_repeated_positions():
    path = Path(__file__).parents[1] / "heavy" / "test_vllm_rollout_capture_gpu.py"
    spec = importlib.util.spec_from_file_location("capture_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = {"hidden_layer_ids": [2], "hidden_states_layout": "aux",
              "hidden_positions": torch.tensor([[1, 2]]),
              "hidden_states": torch.tensor([[[2., 3.], [99., 99.]]])}
    records = [{"positions": torch.tensor([1, 2]), "tokens": torch.tensor([4, 5]),
                "layers": {"2": torch.tensor([[2., 3.], [4., 5.]])}},
               {"positions": torch.tensor([1]), "tokens": torch.tensor([4]),
                "layers": {"2": torch.tensor([[8., 9.]])}}]
    reports = module.compare_forward_audit(sample, records, [0, 4, 5], [6])
    assert reports[0]["matches_any_forward"]
    assert len(reports[0]["candidates"]) == 2
    assert not reports[1]["matches_any_forward"]
    # A different token at the same position cannot vindicate an exported row.
    records[0]["tokens"][0] = 7
    assert not module.compare_forward_audit(sample, records, [0, 4, 5], [6])[0]["matches_any_forward"]


def test_forward_audit_survives_inner_model_call_bypassing_module_hooks():
    path = Path(__file__).parents[1] / "heavy" / "test_vllm_rollout_capture_gpu.py"
    spec = importlib.util.spec_from_file_location("capture_audit_hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual=None):
            return hidden_states + 1

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])
            self.norm = torch.nn.Identity()

        def forward(self, input_ids, positions):
            return self.norm(self.layers[0](positions, input_ids.float().unsqueeze(-1)))

    class Qwen3ForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[0])
            self.model = Body()

        def forward(self, input_ids, positions):
            # vLLM's decorated body bypasses Module.__call__ and its hooks.
            return self.model.forward(input_ids, positions)

    model = Qwen3ForCausalLM()
    module.install_forward_audit(model)
    for token in (4, 5):
        model(input_ids=torch.tensor([token]), positions=torch.tensor([token-4]))
    records = module.collect_forward_audit(model)
    assert len(records) == 2
    assert records[0]["tokens"].tolist() == [4]
    assert records[1]["positions"].tolist() == [1]
    assert records[0]["layers"]["0"].tolist() == [[4.]]
    assert records[1]["layers"]["last"].tolist() == [[6.]]
    assert not model._forward_pre_hooks
    assert not model.model.layers[0]._forward_pre_hooks
    assert not model.model.norm._forward_hooks
