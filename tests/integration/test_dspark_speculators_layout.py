from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn
from verl_speco.integration import oldlogprob_runtime as rt
from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_oldlogprob_aux_layer_ids,
    resolve_oldlogprob_aux_layer_id_space,
)

IDS = [4, 12, 20, 28, 36, 44, 52, 60]
RAW = {"architectures": ["DSparkDraftModel"], "aux_hidden_state_layer_ids": IDS}
DRAFTER = {"speculative_algorithm": "DSPARK", "training": {"dspark_num_target_layers": 8}}


class Increment(nn.Module):
    def forward(self, hidden):
        return hidden + 1


class FinalNorm(nn.Module):
    def forward(self, hidden):
        return hidden * 2


class TinyDecoderStack(nn.Module):
    def __init__(self, depth, final_norm):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([Increment() for _ in range(depth)])
        self.decoder.final_layernorm = FinalNorm() if final_norm else None


class TinyVLPipeline(nn.Module):
    def __init__(self, pp_rank):
        super().__init__()
        # Match the real Qwen3VLModel registration order on PP0 exactly.
        self.vision_model = TinyDecoderStack(27, True) if pp_rank == 0 else None
        self.language_model = TinyDecoderStack(16, pp_rank == 3)

    def forward(self, hidden):
        # Text-only requests never execute the vision stack.
        for layer in self.language_model.decoder.layers:
            hidden = layer(hidden)
        norm = self.language_model.decoder.final_layernorm
        return norm(hidden) if norm is not None else hidden


def test_nonfinal_plain_decoder_allows_absent_final_norm():
    root = nn.Module()
    root.vision_model = TinyDecoderStack(27, True)
    root.decoder = TinyDecoderStack(16, False).decoder
    engine = NS()
    layers, norm = rt._find_megatron_layers_and_final_norm(engine, root)
    assert layers == list(root.decoder.layers)
    assert norm is None
    assert engine._speco_oldlogprob_decoder_path == "decoder.layers"


def test_fallback_refuses_vision_only_module_tree():
    root = nn.Module()
    root.vision_model = TinyDecoderStack(27, True)
    assert rt._find_megatron_layers_and_final_norm(NS(), root) == (None, None)


def test_auto_config_preserves_output_index_space():
    assert resolve_oldlogprob_aux_layer_ids(DRAFTER, target_num_hidden_layers=64, model_configs=[RAW]) == IDS
    assert resolve_oldlogprob_aux_layer_id_space(DRAFTER, model_configs=[RAW]) == "output"
    for stage in range(4):
        mapping = rt._megatron_map_aux_layers_to_stage(IDS, 16, stage, 4, 64, layer_id_space="output")
        assert [(16 * stage + local) for kind, local in mapping if kind == "layer"] == [3 + 16 * stage, 11 + 16 * stage]


def test_legacy_decoder_override_and_conflicts():
    legacy = {**DRAFTER, "target_layer_ids": [1, 4]}
    assert resolve_oldlogprob_aux_layer_ids(legacy, target_num_hidden_layers=64, model_configs=[RAW]) == [1, 4]
    assert resolve_oldlogprob_aux_layer_id_space(legacy, model_configs=[RAW]) == "decoder"
    with pytest.raises(ValueError, match="conflict"):
        resolve_oldlogprob_aux_layer_ids(DRAFTER, target_num_hidden_layers=64, model_configs=[{**RAW, "target_layer_ids": [1, 2]}])


def test_v1_actual_metadata_assignments_roundtrip():
    import ast
    import inspect
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin
    # Execute the actual V1 assignment statements, rather than reconstructing
    # their arguments in the test. The rest of the method needs live workers.
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(SpecoV1Mixin._compute_old_log_prob)))
    statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            args = node.value.args
            if len(args) > 1 and isinstance(args[1], ast.Name) and args[1].id in {
                "OLD_LOGPROB_AUX_LAYER_IDS_KEY", "OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY"
            }:
                statements.append(node)
    assert len(statements) == 2
    control = TensorDict({}, batch_size=[1])
    trainer = NS(_speco_oldlogprob_aux_layer_ids=lambda: IDS,
                 _speco_oldlogprob_aux_layer_id_space=lambda: "output")
    ns = {"tu": tu, "control": control, "self": trainer,
          "OLD_LOGPROB_AUX_LAYER_IDS_KEY": rt.OLD_LOGPROB_AUX_LAYER_IDS_KEY,
          "OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY": rt.OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY}
    exec(compile(ast.Module(body=statements, type_ignores=[]), "actual_v1_metadata", "exec"), ns)
    assert rt._oldlogprob_aux_layer_ids_from_batch(control) == IDS
    assert rt._oldlogprob_aux_layer_id_space(control) == "output"


def test_actual_pp_hooks_collect_buffer_and_backend(monkeypatch, tmp_path):
    import ray
    from verl_speco.trainer import base_trainer as bt
    from verl_speco.trainer.data_buffer import DataBuffer
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

    merged = {}
    final_context = None
    engines = []
    for stage in range(4):
        model = TinyVLPipeline(stage)
        wrapper = nn.Module()
        wrapper.module = model
        context = dict(batch_size=1, output_batch_size=1, hidden_rows=3,
                       local_positions=[0, 1, 2], local_batch_indices=[0, 0, 0], local_row_indices=[0, 1, 2],
                       local_3d_batch_indices=[0, 0, 0], local_3d_seq_positions=[0, 1, 2],
                       selected_owner_rows={0: [0, 1, 2]}, sp_size=1, sp_rank=0, sp_group=None,
                       pp_rank=stage, pp_size=4, is_last_stage=stage == 3)
        monkeypatch.setattr(rt, "_build_megatron_selection_context", lambda *a: dict(context))
        monkeypatch.setattr(rt, "_oldlogprob_aux_layer_ids_from_batch", lambda *a: IDS)
        monkeypatch.setattr(rt, "_oldlogprob_aux_layer_id_space", lambda *a: "output")
        monkeypatch.setattr(rt, "_oldlogprob_hidden_layout", lambda *a: "dflash_aux_plus_last")
        engine = NS(tf_config=NS(num_layers=64), module=[wrapper])
        rt._install_megatron_hidden_hooks(engine, wrapper, {rt.OLD_LOGPROB_COLLECT_MASK_KEY: True})
        assert engine._speco_oldlogprob_decoder_path == "language_model.decoder.layers"
        if model.vision_model is not None:
            assert all(not layer._forward_hooks for layer in model.vision_model.decoder.layers)
        model(torch.full((1, 3, 5120), float(16 * stage), dtype=torch.bfloat16))
        captured = engine._speco_oldlogprob_hidden_context
        assert len(captured["captures"]) == (3 if stage == 3 else 2)
        merged.update(captured["captures"])
        if stage == 3:
            final_context = dict(captured, captures=merged)
            engine._speco_pp_saved_contexts = [dict(captured)]
        rt._megatron_save_stage_captures(engine, None)
        engines.append(engine)
    stored = {}
    monkeypatch.setattr(ray, "put", lambda tensor: stored.setdefault("whole", tensor) is not None and "whole-ref")
    monkeypatch.setattr(rt, "_speco_pp_exchange_dir", lambda: str(tmp_path))
    result = {}
    for stage, engine in enumerate(engines):
        mpu = NS(get_pipeline_model_parallel_rank=lambda: stage,
                 get_tensor_model_parallel_rank=lambda: 0,
                 get_context_parallel_rank=lambda: 0,
                 get_data_parallel_rank=lambda: 0)
        monkeypatch.setattr(rt, "_megatron_mpu", lambda: mpu)
        rt._megatron_post_stage_exchange_and_consume(engine, result)
    full = stored["whole"]
    assert full.shape == (1, 3, 46080)
    assert full[0, 0].view(9, 5120)[:, 0].tolist() == IDS + [128]

    monkeypatch.setattr(bt, "device_name", "cpu")
    trainer = bt.DrafterBaseTrainer.__new__(bt.DrafterBaseTrainer)
    training = {"use_logits": False, "collect_hidden_states_from_sgl": False}
    trainer.backend = NS(model_type="dspark")
    trainer.config = NS(rollout=NS(drafter=NS(training=training, rollout={})))
    trainer.copy_stream = None
    trainer.rank = 0
    trainer.pad_token_id = 0
    trainer.model_config = NS(pad_token_id=0)
    trainer.current_rl_step = 1
    trainer.use_data_buffer = True
    trainer.data_buffer = DataBuffer()
    trainer.buffer_version = 0
    assert trainer.collect_online_data({"input_ids": torch.arange(4).unsqueeze(0),
                                        "hidden_states_layout": "dflash_aux_plus_last"}, full)
    items = trainer.data_buffer.get_all_data()
    assert len(items) == 1 and items[0]["hidden_states"].shape == (3, 46080)
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.config = trainer.config
    model_config = NS(target_hidden_size=5120, hidden_size=5120, num_context_layers=8, pad_token_id=0)
    result = backend.preprocess_individual_items(items, "cpu", model_config)
    assert result["h_states"][0].shape == (3, 40960)
    assert result["target_last_h_states"][0].shape == (3, 5120)
    assert result["h_states"][0][0].view(8, 5120)[:, 0].tolist() == IDS
    assert result["target_last_h_states"][0][0, 0].item() == 128


def test_missing_pp_layers_fail_before_emitting_partial_width():
    captures = {f"aux:{i}": torch.zeros(1, 3, 2) for i in IDS[2:]}
    captures["final"] = torch.zeros(1, 3, 2)
    engine = NS(_speco_oldlogprob_hidden_context={"aux_keys": [f"aux:{i}" for i in IDS],
                 "final_key": "final", "captures": captures, "pp_size": 4, "pp_rank": 3, "num_global_layers": 64})
    with pytest.raises(RuntimeError, match="Refusing to train on a partial hidden layout"):
        rt._consume_oldlogprob_hidden_capture(engine)
    assert not hasattr(engine, "_speco_oldlogprob_hidden_context")


def test_actual_npu_subclass_inherits_all_installed_capture_wrappers():
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    from verl.workers.engine.mindspeed.transformer_impl import MindspeedEngineWithLMHead
    assert rt.install_oldlogprob_hidden_runtime_patch_megatron()
    for name in ("forward_step", "postprocess_micro_batch_func", "forward_backward_batch"):
        assert getattr(MindspeedEngineWithLMHead, name) is getattr(MegatronEngineWithLMHead, name)
        assert hasattr(getattr(MindspeedEngineWithLMHead, name), "__wrapped__")
    print("LOGGER_PROVENANCE", {"disabled": rt.logger.disabled, "level": rt.logger.level,
                                "effective_level": rt.logger.getEffectiveLevel()})
