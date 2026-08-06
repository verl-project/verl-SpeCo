from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.integration.vllm_runtime import (
    SPECO_VLLM_REQUEST_STATS_TRACE_HEADER,
    SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
    SPECO_VLLM_WORKER_EXTENSION_CLS,
    SpecoVLLMColocateWorkerExtension,
    _attach_vllm_request_stats_to_trace_headers,
    _decode_vllm_request_stats_trace_header,
    _describe_vllm_draft_logits,
    _new_vllm_spec_decode_stats,
    _normalize_dflash_target_layer_aliases,
    _pop_vllm_request_stats_for_ids,
    _pop_vllm_request_stats_for_rollout_output,
    _record_vllm_request_acceptance_stats,
    _record_vllm_spec_decode_scheduler_stats,
    _set_vllm_request_stats_on_output,
    _stage_vllm_request_stats_for_rollout_output,
    _vllm_generate_request_id,
    _vllm_request_accept_stats_to_extra_fields,
    _vllm_request_accept_stats_to_scalar_extra_fields,
    _validate_vllm_dflash_drafter_config,
    _vllm_ascend_has_dspark_pr11153_k_query_runtime,
    _vllm_spec_decode_stats_to_metrics,
    attach_update_draft_weights_to_rollout,
    build_vllm_speculative_config_from_drafter,
    configure_vllm_runtime_from_config,
    patch_transformers_attention_layer_type_constants,
    speco_vllm_update_draft_weights,
)


def _drafter(**overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "model_path": "/models/drafter",
        "rollout": {"spec_steps": 3},
        "training": {},
        "vllm": {},
    }
    config.update(overrides)
    return config


def test_vllm_speculative_config_maps_eagle3_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(_drafter())

    assert config == {
        "draft_sample_method": "greedy",
        "method": "eagle3",
        "model": "/models/drafter",
        "num_speculative_tokens": 3,
    }


def test_vllm_fresh_training_does_not_load_checkpoint_output_root() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(checkpoint_path="/checkpoints/run/drafter")
    )

    assert config["model"] == "/models/drafter"


def test_vllm_checkpoint_path_remains_a_fallback_without_model_path() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(model_path=None, checkpoint_path="/checkpoints/draft_step_10")
    )

    assert config["model"] == "/checkpoints/draft_step_10"


def test_vllm_worker_extension_constructs_without_wake_up_fallback() -> None:
    extension = SpecoVLLMColocateWorkerExtension()

    assert isinstance(extension, SpecoVLLMColocateWorkerExtension)


def test_vllm_draft_logits_diagnostic_handles_missing_and_non_tensor_values() -> None:
    assert _describe_vllm_draft_logits(None, missing=True) == "missing"
    assert _describe_vllm_draft_logits(None) == "None(greedy)"
    assert _describe_vllm_draft_logits("MISSING") == "str"


def test_vllm_speculative_config_maps_dflash_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DFLASH",
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": "/models/drafter",
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_native_gpu_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: False)

    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dspark",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_dflash_on_npu_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True)
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_dspark_gpu_probabilistic_sampling_requires_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: False)
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
            vllm={"speculative_config_overrides": {"draft_sample_method": "probabilistic"}},
        )
    )

    assert config["method"] == "dspark"
    assert config["draft_sample_method"] == "probabilistic"


def test_vllm_dflash_validator_rejects_dspark_when_algorithm_is_dflash(tmp_path) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vLLM DFlash requires"):
        _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH")


def test_vllm_dspark_validator_accepts_markov_head_config(tmp_path) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    _validate_vllm_dflash_drafter_config(model_path, algorithm="DSPARK")


def test_vllm_dspark_config_aliases_are_dflash_compatible() -> None:
    config = {
        "architectures": ["DFlashDSparkDraftModel"],
        "markov_head_type": "vanilla",
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }

    assert _normalize_dflash_target_layer_aliases(config) is True

    assert config["dflash_config"] == {
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
    }
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]


def _install_fake_vllm_ascend_modules(monkeypatch, dflash_cls, proposer_cls) -> None:
    root_module = types.ModuleType("vllm_ascend")
    spec_decode_module = types.ModuleType("vllm_ascend.spec_decode")
    dflash_module = types.ModuleType("vllm_ascend.spec_decode.dflash_proposer")
    proposer_module = types.ModuleType("vllm_ascend.spec_decode.llm_base_proposer")

    dflash_module.AscendDflashProposer = dflash_cls
    proposer_module.AscendSpecDecodeBaseProposer = proposer_cls
    spec_decode_module.dflash_proposer = dflash_module
    spec_decode_module.llm_base_proposer = proposer_module
    root_module.spec_decode = spec_decode_module

    monkeypatch.setitem(sys.modules, "vllm_ascend", root_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode", spec_decode_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dflash_proposer", dflash_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.llm_base_proposer", proposer_module)


class _FakePR11153DflashProposer:
    def _num_query_per_req(self):
        return self.num_speculative_tokens if self._is_dspark else 1 + self.num_speculative_tokens

    def set_inputs_first_pass(self):
        return self._num_query_per_req(), "IS_DSPARK"


class _FakePR11153SpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(self.speculative_config.draft_model_config.hf_config, "markov_head_type"):
            blk = self.num_speculative_tokens
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


class _FakeOldDSparkDflashProposer:
    def set_inputs_first_pass(self):
        return 1 + self.num_speculative_tokens


class _FakeOldDSparkSpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(self.speculative_config.draft_model_config.hf_config, "markov_head_type"):
            blk = self.num_speculative_tokens + 1
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


def test_vllm_ascend_dspark_runtime_detector_accepts_pr11153_k_query(monkeypatch) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakePR11153DflashProposer,
        _FakePR11153SpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is True


def test_vllm_ascend_dspark_runtime_detector_rejects_old_full_block_layout(monkeypatch) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakeOldDSparkDflashProposer,
        _FakeOldDSparkSpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is False


def test_vllm_runtime_injects_native_config_and_worker_extension(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "eagle3"
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_vllm_runtime_injects_dspark_as_dflash_on_npu_and_worker_extension(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    monkeypatch.setattr("verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True)
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(
                    speculative_algorithm="DSPARK",
                    model_path=str(model_path),
                    rollout={"spec_steps": 3, "spec_verify_tokens": 16},
                ),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "dflash"
    assert engine_kwargs["speculative_config"]["num_speculative_tokens"] == 16
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_transformers_attention_layer_type_constants_compat(monkeypatch) -> None:
    transformers_module = types.ModuleType("transformers")
    configuration_utils_module = types.ModuleType("transformers.configuration_utils")
    transformers_module.configuration_utils = configuration_utils_module
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.configuration_utils", configuration_utils_module)

    assert patch_transformers_attention_layer_type_constants() is True
    assert configuration_utils_module.ALLOWED_LAYER_TYPES
    assert configuration_utils_module.ALLOWED_LAYER_TYPES == configuration_utils_module.ALLOWED_ATTENTION_LAYER_TYPES
    assert patch_transformers_attention_layer_type_constants() is False


def test_transformers_attention_layer_type_patch_runs_before_vllm_worker_extension_import() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "verl_speco" / "integration" / "vllm_runtime.py"
    ).read_text(encoding="utf-8")

    assert source.index("\npatch_transformers_attention_layer_type_constants()\n") < source.index(
        "from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension"
    )


def test_vllm_acceptance_stats_keep_stable_transport_keys() -> None:
    stats = _new_vllm_spec_decode_stats()
    scheduler_stats = SimpleNamespace(
        spec_decoding_stats=SimpleNamespace(num_drafts=4, num_accepted_tokens=7)
    )

    _record_vllm_spec_decode_scheduler_stats(stats, scheduler_stats)

    assert _vllm_spec_decode_stats_to_metrics(stats) == {
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": 7.0,
    }


def test_vllm_request_acceptance_stats_keep_stable_transport_keys() -> None:
    scheduler = SimpleNamespace()
    _record_vllm_request_acceptance_stats(
        scheduler,
        request_id="req-1",
        num_draft_tokens=7,
        num_accepted_tokens=3,
        num_invalid_spec_tokens=1,
    )
    _record_vllm_request_acceptance_stats(
        scheduler,
        request_id="req-1",
        num_draft_tokens=7,
        num_accepted_tokens=4,
        num_invalid_spec_tokens=0,
    )
    output = SimpleNamespace(
        request_id="req-1",
        _speco_vllm_request_accept_stats=scheduler._speco_vllm_request_accept_stats,
    )

    assert _vllm_request_accept_stats_to_extra_fields(output) == {
        "_speco_vllm_request_id": ["req-1"],
        "_speco_vllm_request_verify_rounds": [2],
        "_speco_vllm_request_draft_tokens": [14],
        "_speco_vllm_request_accepted_tokens": [7],
        "_speco_vllm_request_invalid_spec_tokens": [1],
        "_speco_vllm_request_completion_index": [0],
        "_speco_vllm_request_elapsed_sec": [None],
        "_verl_request_mean_accept_len": [4.5],
    }


def test_vllm_request_acceptance_stats_can_cross_engine_core_trace_headers() -> None:
    scheduler = SimpleNamespace()
    _record_vllm_request_acceptance_stats(
        scheduler,
        request_id="internal-req-1",
        num_draft_tokens=7,
        num_accepted_tokens=6,
        num_invalid_spec_tokens=0,
    )
    summaries, completion_order = _pop_vllm_request_stats_for_ids(scheduler, ["internal-req-1"])

    assert completion_order == ["internal-req-1"]
    assert scheduler._speco_vllm_request_accept_stats == {}

    engine_core_output = SimpleNamespace(request_id="internal-req-1", trace_headers={"trace": "keep"})
    _attach_vllm_request_stats_to_trace_headers(engine_core_output, summaries["internal-req-1"])

    assert engine_core_output.trace_headers["trace"] == "keep"
    decoded = _decode_vllm_request_stats_trace_header(
        engine_core_output.trace_headers[SPECO_VLLM_REQUEST_STATS_TRACE_HEADER]
    )
    assert decoded is not None
    assert decoded["verify_rounds"] == 1
    assert decoded["accepted_tokens"] == 6


def test_vllm_request_acceptance_stats_extra_fields_use_request_output_id() -> None:
    output = SimpleNamespace(request_id="external-req-1")
    summary = {
        "verify_rounds": 2,
        "draft_tokens": 14,
        "accepted_tokens": 8,
        "invalid_spec_tokens": 1,
        "completion_index": 3,
        "elapsed_sec": 0.5,
    }

    assert _set_vllm_request_stats_on_output(output, {"external-req-1": summary}, ["external-req-1"])
    assert _vllm_request_accept_stats_to_extra_fields(output) == {
        "_speco_vllm_request_id": ["external-req-1"],
        "_speco_vllm_request_verify_rounds": [2],
        "_speco_vllm_request_draft_tokens": [14],
        "_speco_vllm_request_accepted_tokens": [8],
        "_speco_vllm_request_invalid_spec_tokens": [1],
        "_speco_vllm_request_completion_index": [3],
        "_speco_vllm_request_elapsed_sec": [0.5],
        "_verl_request_mean_accept_len": [5.0],
    }


def test_vllm_request_acceptance_stats_can_stage_for_token_output() -> None:
    summary = {
        "verify_rounds": 4,
        "draft_tokens": 28,
        "accepted_tokens": 12,
        "invalid_spec_tokens": 0,
        "completion_index": 5,
    }
    _stage_vllm_request_stats_for_rollout_output("server-req-1", {"server-req-1": summary}, ["server-req-1"])

    summaries, completion_order = _pop_vllm_request_stats_for_rollout_output(
        _vllm_generate_request_id(([], {}, "server-req-1"), {})
    )
    output = SimpleNamespace(
        request_id="server-req-1",
        _speco_vllm_request_accept_stats=summaries,
        _speco_vllm_request_completion_order=completion_order,
    )

    scalar_fields = _vllm_request_accept_stats_to_scalar_extra_fields(output)

    assert scalar_fields["_verl_request_mean_accept_len"] == 4.0
    assert scalar_fields["_speco_vllm_request_id"] == "server-req-1"
    assert scalar_fields["_speco_vllm_request_verify_rounds"] == 4


def test_trainer_keeps_public_acceptance_metric_name() -> None:
    trainer_source = (
        Path(__file__).resolve().parents[2] / "verl_speco" / "trainer" / "speco_ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert '"drafter/spec_decode/mean_acceptance_length"' in trainer_source


def test_vllm_draft_update_attachment_is_idempotent() -> None:
    rollout = SimpleNamespace()

    assert attach_update_draft_weights_to_rollout(rollout) is rollout
    first = rollout.update_draft_weights
    assert first.__func__ is speco_vllm_update_draft_weights
    assert attach_update_draft_weights_to_rollout(rollout).update_draft_weights == first
