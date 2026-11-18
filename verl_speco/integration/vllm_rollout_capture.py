# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone Ascend feature collectors and actor-backed serving configuration.

The collectors are not installed in verl rollout workers.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import MethodType

import torch

from verl_speco.integration.ascend_runtime_compat import (
    capture_state,
    require_ascend_runtime,
    validate_capture_runner,
)
from verl_speco.integration.rollout_feature_capture import RolloutFeatureCollector


def capture_enabled(drafter):
    from verl_speco.backends.training_backend import rollout_capture_enabled

    return rollout_capture_enabled(drafter)


def embedded_deepseek_capture(drafter):
    return (
        capture_enabled(drafter) or _actor_capture_enabled(drafter)
    ) and drafter.get("training", {}).get("rollout_capture", {}).get(
        "model_family"
    ) == "deepseek_v4"


def embedded_speculative_config(drafter):
    """DeepSeek V4 uses the checkpoint's embedded DSpark, rather than Qwen aliases."""
    if not embedded_deepseek_capture(drafter):
        return None
    settings = drafter["training"]["rollout_capture"]
    if drafter.get("vllm", {}).get("speculative_config_overrides"):
        raise ValueError(
            "DeepSeek V4 capture requires the native embedded DSpark config without speculative overrides"
        )
    block_size = int(settings["contract"]["block_size"])
    if int(drafter.get("rollout", {}).get("spec_verify_tokens", 0)) != block_size:
        raise ValueError(
            "DeepSeek V4 spec_verify_tokens must equal checkpoint dspark_block_size"
        )
    draft_path = drafter.get("model_path")
    if (
        draft_path
        and Path(draft_path).resolve() != Path(settings["target_model_path"]).resolve()
    ):
        raise ValueError(
            "DeepSeek V4 capture serves the embedded DSpark from the target checkpoint"
        )
    return {
        "method": "dspark",
        "num_speculative_tokens": block_size,
        "enforce_eager": True,
    }


def _actor_capture_enabled(drafter):
    from verl_speco.backends.training_backend import actor_capture_enabled

    return actor_capture_enabled(drafter)


def configure_capture(config, drafter, payload, engine_kwargs):
    training = drafter.get("training", {})
    from verl_speco.backends.training_backend import is_vllm_ascend_drafter

    if is_vllm_ascend_drafter(drafter) and training.get(
        "collect_hidden_states_from_rollout", False
    ):
        raise ValueError(
            "Rollout hidden-state capture has been removed; set "
            "collect_hidden_states_from_rollout=false and use actor old-logprob "
            "capture or standalone trajectory replay"
        )
    actor_source = _actor_capture_enabled(drafter)
    if not actor_source:
        return
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
        resolve_oldlogprob_aux_layer_ids,
    )
    from verl_speco.integration.vllm_runtime import _set_child

    training = drafter.get("training", {})
    if not drafter.get("enable"):
        raise ValueError("Rollout capture currently requires drafter.enable=true")
    if actor_source:
        from verl_speco.integration.deepseek_v4_actor_capture import feature_source

        feature_source(training)
    elif training.get("collect_hidden_states_from_old_logprob") or training.get(
        "collect_hidden_states_from_sgl"
    ):
        raise ValueError("Select one feature source: rollout, old-logprob, or SGLang")
    if training.get("use_logits", False):
        raise ValueError("Rollout capture currently supports use_logits=false")
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.get("pipeline_model_parallel_size", 1)) != 1:
        raise ValueError(
            "Rollout feature capture currently requires pipeline parallel size 1"
        )
    model = str(config.actor_rollout_ref.model.path)
    model_config = json.loads((Path(model) / "config.json").read_text())
    settings = dict(training.get("rollout_capture", {}) or {})
    family = model_config.get("model_type")
    backend = str(training.get("backend", "pytorch"))
    if backend == "vllm_ascend":
        if (
            family != "deepseek_v4"
            or str(drafter.get("speculative_algorithm")).upper() != "DSPARK"
        ):
            raise ValueError("vllm_ascend capture requires DeepSeek V4 DSpark")
        online = training.get("mode", "online") == "online" and drafter.get(
            "enable_drafter_training", False
        )
        if online and (
            model_config.get("quantization_config")
            or engine_kwargs.get("quantization")
            or rollout.get("enable_expert_parallel", False)
        ):
            raise ValueError(
                "Online DeepSeek V4 runtime-shard updates require BF16 without EP"
            )
        if not online and not training.get("feature_store", {}).get("path"):
            raise ValueError(
                "vllm_ascend rollout capture requires training.feature_store.path"
            )
        if not online and (
            training.get("feature_store", {}).get("type", "torch_shard")
            != "torch_shard"
        ):
            raise ValueError(
                "Ascend rollout capture currently writes torch_shard offline stores"
            )
        from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

        contract = DeepSeekV4Contract.from_model(model)
        layers = list(contract.target_layer_ids)
        settings["contract"] = contract.identity()
        settings["target_revision"] = "0731"
        # The embedded draft shares the target checkpoint. Starting it with
        # dummy target weights would leave draft-only tensors uninitialized.
        _set_child(engine_kwargs, "load_format", "auto")
        _set_child(rollout, "load_format", "auto")
    else:
        if family == "deepseek_v4":
            raise ValueError(
                "DeepSeek V4 pre-HC capture requires the vllm_ascend training contract"
            )
        draft_configs = []
        for key in ("model_path", "checkpoint_path"):
            path = drafter.get(key)
            if path and (Path(path) / "config.json").is_file():
                draft_configs.append(
                    json.loads((Path(path) / "config.json").read_text())
                )
        layers = resolve_oldlogprob_aux_layer_ids(
            drafter,
            target_num_hidden_layers=int(model_config["num_hidden_layers"]),
            model_configs=draft_configs,
        )
        if not layers:
            raise ValueError("Rollout capture requires resolvable target layer IDs")
    settings.update(
        {
            "target_model_path": model,
            "model_family": family,
            "target_layer_ids": list(layers),
            "layer_id_space": "output"
            if str(drafter.get("speculative_algorithm")).upper() == "EAGLE3"
            else "decoder",
            "hidden_size": int(model_config["hidden_size"]),
            "hidden_states_layout": resolve_drafter_hidden_states_layout(
                drafter.get("speculative_algorithm"), training
            )
            if family != "deepseek_v4"
            else "deepseek_v4_0731_dspark_pre_hc_v1",
        }
    )
    payload["training"]["rollout_capture"] = settings
    if family == "deepseek_v4":
        existing = engine_kwargs.get("speculative_config") or {}
        if isinstance(existing, str):
            existing = json.loads(existing)
        expected = {
            "method": "dspark",
            "num_speculative_tokens": int(settings["contract"]["block_size"]),
            "enforce_eager": True,
        }
        if any(
            key in existing and existing[key] != value
            for key, value in expected.items()
        ) or existing.get("model"):
            raise ValueError(
                "engine_kwargs speculative_config conflicts with native embedded DeepSeek V4 DSpark"
            )
    runtime_training = rollout.get("drafter", {}).get("training")
    if runtime_training is not None:
        from omegaconf import OmegaConf, open_dict

        if OmegaConf.is_config(runtime_training):
            with open_dict(runtime_training):
                runtime_training.rollout_capture = settings
        else:
            runtime_training["rollout_capture"] = settings
    # Actor features do not install sampler hooks or change prefix caching.
    return


class TargetModelCapture:
    def __init__(self, runner, settings):
        self.runner, self.settings = runner, settings
        self.values = {}
        self.handles = []
        self.is_active = lambda: True
        self.model = runner.get_model()
        if settings["model_family"] == "deepseek_v4":
            backbone = getattr(self.model, "model", None)
            if backbone is None or not callable(
                getattr(self.model, "get_mtp_target_hidden_states", None)
            ):
                raise RuntimeError(
                    "DeepSeek V4 rollout does not expose its target backbone and pre-HC boundary"
                )
            if settings.get("feature_kind") == "mtp":
                return
            active_layers = getattr(backbone, "aux_hidden_state_layers", None)
            expected_layers = tuple(
                int(index) + 1 for index in settings["target_layer_ids"]
            )
            if active_layers is not None and tuple(active_layers) != expected_layers:
                raise ValueError(
                    "DeepSeek V4 rollout auxiliary layers differ from the training contract"
                )
            # The native compilation decorator overrides __call__ and bypasses
            # nn.Module forward hooks, even in eager mode. DSpark aux tensors
            # come from ExecuteModelState instead; pre-HC uses the native getter.
        else:
            from verl_speco.integration.oldlogprob_runtime import (
                _hidden_state_capture_target,
            )
            from verl_speco.trainer.target_feature_replay import (
                _find_layers_and_final_norm,
            )

            layers, norm = _find_layers_and_final_norm(self.model)
            for index in settings["target_layer_ids"]:
                kind, layer = _hidden_state_capture_target(
                    index, len(layers), layer_id_space=settings["layer_id_space"]
                )
                module = norm if kind == "final" else layers[layer]
                self.handles.append(
                    module.register_forward_hook(
                        self._hook(str(index), residual=kind != "final")
                    )
                )
            if settings["hidden_states_layout"].endswith("plus_last"):
                self.handles.append(
                    norm.register_forward_hook(self._hook("final", residual=False))
                )

    def _hook(self, key, *, residual):
        def capture(_module, _args, output):
            if not self.is_active():
                return
            value = output
            if isinstance(output, tuple):
                value = output[0]
                # vLLM's fused decoder returns the residual separately. HF
                # layer-output semantics include that residual exactly once.
                if residual and len(output) == 2 and torch.is_tensor(output[1]):
                    value = value + output[1]
            if not torch.is_tensor(value):
                raise TypeError("Unsupported rollout target layer output")
            # Later fused layers/proposers may reuse the returned storage even
            # before sampling. Own each selected boundary at the hook itself.
            self.values[key] = value.detach().clone()

        return capture

    def tensors(self, total, *, auxiliary_states=None):
        if self.settings.get("feature_kind") == "mtp":
            head = self.model.get_mtp_target_hidden_states()
            if not torch.is_tensor(head) or head.size(0) < total:
                raise RuntimeError("MTP target did not expose complete pre-HC rows")
            width = int(self.settings["hidden_size"]) * int(self.settings["hc_mult"])
            hidden = head[:total].reshape(total, -1)
            if hidden.shape != (total, width):
                raise ValueError("MTP capture requires complete HC channels")
            return hidden.detach().clone(), None
        if self.settings["model_family"] == "deepseek_v4":
            aux = auxiliary_states
            if not isinstance(aux, (tuple, list)) or len(aux) != len(
                self.settings["target_layer_ids"]
            ):
                raise RuntimeError(
                    "DeepSeek V4 execute state has no complete DSpark auxiliary states"
                )
            if any(
                not torch.is_tensor(value) or value.ndim != 2 or value.size(0) < total
                for value in aux
            ):
                raise ValueError(
                    "DeepSeek V4 auxiliary rows do not cover the scheduled batch"
                )
            head = self.model.get_mtp_target_hidden_states()
            if head is None:
                raise RuntimeError("DeepSeek V4 target did not expose pre-HC states")
            hidden = torch.cat([value[:total] for value in aux], dim=-1)
            # Own the native serving buffer before sampler/proposer reuse.
            head = head[:total].detach().clone()
        else:
            parts = [
                self.values[str(index)][:total]
                for index in self.settings["target_layer_ids"]
            ]
            if self.settings["hidden_states_layout"].endswith("plus_last"):
                parts.append(self.values["final"][:total])
            hidden, head = torch.cat(parts, dim=-1), None
        if hidden.size(0) != total or (head is not None and head.size(0) < total):
            raise ValueError(
                "Target feature rows do not cover the scheduled rollout batch"
            )
        width = self.settings.get("hidden_size")
        if width is not None:
            expected = int(width) * (
                len(self.settings["target_layer_ids"])
                + int(self.settings["hidden_states_layout"].endswith("plus_last"))
            )
            if hidden.ndim != 2 or hidden.size(-1) != expected:
                raise ValueError("Target auxiliary states have an unexpected TP layout")
        return hidden, head[:total] if head is not None else None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


class AscendRolloutCapture:
    def __init__(self, runner, settings):
        versions = tuple(
            importlib.metadata.version(name) for name in ("vllm", "vllm-ascend")
        )
        require_ascend_runtime(*versions)
        validate_capture_runner(runner)
        self.runner = runner
        self.collector = RolloutFeatureCollector(
            max_requests=int(settings.get("max_pending_requests", 64)),
            max_bytes=int(settings.get("max_pending_bytes", 1024**3)),
            max_rows=int(settings.get("max_rows_per_request", 8192)),
            timeout=float(settings.get("timeout", 120)),
        )
        self.model = TargetModelCapture(runner, settings)
        self.model.is_active = lambda: bool(self.collector.requests)
        self.pending = None
        original_sample, original_tokens = runner._sample, runner.sample_tokens
        self.original_sample, self.original_tokens = original_sample, original_tokens
        owner = self

        def sample_tokens(_runner, *args, **kwargs):
            try:
                owner.pending = owner.snapshot() if owner.collector.requests else None
            except Exception as error:  # noqa: BLE001 -- finalization rejects this session
                owner.collector.fail(error)
                owner.pending = None
            try:
                return original_tokens(*args, **kwargs)
            finally:
                owner.pending = None
                owner.model.values.clear()

        def sample(_runner, *args, **kwargs):
            output = original_sample(*args, **kwargs)
            try:
                if owner.pending is not None:
                    owner.capture(output.sampled_token_ids)
            except Exception as error:  # noqa: BLE001 -- report through the capture RPC
                owner.collector.fail(error)
            return output

        runner.sample_tokens = MethodType(sample_tokens, runner)
        runner._sample = MethodType(sample, runner)

    def snapshot(self):
        runner = self.runner
        state = capture_state(runner)
        if state is None:
            return None
        schedule = state.scheduler_output
        total = int(schedule.total_num_scheduled_tokens)
        request_ids = tuple(runner.input_batch.req_ids)
        if total < 0:
            raise ValueError("Scheduled rollout token count must be nonnegative")
        scheduled = [
            int(schedule.num_scheduled_tokens.get(key, 0)) for key in request_ids
        ]
        active_ids = [
            key for key, count in zip(request_ids, scheduled, strict=True) if count > 0
        ]
        if total == 0 or not self.collector.watches_any(active_ids):
            return None
        if any(count < 0 for count in scheduled) or sum(scheduled) != total:
            raise ValueError("Scheduled request lengths do not match the target batch")
        hidden, head = self.model.tensors(
            total, auxiliary_states=getattr(state, "aux_hidden_states", None)
        )
        metadata = getattr(state, "spec_decode_metadata", None)
        counts = (
            metadata.num_draft_tokens
            if metadata is not None
            else [0] * len(request_ids)
        )
        if torch.is_tensor(counts):
            raise ValueError("Draft counts must be CPU scheduling metadata")
        if len(counts) != len(request_ids) or total < 0:
            raise ValueError("Ascend draft counts differ from scheduled requests")
        return (
            request_ids,
            scheduled,
            counts,
            runner.input_ids.gpu[:total],
            state.positions[:total],
            hidden,
            head,
        )

    def capture(self, sampled):
        request_ids, scheduled, counts, input_ids, positions, hidden, head = (
            self.pending
        )
        offset = 0
        for index, request_id in enumerate(request_ids):
            count = scheduled[index]
            end = offset + count
            if count == 0:
                continue
            self.collector.submit(
                str(request_id),
                input_ids=input_ids[offset:end],
                positions=positions[offset:end],
                hidden_states=hidden[offset:end],
                target_head_hidden_states=head[offset:end]
                if head is not None
                else None,
                sampled_token_ids=sampled[index],
                num_draft_tokens=int(counts[index]),
            )
            offset = end
        if offset != input_ids.numel():
            raise ValueError("Scheduled request lengths do not match the target batch")

    def close(self):
        self.runner._sample, self.runner.sample_tokens = (
            self.original_sample,
            self.original_tokens,
        )
        self.model.close()
        self.collector.close()
