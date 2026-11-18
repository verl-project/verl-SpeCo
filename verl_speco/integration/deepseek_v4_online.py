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
"""DeepSeek V4 online contracts shared by training and rollout publishing."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

TARGET_OUTPUT_NAMES = {
    "hc_head_fn": "model.hc_head_fn",
    "hc_head_base": "model.hc_head_base",
    "hc_head_scale": "model.hc_head_scale",
    "norm_weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}
TARGET_OUTPUT_HF_NAMES = {
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}
# Some actor exporters apply Transformers' checkpoint conversion before yielding
# weights. Bare names are exact matches only: suffix-matching norm.weight or
# hc_head_fn would accidentally select decoder/compressor or MTP parameters.
TARGET_OUTPUT_CHECKPOINT_NAMES = {
    "hc_head_fn": "hc_head_fn",
    "hc_head_base": "hc_head_base",
    "hc_head_scale": "hc_head_scale",
    "norm_weight": "norm.weight",
    "lm_head.weight": "head.weight",
}


def online_enabled(config):
    section = config.get("actor_rollout_ref", config)
    drafter = section.get("rollout", {}).get("drafter", {})
    training = drafter.get("training", {})
    return (
        training.get("backend", "pytorch") == "vllm_ascend"
        and training.get("mode", "online") == "online"
    )


def execution_config(config):
    """Reuse the offline numerical configuration, with explicit online settings."""
    settings = config.rollout.drafter.training.vllm_ascend
    if not settings.get("work_dir"):
        raise ValueError(
            "Online vllm_ascend requires training.vllm_ascend.work_dir (shared across workers)"
        )
    result = OmegaConf.load(
        Path(__file__).parents[1] / "config/deepseek_v4_0731_draft_trainer.yaml"
    )
    result.actor_rollout_ref.model.path = config.model.path
    result.actor_rollout_ref.rollout.drafter.training.feature_store.path = str(
        settings.work_dir
    )
    result.actor_rollout_ref.rollout.drafter.training.max_steps = int(
        settings.total_optimizer_steps
    )
    for name in (
        "source_root",
        "source_fingerprint",
        "implementation_fingerprint",
        "config_path",
        "overrides",
        "environment",
        "resume_from",
    ):
        if name in settings:
            result.draft_engine[name] = settings[name]
    return result


@torch.no_grad()
def select_target_output(named_tensors, *, include_shared_tokens=False):
    """Consume a full actor export on every rank; preserve HC parameter dtypes."""
    result = {}
    logical_names = dict(TARGET_OUTPUT_NAMES)
    if include_shared_tokens:
        logical_names["embed_tokens.weight"] = "model.embed_tokens.weight"
    for name, tensor in named_tensors:
        for logical, suffix in logical_names.items():
            aliases = (suffix, TARGET_OUTPUT_HF_NAMES.get(logical, suffix))
            checkpoint_name = TARGET_OUTPUT_CHECKPOINT_NAMES.get(
                logical, "embed.weight"
            )
            if name == checkpoint_name or any(
                name == alias or name.endswith("." + alias) for alias in aliases
            ):
                if logical in result:
                    raise ValueError(
                        f"Ambiguous DeepSeek V4 target output tensor: {logical}"
                    )
                full_tensor = getattr(tensor, "full_tensor", None)
                if callable(full_tensor):
                    mesh = getattr(tensor, "device_mesh", None)
                    device = getattr(tensor, "device", None)
                    if (
                        mesh is not None
                        and device is not None
                        and mesh.device_type != device.type
                    ):
                        # Offloaded FSDP2: move only this selected local shard
                        # back to its communication device, leaving the actor
                        # module and all unrelated parameters offloaded.
                        tensor = tensor.to(mesh.device_type)
                    value = tensor.full_tensor()
                else:
                    value = tensor
                result[logical] = value.detach().to("cpu").clone()
    required = set(TARGET_OUTPUT_NAMES)
    if include_shared_tokens:
        required.add("embed_tokens.weight")
    if set(result) != required:
        raise ValueError(
            f"Incomplete DeepSeek V4 target output: {required - set(result)}"
        )
    return result


def apply_shared_token_weights(wrapper, state, *, tp_rank, tp_size):
    """Refresh frozen target-sharing weights using native vocabulary intervals.

    Validate both complete source tensors and destination shards before copying;
    preserve parameter objects, NPU formats and any captured storage references.
    """
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid shared token TP topology")
    runtime = wrapper.model
    modules = {
        "embed_tokens.weight": runtime.model.embed_tokens,
        "lm_head.weight": runtime.lm_head,
    }
    prepared = []
    expected_shape = (
        wrapper.runtime_contract.vocab_size,
        wrapper.runtime_contract.hidden_size,
    )
    for name, module in modules.items():
        value = state.get(name)
        target = module.weight
        if (
            not torch.is_tensor(value)
            or value.ndim != 2
            or tuple(value.shape) != expected_shape
            or target.ndim != 2
            or value.shape[1] != target.shape[1]
            or target.requires_grad
            or value.dtype not in {torch.bfloat16, torch.float32}
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid frozen shared token weight: {name}")
        indices = getattr(module, "shard_indices", None)
        if indices is None:
            width = (value.shape[0] + tp_size - 1) // tp_size
            if target.shape[0] != width:
                raise ValueError(f"Shared token partition differs: {name}")
            start, end = tp_rank * width, min((tp_rank + 1) * width, value.shape[0])
        else:
            start, end = (
                int(indices.org_vocab_start_index),
                int(indices.org_vocab_end_index),
            )
        if not 0 <= start <= end <= value.shape[0] or end - start > target.shape[0]:
            raise ValueError(f"Shared token vocabulary interval differs: {name}")
        shard = torch.zeros(tuple(target.shape), dtype=target.dtype, device="cpu")
        shard[: end - start].copy_(value.detach().cpu()[start:end])
        prepared.append((target, shard))
    with torch.no_grad():
        for target, shard in prepared:
            target.copy_(shard.to(target.device))
    from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

    refresh_fp32_weight_caches(wrapper)


def apply_target_output(teacher, state, *, tp_rank, tp_size):
    """Refresh pre-HC -> logits, including the padded vocab shard when needed."""
    if set(state) != set(TARGET_OUTPUT_NAMES):
        raise ValueError(
            "DeepSeek V4 teacher synchronization requires HC, norm and LM head"
        )
    params = dict(teacher.named_parameters())
    if set(params) != set(state):
        raise ValueError("DeepSeek V4 teacher parameter schema changed")
    prepared = {}
    for name, target in params.items():
        value = state[name]
        if name == "lm_head.weight" and tuple(value.shape) != tuple(target.shape):
            if (
                value.ndim != 2
                or value.shape[1] != target.shape[1]
                or not 0 <= tp_rank < tp_size
            ):
                raise ValueError("Target vocabulary shard topology differs")
            width = target.shape[0]
            if width != (value.shape[0] + tp_size - 1) // tp_size:
                raise ValueError("Target vocabulary partition differs")
            shard = torch.zeros_like(target, device="cpu")
            source = value[tp_rank * width : (tp_rank + 1) * width]
            shard[: len(source)].copy_(source)
            value = shard
        if tuple(value.shape) != tuple(target.shape) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid target output tensor: {name}")
        prepared[name] = value
    with torch.no_grad():
        for name, target in params.items():
            target.copy_(prepared[name].to(device=target.device, dtype=target.dtype))


def runtime_weight_name(rank, size, name):
    return f"__speco_tp_{rank}_of_{size}__/{name}"


class EmbeddedDraftLoader:
    """Copy already processed runtime shards, preserving live NPU storage.

    Training and serving use the same DSpark class and TP degree. Transferring
    runtime tensors avoids loading checkpoint-oriented matrices into parameters
    that Ascend has already transposed/postprocessed.
    """

    def __init__(self, model, tp_rank=0, tp_size=1):
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid embedded draft TP topology")
        self.model, self.tp_rank, self.tp_size = model, tp_rank, tp_size
        self.params = dict(model.named_parameters())
        self.required = {
            name
            for name in self.params
            if "embed_tokens" not in name and not name.startswith("lm_head.")
        }
        self.signatures = {
            name: (tuple(p.shape), p.data_ptr()) for name, p in self.params.items()
        }
        self.received = set()
        self.loaded = set()
        self.lanes = set()
        from verl_speco.integration.deepseek_v4_expert_weights import (
            RuntimeExpertAssembler,
        )

        self.experts = RuntimeExpertAssembler(model, tp_size)
        from verl_speco.ops.ascend_weight_cache import FP32WeightCaches

        self.caches = FP32WeightCaches(model)
        self.pending = {}
        self.failed = False
        self.committed = False

    def receive(self, weights):
        if self.failed or self.committed:
            raise RuntimeError("Embedded draft update session is closed")
        try:
            self._receive(weights)
        except Exception:
            self.failed = True
            self.pending.clear()
            self.experts.clear()
            raise

    def _receive(self, weights):
        import re

        for encoded, value in weights:
            match = re.fullmatch(r"__speco_tp_(\d+)_of_(\d+)__/(.+)", encoded)
            if match is None:
                raise ValueError(
                    "Embedded DS4 update requires versioned runtime TP shards"
                )
            rank, size, name = int(match[1]), int(match[2]), match[3]
            if size != self.tp_size or not 0 <= rank < size:
                raise ValueError("Embedded DS4 training/serving TP degrees differ")
            if encoded in self.received:
                raise ValueError("Duplicate embedded draft update tensor")
            self.received.add(encoded)
            self.lanes.add(rank)
            target = self.params.get(name)
            if name not in self.required or target is None:
                raise ValueError(f"Unexpected embedded DS4 runtime tensor: {name}")
            if name in self.experts.specs:
                result = self.experts.add(name, rank, value)
                if result is not None:
                    self.pending[name] = result
                    self.loaded.add(name)
                continue
            if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                raise ValueError(f"Embedded DS4 runtime layout differs: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite embedded DS4 update: {name}")
            if rank == self.tp_rank:
                # IPC buckets are reused: own a CPU snapshot until validation completes.
                self.pending[name] = value.detach().to("cpu").clone()
                self.loaded.add(name)

    def finish(self):
        missing = self.required - self.loaded
        expected = {
            runtime_weight_name(rank, self.tp_size, name)
            for rank in range(self.tp_size)
            for name in self.required
        }
        if (
            self.failed
            or self.committed
            or not self.required
            or missing
            or self.received != expected
        ):
            raise RuntimeError(
                f"Incomplete embedded DS4 draft update: {sorted(missing)[:8]}"
            )
        current = {
            name: (tuple(p.shape), p.data_ptr())
            for name, p in self.model.named_parameters()
        }
        if current != self.signatures:
            raise RuntimeError("Embedded DS4 update changed parameter storage")
        self.experts.validate()
        previous_caches = self.caches.snapshot()
        # Keep live parameter storage stable, and restore this rank on copy failure.
        previous = {
            name: self.params[name].detach().to("cpu").clone() for name in self.pending
        }
        try:
            with torch.inference_mode():
                for name, value in self.pending.items():
                    self.params[name].copy_(value)
                self.caches.refresh()
        except Exception:
            self.failed = True
            with torch.inference_mode():
                for name, value in previous.items():
                    self.params[name].copy_(value)
                self.caches.restore(previous_caches)
            raise
        finally:
            self.pending.clear()
            self.experts.clear()
        self.committed = True
        return len(self.loaded)


def prepare_online_resume(training, checkpoint_root, actor_step):
    """Resolve the existing RL checkpoint boundary without replacing model_path."""
    import json

    if (
        not training.get("resume_trainer_state_from_checkpoint", True)
        or actor_step is None
    ):
        return
    settings = training.vllm_ascend
    path = settings.get("resume_from")
    if not path and checkpoint_root:
        path = str(Path(checkpoint_root) / f"draft_step_{int(actor_step)}")
    if not path:
        raise ValueError(
            "Resuming RL with online Ascend requires a matching drafter checkpoint"
        )
    metadata = json.loads((Path(path) / "speco_online.json").read_text())
    if int(metadata["global_step"]) != int(actor_step):
        raise ValueError("Online Ascend checkpoint does not match the actor RL step")
    settings.resume_from = path


def export_target_output(worker, *, keep_model_on_device=False):
    """Reuse actor engine export or VeOmni's existing activation helpers."""
    from verl_speco.integration.rollout_publish import actor_training_backend_name

    engine = worker.actor.engine
    backend = actor_training_backend_name(worker.config)
    if backend == "fsdp2":
        # FSDP2 exposes DTensors here. Gather only the selected output/token
        # parameters, rather than consuming a complete target weight export.
        offloaded = bool(getattr(engine, "_is_offload_param", False))
        if offloaded and keep_model_on_device:
            from verl.utils.device import get_device_name

            engine.to(device=get_device_name(), model=True, optimizer=False, grad=False)
        return select_target_output(
            engine.module.named_parameters(), include_shared_tokens=True
        )
    if backend != "veomni":
        tensors, _ = engine.get_per_tensor_param(
            layered_summon=getattr(worker, "layered_summon", False), base_sync_done=True
        )
        return select_target_output(tensors, include_shared_tokens=True)
    module = engine.module
    offloaded = bool(getattr(engine, "_is_offload_param", False))
    if offloaded:
        from verl.workers.engine.veomni.utils import (
            load_veomni_model_to_gpu,
            offload_veomni_model_to_cpu,
        )

        load_veomni_model_to_gpu(module)
    try:
        return select_target_output(
            module.named_parameters(), include_shared_tokens=True
        )
    finally:
        if offloaded and not keep_model_on_device:
            offload_veomni_model_to_cpu(module)
