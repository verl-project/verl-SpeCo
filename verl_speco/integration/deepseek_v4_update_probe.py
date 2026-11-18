# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone prepare/commit/rollback probe using the production draft loader."""

import hashlib
from pathlib import Path

import torch

from verl_speco.integration.deepseek_v4_online import (
    EmbeddedDraftLoader,
    runtime_weight_name,
)


def tensor_digest(tensor):
    value = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for start in range(0, value.numel(), 4 * 1024**2):
        digest.update(value[start : start + 4 * 1024**2].numpy().tobytes())
    return digest.hexdigest()


def _canonical(name):
    while name.startswith("model.model."):
        name = name[len("model.") :]
    return name.replace(".experts.routed_experts.", ".experts.")


class DraftUpdateSession:
    def __init__(self, model, checkpoint, tp_rank, tp_size):
        from verl_speco.models.deepseek_v4.export import (
            _rank_state_files,
            _topology_rank_state_files,
        )

        self.model = model
        self.loader = EmbeddedDraftLoader(model, tp_rank, tp_size)
        self.backup = {}
        self.committed = False
        params = self.loader.params
        primary = {id(param): name for name, param in params.items()}
        aliases = {
            _canonical(alias): primary[id(param)]
            for alias, param in model.named_parameters(remove_duplicate=False)
        }
        self.aliases = aliases
        self.max_tensor_bytes = 0
        baseline = {name: tensor_digest(params[name]) for name in self.loader.required}
        checkpoint = Path(checkpoint)
        files = _topology_rank_state_files(checkpoint, _rank_state_files(checkpoint), 0)
        if len(files) != tp_size:
            raise ValueError("Training and serving attention TP must match")
        for rank, filename in enumerate(files):
            state = torch.load(
                filename, map_location="cpu", weights_only=True, mmap=True
            )
            seen = {}
            for source, value in state.items():
                name = aliases.get(_canonical(source))
                if name not in self.loader.required:
                    continue
                if name in seen:
                    if not torch.equal(seen[name], value):
                        raise ValueError(
                            f"Conflicting checkpoint parameter aliases: {name}"
                        )
                    continue
                seen[name] = value
                self.max_tensor_bytes = max(
                    self.max_tensor_bytes, value.numel() * value.element_size()
                )
                self.loader.receive([(runtime_weight_name(rank, tp_size, name), value)])
            del state, seen
        # Validate completeness without committing the prepared tensors.
        expected = {
            runtime_weight_name(rank, tp_size, name)
            for rank in range(tp_size)
            for name in self.loader.required
        }
        if (
            self.loader.received != expected
            or self.loader.loaded != self.loader.required
        ):
            raise ValueError("Checkpoint does not cover all draft runtime parameters")
        self.loader.experts.validate()
        if any(
            tensor_digest(params[name]) != value for name, value in baseline.items()
        ):
            raise RuntimeError("Draft changed during preparation")
        self.expected = {
            name: tensor_digest(value) for name, value in self.loader.pending.items()
        }

    def refresh(self):
        from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

        refresh_fp32_weight_caches(self.model)
        if callable(
            getattr(getattr(self.model, "model", None), "_build_fused_kv_buffers", None)
        ):
            from verl_speco.integration.vllm_runtime import (
                SpecoVLLMColocateWorkerExtension,
            )

            SpecoVLLMColocateWorkerExtension._speco_rebuild_draft_metadata_buffers(
                self.model
            )

    def commit(self):
        if self.committed:
            raise RuntimeError("Probe update already committed")
        self.backup = {
            name: self.loader.params[name].detach().cpu().clone()
            for name in self.loader.required
        }
        try:
            count = self.loader.finish()
            if any(
                tensor_digest(self.loader.params[name]) != value
                for name, value in self.expected.items()
            ):
                raise RuntimeError("Committed draft differs from staged weights")
            self.refresh()
            self.committed = True
            return count
        except Exception:
            self.rollback()
            raise

    def rollback(self):
        with torch.inference_mode():
            for name, value in self.backup.items():
                self.loader.params[name].copy_(value)
        if self.backup:
            self.refresh()
        self.backup.clear()
        self.loader.pending.clear()
        self.loader.experts.clear()


def worker_update_probe(worker, operation, checkpoint=None, *, tp_rank, tp_size):
    """Return builtin results through vLLM RPC; keep rollback state until finalize."""
    session = getattr(worker, "_speco_draft_update_probe", None)
    try:
        if operation == "prepare":
            if session is not None:
                raise RuntimeError("Another draft update is pending")
            runner = worker.model_runner
            proposer = getattr(runner, "drafter", None)
            if proposer is None:
                proposer = getattr(runner, "speculator", None)
            if proposer is None:
                raise RuntimeError("No DSpark proposer found")
            getter = getattr(proposer, "get_model", None)
            model = getter() if callable(getter) else proposer.model
            session = DraftUpdateSession(model, checkpoint, tp_rank, tp_size)
            worker._speco_draft_update_probe = session
            return {
                "status": "prepared",
                "tp_rank": tp_rank,
                "expert_tensors": len(session.loader.experts.specs),
            }
        if operation == "rollback":
            if session is not None:
                session.rollback()
                worker._speco_draft_update_probe = None
            return {"status": "rolled_back", "tp_rank": tp_rank}
        if session is None:
            raise RuntimeError("No prepared draft update")
        if operation == "commit":
            return {
                "status": "committed",
                "tp_rank": tp_rank,
                "num_tensors": session.commit(),
            }
        if operation == "finalize":
            if not session.committed:
                raise RuntimeError("Cannot finalize uncommitted update")
            session.backup.clear()
            worker._speco_draft_update_probe = None
            return {"status": "finalized", "tp_rank": tp_rank}
        raise ValueError("Unknown draft update phase")
    except Exception as error:  # noqa: BLE001 -- report worker failures to the coordinator
        return {"status": "failed", "tp_rank": tp_rank, "error": repr(error)}
