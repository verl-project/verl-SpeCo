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
"""SpecoTrainingWorker — SFT TrainingWorker with SPECO hidden-state collection patch.

Uses INSTANCE-LEVEL patching (not class-level) to ensure the patch is always
active regardless of MRO or import-order issues. This is the same approach
validated in the working SFT-CoTrain implementation.

策略：
1. super().__init__ 创建 engine 后，直接 patch engine 实例的 prepare_model_inputs/outputs
2. patched prepare_model_outputs 将 hidden refs 存入 engine._speco_pending_hidden
3. 重写 _postprocess_output，从 engine._speco_pending_hidden 注入 hidden refs 到 final_output
"""

from __future__ import annotations

import logging
import os
import types

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_SPECO_LOGGING_LEVEL", "WARN"))


def _direct_patch_engine_instance(engine) -> bool:
    """Directly patch the engine INSTANCE's prepare_model_inputs/outputs.

    This bypasses class-level patching issues (method signature changes,
    MRO ordering, import order) by patching the bound instance methods directly.
    """
    try:
        import importlib

        from verl_speco.integration.oldlogprob_runtime import (
            OLD_LOGPROB_COLLECT_MASK_KEY,
            OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
            OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
            OLD_LOGPROB_HIDDEN_REF_META_KEY,
            OLD_LOGPROB_HIDDEN_REFS_KEY,
            OLD_LOGPROB_HIDDEN_STATES_KEY,
            OLD_LOGPROB_SAMPLE_INDICES_KEY,
            _consume_oldlogprob_hidden_capture,
            _install_oldlogprob_fsdp_batch_postprocess_patch,
            _install_oldlogprob_hidden_hooks,
            _oldlogprob_capture_impl,
            _oldlogprob_hidden_object_ref_enabled,
            _put_oldlogprob_hidden_refs,
            _select_oldlogprob_hidden_states,
            _tensor_key_present,
        )

        # Install batch postprocess patch (needed for ObjectRef handling)
        try:
            transformer_module = importlib.import_module(
                "verl.workers.engine.fsdp.transformer_impl"
            )
            _install_oldlogprob_fsdp_batch_postprocess_patch(transformer_module)
        except Exception:
            pass

        if getattr(engine, "_speco_instance_patched", False):
            return True

        orig_inputs = engine.prepare_model_inputs
        orig_outputs = engine.prepare_model_outputs
        logger.warning(
            "[SpecoTrainingWorker] Patching engine instance: %s", type(engine).__name__
        )

        def patched_prepare_inputs(self, micro_batch, *args, **kwargs):
            model_inputs, output_args = orig_inputs.__func__(
                self, micro_batch, *args, **kwargs
            )
            if _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
                capture_impl = _oldlogprob_capture_impl(micro_batch)
                model_inputs["return_dict"] = True
                if capture_impl == "output_hidden_states":
                    model_inputs["output_hidden_states"] = True
                elif capture_impl == "forward_hook":
                    _install_oldlogprob_hidden_hooks(engine, output_args, micro_batch)
            return model_inputs, output_args

        def patched_prepare_outputs(
            self,
            output,
            output_args,
            micro_batch,
            logits_processor_func,
            *args,
            **kwargs,
        ):
            model_output = orig_outputs.__func__(
                self,
                output,
                output_args,
                micro_batch,
                logits_processor_func,
                *args,
                **kwargs,
            )
            if _tensor_key_present(micro_batch, OLD_LOGPROB_COLLECT_MASK_KEY):
                capture_impl = _oldlogprob_capture_impl(micro_batch)
                if capture_impl == "forward_hook":
                    hidden_output = _consume_oldlogprob_hidden_capture(engine)
                elif capture_impl == "output_hidden_states":
                    hidden_output = _select_oldlogprob_hidden_states(
                        engine, output, output_args, micro_batch
                    )
                else:
                    raise ValueError(f"Unknown capture_impl: {capture_impl}")

                if not hidden_output:
                    if _oldlogprob_hidden_object_ref_enabled(micro_batch):
                        return model_output
                    raise RuntimeError(
                        "SPECO old-logprob hidden collection produced no hidden states"
                    )

                if _oldlogprob_hidden_object_ref_enabled(micro_batch):
                    hidden_output = _put_oldlogprob_hidden_refs(
                        hidden_output, micro_batch
                    )
                    # Stamp each ref meta with the sample's original batch index.
                    # Dynamic micro-batching may reorder samples, so the list
                    # order of refs/metas no longer matches the original batch.
                    # The driver maps by meta["batch_idx"] rather than list index.
                    sample_indices = micro_batch.get(OLD_LOGPROB_SAMPLE_INDICES_KEY)
                    if sample_indices is not None:
                        metas = hidden_output.get(OLD_LOGPROB_HIDDEN_REF_META_KEY)
                        if isinstance(metas, list):
                            indices_list = (
                                sample_indices.detach().cpu().reshape(-1).tolist()
                                if hasattr(sample_indices, "detach")
                                else list(sample_indices)
                            )
                            for mb_pos, meta in enumerate(metas):
                                if meta is None:
                                    continue
                                if mb_pos < len(indices_list):
                                    meta["batch_idx"] = int(indices_list[mb_pos])
                else:
                    hidden_output = {
                        k: v for k, v in hidden_output.items() if v is not None
                    }

                # Store hidden refs on engine (accumulate across micro-batches)
                hidden_refs_to_save = {}
                for key in (
                    OLD_LOGPROB_HIDDEN_REFS_KEY,
                    OLD_LOGPROB_HIDDEN_REF_META_KEY,
                    OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY,
                    OLD_LOGPROB_HIDDEN_CHUNK_META_KEY,
                    OLD_LOGPROB_HIDDEN_STATES_KEY,
                    "speco_oldlogprob_timing",
                ):
                    if key in hidden_output:
                        hidden_refs_to_save[key] = hidden_output[key]

                if hidden_refs_to_save:
                    if (
                        not hasattr(engine, "_speco_pending_hidden")
                        or engine._speco_pending_hidden is None
                    ):
                        engine._speco_pending_hidden = hidden_refs_to_save
                    else:
                        for key, value in hidden_refs_to_save.items():
                            if key not in engine._speco_pending_hidden:
                                engine._speco_pending_hidden[key] = value
                            elif isinstance(value, list):
                                engine._speco_pending_hidden[key].extend(value)
                            elif isinstance(value, torch.Tensor):
                                engine._speco_pending_hidden[key] = torch.cat(
                                    [engine._speco_pending_hidden[key], value], dim=0
                                )

            return model_output

        engine.prepare_model_inputs = types.MethodType(patched_prepare_inputs, engine)
        engine.prepare_model_outputs = types.MethodType(patched_prepare_outputs, engine)
        engine._speco_instance_patched = True
        logger.warning("[SpecoTrainingWorker] Engine instance patched successfully")
        return True

    except Exception as exc:
        logger.error("[SpecoTrainingWorker] Direct patch FAILED: %s", exc)
        import traceback

        traceback.print_exc()
        return False


class SpecoTrainingWorker(TrainingWorker):
    """SFT TrainingWorker with SPECO old-logprob hidden-state patch applied."""

    def __init__(self, config):
        # 先创建 engine
        super().__init__(config)
        # 实例级 patch engine（必须在 super().__init__ 之后，因为需要 self.engine）
        success = _direct_patch_engine_instance(self.engine)
        logger.warning("[SpecoTrainingWorker] Instance patch result: %s", success)

    def _postprocess_output(self, output, **kwargs):
        """Override to inject hidden refs from engine._speco_pending_hidden."""
        hidden_data = None
        if (
            hasattr(self.engine, "_speco_pending_hidden")
            and self.engine._speco_pending_hidden is not None
        ):
            hidden_data = self.engine._speco_pending_hidden
            self.engine._speco_pending_hidden = None  # Reset for next step
            logger.debug(
                "[SpecoTrainingWorker] _postprocess_output: captured keys=%s",
                list(hidden_data.keys()),
            )

        final_output = super()._postprocess_output(output, **kwargs)

        if hidden_data:
            for key, value in hidden_data.items():
                try:
                    tu.assign_non_tensor_data(final_output, key, value)
                except Exception as e:
                    logger.warning(
                        "[SpecoTrainingWorker] Failed to inject %s: %s", key, e
                    )

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_lm_head_weight_for_drafter(self, row_indices=None):
        """Export current target model's lm_head weight for drafter backend sync."""
        import sys

        try:
            torch = __import__("torch")

            selected_weight = None
            selected_name = None

            # Strategy 1: engine.get_per_tensor_param() (FSDP engine, works with all-gather)
            if hasattr(self.engine, "get_per_tensor_param"):
                per_tensor_param, _ = self.engine.get_per_tensor_param(
                    layered_summon=False,
                    base_sync_done=True,
                )
                for name, tensor in per_tensor_param:
                    if not torch.is_tensor(tensor):
                        continue
                    name = str(name)
                    if name.endswith(".lm_head.weight") or name == "lm_head.weight":
                        selected_name = name
                        selected_weight = tensor
                        break
                # fallback to embed_tokens.weight if lm_head not found
                if selected_weight is None:
                    for name, tensor in per_tensor_param:
                        if torch.is_tensor(tensor) and (
                            name.endswith(".embed_tokens.weight")
                            or name == "model.embed_tokens.weight"
                        ):
                            selected_name = name
                            selected_weight = tensor
                            break

            # Strategy 2: direct model access
            if selected_weight is None and hasattr(self.engine, "module"):
                model = self.engine.module
                if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                    selected_weight = model.lm_head.weight
                    selected_name = "lm_head.weight"
            if selected_weight is None and hasattr(self.engine, "model"):
                model = self.engine.model
                if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                    selected_weight = model.lm_head.weight
                    selected_name = "lm_head.weight"

            if selected_weight is None:
                print(
                    "[SpecoTrainingWorker] export_lm_head_weight: FAILED - no lm_head found",
                    file=sys.stderr,
                    flush=True,
                )
                return None

            # Only rank 0 exports
            rank = getattr(self, "rank", None)
            if rank is not None and rank != 0:
                return None

            source_vocab_size = int(selected_weight.shape[0])

            # Apply row selection
            if row_indices is not None:
                if isinstance(row_indices, (list, tuple)):
                    row_indices = torch.tensor(
                        [int(i) for i in row_indices], dtype=torch.long
                    )
                if torch.is_tensor(row_indices) and row_indices.numel() > 0:
                    row_indices = row_indices.to(
                        device=selected_weight.device, dtype=torch.long
                    )
                    if row_indices.numel() < source_vocab_size:
                        selected_weight = selected_weight.index_select(0, row_indices)
                        row_indices = (
                            row_indices.detach()
                            .to(device="cpu", dtype=torch.long)
                            .contiguous()
                        )

            # 强制转 bfloat16
            try:
                target_dtype = None
                if hasattr(self.engine, "module"):
                    for _p in list(self.engine.module.parameters())[:5]:
                        if _p.dtype in (torch.bfloat16, torch.float16):
                            target_dtype = _p.dtype
                            break
                if target_dtype is None and hasattr(self.engine, "model"):
                    for _p in list(self.engine.model.parameters())[:5]:
                        if _p.dtype in (torch.bfloat16, torch.float16):
                            target_dtype = _p.dtype
                            break
                if target_dtype is None:
                    target_dtype = getattr(self.engine, "torch_dtype", None)
                if target_dtype is not None and selected_weight.dtype != target_dtype:
                    selected_weight = selected_weight.to(target_dtype)
            except Exception as _dtype_err:
                print(
                    f"[SpecoTrainingWorker] dtype detection error (non-fatal): {_dtype_err}",
                    file=sys.stderr,
                    flush=True,
                )

            payload = {
                "weight": selected_weight.detach().cpu().contiguous(),
                "row_indices": row_indices.detach().cpu()
                if torch.is_tensor(row_indices)
                else None,
                "source_vocab_size": source_vocab_size,
                "name": selected_name,
                "export_strategy": "direct_sparse"
                if row_indices is not None
                and torch.is_tensor(row_indices)
                and row_indices.numel() > 0
                else "full",
            }
            print(
                f"[SpecoTrainingWorker] export_lm_head_weight: OK shape={tuple(payload['weight'].shape)} name={selected_name}",
                file=sys.stderr,
                flush=True,
            )
            return payload
        except Exception as e:
            import traceback

            print(
                f"[SpecoTrainingWorker] export_lm_head_weight: ERROR {e}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            return None
