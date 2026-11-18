# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone vLLM worker extension sharing the live rollout capture adapter."""

from verl_speco.integration.vllm_rollout_capture import AscendRolloutCapture
from verl_speco.transport.rollout_feature_payload import encode_feature_payload


class StandaloneCaptureWorkerExtension:
    def speco_standalone_capture(
        self, operation, request_id=None, prompt_ids=None, final_ids=None, settings=None
    ):
        from vllm.distributed import get_tensor_model_parallel_rank

        # All TP ranks join the RPC; only the leader owns feature transfer.
        if get_tensor_model_parallel_rank() != 0:
            return None
        capture = getattr(self, "_speco_standalone_capture", None)
        if operation == "begin":
            if capture is None:
                capture = AscendRolloutCapture(self.model_runner, settings)
                self._speco_standalone_capture = capture
            capture.collector.begin(request_id, prompt_ids)
        elif operation == "finish":
            return (
                encode_feature_payload(capture.collector.finish(request_id, final_ids))
                if capture
                else None
            )
        elif operation == "cancel":
            if capture:
                capture.collector.cancel(request_id)
        elif operation == "close":
            if capture:
                capture.close()
                self._speco_standalone_capture = None
        else:
            raise ValueError(f"Unknown standalone capture operation: {operation!r}")
        return None

    def speco_standalone_draft_update(self, operation, checkpoint=None):
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        from verl_speco.integration.deepseek_v4_update_probe import worker_update_probe

        return worker_update_probe(
            self,
            operation,
            checkpoint,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
        )
