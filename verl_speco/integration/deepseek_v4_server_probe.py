# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real verl standalone HTTP/Ray server, without an RL trainer or actor model."""

import asyncio
import json
import os

from verl_speco.integration.serving_drafter_commit import collective_rpc_all_dp


class ModelVocabularyTokenizer:
    """Delegate tokenization while keeping verl's model-mask size authoritative."""

    def __init__(self, tokenizer, vocab_size):
        if len(tokenizer) < vocab_size:
            raise ValueError("Tokenizer does not cover the model vocabulary")
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size

    def __len__(self):
        return self._vocab_size

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tokenizer"), name)


def probe_drafter_config(options):
    """Keep embedded weight updates; direct rollout capture was removed."""
    if options.get("feature_store"):
        raise ValueError(
            "Rollout hidden-state capture has been removed; use standalone replay"
        )
    return {
        "training": {
            "backend": "vllm_ascend",
            "collect_hidden_states_from_rollout": False,
            "rollout_capture": {"model_family": "deepseek_v4"},
        }
    }


def server_configs(model, tp_size, dp_size, max_model_len):
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )

    install_verl_npu_vllm_import_compat()
    from importlib.metadata import version

    from packaging.version import Version

    if Version(version("verl")) != Version("0.9.0") or Version(
        version("vllm")
    ).release not in {(0, 26, 0), (0, 27, 1)}:
        raise RuntimeError(
            "Standalone server probe requires verl 0.9.0 and a reviewed "
            "vLLM 0.26.0 or 0.27.1 Ascend runtime"
        )
    from verl_speco.integration.ascend_runtime_compat import require_ascend_runtime

    require_ascend_runtime(version("vllm"), version("vllm-ascend"))
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.config.model import MtpConfig
    from vllm.tokenizers import get_tokenizer

    from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

    contract = DeepSeekV4Contract.from_model(model)
    config = RolloutConfig(
        name="vllm",
        nnodes=1,
        n_gpus_per_node=tp_size * dp_size,
        tensor_model_parallel_size=tp_size,
        data_parallel_size=dp_size,
        expert_parallel_size=tp_size * dp_size,
        max_model_len=max_model_len,
        prompt_length=max_model_len - 8,
        response_length=8,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enable_sleep_mode=False,
        load_format="auto",
        engine_kwargs={
            "vllm": {
                "tokenizer_mode": "deepseek_v4",
                "block_size": 128,
                "async_scheduling": False,
                "additional_config": {
                    "enable_flashcomm1": False,
                    "enable_fused_mc2": False,
                },
                "speculative_config": {
                    "method": "dspark",
                    "num_speculative_tokens": contract.block_size,
                    "enforce_eager": True,
                },
            }
        },
    )
    # Preserve DSpark stage counts; this enables parameter loading, not MTP rollout.
    model_config = HFModelConfig(
        path=model,
        load_tokenizer=False,
        mtp=MtpConfig(enable=True, enable_rollout=False),
    )
    model_config.tokenizer = ModelVocabularyTokenizer(
        get_tokenizer(model, tokenizer_mode="deepseek_v4"), contract.vocab_size
    )
    return config, model_config


def create_server_actor_class():
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )

    # Importing the upstream actor already imports verl's legacy NPU patch;
    # waiting until server_configs/__init__ is too late in the driver process.
    install_verl_npu_vllm_import_compat()
    import ray
    from verl.workers.rollout.vllm_rollout import vllm_async_server as upstream

    from verl_speco.integration.vllm_runtime import (
        _build_speco_vllm_http_server_class,
        _remote_speco_vllm_http_server,
    )

    base = _build_speco_vllm_http_server_class(upstream)

    class DraftServerProbe(base):
        def __init__(self, options):
            from verl.workers.rollout.replica import RolloutMode

            from verl_speco.integration.drafter_config_env import set_drafter_config_env

            for name, value in options.get("hccl_environment", {}).items():
                if name not in {
                    "HCCL_NPU_SOCKET_PORT_RANGE",
                    "HCCL_HOST_SOCKET_PORT_RANGE",
                }:
                    raise ValueError(
                        f"Unexpected probe communication environment: {name}"
                    )
                os.environ[name] = value
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = options["devices"]
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
            os.environ.setdefault("VLLM_ASCEND_APPLY_DSV4_PATCH", "1")
            set_drafter_config_env(json.dumps(probe_drafter_config(options)))
            config, model_config = server_configs(
                options["model"],
                options["tp_size"],
                options["dp_size"],
                options["max_model_len"],
            )
            super().__init__(
                config,
                model_config,
                RolloutMode.STANDALONE,
                [None] * (options["tp_size"] * options["dp_size"]),
                replica_rank=0,
                node_rank=0,
                gpus_per_node=options["tp_size"] * options["dp_size"],
                nnodes=1,
                cuda_visible_devices=options["devices"],
            )
            self._server_address = "127.0.0.1"

        # Explicit public overrides keep Ray's method table on the SpeCo
        # wrapper instead of an inherited upstream implementation.
        async def launch_server(self, *args, **kwargs):
            return await super().launch_server(*args, **kwargs)

        async def run_server(self, args):
            return await super().run_server(args)

        async def generate(self, *args, **kwargs):
            return await super().generate(*args, **kwargs)

        async def probe_generate_on_dp(
            self, prompt_ids, sampling_params, request_id, dp_rank
        ):
            """Exercise each DP leader through the real verl generate/capture path."""
            if not 0 <= dp_rank < self.config.data_parallel_size:
                raise ValueError("Probe DP rank is out of range")
            if not hasattr(self, "_probe_request_routes"):
                self._probe_request_routes = {}
                original = self.engine.generate

                def routed(*args, **kwargs):
                    route = self._probe_request_routes.get(kwargs.get("request_id"))
                    if route is not None:
                        kwargs["data_parallel_rank"] = route
                    return original(*args, **kwargs)

                self.engine.generate = routed
            if request_id in self._probe_request_routes:
                raise ValueError("Duplicate routed probe request")
            self._probe_request_routes[request_id] = dp_rank
            try:
                return await self.generate(prompt_ids, sampling_params, request_id)
            finally:
                self._probe_request_routes.pop(request_id, None)

        def _get_worker_extension_cls(self):
            return "verl_speco.integration.deepseek_v4_ipc_probe.StandaloneIPCDraftWorkerExtension"

        async def probe_rpc(self, method, kwargs=None):
            return await collective_rpc_all_dp(
                self.engine,
                method,
                dp_size=self.config.data_parallel_size,
                kwargs=kwargs,
            )

        async def probe_shutdown(self):
            task = getattr(self, "_server_task", None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            engine = getattr(self, "engine", None)
            if engine is not None:
                engine.shutdown(timeout=15)
            for name in ("_master_sock", "_dp_rpc_sock", "_dp_master_sock"):
                sock = getattr(self, name, None)
                if sock is not None:
                    sock.close()
            return {"status": "closed"}

    return _remote_speco_vllm_http_server(ray, DraftServerProbe)
