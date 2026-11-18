# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Run real verl FSDP old-logprob capture without a rollout or RL task.

Loads the full target checkpoint. Synthetic short token sequences exercise
unequal-length padding and a middle feature window; no training data is invented.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


def write_api_fixture(output):
    """Small real HF model for API validation; never a target-weight substitute."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )

    root = Path(output)
    if root.exists():
        raise FileExistsError("Refusing to overwrite actor API fixture")
    torch.manual_seed(123)
    cfg = DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=3,
        moe_intermediate_size=64,
        num_attention_heads=4,
        head_dim=512,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
        compress_ratios=[0, 4, 128],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        hc_mult=4,
        hc_sinkhorn_iters=3,
        max_position_embeddings=128,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_block_size=5,
        attn_implementation="eager",
    )
    model = DeepseekV4ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(root)
    (root / "api_fixture.json").write_text(
        json.dumps({"purpose": "api_only_not_trained_model"})
    )


def run(model, output, *, export_transport=False):
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )

    install_verl_npu_vllm_import_compat()
    from tensordict import TensorDict
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    from verl_speco.integration import oldlogprob_runtime as runtime
    from verl_speco.integration.deepseek_v4_actor_capture import CONTRACT_KEY
    from verl_speco.probe_deepseek_v4_recovery import differences, persistent_state
    from verl_speco.trainer.deepseek_v4_contract import LAYOUT, DeepSeekV4Contract
    from verl_speco.trainer.vllm_ascend_training_loop import _atomic_report

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    root = Path(output).resolve()
    torch.npu.set_device(local_rank)
    torch.set_num_threads(2)
    dist.init_process_group("hccl")
    control = dist.new_group(backend="gloo")
    phase = "initialize"
    handles = []
    try:
        if rank == 0:
            if root.exists() and any(root.iterdir()):
                raise FileExistsError("Use a fresh actor probe output")
            root.mkdir(parents=True, exist_ok=True)
        dist.barrier(group=control)
        contract = DeepSeekV4Contract.from_model(model)
        model_config = HFModelConfig(
            path=model,
            load_tokenizer=False,
            use_remove_padding=False,
            enable_gradient_checkpointing=False,
            use_fused_kernels=False,
            override_config={"attn_implementation": "eager"},
        )
        engine_config = FSDPEngineConfig(
            strategy="fsdp2",
            forward_only=True,
            model_dtype="bf16",
            fsdp_size=dist.get_world_size(),
            use_torch_compile=False,
            ulysses_sequence_parallel_size=1,
            forward_prefetch=False,
            wrap_policy={"transformer_layer_cls_to_wrap": ["DeepseekV4DecoderLayer"]},
        )
        engine = FSDPEngineWithLMHead(
            model_config, engine_config, FSDPOptimizerConfig(), CheckpointConfig()
        )
        engine.initialize()
        if not runtime.install_oldlogprob_hidden_runtime_patch(actor_backend="fsdp2"):
            raise RuntimeError("Could not install production actor capture adapter")
        layers, _ = runtime._find_layers_and_final_norm(engine)
        if len(layers) != model_config.hf_config.num_hidden_layers:
            raise RuntimeError("Actor decoder discovery does not match full target")
        ids_list = [torch.arange(10, 26), torch.arange(40, 52)]
        ids = torch.nested.nested_tensor(ids_list, layout=torch.jagged)
        positions = torch.nested.nested_tensor(
            [torch.arange(len(x)) for x in ids_list], layout=torch.jagged
        )
        batch = TensorDict(
            {"input_ids": ids, "position_ids": positions}, batch_size=[2]
        )
        tu.assign_non_tensor_data(batch, "temperature", 1.0)
        tu.assign_non_tensor_data(batch, "use_remove_padding", False)
        snapshots = {}
        head_snapshots = {}

        if export_transport and rank == 0:

            def capture_logits(_module, _args, value):
                head_snapshots[phase] = value.detach().cpu().clone()

            handles.append(
                engine.module.get_output_embeddings().register_forward_hook(
                    capture_logits
                )
            )

        def tap(index):
            def hook(_module, _args, value):
                raw = value[0] if isinstance(value, tuple) else value
                snapshots[index] = raw.detach().cpu().clone()

            return hook

        for index in sorted(set(contract.target_layer_ids + (len(layers) - 1,))):
            handles.append(layers[index].register_forward_hook(tap(index)))
        phase = "reference_forward"
        with torch.no_grad(), engine.eval_mode():
            _, plain = engine.forward_step(batch, loss_function=None, forward_only=True)
        direct = dict(snapshots)
        window = torch.arange(3, 11).repeat(2, 1)
        batch[runtime.OLD_LOGPROB_COLLECT_MASK_KEY] = torch.tensor([True, True])
        batch[runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = window
        batch[runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = torch.ones_like(
            window, dtype=torch.bool
        )
        for key, value in {
            runtime.OLD_LOGPROB_HIDDEN_LAYOUT_KEY: LAYOUT,
            runtime.OLD_LOGPROB_AUX_LAYER_IDS_KEY: list(contract.target_layer_ids),
            runtime.OLD_LOGPROB_AUX_LAYER_ID_SPACE_KEY: "decoder",
            runtime.OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY: "forward_hook",
            CONTRACT_KEY: contract.identity(),
        }.items():
            tu.assign_non_tensor_data(batch, key, value)
        phase = "captured_forward"
        with torch.no_grad(), engine.eval_mode():
            _, captured = engine.forward_step(
                batch, loss_function=None, forward_only=True
            )
        plain_logprobs = plain["model_output"]["log_probs"].values().cpu()
        captured_logprobs = captured["model_output"]["log_probs"].values().cpu()
        hidden = captured["model_output"][runtime.OLD_LOGPROB_HIDDEN_STATES_KEY].cpu()
        expected_parts = [
            direct[i].reshape(2, 16, contract.hc_mult, contract.hidden_size).mean(-2)
            for i in contract.target_layer_ids
        ]
        expected_parts.append(direct[len(layers) - 1].reshape(2, 16, -1))
        expected = torch.cat(expected_parts, -1)[:, 3:11]
        changes = differences(
            persistent_state({"logprobs": plain_logprobs, "features": expected}),
            persistent_state({"logprobs": captured_logprobs, "features": hidden}),
        )
        result = {
            "rank": rank,
            "status": "passed" if not changes else "different",
            "contract": contract.identity(),
            "different_paths": changes,
            "feature_shape": list(hidden.shape),
            "feature_max_abs": float((hidden.float() - expected.float()).abs().max()),
            "logprob_max_abs": float((plain_logprobs - captured_logprobs).abs().max()),
            "scope": "miniature_api_fixture"
            if (Path(model) / "api_fixture.json").exists()
            else "full_target_real_verl_fsdp2_oldlogprob_synthetic_short_inputs",
            "model_path": str(Path(model).resolve()),
            "rl_training_validated": False,
        }
        if export_transport and not changes:
            from verl_speco.integration.deepseek_v4_online import export_target_output

            phase = "export_actor_target"
            worker = SimpleNamespace(
                actor=SimpleNamespace(engine=engine),
                config={"actor": {"strategy": "fsdp2"}},
            )
            state = export_target_output(worker)
            if any(
                value.requires_grad
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
                for value in state.values()
            ):
                raise RuntimeError("Actor export must own finite detached CPU tensors")
            result["target_output_state"] = persistent_state(state)
            if rank == 0:
                raw_logits = head_snapshots["reference_forward"]
                if tuple(raw_logits.shape) != (2, 16, contract.vocab_size):
                    raise ValueError(
                        f"Unexpected actor LM-head output: {raw_logits.shape}"
                    )
                torch.save(
                    {
                        "contract": contract.identity(),
                        "input_ids": ids_list,
                        "positions": [window[i] for i in range(2)],
                        "logits": [raw_logits[i, 3:11].clone() for i in range(2)],
                        "old_logprobs": plain_logprobs,
                        "scope": "actual_fsdp2_lm_head_output_before_log_softmax",
                        "model_dtype": "bf16",
                    },
                    root / "actor_alignment_reference.pt",
                )
                torch.save(
                    {
                        "ds4_target_output": state,
                        "global_step": 0,
                        "contract": contract.identity(),
                        "scope": result["scope"],
                    },
                    root / "actor_target_output.pt",
                )
                torch.save(
                    {
                        "contract": contract.identity(),
                        "scope": result["scope"],
                        "synthetic_tokens": True,
                        "target_version": 0,
                        "samples": [
                            {
                                "batch": {
                                    "input_ids": ids_list[i].unsqueeze(0),
                                    "prompts": ids_list[i][:4].unsqueeze(0),
                                    "hidden_positions": window[i].unsqueeze(0),
                                    "hidden_states_layout": LAYOUT,
                                    "global_step": 0,
                                },
                                "hidden_states": hidden[i : i + 1],
                            }
                            for i in range(2)
                        ],
                    },
                    root / "actor_feature_transport.pt",
                )
            result["transport_exported"] = True
        _atomic_report(root / f"rank_{rank}.json", result)
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result, group=control)
        if rank == 0:
            _atomic_report(
                root / "actor_capture_report.json",
                {
                    "workers": results,
                    "status": "passed"
                    if all(x["status"] == "passed" for x in results)
                    else "different",
                },
            )
        if any(x["status"] != "passed" for x in results):
            raise RuntimeError("Actor capture differs from independent decoder taps")
    except BaseException as error:
        _atomic_report(
            root / f"failure_rank_{rank}.json", {"phase": phase, "error": repr(error)}
        )
        raise
    finally:
        for handle in handles:
            handle.remove()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--create-api-fixture", action="store_true")
    parser.add_argument("--export-transport", action="store_true")
    args = parser.parse_args()
    if args.create_api_fixture:
        write_api_fixture(args.output)
    elif args.model:
        run(args.model, args.output, export_transport=args.export_transport)
    else:
        parser.error("--model is required for actor execution")


if __name__ == "__main__":
    main()
