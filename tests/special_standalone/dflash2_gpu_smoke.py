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
"""Hardware smoke test for the DFlash2 drafter training backend.

Drives the real training path on GPU with a real target model (default
Qwen3-4B): it runs the frozen target forward to collect the DFlash-style
multi-layer context hidden states, builds the DFlash2 draft via
``DFlash2TrainerBackend.build_model``, and runs several optimizer steps through
``compute_loss`` (which invokes the block-drafter forward with the two-tap
dynamic convolutions and the candidate-selector objective).

The draft is cold-started, so the useful signals are:
  * ``loss`` trending down and ``accuracy`` rising (the DFlash CE path, now
    routed through the dynamic convolutions),
  * ``selector_loss`` trending down and ``selector_acc`` rising (the DFlash2
    candidate selector learning to re-rank the drafter's own top-k),
  * ``selector_coverage`` (fraction of scored rows whose ground truth survived
    the drafter's top-k) rising as the backbone improves.

Run:
    python tests/special_standalone/dflash2_gpu_smoke.py \
        --target /path/to/target-model --steps 120
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain why the sky appears blue during the day, in a few sentences.",
    "Write a short Python function that returns the nth Fibonacci number and explain it.",
    "Summarize the water cycle and its main stages in a short paragraph.",
    "Describe the main differences between TCP and UDP for a networking student.",
]


def _build_batch(target, tokenizer, target_layer_ids, device):
    """One packed batch: input_ids, loss_mask, and concatenated context hidden states."""
    id_chunks, mask_chunks, hidden_chunks = [], [], []
    for text in PROMPTS:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = target(input_ids=enc["input_ids"], output_hidden_states=True)
        layers = [out.hidden_states[i][0] for i in target_layer_ids]  # each [S, H]
        id_chunks.append(enc["input_ids"][0])
        mask_chunks.append(torch.ones(enc["input_ids"].size(1), device=device))
        hidden_chunks.append(torch.cat(layers, dim=-1))  # [S, num_ctx*H]

    input_ids = torch.cat(id_chunks).unsqueeze(0)
    loss_mask = torch.cat(mask_chunks).unsqueeze(0)
    hidden = torch.cat(hidden_chunks).unsqueeze(0).to(torch.bfloat16)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden,
        "attention_mask": torch.ones_like(input_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", required=True, help="path or HF id of the target causal LM"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-context-layers", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--selector-top-k", type=int, default=16)
    parser.add_argument("--selector-rank", type=int, default=256)
    parser.add_argument("--selector-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    print(f"[smoke] loading target {args.target}")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = (
        AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    target_cfg = AutoConfig.from_pretrained(args.target)

    from verl_speco.models.dflash import build_target_layer_ids

    target_layers = int(getattr(target_cfg, "num_hidden_layers"))
    target_layer_ids = build_target_layer_ids(args.num_context_layers, target_layers)
    print(f"[smoke] context layers={target_layer_ids} (of {target_layers})")

    batch = _build_batch(target, tokenizer, target_layer_ids, device)
    print(
        f"[smoke] batch seq_len={batch['input_ids'].size(1)} "
        f"hidden={batch['hidden_states'].size(-1)}"
    )

    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DFLASH2",
                    "model_path": "/dev/null/does-not-exist",
                    "training": {
                        "dflash2_block_size": args.block_size,
                        "dflash2_num_anchors": 128,
                        "dflash2_num_target_layers": args.num_context_layers,
                        "dflash2_num_hidden_layers": 1,
                        "dflash2_selector_top_k": args.selector_top_k,
                        "dflash2_selector_rank": args.selector_rank,
                        "dflash2_selector_loss_weight": args.selector_loss_weight,
                        "lr": args.lr,
                    },
                }
            },
            "model": {"path": args.target},
        }
    )

    from verl_speco.backends.dflash2_trainer_backend import DFlash2TrainerBackend

    backend = DFlash2TrainerBackend(cfg, target_cfg)
    model, drafter_cfg = backend.build_model()
    # Match the real training path, which stacks two things: FSDP
    # MixedPrecision(param_dtype=bf16) gives the forward bf16 parameters (so the
    # RMSNorm weights stay bf16 and attention q/k/v agree), and the surrounding
    # torch.amp.autocast keeps cross_entropy in fp32 (so its result can be
    # scattered into the fp32 loss_per_token buffer). Emulate both here.
    model = model.to(device).to(torch.bfloat16).train()
    backend.target_lm_head = backend.target_lm_head.to(device).to(torch.bfloat16)
    optimizer = backend.setup_optimizer(model, cfg.rollout.drafter.training)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv = model.draft_model.layers[0].attention_conv
    print(
        f"[smoke] block_size={model.block_size} conv_kernel={conv.kernel_size} "
        f"conv_group={conv.group_size} selector_rank={drafter_cfg.selector_rank} "
        f"selector_top_k={drafter_cfg.selector_top_k} trainable_params={n_params:,}"
    )

    first = None
    for step in range(args.steps):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = backend.compute_loss(model, batch, 0)
        num_tokens = out["local_num_tokens"].clamp_min(1)
        loss = out["total_local_ploss"] / num_tokens
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            d = out["diagnostics"]
            total = float(loss)
            acc = float(out["accuracy"])
            sel_loss = float(d["selector_loss"])
            sel_tokens = max(float(d["selector_token_count"]), 1.0)
            sel_acc = float(d["selector_correct_count"]) / sel_tokens
            # Unary-only ranking on the same rows. The selector's score starts
            # from the drafter's own logit, so this is the baseline it has to
            # beat; sel_acc alone says nothing once the backbone converges.
            base_acc = float(d["selector_base_correct_count"]) / sel_tokens
            coverage = float(d["selector_coverage_count"]) / max(
                float(d["selector_active_count"]), 1.0
            )
            if first is None:
                first = (total, acc, sel_loss, sel_acc, coverage)
            print(
                f"[smoke] step {step:3d}  loss={total:.4f}  acc={acc:.4f}  "
                f"selector_loss={sel_loss:.4f}  selector_acc={sel_acc:.4f}  "
                f"unary_only_acc={base_acc:.4f}  lift={sel_acc - base_acc:+.4f}  "
                f"selector_coverage={coverage:.4f}"
            )

    print(
        f"[smoke] DONE  loss {first[0]:.4f}->{total:.4f}  acc {first[1]:.4f}->{acc:.4f}  "
        f"selector_loss {first[2]:.4f}->{sel_loss:.4f}  "
        f"selector_acc {first[3]:.4f}->{sel_acc:.4f}  "
        f"selector_coverage {first[4]:.4f}->{coverage:.4f}"
    )


if __name__ == "__main__":
    main()
