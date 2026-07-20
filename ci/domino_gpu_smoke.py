"""Hardware smoke test for the Domino drafter training backend.

Drives the real training path on GPU with a real target model (default
Qwen3-4B): it runs the frozen target forward to collect the DFlash-style
multi-layer context hidden states, builds the Domino draft via
``DominoTrainerBackend.build_model``, and runs several optimizer steps through
``compute_loss`` (which invokes the block-drafter forward with the causal GRU
correction head and the dual-logit base-anchor curriculum).

The draft is cold-started, so the useful signals are:
  * ``loss`` / ``final_loss`` (Domino-refined logits CE) trending down,
  * ``base_loss`` (backbone-only logits CE) trending down,
  * ``accuracy`` (final) and ``base_accuracy`` rising,
  * ``lambda_base`` decaying from 1 -> 0 (curriculum handing over to the head).

Run:
    python ci/domino_gpu_smoke.py --target /path/to/target-model --steps 120
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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = target(input_ids=enc["input_ids"], output_hidden_states=True)
        # hidden_states is a tuple of (num_layers + 1) tensors; pick the context layers.
        layers = [out.hidden_states[i][0] for i in target_layer_ids]  # each [S, H]
        id_chunks.append(enc["input_ids"][0])
        mask_chunks.append(torch.ones(enc["input_ids"].size(1), device=device))
        hidden_chunks.append(torch.cat(layers, dim=-1))  # [S, num_ctx*H]

    input_ids = torch.cat(id_chunks).unsqueeze(0)
    loss_mask = torch.cat(mask_chunks).unsqueeze(0)
    hidden = torch.cat(hidden_chunks).unsqueeze(0).to(torch.bfloat16)
    return {"input_ids": input_ids, "loss_mask": loss_mask, "hidden_states": hidden, "attention_mask": torch.ones_like(input_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="path or HF id of the target causal LM")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-context-layers", type=int, default=5)
    parser.add_argument("--lambda-decay-steps", type=int, default=60)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    print(f"[smoke] loading target {args.target}")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16).to(device).eval()
    target_cfg = AutoConfig.from_pretrained(args.target)

    from verl_speco.models.dflash import build_target_layer_ids

    target_layers = int(getattr(target_cfg, "num_hidden_layers"))
    target_layer_ids = build_target_layer_ids(args.num_context_layers, target_layers)
    print(f"[smoke] context layers={target_layer_ids} (of {target_layers})")

    batch = _build_batch(target, tokenizer, target_layer_ids, device)
    print(f"[smoke] batch seq_len={batch['input_ids'].size(1)} hidden={batch['hidden_states'].size(-1)}")

    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DOMINO",
                    "model_path": "/dev/null/does-not-exist",
                    "training": {
                        "domino_block_size": 8,
                        "domino_num_anchors": 128,
                        "domino_num_target_layers": args.num_context_layers,
                        "domino_num_hidden_layers": 1,
                        "domino_lambda_base_decay_steps": args.lambda_decay_steps,
                        "lr": args.lr,
                    },
                }
            },
            "model": {"path": args.target},
        }
    )

    from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend

    backend = DominoTrainerBackend(cfg, target_cfg)
    model, drafter_cfg = backend.build_model()
    model = model.to(device).to(torch.bfloat16).train()
    backend.target_lm_head = backend.target_lm_head.to(device).to(torch.bfloat16)
    optimizer = backend.setup_optimizer(model, cfg.rollout.drafter.training)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[smoke] block_size={model.block_size} gru={model.draft_model.gru_hidden_dim} "
        f"emb_dim={model.draft_model.emb_dim} trainable_params={n_params:,}"
    )

    first = None
    for step in range(args.steps):
        out = backend.compute_loss(model, batch, 0)
        num_tokens = out["local_num_tokens"].clamp_min(1)
        loss = out["total_local_ploss"] / num_tokens
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            d = out["diagnostics"]
            fl = float(d["domino_final_loss"])
            bl = float(d["domino_base_loss"])
            acc = float(out["accuracy"])
            bacc = float(d["domino_base_accuracy"])
            lam = float(d["domino_lambda_base"])
            if first is None:
                first = (fl, bl, acc)
            print(
                f"[smoke] step {step:3d}  final_loss={fl:.4f}  base_loss={bl:.4f}  "
                f"final_acc={acc:.4f}  base_acc={bacc:.4f}  lambda_base={lam:.3f}"
            )

    print(
        f"[smoke] DONE  final_loss {first[0]:.4f}->{fl:.4f}  base_loss {first[1]:.4f}->{bl:.4f}  "
        f"final_acc {first[2]:.4f}->{acc:.4f}"
    )


if __name__ == "__main__":
    main()
