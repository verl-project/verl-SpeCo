"""Hardware smoke test for the P-EAGLE (parallel-drafting) drafter backend.

Drives the real training path on GPU with a real target model (default Qwen3-4B):
it collects the UNSHIFTED per-position data P-EAGLE needs (num_aux target aux
hidden states, the final-layer hidden for the target distribution, tokens, loss
mask) from the frozen target, builds the P-EAGLE draft via
``PEagleTrainerBackend.build_model``, and runs several optimizer steps through
``compute_loss`` (COD sampling -> flat multi-depth flex-attention forward ->
count-normalized KL(target || draft)).

The draft is cold-started, so the useful signal is convergence: the KL loss falls
and draft-vs-target top-1 agreement rises across depths.

Run:
    python ci/peagle_gpu_smoke.py --target /path/to/target-model --steps 150
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain why the sky appears blue during the day, in a few sentences.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Summarize the water cycle and its main stages in a short paragraph.",
    "Describe the main differences between TCP and UDP for a networking student.",
]


def _build_batch(target, tokenizer, aux_layer_ids, device):
    id_chunks, mask_chunks, aux_chunks, last_chunks = [], [], [], []
    for text in PROMPTS:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = target(input_ids=enc["input_ids"], output_hidden_states=True)
        aux = torch.cat(
            [out.hidden_states[i][0] for i in aux_layer_ids], dim=-1
        )  # [S, num_aux*H]
        id_chunks.append(enc["input_ids"][0])
        mask_chunks.append(torch.ones(enc["input_ids"].size(1), device=device))
        aux_chunks.append(aux)
        last_chunks.append(out.hidden_states[-1][0])  # [S, H]

    input_ids = torch.cat(id_chunks).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "loss_mask": torch.cat(mask_chunks).unsqueeze(0),
        "hidden_states": torch.cat(aux_chunks).unsqueeze(0).to(torch.bfloat16),
        "last_hidden_states": torch.cat(last_chunks).unsqueeze(0).to(torch.bfloat16),
        "attention_mask": torch.ones_like(input_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-aux", type=int, default=3)
    parser.add_argument("--num-depths", type=int, default=8)
    parser.add_argument("--num-draft-layers", type=int, default=2)
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

    num_layers = int(getattr(target_cfg, "num_hidden_layers"))
    aux_layer_ids = [2, num_layers // 2, num_layers - 3][: args.num_aux]
    print(f"[smoke] aux layers={aux_layer_ids} (of {num_layers})")

    batch = _build_batch(target, tokenizer, aux_layer_ids, device)
    print(
        f"[smoke] batch seq_len={batch['input_ids'].size(1)} aux={batch['hidden_states'].size(-1)}"
    )

    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "PEAGLE",
                    "model_path": "/dev/null/does-not-exist",
                    "training": {
                        "use_logits": False,
                        "peagle_num_draft_layers": args.num_draft_layers,
                        "peagle_num_aux_hidden_states": args.num_aux,
                        "peagle_num_depths": args.num_depths,
                        "lr": args.lr,
                    },
                }
            },
            "model": {"path": args.target},
        }
    )

    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    backend = PEagleTrainerBackend(cfg, target_cfg)
    model, draft_cfg = backend.build_model()
    model = model.to(device).to(torch.bfloat16).train()
    backend.target_model = backend.target_model.to(device).to(torch.bfloat16)
    optimizer = backend.setup_optimizer(model, cfg.rollout.drafter.training)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[smoke] draft_layers={len(model.layers)} num_depths={model.config.num_depths} "
        f"fc={model.fc.in_features}->{model.fc.out_features} trainable_params={n_params:,}"
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

        if step % 15 == 0 or step == args.steps - 1:
            kl = float(loss)
            acc = float(out["accuracy"])
            if first is None:
                first = (kl, acc)
            print(
                f"[smoke] step {step:3d}  kl_loss={kl:.4f}  draft_vs_target_top1={acc:.4f}"
            )

    print(
        f"[smoke] DONE  kl_loss {first[0]:.4f}->{kl:.4f}  top1 {first[1]:.4f}->{acc:.4f}"
    )


if __name__ == "__main__":
    main()
