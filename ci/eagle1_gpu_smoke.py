"""Hardware smoke test for the EAGLE-1 / EAGLE-2 drafter training backend.

Drives the real training path on GPU with a real target model (default
Qwen3-4B): it runs the frozen target forward to produce genuine last-layer
hidden states, assembles the batch exactly the way ``base_trainer`` does for the
EAGLE (``model_type == "eagle3"``) shifted-input path, then builds the draft via
``Eagle1TrainerBackend.build_model`` and runs several optimizer steps.

The draft is cold-started (no pretrained checkpoint), so the useful signals are:
  * ``hidden_loss`` (SmoothL1 feature regression) trending down,
  * ``token_loss`` (soft-CE distillation) trending down,
  * draft-vs-target top-1 agreement trending up.

Run:
    python ci/eagle1_gpu_smoke.py --target /path/to/target-model \
        --algorithm EAGLE1 --steps 100
"""

from __future__ import annotations

import argparse
import tempfile

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain why the sky appears blue during the day.",
    "Write a short function that returns the nth Fibonacci number.",
    "Summarize the water cycle in three sentences.",
    "What are the main differences between TCP and UDP?",
]


def _build_shifted_batch(target, tokenizer, device):
    """Produce one concatenated EAGLE batch from real target hidden states.

    Mirrors base_trainer's eagle3 (shifted) assembly per item:
      hidden_states[t]      = final_hidden[t]        (fc input feature)
      input_ids[t]          = ids[t + 1]             (next token embedded)
      last_hidden_states[t] = final_hidden[t + 1]    (regression target)
      loss_mask[t]          = mask[t + 2]
    """
    hidden_chunks, id_chunks, last_hidden_chunks, mask_chunks, pos_chunks = (
        [],
        [],
        [],
        [],
        [],
    )
    for text in PROMPTS:
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        ids = enc["input_ids"][0]
        with torch.no_grad():
            out = target(input_ids=enc["input_ids"], output_hidden_states=True)
        final_hidden = out.hidden_states[-1][0]  # [S, H]
        seq = ids.size(0)
        train = seq - 2
        if train < 1:
            continue
        hidden_chunks.append(final_hidden[:train])
        id_chunks.append(ids[1 : 1 + train])
        last_hidden_chunks.append(final_hidden[1 : 1 + train])
        mask_chunks.append(torch.ones(train, device=device))
        pos_chunks.append(torch.arange(train, device=device, dtype=torch.long))

    return {
        "input_ids": torch.cat(id_chunks).unsqueeze(0),
        "hidden_states": torch.cat(hidden_chunks).unsqueeze(0),
        "last_hidden_states": torch.cat(last_hidden_chunks).unsqueeze(0),
        "attention_mask": torch.ones(
            1, sum(c.size(0) for c in id_chunks), dtype=torch.long, device=device
        ),
        "loss_mask": torch.cat(mask_chunks).unsqueeze(0),
        "position_ids": torch.cat(pos_chunks).unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", required=True, help="path or HF id of the target causal LM"
    )
    parser.add_argument("--algorithm", default="EAGLE1", choices=["EAGLE1", "EAGLE2"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    batch = _build_shifted_batch(target, tokenizer, device)
    print(
        f"[smoke] batch seq_len={batch['input_ids'].size(1)} hidden={batch['hidden_states'].size(-1)}"
    )

    cfg = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "speculative_algorithm": args.algorithm,
                    "model_path": tempfile.mkdtemp(prefix="eagle1_smoke_cold_start_"),
                    "training": {
                        "use_logits": False,
                        "eagle1_num_hidden_layers": 1,
                        "eagle1_hidden_loss_weight": 1.0,
                        "eagle1_token_loss_weight": 0.1,
                        "eagle1_feature_noise": 0.1,
                        "lr": args.lr,
                    },
                }
            },
            "model": {"path": args.target},
        }
    )

    from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend

    backend = Eagle1TrainerBackend(cfg, target_cfg)
    draft, draft_cfg = backend.build_model()
    draft = draft.to(device).to(torch.bfloat16).train()
    backend.target_model = backend.target_model.to(device).to(torch.bfloat16).eval()
    optimizer = backend.setup_optimizer(draft, cfg.rollout.drafter.training)
    print(
        f"[smoke] algorithm={args.algorithm} draft_layers={len(draft.layers)} "
        f"fc={draft.fc.in_features}->{draft.fc.out_features} "
        f"trainable_params={sum(p.numel() for p in draft.parameters() if p.requires_grad):,}"
    )

    first_hidden = first_token = None
    for step in range(args.steps):
        loss_dict = backend.compute_loss(draft, batch, 0)
        num_tokens = loss_dict["local_num_tokens"].clamp_min(1)
        vloss = loss_dict["total_local_vloss"] / num_tokens
        ploss = loss_dict["total_local_ploss"] / num_tokens
        loss = loss_dict["v_weight"] * vloss + loss_dict["p_weight"] * ploss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            with torch.no_grad():
                predicted_hidden = draft(
                    input_ids=batch["input_ids"],
                    hidden_states=batch["hidden_states"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                )
                pred_logits = backend.target_model(predicted_hidden).float()
                tgt_logits = backend.target_model(batch["last_hidden_states"]).float()
                valid = batch["loss_mask"].bool()
                acc = (
                    (pred_logits.argmax(-1) == tgt_logits.argmax(-1)) & valid
                ).float().sum() / valid.float().sum()
            hv, pv, av = float(vloss), float(ploss), float(acc)
            if first_hidden is None:
                first_hidden, first_token = hv, pv
            print(
                f"[smoke] step {step:3d}  hidden_loss={hv:.4f}  token_loss={pv:.4f}  "
                f"loss={float(loss):.4f}  draft_vs_target_top1={av:.4f}"
            )

    print(
        f"[smoke] DONE  hidden_loss {first_hidden:.4f}->{hv:.4f}  "
        f"token_loss {first_token:.4f}->{pv:.4f}  top1={av:.4f}"
    )


if __name__ == "__main__":
    main()
