"""Build a standalone SPECO feature store from a frozen target model.

Stage 1 of the separate-training workflow normally fills this store from an RL
rollout. This script produces the same store from a frozen target so stage 2
(``python -m verl_speco.draft_train_launcher``) can be exercised on one node,
which is how the offline path is smoke tested for every drafter family.

The layout written here is the one the trainer resolves for the algorithm:
DFlash-family drafters (DFlash, DSpark, Domino) get ``dflash_aux`` context
layers, EAGLE-family drafters (EAGLE-1/2/3, P-EAGLE) get ``eagle3_aux_plus_last``.

Run:
    python ci/standalone_feature_store_smoke.py --target /path/to/target-model \
        --algorithm DOMINO --out /path/to/features --num-samples 32
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from verl_speco.integration.oldlogprob_layer_ids import (
    DFLASH_FAMILY_ALGORITHMS,
    resolve_drafter_hidden_states_layout,
)
from verl_speco.models.dflash.modeling_dflash import build_target_layer_ids
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore

PROMPTS = [
    "Explain why the sky appears blue during the day, in a few sentences.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Summarize the water cycle and its main stages in a short paragraph.",
    "Describe the main differences between TCP and UDP for a networking student.",
    "Give three practical tips for reviewing a large pull request.",
    "Explain gradient accumulation to someone who just learned about batch size.",
    "What is speculative decoding, and why does it speed up inference?",
    "Outline the steps of a code review for a performance-sensitive change.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True, help="Feature store directory")
    parser.add_argument("--algorithm", required=True, help="EAGLE3, EAGLE1, PEAGLE, DFLASH, DSPARK or DOMINO")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-context-layers", type=int, default=5, help="DFlash-family context layers")
    parser.add_argument("--num-aux", type=int, default=3, help="EAGLE-family aux hidden states")
    parser.add_argument("--max-samples-per-shard", type=int, default=16)
    args = parser.parse_args()

    algorithm = str(args.algorithm).strip().upper()
    layout = resolve_drafter_hidden_states_layout(algorithm, {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[features] loading target {args.target}")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16).to(device).eval()
    target_cfg = AutoConfig.from_pretrained(args.target)
    num_layers = int(getattr(target_cfg, "num_hidden_layers"))

    if algorithm in DFLASH_FAMILY_ALGORITHMS:
        target_layer_ids = build_target_layer_ids(args.num_context_layers, num_layers)
    else:
        # EAGLE-family aux layers, matching the per-backend GPU smokes.
        target_layer_ids = [2, num_layers // 2, num_layers - 3][: args.num_aux]
    print(f"[features] algorithm={algorithm} layout={layout} target_layer_ids={target_layer_ids} (of {num_layers})")

    store = TorchShardFeatureStore(
        args.out,
        max_samples_per_shard=int(args.max_samples_per_shard),
        metadata={
            "algorithm": algorithm,
            "target_model_path": args.target,
            "source": "ci_standalone_feature_store_smoke",
        },
        shard_prefix="smoke",
    )

    for index in range(int(args.num_samples)):
        text = PROMPTS[index % len(PROMPTS)]
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = target(input_ids=enc["input_ids"], output_hidden_states=True)

        # HF hidden_states[0] is the embedding output, so layer L lands at L + 1.
        blocks = [out.hidden_states[layer_id + 1][0] for layer_id in target_layer_ids]
        if layout.endswith("_plus_last"):
            blocks.append(out.hidden_states[-1][0])
        hidden_states = torch.cat(blocks, dim=-1).to(torch.bfloat16).cpu()

        input_ids = enc["input_ids"][0].cpu()
        seq_len = int(input_ids.numel())
        sample = DraftFeatureSample(
            algorithm=algorithm,
            input_ids=input_ids,
            loss_mask=torch.ones(seq_len, dtype=torch.float32),
            hidden_states=hidden_states,
            position_ids=torch.arange(seq_len, dtype=torch.long),
            metadata={
                "source": "ci_standalone_feature_store_smoke",
                "global_step": 0,
                "target_model_path": args.target,
                "hidden_states_layout": layout,
                "target_layer_ids": target_layer_ids,
                "use_logits": False,
                "sequence_length": seq_len,
                "loss_tokens": seq_len,
                "full_sequence_length": seq_len,
                "feature_start": 0,
                "feature_end": seq_len,
            },
        )
        store.write_many([sample])

    store.close()
    metadata = store.get_metadata()
    print(
        f"[features] wrote num_samples={metadata['num_samples']} num_shards={metadata['num_shards']} "
        f"hidden_dim={hidden_states.size(-1)} to {args.out}"
    )


if __name__ == "__main__":
    main()
