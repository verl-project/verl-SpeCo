"""Compare real cached draft calls to causal replay, including masked depths."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn.attention.flex_attention import create_block_mask

from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="tiny-peagle")
parser.add_argument(
    "--reference", type=Path, default=Path("/experiment/tiny-peagle/draft-original")
)
parser.add_argument(
    "--evidence-root", type=Path, default=Path("/experiment/evidence/l20-20260919")
)
args = parser.parse_args()
evidence = args.evidence_root
records = torch.load(evidence / f"{args.prefix}-forwards.pt", weights_only=True)
model = LlamaForCausalLMPeagle(PeagleConfig.from_pretrained(args.reference))
model.load_state_dict(load_file(args.reference / "model.safetensors"))
model = model.cuda().to(torch.bfloat16).eval()
history = {}
results = []


def causal(batch, head, query, key):
    return query >= key


for record in records:
    metadata_rows = list(record["metadata"].values())
    valid = metadata_rows[0]["slot_mapping"] >= 0
    padding = int((~valid).sum())
    inputs = {key: value[valid] for key, value in record["inputs"].items()}
    positions = inputs["positions"]
    start, end = int(positions[0]), int(positions[-1]) + 1
    torch.testing.assert_close(positions, torch.arange(start, end))
    for metadata in record["metadata"].values():
        torch.testing.assert_close(metadata["slot_mapping"] >= 0, valid)
        assert metadata["query_start_loc"].tolist() == [0, valid.numel()]
        assert metadata["seq_lens"].tolist() == [end + padding]
    if start == 0:
        history = {key: value for key, value in inputs.items()}
    else:
        assert history["input_ids"].numel() >= start
        history = {
            key: torch.cat((history[key][:start], value))
            for key, value in inputs.items()
        }
    mask = create_block_mask(causal, B=1, H=None, Q_LEN=end, KV_LEN=end, device="cuda")
    with torch.no_grad():
        hidden = model.forward_peagle(
            history["input_ids"].long().cuda().unsqueeze(0),
            history["hidden_states"].cuda().unsqueeze(0),
            history["positions"].cuda().unsqueeze(0),
            mask,
        )
        actual = model.compute_logits(hidden)[0, start:end].float().cpu()
    expected = record["logits"][valid].float()
    result = {
        "start": start,
        "end": end,
        "padding_rows": padding,
        "masked_rows": int((inputs["input_ids"] == model.config.mask_token_id).sum()),
        "max_logit_error": float((actual - expected).abs().max()),
        "argmax_equal": torch.equal(actual.argmax(-1), expected.argmax(-1)),
    }
    results.append(result)
    (evidence / f"{args.prefix}-decode-parity.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    assert result["argmax_equal"]
print(
    json.dumps(
        {
            "calls": len(results),
            "max_logit_error": max(row["max_logit_error"] for row in results),
        }
    )
)
