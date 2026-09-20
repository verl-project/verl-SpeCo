"""Compare independently started hot and cold trained-checkpoint engines."""

import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("evidence", type=Path)
args = parser.parse_args()
rows = []
for tp in (1, 2):
    for mode in ("eager", "graph"):
        prefix = args.evidence / f"tp{tp}-{mode}"
        hot = json.loads(Path(f"{prefix}-hot.json").read_text())
        cold = json.loads(Path(f"{prefix}-cold.json").read_text())
        assert hot["tokens"] == cold["tokens"]
        assert hot["target_hashes"] == cold["target_hashes"]
        assert len(hot["updates"]) == tp
        errors = []
        if mode == "eager":
            for rank in range(tp):
                a = torch.load(f"{prefix}-hot.rank{rank}.pt", weights_only=True)
                b = torch.load(f"{prefix}-cold.rank{rank}.pt", weights_only=True)
                torch.testing.assert_close(a, b, atol=0.01, rtol=0.02)
                assert torch.equal(a.argmax(-1), b.argmax(-1))
                errors.append((a - b).abs().max().item())
        rows.append(
            {
                "tp": tp,
                "mode": mode,
                "tokens_equal": True,
                "target_equal": True,
                "rank_draft_logits_max_error": errors,
                "logits_checked": mode == "eager",
            }
        )
(args.evidence / "all-rank-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
