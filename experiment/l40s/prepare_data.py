"""Small deterministic arithmetic dataset for the SpeCo online loop."""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("destination", type=Path)
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=True)

rows = []
for index in range(16):
    left, right = 173 + index, 29 + index
    rows.append(
        {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Calculate {left} times {right}. Explain your calculation in two sentences, "
                        "then write #### followed by the final number."
                    ),
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(left * right)},
            "extra_info": {"split": "train", "index": index},
        }
    )
pd.DataFrame(rows).to_parquet(args.destination / "train.parquet")
pd.DataFrame(rows[:4]).to_parquet(args.destination / "val.parquet")
print(f"wrote {len(rows)} train rows and 4 val rows to {args.destination}")
