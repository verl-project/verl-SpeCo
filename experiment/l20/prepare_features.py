"""Freeze real target forwards for the standalone drafter lifecycle test."""

from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore

root = Path("/experiment/tiny-peagle")
target = LlamaForCausalLM.from_pretrained(root / "target").eval()
store = TorchShardFeatureStore(root / "features-packed")
torch.manual_seed(23)
with torch.no_grad():
    for index in range(8):
        ids = torch.randint(1, 253, (1, 33 + index))
        output = target(ids, output_hidden_states=True)
        store.write_many(
            [
                DraftFeatureSample(
                    input_ids=ids[0],
                    loss_mask=torch.ones_like(ids[0]),
                    hidden_states=torch.cat(
                        (*output.hidden_states[1:4], output.hidden_states[-1]), dim=-1
                    )[0],
                    last_hidden_states=output.hidden_states[-1][0],
                    algorithm="PEAGLE",
                )
            ]
        )
store.flush()
