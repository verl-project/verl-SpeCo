# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Reuse SpeCo store segmentation/DP sampling with an explicit MTP contract."""

import torch

from verl_speco.trainer.deepseek_v4_dataset import StoreDataset


class MTPStoreDataset(StoreDataset):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            raise

    def __getitem__(self, index):
        segment = self.segments[index]
        sample = self.store.read(self.keys[segment.sample_index])
        self.contract.validate(sample)
        start, end = segment.start, segment.end
        mask = torch.zeros(end - start, dtype=torch.float32)
        a, b = segment.loss_start - start, segment.loss_end - start
        mask[a:b] = sample.loss_mask[segment.loss_start : segment.loss_end]
        hidden = sample.hidden_states[start:end]
        return {
            "input_ids": sample.input_ids[start:end],
            "positions": sample.position_ids[start:end],
            "loss_mask": mask,
            "hidden_states": hidden,
            # Reuse the existing collator ABI; this is the same pre-HC tensor,
            # not a second feature or a DSpark auxiliary-state conversion.
            "target_head_hidden_states": hidden,
        }
