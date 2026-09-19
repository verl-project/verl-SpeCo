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
"""Fixed-COD loss, gradient and optimizer parity for sequence partitions."""

from copy import deepcopy
import os

import pytest
import torch

from verl_speco.backends import peagle_trainer_backend as backend
from verl_speco.models.peagle import LlamaForCausalLMPeagle, PeagleConfig


@pytest.fixture(autouse=True)
def tensor_device():
    with torch.device(os.environ.get("SPECO_TEST_DEVICE", "cpu")):
        yield


@pytest.mark.parametrize("partitions", [1, 2, 3, 20])
@pytest.mark.parametrize("lengths", [[1], [8], [3, 5]])
@pytest.mark.parametrize("draft_vocab", [16, 32])
def test_partition_training_parity(monkeypatch, partitions, lengths, draft_vocab):
    torch.manual_seed(17)
    seq_len = sum(lengths)
    config = PeagleConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_draft_layers=2,
        target_hidden_size=16,
        num_aux_hidden_states=3,
        vocab_size=32,
        draft_vocab_size=draft_vocab,
        num_depths=3,
        mask_token_id=31,
        max_position_embeddings=64,
    )
    flat = backend.PEagleTrainingModel(LlamaForCausalLMPeagle(config), num_depths=3)
    partitioned = deepcopy(flat)
    partitioned.sequence_partitions = partitions
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0][:seq_len]])
    cod = backend.generate_cod_sample_indices(seq_len, mask, num_depths=3)
    monkeypatch.setattr(backend, "generate_cod_sample_indices", lambda **kwargs: cod)
    batch = dict(
        input_ids=torch.randint(0, 31, (1, seq_len)),
        aux_hidden=torch.randn(1, seq_len, 48),
        loss_mask=mask,
        attention_mask=torch.ones(1, seq_len),
        target_logits=torch.randn(1, seq_len, 32),
        seq_lengths=torch.tensor(lengths),
    )
    outputs = []
    for model in (flat, partitioned):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        num, den, hits = model(**batch)
        (num / den.clamp_min(1)).backward()
        outputs.append((num.detach(), den, hits))
        optimizer.step()
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-5)
    for (name, reference), (_, actual) in zip(
        flat.named_parameters(), partitioned.named_parameters()
    ):
        torch.testing.assert_close(
            actual.grad, reference.grad, atol=2e-5, rtol=2e-4, msg=name
        )
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4, msg=name)


def test_partition_count_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        backend.PEagleTrainingModel(
            LlamaForCausalLMPeagle(
                PeagleConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    num_draft_layers=1,
                    vocab_size=32,
                )
            ),
            sequence_partitions=0,
        )
