# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fixed-trajectory replay, positive masks and the existing training store."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.producer.trajectory_reader import (
    iter_trajectories,
    normalize_trajectory,
    prepare_trajectories,
)
from verl_speco.replay_deepseek_v4_features import replay_features
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract
from verl_speco.trainer.deepseek_v4_dataset import StoreDataset


class Tokenizer:
    def encode(self, text, *, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c) % 60 for c in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text, **kwargs)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        result = []
        for message in messages:
            result += [20 if message["role"] == "user" else 21]
            result += self.encode(message["content"], add_special_tokens=False) + [22]
        return result + ([21] if add_generation_prompt else [])


def normalize(payload, **kwargs):
    return normalize_trajectory(
        payload,
        Tokenizer(),
        line_number=1,
        vocab_size=64,
        max_model_len=kwargs.get("max_model_len", 64),
    )


def test_existing_token_ids_and_explicit_mask_are_preserved():
    payload = {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [0, 1, 0, 1, 1]}
    record = normalize(payload)
    assert record.input_ids == payload["input_ids"]
    assert record.loss_mask == payload["loss_mask"]


def test_split_tokens_use_speco_prompt_and_response_masks():
    record = normalize({"prompt_token_ids": [1, 2], "response_token_ids": [3, 4, 5]})
    assert record.input_ids == [1, 2, 3, 4, 5]
    assert record.loss_mask == [0, 0, 1, 1, 1]


def test_text_and_messages_reuse_existing_speco_tokenization_boundaries():
    record = normalize({"prompt": "abc", "response": "def"})
    assert record.input_ids == Tokenizer().encode("abcdef", add_special_tokens=False)
    assert record.loss_mask == [0, 0, 0, 1, 1, 1]
    messages = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "def"},
    ]
    record = normalize({"messages": messages})
    assert record.input_ids == Tokenizer().apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    assert record.loss_mask == [0] * 6 + [1] * 4


def test_serialized_air_chat_uses_positive_explicit_mask():
    record = normalize({"chat": "abc", "loss_mask": [0, 0, 1, 1]})
    assert record.input_ids == Tokenizer().encode("abc")
    assert record.loss_mask == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "payload",
    [
        {"input_ids": [1, 2, 3], "prompt_generation_mask": [1, 0, 0]},
        {"chat": "abc"},
        {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 0]},
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1]},
        {"input_ids": [1, 2, 64], "loss_mask": [0, 1, 1]},
        {"input_ids": [True, 2, 3], "loss_mask": [0, 1, 1]},
        {"input_ids": [1, 2, 3], "loss_mask": [0, 0.5, 1]},
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "prompt": "abc"},
    ],
)
def test_bad_or_ambiguous_trajectories_are_rejected(payload):
    with pytest.raises(ValueError):
        normalize(payload)


def test_long_trajectories_are_never_silently_truncated():
    with pytest.raises(ValueError, match="rather than truncating"):
        normalize({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}, max_model_len=3)


def test_streaming_limit_and_source_line_errors(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"input_ids":[1,2,3],"loss_mask":[0,1,1]}\ninvalid json\n')
    assert (
        len(
            list(
                iter_trajectories(
                    path, Tokenizer(), vocab_size=64, max_model_len=64, max_samples=1
                )
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match=r"trajectories.jsonl:2"):
        list(iter_trajectories(path, Tokenizer(), vocab_size=64, max_model_len=64))


def test_duplicate_source_ids_are_rejected(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    record = {"sample_id": "same", "input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate trajectory ID"):
        list(iter_trajectories(path, Tokenizer(), vocab_size=64, max_model_len=64))


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_replay_prefills_recorded_answer_and_store_is_directly_trainable(
    tmp_path, chunk_size
):
    from test_standalone_rollout_capture import Engine

    class ReplayEngine(Engine):
        def step(self):
            self.forwards += 1
            start = getattr(self, "offset", 0)
            self.offset = min(
                len(self.prompt), start + (chunk_size or len(self.prompt))
            )
            ids = torch.tensor(self.prompt[start : self.offset])
            self.collector.submit(
                self.request_id + "-internal",
                input_ids=ids,
                positions=torch.arange(start, self.offset),
                hidden_states=torch.ones(len(ids), 8),
                target_head_hidden_states=torch.ones(len(ids), 8),
                sampled_token_ids=torch.tensor([13]),
                num_draft_tokens=0,
            )
            self.running = self.offset < len(self.prompt)
            if self.running:
                return []
            return [
                SimpleNamespace(
                    finished=True,
                    prompt_token_ids=self.prompt,
                    outputs=[SimpleNamespace(token_ids=[13])],
                )
            ]

    record = normalize(
        {"input_ids": [1, 2, 3, 4, 5, 6, 7], "loss_mask": [0, 0, 1, 0, 1, 1, 1]}
    )
    contract = DeepSeekV4Contract(
        hidden_size=4,
        hc_mult=2,
        vocab_size=64,
        block_size=2,
        target_layer_ids=(0, 2),
        config_fingerprint="fixture",
    )
    engine = ReplayEngine()
    output = tmp_path / "features"
    report = replay_features(
        SimpleNamespace(llm_engine=engine),
        [record],
        contract=contract,
        output=output,
        sampling_params=object(),
    )
    assert engine.forwards == (
        len(record.input_ids) + (chunk_size or len(record.input_ids)) - 1
    ) // (chunk_size or len(record.input_ids))
    assert engine.prompt == record.input_ids
    assert report["samples"][0]["feature_rows"] == 7
    assert not engine.collector.requests and engine.calls[-1] == "close"
    cfg = OmegaConf.load(
        Path(__file__).parents[2] / "verl_speco/config/deepseek_v4_dspark_backend.yaml"
    )
    cfg.train.segment_length = None
    cfg.train.max_len = None
    dataset = StoreDataset(str(output), contract, cfg)
    try:
        assert dataset[0]["input_ids"].tolist() == record.input_ids
        assert dataset[0]["loss_mask"].tolist() == record.loss_mask
        assert dataset[0]["hidden_states"].shape == (7, 8)
        assert dataset[0]["target_head_hidden_states"].shape == (7, 8)
        assert dataset.store.get_metadata()["source"] == "trajectory_replay"
    finally:
        dataset.store.close()


def test_help_works_without_npu_or_rl_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "verl_speco.replay_deepseek_v4_features", "--help"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--trajectories" in result.stdout


class ToolTokenizer:
    """Prefix-stable template which keeps tools as their own context rows."""

    def __init__(self):
        self.seen = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, drop_thinking
    ):
        assert tokenize and drop_thinking is False
        self.seen.append(deepcopy(messages))
        result = [1]
        roles = {"user": 10, "assistant": 20, "tool": 30, "system": 40}
        for message in messages:
            result += [roles[message["role"]]]
            result += [41 + ord(c) % 10 for c in message["content"]]
            if message.get("tool_calls"):
                result += [51] * len(json.dumps(message["tool_calls"]))
            result += [5]
        if add_generation_prompt:
            result += [20]
        return result


def alpaca_row():
    return {
        "instruction": "last tool result",
        "input": "",
        "output": "",
        "history": [],
        "metadata": {"schema": "alpaca-role-preserving-v1"},
        "messages": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<think>reasoning</think>",
                "tool_calls": [
                    {"id": "call1", "function": {"name": "read", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "tool result", "tool_call_id": "call1"},
            {"role": "assistant", "content": "next step"},
            {"role": "tool", "content": "final tool result", "tool_call_id": "call2"},
        ],
    }


def test_alpaca_masks_all_assistant_turns_and_preserves_final_tool_and_call_fields():
    row = alpaca_row()
    original = deepcopy(row)
    tokenizer = ToolTokenizer()
    record = normalize_trajectory(
        row, tokenizer, line_number=1, vocab_size=64, max_model_len=1024
    )
    assert row == original
    assert tokenizer.seen[0] == row["messages"]
    full = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False, drop_thinking=False
    )
    assert record.input_ids == full
    for index, message in enumerate(row["messages"]):
        before = tokenizer.apply_chat_template(
            row["messages"][:index],
            tokenize=True,
            add_generation_prompt=False,
            drop_thinking=False,
        )
        until = tokenizer.apply_chat_template(
            row["messages"][: index + 1],
            tokenize=True,
            add_generation_prompt=False,
            drop_thinking=False,
        )
        span = record.loss_mask[len(before) + 1 : len(until)]
        assert span and set(span) == (
            {1.0} if message["role"] == "assistant" else {0.0}
        )


def test_alpaca_rejects_template_that_rewrites_history():
    class Unstable(ToolTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            values = super().apply_chat_template(messages, **kwargs)
            if len(messages) > 3:
                values[0] = 2
            return values

    with pytest.raises(ValueError, match="rewrites an earlier assistant"):
        normalize_trajectory(
            alpaca_row(), Unstable(), line_number=1, vocab_size=64, max_model_len=1024
        )


def test_directory_ids_order_global_limit_and_prepare_round_trip(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for name in ("b.jsonl", "a.jsonl"):
        (source / name).write_text(json.dumps(alpaca_row()) + "\n")
    kwargs = {"vocab_size": 64, "max_model_len": 1024}
    records = list(iter_trajectories(source, ToolTokenizer(), **kwargs))
    assert [r.sample_id for r in records] == ["a.jsonl:1", "b.jsonl:1"]
    assert (
        len(list(iter_trajectories(source, ToolTokenizer(), max_samples=1, **kwargs)))
        == 1
    )
    output = tmp_path / "prepared"
    report = prepare_trajectories(records, output, metadata={"model": "0731"})
    assert report["sample_count"] == 2 and not report["npu_forward_validated"]
    prepared = list(iter_trajectories(output / "trajectories.jsonl", None, **kwargs))
    assert [(r.sample_id, r.input_ids, r.loss_mask) for r in prepared] == [
        (r.sample_id, r.input_ids, r.loss_mask) for r in records
    ]
    assert (source / "a.jsonl").read_text() == json.dumps(alpaca_row()) + "\n"


def test_prepare_failure_keeps_incomplete_output_and_no_success_report(tmp_path):
    def records():
        yield normalize({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]})
        raise ValueError("bad later trajectory")

    with pytest.raises(ValueError, match="bad later"):
        prepare_trajectories(records(), tmp_path / "prepared")
    assert (tmp_path / "prepared/trajectories.jsonl.incomplete").exists()
    assert not (tmp_path / "prepared/prepare_report.json").exists()
