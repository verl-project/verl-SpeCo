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
"""Benchmark the standalone Producer path without starting Ray or TQ.

The benchmark deliberately reuses the production input reader, asynchronous
vLLM client pool, hidden-state alignment, and feature conversion.  In ``both``
mode, each vLLM result is converted twice: once with the real target final norm
and once with ``torch.nn.Identity``.  This keeps the HTTP response and hidden
tensor identical, so the conversion-time difference isolates final-norm cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_speco.integration.oldlogprob_layer_ids import (  # noqa: E402
    resolve_drafter_hidden_states_layout,
)
from verl_speco.producer.input_reader import (  # noqa: E402
    GenerationRequest,
    TokenizedRequest,
    iter_input_records,
    prepare_generated_prefill_request,
    prepare_generation_request,
    tokenize_record,
)
from verl_speco.producer.vllm_feature_client import (  # noqa: E402
    VllmEndpoint,
    VllmFeatureClientPool,
    delete_temporary_result,
)
from verl_speco.trainer.target_feature_replay import (  # noqa: E402
    FeatureContract,
    HiddenStateAlignmentError,
    feature_from_vllm_payload,
    load_vllm_final_norm,
)


logger = logging.getLogger("producer_benchmark")
_INPUT_DONE = object()


@dataclass(frozen=True)
class QueuedRequest:
    request: GenerationRequest | TokenizedRequest
    queued_at: float


class Timings:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)
        self.completed = 0
        self.dropped = 0
        self.generated = 0
        self.hidden_bytes = 0
        self.feature_tokens = 0
        self._lock = asyncio.Lock()

    async def add(self, **values: float) -> None:
        async with self._lock:
            for name, value in values.items():
                self.values[name].append(float(value))

    async def complete(
        self, *, generated: bool, hidden_bytes: int, feature_tokens: int
    ) -> int:
        async with self._lock:
            self.completed += 1
            self.generated += int(generated)
            self.hidden_bytes += int(hidden_bytes)
            self.feature_tokens += int(feature_tokens)
            return self.completed

    async def drop(self) -> None:
        async with self._lock:
            self.dropped += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time the production standalone vLLM Producer path without TQ."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="HF checkpoint used to load only the target final-norm parameters.",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Model name sent to vLLM; defaults to --target-model-path.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["http://127.0.0.1:8000/v1"],
        help="One or more OpenAI-compatible vLLM base URLs.",
    )
    parser.add_argument("--algorithm", default="DSPARK")
    parser.add_argument(
        "--target-layer-ids",
        required=True,
        help="Comma-separated auxiliary layer IDs, excluding the final layer.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=("both", "norm", "no-norm"),
        default="both",
        help="both reuses each response and converts it through both paths.",
    )
    parser.add_argument(
        "--hidden-dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument("--dspark-l1-loss-alpha", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-inflight-requests", type=int, default=64)
    parser.add_argument("--per-endpoint-concurrency", type=int, default=64)
    parser.add_argument("--input-queue-size", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--max-feature-length", type=int, default=512)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _producer_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_sequence_length": args.max_sequence_length,
        "max_feature_length": args.max_feature_length,
        "generation_max_tokens": args.generation_max_tokens,
    }


def _load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(math.ceil(percentile * len(sorted_values)) - 1, 0)
    return sorted_values[min(index, len(sorted_values) - 1)]


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total_s": sum(ordered),
        "mean_ms": statistics.fmean(ordered) * 1000 if ordered else 0.0,
        "p50_ms": _percentile(ordered, 0.50) * 1000,
        "p95_ms": _percentile(ordered, 0.95) * 1000,
        "max_ms": ordered[-1] * 1000 if ordered else 0.0,
    }


def _optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.3f}"


def _print_report(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print("\nProducer benchmark result")
    print(
        f"completed={totals['completed']} dropped={totals['dropped']} "
        f"generated={totals['generated']} wall={totals['wall_seconds']:.3f}s "
        f"samples/s={totals['samples_per_second']:.3f} "
        f"feature_tokens/s={totals['feature_tokens_per_second']:.1f}"
    )
    print(
        "stage".ljust(30)
        + "count".rjust(8)
        + "mean_ms".rjust(12)
        + "p50_ms".rjust(12)
        + "p95_ms".rjust(12)
        + "max_ms".rjust(12)
        + "sum_s".rjust(12)
    )
    for name, item in report["stages"].items():
        print(
            name.ljust(30)
            + str(item["count"]).rjust(8)
            + f"{item['mean_ms']:.3f}".rjust(12)
            + f"{item['p50_ms']:.3f}".rjust(12)
            + f"{item['p95_ms']:.3f}".rjust(12)
            + f"{item['max_ms']:.3f}".rjust(12)
            + f"{item['total_s']:.3f}".rjust(12)
        )
    comparison = report.get("norm_comparison")
    if comparison:
        print(
            "\nconversion mean delta (norm - no_norm): "
            f"{comparison['mean_delta_ms']:.3f} ms/sample; "
            f"ratio={comparison['mean_ratio']:.3f}x"
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_inflight_requests <= 0 or args.per_endpoint_concurrency <= 0:
        raise ValueError("request concurrency values must be positive")
    target_layer_ids = [
        int(value.strip())
        for value in args.target_layer_ids.split(",")
        if value.strip()
    ]
    if not target_layer_ids:
        raise ValueError("--target-layer-ids must contain at least one layer")

    producer_config = _producer_config(args)
    algorithm = args.algorithm.strip().upper()
    training_config = {
        "speculative_algorithm": algorithm,
        "dspark_l1_loss_alpha": args.dspark_l1_loss_alpha,
    }
    layout = resolve_drafter_hidden_states_layout(algorithm, training_config)
    include_final = layout.endswith("_plus_last")
    if args.norm_mode in {"both", "norm"} and not include_final:
        raise ValueError(
            f"algorithm/config resolves to {layout!r}, which does not consume a final "
            "hidden block; norm comparison is not part of that production path"
        )
    feature_contract = FeatureContract(
        algorithm=algorithm,
        target_layer_ids=target_layer_ids,
        hidden_states_layout=layout,
        dtype=_dtype(args.hidden_dtype),
        target_model_id=args.target_model_path,
        target_model_revision=None,
        tokenizer_fingerprint="producer-benchmark",
        use_logits=False,
        source="producer_benchmark",
        require_full_alignment=True,
    )

    timings = Timings()
    startup_begin = time.perf_counter()
    tokenizer_begin = time.perf_counter()
    tokenizer = await asyncio.to_thread(_load_tokenizer, args)
    await timings.add(tokenizer_load=time.perf_counter() - tokenizer_begin)

    real_norm: nn.Module | None = None
    if args.norm_mode in {"both", "norm"}:
        norm_begin = time.perf_counter()
        real_norm = await asyncio.to_thread(
            load_vllm_final_norm,
            args.target_model_path,
            dtype=feature_contract.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        await timings.add(final_norm_load=time.perf_counter() - norm_begin)
    identity_norm = nn.Identity()

    pool = VllmFeatureClientPool(
        [
            VllmEndpoint(url.rstrip("/"), args.per_endpoint_concurrency)
            for url in args.endpoints
        ],
        model=args.vllm_model or args.target_model_path,
        max_inflight_requests=args.max_inflight_requests,
        request_timeout=args.request_timeout,
    )
    pool_begin = time.perf_counter()
    await pool.start()
    await timings.add(client_pool_start=time.perf_counter() - pool_begin)
    queue: asyncio.Queue[QueuedRequest | object] = asyncio.Queue(
        maxsize=args.input_queue_size
    )

    async def read_inputs() -> None:
        count = 0
        for record in iter_input_records(args.input_path):
            if count >= args.max_samples:
                break
            begin = time.perf_counter()
            request = (
                prepare_generation_request(record, tokenizer, producer_config)
                if record.response is None
                else tokenize_record(record, tokenizer, producer_config)
            )
            preparation = time.perf_counter() - begin
            await timings.add(input_prepare=preparation)
            await queue.put(
                QueuedRequest(request=request, queued_at=time.perf_counter())
            )
            count += 1
        if count == 0:
            raise ValueError("input contains no usable samples")
        for _ in range(args.max_inflight_requests):
            await queue.put(_INPUT_DONE)

    async def request_worker() -> None:
        while True:
            queued = await queue.get()
            if queued is _INPUT_DONE:
                return
            assert isinstance(queued, QueuedRequest)
            request = queued.request
            worker_begin = time.perf_counter()
            await timings.add(input_queue_wait=worker_begin - queued.queued_at)
            raw = None
            generated_raw = None
            was_generated = isinstance(request, GenerationRequest)
            try:
                if was_generated:
                    # Re-assert to narrow the type for mypy: the bool above is
                    # not tracked as a type guard.
                    assert isinstance(request, GenerationRequest)
                    begin = time.perf_counter()
                    generated_raw = await pool.generate(request)
                    await timings.add(
                        vllm_generate_and_load=time.perf_counter() - begin
                    )

                    begin = time.perf_counter()
                    request = prepare_generated_prefill_request(
                        request,
                        generated_raw.generated_token_ids,
                        producer_config,
                    )
                    await timings.add(
                        generated_request_prepare=time.perf_counter() - begin
                    )

                    begin = time.perf_counter()
                    await asyncio.to_thread(delete_temporary_result, generated_raw)
                    await timings.add(
                        generation_file_cleanup=time.perf_counter() - begin
                    )
                    generated_raw = None

                begin = time.perf_counter()
                raw = await pool.prefill(request)
                prefill_seconds = time.perf_counter() - begin
                await timings.add(vllm_prefill_and_load=prefill_seconds)
                assert isinstance(request, TokenizedRequest)
                endpoint_url = raw.endpoint_url

                feature_tokens = 0
                conversion_seconds: dict[str, float] = {}

                async def convert(name: str, module: nn.Module | None) -> None:
                    nonlocal feature_tokens
                    begin = time.perf_counter()
                    sample = feature_from_vllm_payload(
                        raw,
                        request,
                        feature_contract,
                        final_norm=module,
                    )
                    elapsed = time.perf_counter() - begin
                    conversion_seconds[name] = elapsed
                    await timings.add(**{name: elapsed})
                    feature_tokens = int(sample.input_ids.numel())
                    del sample

                # Alternate order in comparison mode so CPU cache warmth does
                # not systematically favor either conversion path.
                conversion_plan: list[tuple[str, nn.Module | None]] = []
                if args.norm_mode == "both":
                    conversion_plan = [
                        ("convert_no_norm", identity_norm),
                        ("convert_with_norm", real_norm),
                    ]
                    if request.sequence_no % 2:
                        conversion_plan.reverse()
                elif args.norm_mode == "no-norm":
                    conversion_plan = [
                        (
                            "convert_no_norm",
                            identity_norm if include_final else None,
                        )
                    ]
                else:
                    conversion_plan = [("convert_with_norm", real_norm)]
                for conversion_name, norm_module in conversion_plan:
                    await convert(conversion_name, norm_module)

                begin = time.perf_counter()
                await asyncio.to_thread(delete_temporary_result, raw)
                await timings.add(prefill_file_cleanup=time.perf_counter() - begin)
                hidden_bytes = int(raw.byte_size)
                raw = None
                await timings.add(
                    worker_total=time.perf_counter() - worker_begin,
                    queued_to_complete=time.perf_counter() - queued.queued_at,
                )
                completed = await timings.complete(
                    generated=was_generated,
                    hidden_bytes=hidden_bytes,
                    feature_tokens=feature_tokens,
                )
                if completed <= 3 or completed % args.progress_interval == 0:
                    logger.info(
                        "completed=%s/%s sample_id=%s generated=%s endpoint=%s "
                        "hidden_mib=%.2f",
                        completed,
                        args.max_samples,
                        request.sample_id,
                        was_generated,
                        endpoint_url,
                        hidden_bytes / 1024**2,
                    )
                    logger.info(
                        "sample timing sample_id=%s prefill_and_load_ms=%.3f "
                        "convert_no_norm_ms=%s convert_with_norm_ms=%s",
                        request.sample_id,
                        prefill_seconds * 1000,
                        _optional_ms(conversion_seconds.get("convert_no_norm")),
                        _optional_ms(conversion_seconds.get("convert_with_norm")),
                    )
            except HiddenStateAlignmentError as exc:
                await timings.drop()
                logger.warning(
                    "dropped sample_id=%s because hidden states do not align: %s",
                    request.sample_id,
                    exc,
                )
            finally:
                if generated_raw is not None:
                    await asyncio.to_thread(delete_temporary_result, generated_raw)
                if raw is not None:
                    await asyncio.to_thread(delete_temporary_result, raw)

    benchmark_begin = time.perf_counter()
    try:
        await asyncio.gather(
            read_inputs(),
            *(request_worker() for _ in range(args.max_inflight_requests)),
        )
    finally:
        await pool.close()
    wall_seconds = time.perf_counter() - benchmark_begin
    startup_seconds = benchmark_begin - startup_begin
    stages = {name: _summary(values) for name, values in timings.values.items()}
    report: dict[str, Any] = {
        "configuration": {
            "input_path": args.input_path,
            "endpoints": args.endpoints,
            "algorithm": algorithm,
            "hidden_states_layout": layout,
            "target_layer_ids": target_layer_ids,
            "norm_mode": args.norm_mode,
            "max_samples": args.max_samples,
            "max_inflight_requests": args.max_inflight_requests,
            "per_endpoint_concurrency": args.per_endpoint_concurrency,
            "max_sequence_length": args.max_sequence_length,
            "max_feature_length": args.max_feature_length,
        },
        "totals": {
            "completed": timings.completed,
            "dropped": timings.dropped,
            "generated": timings.generated,
            "hidden_bytes": timings.hidden_bytes,
            "feature_tokens": timings.feature_tokens,
            "startup_seconds": startup_seconds,
            "wall_seconds": wall_seconds,
            "samples_per_second": timings.completed / wall_seconds,
            "feature_tokens_per_second": timings.feature_tokens / wall_seconds,
        },
        "stages": stages,
    }
    if "convert_with_norm" in stages and "convert_no_norm" in stages:
        norm_mean = float(stages["convert_with_norm"]["mean_ms"])
        no_norm_mean = float(stages["convert_no_norm"]["mean_ms"])
        report["norm_comparison"] = {
            "mean_delta_ms": norm_mean - no_norm_mean,
            "mean_ratio": norm_mean / no_norm_mean if no_norm_mean else math.inf,
        }
    return report


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = asyncio.run(_run(args))
    _print_report(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON report written to {output}")


if __name__ == "__main__":
    main()
