#!/usr/bin/env python3
"""Generate GLM-53 JSONL using streaming HTTP. Dependencies: httpx, tqdm.

Run: python -u examples/glm53-train/generate_responses.py
All user settings are below. Relative data paths are relative to this script.
"""

import asyncio
import hashlib
import heapq
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from tqdm import tqdm


# ======================== 配置区 ========================
INPUT_FILE = "input.jsonl"
OUTPUT_FILE = "glm53_first_pair_responses.jsonl"
MODEL_NAME = "glm-53"

# 每个实例分别配置并发和北京时间运行时段。
# 全天用 [(0, 24)]；跨午夜的时段拆为 [(0, 7), (22, 24)]。
INSTANCES = [
    {
        "name": "instance1",
        "url": "http://REPLACE_INSTANCE_1_HOST:9000/v1/chat/completions",
        "concurrency": 4,
        "windows": [(0, 24)],
    },
    {
        "name": "instance2",
        "url": "http://REPLACE_INSTANCE_2_HOST:9000/v1/chat/completions",
        "concurrency": 4,
        "windows": [(0, 7)],
    },
    {
        "name": "instance3",
        "url": "http://REPLACE_INSTANCE_3_HOST:9000/v1/chat/completions",
        "concurrency": 4,
        "windows": [(0, 7)],
    },
    {
        "name": "instance4",
        "enabled": False,  # 填写实际地址后开启。
        "url": "http://REPLACE_INSTANCE_4_HOST:9000/v1/chat/completions",
        "concurrency": 4,
        "windows": [(0, 7)],
    },
]

REQUEST_PARAMETERS = {
    "max_tokens": 10000,
    "temperature": 0,
    "reasoning_effort": "low",
    "chat_template_kwargs": {"enable_thinking": True},
}
HEADERS = {"Content-Type": "application/json"}
# 如需鉴权，可在 HEADERS 中添加 Authorization。
CONNECT_TIMEOUT_SECONDS = 30
STREAM_READ_TIMEOUT_SECONDS = 180  # 等待下一段网络数据，不是总生成时限。
STREAM_INCLUDE_USAGE = True  # 服务不支持 stream_options 时设为 False。
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1
RETRY_MAX_SECONDS = 30
FAILURE_RETRY_SECONDS = 60
CHECK_INTERVAL_SECONDS = 1
SAVE_RAW_RESPONSE = False  # 保存流合并后的响应，不保存逐条 SSE。
# =======================================================

BASE_DIR = Path(__file__).resolve().parent
TIMEZONE = timezone(timedelta(hours=8))


class TruncatedResponseError(Exception):
    """Permanently discard a length-truncated answer."""


class WindowClosed(Exception):
    """Return unfinished work to the shared task pool."""


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def now():
    return datetime.now(TIMEZONE)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def active_instances():
    return [item for item in INSTANCES if item.get("enabled", True)]


def merged_windows(instance):
    merged = []
    for start, end in sorted(instance["windows"]):
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def window_remaining(instance):
    """Seconds to the end of the current continuous window; zero when closed."""
    current = now()
    midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
    windows = merged_windows(instance)
    if windows == [(0, 24)]:
        return math.inf
    for start, end in windows:
        begin = midnight + timedelta(hours=start)
        finish = midnight + timedelta(hours=end)
        if begin <= current < finish:
            # Join a split overnight window across midnight.
            if end == 24 and windows[0][0] == 0:
                finish += timedelta(hours=windows[0][1])
            return (finish - current).total_seconds()
    return 0


def validate_config():
    instances = active_instances()
    if not instances or not MODEL_NAME:
        raise ValueError("至少启用一个实例，并配置 MODEL_NAME")
    if len({item["name"] for item in instances}) != len(instances):
        raise ValueError("实例 name 必须唯一")
    for item in instances:
        if "REPLACE_" in item["url"]:
            raise ValueError(f"请先配置实例 {item['name']} 的实际服务地址")
        if not item["url"].startswith(("http://", "https://")):
            raise ValueError("实例 url 必须为完整 HTTP/HTTPS 地址")
        if type(item["concurrency"]) is not int or item["concurrency"] < 1:
            raise ValueError("concurrency 必须为正整数")
        previous_end = -1
        if not item["windows"]:
            raise ValueError("运行时段不能为空")
        for start, end in sorted(item["windows"]):
            if not 0 <= start < end <= 24 or start < previous_end:
                raise ValueError("时段须满足 0 <= start < end <= 24，且不能重叠")
            previous_end = end
    for value in (CONNECT_TIMEOUT_SECONDS, STREAM_READ_TIMEOUT_SECONDS,
                  FAILURE_RETRY_SECONDS, CHECK_INTERVAL_SECONDS):
        if value <= 0:
            raise ValueError("超时和检查间隔必须为正数")
    if type(MAX_RETRIES) is not int or MAX_RETRIES < 0:
        raise ValueError("MAX_RETRIES 必须为非负整数")
    if min(RETRY_BASE_SECONDS, RETRY_MAX_SECONDS) < 0:
        raise ValueError("重试等待不能为负数")
    if {"model", "messages", "stream", "stream_options"} & REQUEST_PARAMETERS.keys():
        raise ValueError("REQUEST_PARAMETERS 不能覆盖 model/messages/stream/stream_options")
    source = resolve_path(INPUT_FILE).resolve()
    output = resolve_path(OUTPUT_FILE).resolve()
    for suffix in ("", ".meta.json", ".errors.jsonl", ".discarded.jsonl", ".lock"):
        if Path(str(output) + suffix) in {source, Path(__file__).resolve()}:
            raise ValueError("输出及附属文件不能覆盖输入或脚本")
    if not source.is_file():
        raise FileNotFoundError(source)


def iter_prompts(path):
    """One task per row: first human only; skip empty human; no assistant required."""
    with path.open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                turns = json.loads(line)["conversations"]
                if not isinstance(turns, list):
                    raise ValueError("conversations 必须为列表")
                for index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        raise ValueError("消息必须为对象")
                    if turn.get("from") == "human":
                        content = turn.get("value")
                        if not isinstance(content, str):
                            raise ValueError("human.value 必须为字符串")
                        if content.strip():
                            yield line_no, index, [{"role": "user", "content": content}]
                        break
                else:
                    raise ValueError("没有 human 消息")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"输入第 {line_no} 行有误：{exc}") from exc


def extract_answer(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise TruncatedResponseError()
    message = choice["message"]
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("响应没有有效正文")
    reasoning = ""
    for value in (choice.get("reasoning"), message.get("reasoning_content"), message.get("reasoning")):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("reasoning 不是字符串")
        if value.strip():
            reasoning = value
            break
    return (f"{reasoning}</think>{content}" if reasoning else content), choice.get("finish_reason")


async def read_stream_response(client, instance, payload):
    reasoning_parts, content_parts = [], []
    finish_reason, usage, received_done = None, None, False

    def process_event(data):
        nonlocal finish_reason, usage, received_done
        if data.strip() == "[DONE]":
            received_done = True
            return
        chunk = json.loads(data)
        if chunk.get("error"):
            raise ValueError(f"服务端流式错误：{dumps(chunk['error'])}")
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            reason = choice.get("finish_reason")
            if reason == "length":
                raise TruncatedResponseError()
            if reason is not None:
                finish_reason = reason
            delta = choice.get("delta") or {}
            for value in (choice.get("reasoning"), delta.get("reasoning_content"), delta.get("reasoning")):
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError("流式 reasoning 不是字符串")
                if value:
                    reasoning_parts.append(value)
                    break
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("流式 content 不是字符串")
                content_parts.append(content)

    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT_SECONDS, read=STREAM_READ_TIMEOUT_SECONDS,
                            write=30, pool=30)
    async with client.stream("POST", instance["url"], json=payload, timeout=timeout) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", "").lower():
            raise ValueError("服务未返回 text/event-stream")
        event_data = []
        async for line in response.aiter_lines():
            if line == "":
                if event_data:
                    process_event("\n".join(event_data))
                    event_data.clear()
                if received_done:
                    break
            elif line.startswith("data:"):
                value = line[5:]
                event_data.append(value[1:] if value.startswith(" ") else value)
        if event_data and not received_done:
            process_event("\n".join(event_data))
    if not received_done and finish_reason is None:
        raise ValueError("流中断，没有收到结束标识")
    result = {"model": MODEL_NAME, "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "".join(content_parts)},
              "reasoning": "".join(reasoning_parts), "finish_reason": finish_reason}], "usage": usage}
    extract_answer(result)
    return result


async def request_with_retries(client, instance, messages):
    payload = {**REQUEST_PARAMETERS, "model": MODEL_NAME, "messages": messages, "stream": True}
    if STREAM_INCLUDE_USAGE:
        payload["stream_options"] = {"include_usage": True}
    for attempt in range(MAX_RETRIES + 1):
        remaining = window_remaining(instance)
        if remaining <= 0:
            raise WindowClosed()
        delay = None
        try:
            if math.isinf(remaining):
                return await read_stream_response(client, instance, payload)
            # Only the instance's closing time limits total request duration.
            return await asyncio.wait_for(read_stream_response(client, instance, payload), remaining)
        except TruncatedResponseError:
            raise
        except asyncio.TimeoutError:
            raise WindowClosed() from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in (408, 409, 425, 429) and status < 500:
                raise RuntimeError(f"HTTP {status}") from None
            error = f"HTTP {status}"
            retry_after = exc.response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = min(float(retry_after), 120)
        except httpx.RequestError as exc:
            error = type(exc).__name__
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            error = f"流式响应异常：{type(exc).__name__}: {exc}"
        if window_remaining(instance) <= 0:
            raise WindowClosed()
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"{error}，已尝试 {MAX_RETRIES + 1} 次")
        if delay is None:
            delay = min(RETRY_BASE_SECONDS * 2 ** attempt, RETRY_MAX_SECONDS) + random.random()
        await asyncio.sleep(min(delay, window_remaining(instance)))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_records(path):
    keys = set()
    if not path.exists():
        return keys
    with path.open("rb+") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                source.truncate(offset)
                break
            key = tuple(json.loads(line)["request_key"])
            if len(key) != 2 or not all(type(value) is int for value in key):
                raise ValueError(f"{path} 中 request_key 格式错误")
            if key in keys:
                raise ValueError(f"{path} 中存在重复任务：{key}")
            keys.add(key)
    return keys


def append_jsonl(stream, record):
    # No await here: one event loop serializes writes and completion bookkeeping.
    stream.write(dumps(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


async def run_workers(tasks, total, successful, discarded, output, errors, discarded_output):
    completed = successful | discarded
    finished_event = asyncio.Event()
    retry_heap = []
    sequence, failures = 0, 0
    source_exhausted = False

    def retry_later(task, delay):
        nonlocal sequence
        sequence += 1
        heapq.heappush(retry_heap, (time.monotonic() + delay, sequence, task))

    def claim_task():
        nonlocal source_exhausted
        if retry_heap and retry_heap[0][0] <= time.monotonic():
            return heapq.heappop(retry_heap)[2]
        if not source_exhausted:
            task = next(tasks, None)
            if task is not None:
                return task
            source_exhausted = True
        return None

    with tqdm(total=total, initial=len(completed), desc="处理进度", unit="条",
              dynamic_ncols=True, mininterval=0.5, disable=False) as progress:
        def update_progress():
            progress.set_postfix(success=len(successful), discarded=len(discarded),
                                 failed=failures, refresh=False)

        update_progress()

        async def worker(instance, client):
            nonlocal failures
            while not finished_event.is_set():
                if window_remaining(instance) <= 0:
                    await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                    continue
                task = claim_task()
                if task is None:
                    await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                    continue
                line_no, human_index, messages = task
                key = [line_no, human_index]
                try:
                    response = await request_with_retries(client, instance, messages)
                    answer, finish_reason = extract_answer(response)
                except WindowClosed:
                    retry_later(task, 0)
                    continue
                except TruncatedResponseError:
                    append_jsonl(discarded_output, {"request_key": key, "instance": instance["name"],
                                 "reason": "length", "timestamp": time.time()})
                    discarded.add(tuple(key))
                    completed.add(tuple(key))
                except Exception as exc:
                    append_jsonl(errors, {"request_key": key, "instance": instance["name"],
                                 "error": str(exc), "timestamp": time.time()})
                    failures += 1
                    update_progress()
                    tqdm.write(f"失败 {key} [{instance['name']}]：{exc}")
                    retry_later(task, FAILURE_RETRY_SECONDS)
                    continue
                else:
                    record = {"request_key": key, "model": MODEL_NAME, "instance": instance["name"],
                              "conversations": [{"from": "human", "value": messages[0]["content"]},
                                                {"from": "assistant", "value": answer}],
                              "finish_reason": finish_reason, "usage": response.get("usage")}
                    if SAVE_RAW_RESPONSE:
                        record["raw_response"] = response
                    append_jsonl(output, record)
                    successful.add(tuple(key))
                    completed.add(tuple(key))
                update_progress()
                progress.update(1)
                if len(completed) == total:
                    finished_event.set()

        async def monitor():
            previous_active = None
            while not finished_event.is_set():
                active = tuple(item["name"] for item in active_instances() if window_remaining(item) > 0)
                if active != previous_active:
                    tqdm.write(f"[{now():%Y-%m-%d %H:%M:%S}] 当前运行实例："
                               + (", ".join(active) if active else "无，等待运行时段"))
                    previous_active = active
                progress.refresh()
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)

        clients, background = [], []
        try:
            for instance in active_instances():
                concurrency = instance["concurrency"]
                client = httpx.AsyncClient(headers=HEADERS, limits=httpx.Limits(
                    max_connections=concurrency, max_keepalive_connections=concurrency))
                clients.append(client)
                background.extend(asyncio.create_task(worker(instance, client)) for _ in range(concurrency))
            background.append(asyncio.create_task(monitor()))
            completion_waiter = asyncio.create_task(finished_event.wait())
            background.append(completion_waiter)
            done, _ = await asyncio.wait(background, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if not finished_event.is_set():
                raise RuntimeError("工作协程提前退出")
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            for client in clients:
                await client.aclose()


async def main():
    validate_config()
    input_path, output_path = resolve_path(INPUT_FILE), resolve_path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(output_path) + ".meta.json")
    discarded_path = Path(str(output_path) + ".discarded.jsonl")
    errors_path = Path(str(output_path) + ".errors.jsonl")
    lock_path = Path(str(output_path) + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"锁文件已存在：{lock_path}。确认旧进程已停止后再删除锁文件。") from None
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        print("正在校验输入和恢复进度……", flush=True)
        # Preserve compatibility with the previously supplied first-pair script.
        signature_data = {"input_sha256": file_hash(input_path), "model": MODEL_NAME,
                          "parameters": REQUEST_PARAMETERS, "save_raw": SAVE_RAW_RESPONSE,
                          "prompt_mode": "first_human_assistant_pair", "format": "first-pair-discard-length-v1"}
        signature = hashlib.sha256(dumps(signature_data).encode("utf-8")).hexdigest()
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
                raise ValueError("输入、模型或生成参数已变化，请使用新的 OUTPUT_FILE")
        else:
            for path in (output_path, discarded_path):
                if path.exists() and path.stat().st_size:
                    raise ValueError("已有结果缺少 meta 文件，请使用新的输出路径")
        total = sum(1 for _ in iter_prompts(input_path))
        successful, discarded = recover_records(output_path), recover_records(discarded_path)
        if successful & discarded:
            raise ValueError("同一任务同时出现在成功和丢弃文件中")
        completed = successful | discarded
        matched = sum((line, index) in completed for line, index, _ in iter_prompts(input_path))
        if matched != len(completed):
            raise ValueError("结果中包含不属于当前输入的任务")
        if not manifest_path.exists():
            manifest_path.write_text(dumps({"signature": signature}) + "\n", encoding="utf-8")
        print(f"总任务 {total}，成功 {len(successful)}，丢弃 {len(discarded)}，"
              f"待处理 {total - len(completed)}", flush=True)
        if len(completed) == total:
            return
        already_processed = frozenset(completed)
        tasks = (task for task in iter_prompts(input_path) if (task[0], task[1]) not in already_processed)
        with (output_path.open("a", encoding="utf-8", newline="\n") as output,
              errors_path.open("a", encoding="utf-8", newline="\n") as errors,
              discarded_path.open("a", encoding="utf-8", newline="\n") as discarded_output):
            await run_workers(tasks, total, successful, discarded, output, errors, discarded_output)
        print(f"全部处理完成：成功 {len(successful)}，超长丢弃 {len(discarded)}。", flush=True)
    finally:
        os.close(lock_fd)
        lock_path.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已停止，重新运行可续跑。", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n错误：{type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
