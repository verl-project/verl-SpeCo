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
"""Wait until every OpenAI-compatible vLLM endpoint is ready."""

from __future__ import annotations

import argparse
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def parse_endpoint_list(value: str) -> list[str]:
    """Parse the Hydra-style ``[url0,url1]`` used by the training launcher."""

    raw = value.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError("endpoints must use [url0,url1] syntax")
    endpoints = [
        item.strip().strip("'\"").rstrip("/")
        for item in raw[1:-1].split(",")
        if item.strip()
    ]
    if not endpoints:
        raise ValueError("endpoints must contain at least one URL")
    return endpoints


def wait_for_endpoints(
    endpoints: list[str],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    request_timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = set(endpoints)
    while pending:
        for endpoint in list(pending):
            try:
                with urlopen(
                    f"{endpoint}/models", timeout=request_timeout_seconds
                ) as response:
                    if 200 <= int(response.status) < 300:
                        print(f"EXTERNAL_VLLM_READY endpoint={endpoint}", flush=True)
                        pending.remove(endpoint)
            except (HTTPError, OSError, URLError):
                pass
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "external hidden-state vLLM is not ready at: "
                + ", ".join(sorted(pending))
            )
        time.sleep(poll_interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=2.0)
    args = parser.parse_args()

    try:
        endpoints = parse_endpoint_list(args.endpoints)
        wait_for_endpoints(
            endpoints,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except (TimeoutError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
