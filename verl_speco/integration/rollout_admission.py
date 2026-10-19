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

"""A serving-RPC barrier, not an agent-loop completion barrier.

Async agent loops may retry aborted requests while the trainer owns the cards.
Waiting for those loops to finish would deadlock before generation resumes.
Only already-admitted serving RPCs must settle at the weight/probe boundary.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class RolloutAdmission:
    def __init__(self):
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active = 0

    def close(self):
        # No await between checking admission and incrementing active below.
        self._open.clear()

    def resume(self):
        self._open.set()

    async def wait_idle(self):
        await self._idle.wait()

    @asynccontextmanager
    async def request(self):
        while not self._open.is_set():
            await self._open.wait()
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()
