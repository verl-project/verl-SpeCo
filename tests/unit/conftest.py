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
"""Shared fixtures for unit tests."""

from __future__ import annotations

import sys

import pytest

_EP_MODULE = "verl_speco.models.dsv4_dspark.backbone.moe_ep"
_KERNELS_MODULE = "verl_speco.models.dsv4_dspark.backbone.kernels"


@pytest.fixture(autouse=True)
def _reset_ep_context():
    """Restore the process-global DSV4 backbone state after each test.

    The expert-parallel context (``moe_ep._EP``) and the active kernel backend
    (``kernels._ACTIVE_BACKEND``, flipped by ``moe_ep.enable()``) are module
    globals; tests that touch them must not leak into other tests. Only resets
    when the modules were already imported, so unrelated tests do not pull in
    the DSV4 tree.
    """
    ep_module = sys.modules.get(_EP_MODULE)
    kernels_module = sys.modules.get(_KERNELS_MODULE)
    if ep_module is None and kernels_module is None:
        yield
        return
    saved_ep = getattr(ep_module, "_EP", None) if ep_module is not None else None
    saved_backend = (
        getattr(kernels_module, "_ACTIVE_BACKEND", None)
        if kernels_module is not None
        else None
    )
    yield
    if ep_module is not None:
        ep_module._EP = saved_ep
    if kernels_module is not None and saved_backend is not None:
        kernels_module._ACTIVE_BACKEND = saved_backend
