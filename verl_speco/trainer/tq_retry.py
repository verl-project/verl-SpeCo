# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Compatibility exports for legacy TQ retry callers."""

import random  # retained for callers patching retry jitter
import time  # retained for callers patching retry backoff

from verl_speco.trainer.standalone_retry import is_transient_io_error, retry_io

__all__ = ["is_transient_io_error", "random", "retry_io", "time"]
