# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Shared environment flag semantics."""

import os


def env_flag(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes"}
