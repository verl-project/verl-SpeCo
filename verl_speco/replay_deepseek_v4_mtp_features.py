# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Standalone native pre-HC replay for MTP; never installs verl rollout hooks."""

from verl_speco.replay_deepseek_v4_features import main

if __name__ == "__main__":
    main(algorithm="mtp")
