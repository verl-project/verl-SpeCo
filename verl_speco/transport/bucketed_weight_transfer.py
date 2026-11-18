# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Small policies around verl's existing bucketed transport."""


def required_bucket_megabytes(weights, configured):
    """SHM cannot directly send tensors larger than its bucket."""
    if configured <= 0:
        raise ValueError("IPC bucket size must be positive")
    values = getattr(weights, "values", None)
    if not callable(values):
        return configured
    largest = max((int(getattr(value, "nbytes", 0)) for value in values()), default=0)
    return max(configured, (largest + (1 << 20) - 1) // (1 << 20))


def configure_socket_timeout(transfer, seconds):
    import zmq

    if seconds <= 0:
        raise ValueError("IPC timeout must be positive")
    original = transfer._init_socket

    def initialize():
        original()
        transfer.socket.setsockopt(zmq.RCVTIMEO, int(seconds * 1000))
        transfer.socket.setsockopt(zmq.SNDTIMEO, int(seconds * 1000))
        transfer.socket.setsockopt(zmq.LINGER, 0)

    transfer._init_socket = initialize
