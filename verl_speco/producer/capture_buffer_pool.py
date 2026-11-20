# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded host-buffer reuse; callers release only after D2H and serialization."""

import threading


class CaptureBufferPool:
    def __init__(self, max_bytes=64 << 20):
        self.max_bytes = max_bytes
        self._bytes = 0
        self._free = {}
        self._events = {}
        self._lock = threading.Lock()

    def acquire(self, tensor):
        import torch

        key = (tuple(tensor.shape), tensor.dtype)
        with self._lock:
            entries = self._free.get(key)
            if entries:
                result = entries.pop()
                self._bytes -= result.numel() * result.element_size()
                if not entries:
                    del self._free[key]
                return result
        return torch.empty_like(tensor, device="cpu", pin_memory=True)

    def release(self, tensors):
        with self._lock:
            for tensor in tensors:
                size = tensor.numel() * tensor.element_size()
                if self._bytes + size <= self.max_bytes:
                    key = (tuple(tensor.shape), tensor.dtype)
                    self._free.setdefault(key, []).append(tensor)
                    self._bytes += size

    def acquire_event(self, device, timing=False):
        import torch

        key = (str(device), timing)
        with self._lock:
            events = self._events.get(key)
            if events:
                return events.pop()
        return torch.get_device_module(device).Event(enable_timing=timing)

    def release_events(self, device, events, timing=False):
        key = (str(device), timing)
        with self._lock:
            target = self._events.setdefault(key, [])
            target.extend(events[: max(0, 64 - len(target))])
