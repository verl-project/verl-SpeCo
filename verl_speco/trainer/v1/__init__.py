# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""verl V1 trainer adapters for SPECO."""

__all__ = ["get_speco_v1_trainer_cls"]


def __getattr__(name):
    if name == "get_speco_v1_trainer_cls":
        from .factory import get_speco_v1_trainer_cls

        return get_speco_v1_trainer_cls
    raise AttributeError(name)
