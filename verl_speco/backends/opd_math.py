# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Distribution distance and prefix survival shared by both OPD methods."""


def total_variation(p, q):
    return (p - q).abs().sum(-1) * 0.5


def prefix_survival(overlap):
    return overlap.clamp(0, 1).cumprod(-1)
