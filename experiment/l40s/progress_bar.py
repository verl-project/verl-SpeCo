#!/usr/bin/env python3
"""Render centered, full-width progress bars for the tracked workstreams.

Usage: progress_bar.py [percent ...]
A single bar spans the width so the filled portion is visually honest; scale and label are
shown when a bar is sampled.
"""

import sys

FULL = "\u2588"   # █
EMPTY = "\u2591"  # ░


def bar(percent: float, width: int = 24) -> str:
    filled = round(width * max(0.0, min(100.0, percent)) / 100.0)
    return f"{FULL * filled}{EMPTY * (width - filled)} {percent:3.0f}%"


def center(text: str, width: int = 56) -> str:
    pad = max(0, width - len(text))
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def block(label: str, percent: float, detail: str = "", width: int = 56) -> str:
    lines = [center(label, width), center(bar(percent), width)]
    if detail:
        lines.append(center(detail, width))
    return "\n".join(lines)


if __name__ == "__main__":
    for value in sys.argv[1:]:
        print(block("progress", float(value)))
