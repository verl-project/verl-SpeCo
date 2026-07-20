from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = os.getenv("VERL_SPECO_UPSTREAM_ROOT")

sys.path.insert(0, str(REPO_ROOT))
if UPSTREAM_ROOT:
    sys.path.insert(1, UPSTREAM_ROOT)
