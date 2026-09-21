#!/usr/bin/env python3
"""Add a retention probe to a SpeCo source checkout without changing loader behaviour.

The probe fingerprints the rollout drafter after every committed online draft update and
re-fingerprints it after each target weight sync, so a target sync that rolls the drafter back
to its startup checkpoint is visible in the log.

Usage:
  prepare_retention_probe.py <source-checkout> <destination-overlay>
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PUBLISH_ANCHOR = '        self._speco_draft_weight_source = "online"\n'
SYNC_ANCHOR = '            self._speco_diag_draft_state("after_dspark_lm_head_sync")\n'
AUDIT_BLOCK = '''

import hashlib as _pr_audit_hashlib
import os as _pr_audit_os

_pr_audit_published: dict[int, str] = {}
logger.warning("[pr audit] runtime_source=%s pid=%s", __file__, _pr_audit_os.getpid())


def _pr_audit_fingerprint(draft_model) -> str:
    import torch

    tensor = draft_model.model.fc.weight.detach().cpu().contiguous().view(torch.uint8)
    return _pr_audit_hashlib.sha256(tensor.numpy()).hexdigest()


def _pr_audit_after_target_sync(worker) -> None:
    import torch

    rank = torch.distributed.get_rank()
    expected = _pr_audit_published.get(rank)
    if expected is None:
        return
    draft_model, _ = worker._speco_resolve_draft_model()
    if draft_model is None:
        logger.warning("[pr audit] after_target_sync rank=%s draft_model=None", rank)
        return
    actual = _pr_audit_fingerprint(draft_model)
    logger.warning(
        "[pr audit] after_target_sync rank=%s latest_draft_retained=%s expected=%s actual=%s",
        rank,
        actual == expected,
        expected,
        actual,
    )
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.destination.exists():
        shutil.rmtree(args.destination)
    args.destination.mkdir(parents=True)
    runtime_relative = Path("verl_speco/integration/vllm_runtime.py")
    target = args.destination / runtime_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    original = (args.source / runtime_relative).read_text()

    assert original.count(PUBLISH_ANCHOR) == 1, "online publish anchor changed"
    updated = original.replace(
        PUBLISH_ANCHOR,
        PUBLISH_ANCHOR
        + "        _pr_rank = __import__('torch').distributed.get_rank()\n"
        + "        _pr_audit_published[_pr_rank] = _pr_audit_fingerprint(draft_model)\n"
        + '        logger.warning("[pr audit] public_load rank=%s fp=%s", _pr_rank, '
        + "_pr_audit_published[_pr_rank])\n",
    )

    assert updated.count(SYNC_ANCHOR) == 1, "target sync anchor changed"
    updated = updated.replace(
        SYNC_ANCHOR,
        SYNC_ANCHOR + "            _pr_audit_after_target_sync(self)\n",
    )
    updated += AUDIT_BLOCK
    target.write_text(updated)

    (args.destination / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "audited_sha256": hashlib.sha256(updated.encode()).hexdigest(),
                "change": "Log the public draft fingerprint after a committed draft update and "
                "after target sync; no loader logic changes",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"overlay": str(args.destination), "runtime": str(target)}, indent=2))


if __name__ == "__main__":
    main()
