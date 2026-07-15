"""Standalone SPECO draft model training entrypoint.

The user-facing launcher is ``python -m verl_speco.draft_train_launcher``.  It
starts this module through PyTorch distributed launch so each rank can
participate in draft model training.
"""

from __future__ import annotations

import logging

import hydra

from verl_speco.trainer.draft_training_loop import log_resolved_config, run_standalone_draft_training


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="draft_trainer", version_base=None)
def main(config):
    """Run standalone draft model training."""

    logging.basicConfig(level=logging.INFO)
    log_resolved_config(config)
    result = run_standalone_draft_training(config)
    if result.get("rank", 0) == 0:
        logger.warning("Standalone SPECO draft training finished: %s", result)


if __name__ == "__main__":
    main()
