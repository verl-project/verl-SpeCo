"""Shared drafter trainer backend factory.

Online PPO training (``verl_speco.workers.speco_worker``) and standalone
feature-store training (``verl_speco.trainer.draft_training_loop``) build the
same backends from the same ``speculative_algorithm`` string, so the dispatch
lives here instead of being duplicated per entrypoint.

Backend modules are imported lazily so that resolving one algorithm never pulls
in the model code of the others.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_DRAFTER_ALGORITHMS = (
    "EAGLE1",
    "EAGLE2",
    "EAGLE3",
    "DFLASH",
    "DSPARK",
    "DOMINO",
    "PEAGLE",
)


def build_trainer_backend(config, model_config) -> Any:
    """Build the drafter trainer backend selected by the drafter config.

    Args:
        config: The ``actor_rollout_ref`` config carrying
            ``rollout.drafter.speculative_algorithm``.
        model_config: The target model config passed to every backend.
    """

    raw_algorithm = config.rollout.drafter.speculative_algorithm
    algorithm = str(raw_algorithm).strip().upper()

    if algorithm == "EAGLE3":
        from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend

        return Eagle3TrainerBackend(config, model_config)
    if algorithm in ("EAGLE1", "EAGLE2"):
        from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend

        return Eagle1TrainerBackend(config, model_config)
    if algorithm == "DFLASH":
        from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend

        return DFlashTrainerBackend(config, model_config)
    if algorithm == "DSPARK":
        from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

        return DSparkTrainerBackend(config, model_config)
    if algorithm == "DOMINO":
        from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend

        return DominoTrainerBackend(config, model_config)
    if algorithm == "PEAGLE":
        from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

        return PEagleTrainerBackend(config, model_config)

    raise ValueError(
        f"Unsupported drafter algorithm {raw_algorithm!r}; supported algorithms are "
        f"{', '.join(SUPPORTED_DRAFTER_ALGORITHMS)}"
    )
