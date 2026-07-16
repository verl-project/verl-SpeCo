"""Layer-id selection for SPECO old-logprob hidden-state collection."""

from __future__ import annotations

from typing import Any


def _get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _normalize_layer_ids(layer_ids: Any) -> list[int] | None:
    if layer_ids is None:
        return None
    if isinstance(layer_ids, int):
        return [int(layer_ids)]
    if isinstance(layer_ids, str):
        raw = layer_ids.strip()
        if not raw:
            return None
        if raw.startswith("["):
            import json

            layer_ids = json.loads(raw)
        else:
            layer_ids = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(layer_id) for layer_id in list(layer_ids)]


def _config_architectures(config: Any) -> list[str]:
    architectures = _get_nested(config, ("architectures",), []) or []
    if isinstance(architectures, str):
        return [architectures]
    return [str(architecture) for architecture in architectures]


def _drafter_algorithm(drafter_cfg: Any) -> str:
    return str(_get_nested(drafter_cfg, ("speculative_algorithm",), "") or "").upper()


def _is_dspark_config(config: Any) -> bool:
    algorithm = _drafter_algorithm(config)
    if algorithm == "DSPARK":
        return True
    return any(
        architecture in {"DSparkDraftModel", "Qwen3DSparkModel"}
        for architecture in _config_architectures(config)
    )


def _is_dflash_config(drafter_cfg: Any, model_configs: tuple[Any, ...]) -> bool:
    algorithm = _drafter_algorithm(drafter_cfg)
    if algorithm in {"DFLASH", "DSPARK"}:
        return True
    return any(
        architecture in {"DFlashDraftModel", "DSparkDraftModel", "Qwen3DSparkModel"}
        for config in model_configs
        for architecture in _config_architectures(config)
    )


def _generic_aux_layer_ids_from_config(config: Any) -> list[int] | None:
    candidates = (
        ("model", "eagle_config", "target_hidden_layer_ids"),
        ("model", "eagle_config", "eagle_aux_hidden_state_layer_ids"),
        ("eagle_config", "target_hidden_layer_ids"),
        ("eagle_config", "eagle_aux_hidden_state_layer_ids"),
        ("target_hidden_layer_ids",),
        ("eagle_aux_hidden_state_layer_ids",),
        ("target_layer_ids",),
    )
    for path in candidates:
        layer_ids = _normalize_layer_ids(_get_nested(config, path, None))
        if layer_ids is not None:
            return layer_ids
    return None


def eagle3_num_aux_hidden_states_from_config(config: Any) -> int | None:
    layer_ids = _generic_aux_layer_ids_from_config(config)
    return len(layer_ids) if layer_ids is not None else None


def _dflash_target_layer_ids_from_config(config: Any) -> list[int] | None:
    top_level = _normalize_layer_ids(_get_nested(config, ("target_layer_ids",), None))
    nested = _normalize_layer_ids(_get_nested(config, ("dflash_config", "target_layer_ids"), None))
    dspark_nested = _normalize_layer_ids(_get_nested(config, ("dspark_config", "target_layer_ids"), None))
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError(f"DFlash target_layer_ids conflict with dflash_config.target_layer_ids: {top_level} != {nested}")
    if top_level is not None and dspark_nested is not None and top_level != dspark_nested:
        raise ValueError(f"DSpark target_layer_ids conflict with dspark_config.target_layer_ids: {top_level} != {dspark_nested}")
    if nested is not None and dspark_nested is not None and nested != dspark_nested:
        raise ValueError(f"DFlash target_layer_ids conflict with dspark_config.target_layer_ids: {nested} != {dspark_nested}")
    if top_level is not None:
        return top_level
    return nested if nested is not None else dspark_nested


def _build_dflash_target_layer_ids(num_context_layers: int, num_hidden_layers: int) -> list[int]:
    num_context_layers = int(num_context_layers)
    num_hidden_layers = int(num_hidden_layers)
    if num_context_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_context_layers - 1))) for i in range(num_context_layers)]


def _dflash_num_context_layers(drafter_cfg: Any, model_configs: tuple[Any, ...], *, is_dspark: bool = False) -> int:
    training_cfg = _get_nested(drafter_cfg, ("training",), {}) or {}
    candidates = []
    if is_dspark:
        candidates.append(_get_nested(training_cfg, ("dspark_num_target_layers",), None))
    candidates.extend(
        (
            _get_nested(training_cfg, ("dflash_num_target_layers",), None),
            _get_nested(drafter_cfg, ("num_context_layers",), None),
            _get_nested(drafter_cfg, ("dflash_config", "num_context_layers"), None),
        )
    )
    for config in model_configs:
        if is_dspark:
            candidates.append(_get_nested(config, ("dspark_config", "num_context_layers"), None))
        candidates.extend(
            (
                _get_nested(config, ("num_context_layers",), None),
                _get_nested(config, ("dflash_config", "num_context_layers"), None),
            )
        )
    for value in candidates:
        if value is not None:
            return int(value)
    return 5


def _default_eagle3_aux_layer_ids(num_hidden_layers: int) -> list[int]:
    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0:
        raise RuntimeError(f"SPECO cannot derive EAGLE3 aux hidden layers from num_hidden_layers={num_hidden_layers}")
    return [2, num_hidden_layers // 2, num_hidden_layers - 3]


def assert_sglang_aux_last_layer_norm_safe(
    layer_ids: Any,
    num_hidden_layers: int | None,
    *,
    collect_from_sgl: bool,
    allow_prenorm_last: bool,
) -> None:
    """Fail closed when SGLang aux collection would capture the last layer pre-norm.

    SGLang's aux/context capture never applies the target's final norm, so a
    ``target_layer_id`` equal to the last layer (``num_hidden_layers - 1``) or the
    embedding id ``-1`` is captured with different semantics than the offline /
    old-logprob paths (post-norm for the last layer, embedding for ``-1``). Training
    a drafter offline (or via old-logprob) and then collecting/serving via SGLang
    would then feed the ``fc`` an inconsistent feature for that slot. This refuses
    the combination unless the user opts in (e.g. a self-consistent SGLang-only
    train+serve setup, where pre-norm on both sides is fine).
    """
    if not collect_from_sgl or allow_prenorm_last or not layer_ids:
        return
    # ``-1`` (the embedding) is divergent regardless of target depth, so it is
    # rejected even when ``num_hidden_layers`` is unresolved; the last layer can
    # only be flagged once the depth is known.
    last = int(num_hidden_layers) - 1 if num_hidden_layers is not None else None
    offenders = sorted({int(lid) for lid in layer_ids if int(lid) == -1 or (last is not None and int(lid) == last)})
    if offenders:
        last_desc = f"the last layer {last}" if last is not None else "the last layer"
        raise ValueError(
            "SGLang hidden-state collection (collect_hidden_states_from_sgl=true) captures aux/context "
            f"features WITHOUT the target's final norm, but the resolved target_layer_ids {list(layer_ids)} "
            f"include layer(s) {offenders} whose semantics differ from the offline / old-logprob paths "
            f"({last_desc} is post-norm there, and -1 is the embedding). A drafter trained "
            "off-policy of this convention would see a mismatched feature for that slot. Either set "
            "collect_hidden_states_from_old_logprob=true, drop the last layer / -1 from target_layer_ids, "
            "or set actor_rollout_ref.rollout.drafter.training.allow_sglang_prenorm_last_layer=true if you "
            "train and serve entirely on SGLang (self-consistent pre-norm)."
        )


def resolve_oldlogprob_aux_layer_ids(
    drafter_cfg: Any,
    *,
    target_num_hidden_layers: int | None,
    model_configs: list[Any] | tuple[Any, ...] = (),
) -> list[int] | None:
    """Resolve target layer ids to capture during old-logprob hidden collection."""

    model_configs = tuple(config for config in model_configs if config is not None)
    if _is_dflash_config(drafter_cfg, model_configs):
        algorithm = _drafter_algorithm(drafter_cfg)
        is_dspark = algorithm == "DSPARK" or (
            algorithm != "DFLASH" and any(_is_dspark_config(config) for config in model_configs)
        )
        for config in (drafter_cfg, *model_configs):
            layer_ids = _dflash_target_layer_ids_from_config(config)
            if layer_ids is not None:
                return layer_ids
        if target_num_hidden_layers is None:
            return None
        return _build_dflash_target_layer_ids(
            _dflash_num_context_layers(drafter_cfg, model_configs, is_dspark=is_dspark),
            int(target_num_hidden_layers),
        )

    # EAGLE-1/2 fuse a single (last) target hidden layer, unlike EAGLE-3's
    # low/mid/high triple. The last-layer feature is also what the frozen target
    # head distills from, so collect only the final layer as the single aux. This
    # is resolved before the generic multi-layer config lookup so a stray
    # eagle3-style eagle_aux_hidden_state_layer_ids cannot silently select the
    # wrong (multi-layer) set while the draft fixes num_aux_hidden_states=1.
    if _drafter_algorithm(drafter_cfg) in {"EAGLE1", "EAGLE2"}:
        if target_num_hidden_layers is None:
            return None
        return [int(target_num_hidden_layers) - 1]
    for config in (drafter_cfg, *model_configs):
        layer_ids = _generic_aux_layer_ids_from_config(config)
        if layer_ids is not None:
            return layer_ids
    if target_num_hidden_layers is None:
        return None
    return _default_eagle3_aux_layer_ids(target_num_hidden_layers)
