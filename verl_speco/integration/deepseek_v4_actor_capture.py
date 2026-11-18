# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 actor boundaries using SpeCo's existing old-logprob transport."""

import torch

from verl_speco.trainer.deepseek_v4_contract import LAYOUT

CONTRACT_KEY = "speco_oldlogprob_deepseek_v4_contract"


def padded_selection_context(engine, micro_batch):
    """Map unpadded sample positions into verl's right-padded eager forward.

    The external NO_PADDING batch still stores jagged tensors. The native FSDP
    engine converts them to independent, right-padded sequences with attention
    masks when use_remove_padding=False; no forward inputs are changed here.
    """
    from verl_speco.integration import oldlogprob_runtime as runtime

    if getattr(engine, "use_ulysses_sp", False):
        raise ValueError("Padded DeepSeek V4 actor capture requires Ulysses SP=1")
    ids = micro_batch["input_ids"]
    lengths = ids.offsets().diff().tolist()
    positions = micro_batch[runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY]
    valid = micro_batch[runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY].bool()
    collect = micro_batch[runtime.OLD_LOGPROB_COLLECT_MASK_KEY].bool()
    if positions.ndim != 2 or tuple(valid.shape) != tuple(positions.shape):
        raise ValueError("Invalid DeepSeek V4 actor hidden selection shape")
    batch_size, rows = positions.shape
    if len(lengths) != batch_size or tuple(collect.shape) != (batch_size,):
        raise ValueError("DeepSeek V4 actor microbatch selection does not align")
    width = max(lengths, default=0)
    flat, batches, row_indices, sequence_positions = [], [], [], []
    owners = {}
    for batch, length in enumerate(lengths):
        if not bool(collect[batch]):
            continue
        for row in range(rows):
            pos = int(positions[batch, row])
            if not bool(valid[batch, row]) or pos < 0 or pos >= length:
                continue
            flat.append(batch * width + pos)
            batches.append(batch)
            row_indices.append(row)
            sequence_positions.append(pos)
            owners.setdefault(batch, []).append(row)
    return {
        "batch_size": batch_size,
        "output_batch_size": batch_size,
        "hidden_rows": rows,
        "local_positions": flat,
        "local_batch_indices": batches,
        "local_row_indices": row_indices,
        "local_3d_batch_indices": batches,
        "local_3d_seq_positions": sequence_positions,
        "max_local_position": max(flat, default=-1),
        "selected_owner_rows": owners,
        "selected_batch_indices": sorted(owners),
        "compact_selected": False,
        "sparse_sp_merge": False,
        "sp_group": None,
        "sp_size": 1,
        "sp_rank": 0,
        "timing_us": {"select": 0.0, "sp_merge": 0.0, "concat": 0.0},
    }


def feature_source(training):
    """Require one source; execution backend does not select the feature producer."""
    if training.get("collect_hidden_states_from_rollout", False):
        raise ValueError(
            "Rollout hidden-state capture has been removed; use actor old-logprob capture"
        )
    actor = bool(training.get("collect_hidden_states_from_old_logprob", False))
    if not actor or training.get("collect_hidden_states_from_sgl", False):
        raise ValueError("Online vllm_ascend requires actor old-logprob capture")
    if (
        actor
        and training.get("old_logprob_hidden_capture_impl", "forward_hook")
        != "forward_hook"
    ):
        raise ValueError(
            "DeepSeek V4 actor capture requires forward_hook for pre-HC states"
        )
    return "actor"


def install_hooks(engine, layers, selection, micro_batch):
    """Capture mean-over-HC decoder outputs and raw final decoder residual.

    The packed tensor travels through the unmodified old-logprob object/ref/SP
    transport; the training adapter splits its final hc_mult * hidden_size columns.
    """
    from verl.utils import tensordict_utils as tu

    from verl_speco.integration import oldlogprob_runtime as runtime

    contract = tu.get_non_tensor_data(data=micro_batch, key=CONTRACT_KEY, default=None)
    if runtime._oldlogprob_aux_layer_id_space(micro_batch) != "decoder":
        raise ValueError("DeepSeek V4 actor requires decoder layer IDs")
    install_contract_hooks(
        engine,
        layers,
        selection,
        contract,
        runtime._oldlogprob_aux_layer_ids_from_batch(micro_batch),
    )


def install_contract_hooks(engine, layers, selection, contract, requested_layers):
    from verl_speco.integration import oldlogprob_runtime as runtime

    if int(selection.get("pp_size", 1)) != 1:
        raise ValueError(
            "DeepSeek V4 actor pre-HC capture currently requires actor PP=1"
        )
    if not isinstance(contract, dict) or contract.get("layout") != LAYOUT:
        raise ValueError("Missing DeepSeek V4 actor feature contract")
    layer_ids = list(contract["target_layer_ids"])
    if not layer_ids or min(layer_ids) < 0 or max(layer_ids) >= len(layers):
        raise ValueError("DeepSeek V4 actor layers differ from target contract")
    if requested_layers != layer_ids:
        raise ValueError("DeepSeek V4 actor auxiliary layer IDs differ")
    width, mult = int(contract["hidden_size"]), int(contract["hc_mult"])
    runtime._cleanup_oldlogprob_hidden_capture(engine)
    context = {
        **selection,
        "aux_keys": [f"layer:{i}" for i in layer_ids],
        "final_key": "pre_hc",
        "hidden_layout": LAYOUT,
        "captures": {},
        "handles": [],
    }

    def capture(index):
        def hook(_module, _args, output):
            raw = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(raw):
                raise ValueError("DeepSeek V4 decoder must expose its raw HC residual")
            if raw.ndim >= 3 and tuple(raw.shape[-2:]) == (mult, width):
                raw = raw.flatten(-2)
            if raw.ndim not in (2, 3) or raw.shape[-1] != mult * width:
                raise ValueError(
                    "DeepSeek V4 actor output is not a complete pre-HC residual"
                )
            # Match native Ascend DSpark: hidden_states.mean(dim=HC), before head/norm.
            aux = raw.reshape(*raw.shape[:-1], mult, width).mean(dim=-2)
            for key, value in ((f"layer:{index}", aux), ("pre_hc", raw)):
                if (key == "pre_hc" and index != len(layers) - 1) or (
                    key != "pre_hc" and index not in layer_ids
                ):
                    continue
                selected = runtime._select_rows_from_local_hidden(context, value)
                if selected is not None:
                    if isinstance(selected, dict):
                        selected = {**selected, "rows": selected["rows"].clone()}
                    else:
                        selected = selected.clone()
                    context["captures"][key] = selected

        return hook

    try:
        for index in sorted(set(layer_ids + [len(layers) - 1])):
            context["handles"].append(
                layers[index].register_forward_hook(capture(index))
            )
    except Exception:
        for handle in context["handles"]:
            handle.remove()
        raise
    engine._speco_oldlogprob_hidden_context = context


def unpack_features(hidden, contract):
    expected = contract.hidden_size * (
        len(contract.target_layer_ids) + contract.hc_mult
    )
    if hidden.shape[-1] != expected:
        raise ValueError(
            "Actor features must include auxiliary layers and complete pre-HC teacher states"
        )
    boundary = contract.hidden_size * len(contract.target_layer_ids)
    return hidden[..., :boundary], hidden[..., boundary:]
