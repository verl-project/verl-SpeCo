#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert a trained DSV4-DSpark draft checkpoint into the ``mtp.*`` layout.

Inverse of ``verl_speco/models/dsv4_dspark/weights.py::map_released_key``:
  - ``layers.{n}.*`` → ``mtp.{n}.*``
  - Stacked ``ffn.experts.w{1,2,3}`` [E, out, in] → per-expert ``mtp.{n}.ffn.experts.{e}.w{1,2,3}.weight``
  - ``fc.weight`` → ``mtp.0.main_proj.weight``
  - ``hidden_norm.weight`` → ``mtp.0.main_norm.weight``
  - etc.

Usage:
    python -m verl_speco.tools.convert_checkpoint --in <ckpt_dir> --out <out_dir> \
        [--config-from <released_config.json>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SKIP_PREFIXES = ("verifier_lm_head", "verifier_norm", "target_lm_head")
_SKIP_KEYWORDS = ("freqs_cis",)


def _map_key(key: str, n_layers: int) -> list[str] | None:
    """Map verl-SpeCo internal key → released mtp.* key(s)."""
    if any(key.startswith(p) for p in _SKIP_PREFIXES) or any(
        kw in key for kw in _SKIP_KEYWORDS
    ):
        return None
    last = n_layers - 1
    # The trainer loads its embedding from the target and has no lm_head, so the
    # converted checkpoint writes these at the top level (loadable by
    # ``map_released_key``) rather than under ``mtp.{n}`` like the released
    # package. Serving-side parity is not exercised by this tool.
    if key == "embed_tokens.weight":
        return ["embed.weight"]
    if key == "lm_head.weight":
        return ["head.weight"]
    if key in ("fc.weight", "main_proj.weight"):
        return ["mtp.0.main_proj.weight"]
    if key in ("hidden_norm.weight", "main_norm.weight"):
        return ["mtp.0.main_norm.weight"]
    if key == "norm.weight":
        return [f"mtp.{last}.norm.weight"]
    if key.startswith("markov_head."):
        return [f"mtp.{last}.{key}"]
    if key in ("confidence_head.proj.weight", "confidence_head.proj.bias"):
        return [f"mtp.{last}.{key}"]
    m = re.fullmatch(r"hc_head\.hc_(fn|base|scale)", key)
    if m:
        return [f"mtp.{last}.hc_head_{m.group(1)}"]

    lm = re.fullmatch(r"layers\.(\d+)\.(.*)", key)
    if not lm:
        return None
    n, rest = int(lm.group(1)), lm.group(2)

    hc = re.fullmatch(r"(attn_hc|ffn_hc)\.(fn|base|scale)", rest)
    if hc:
        site = "hc_attn" if hc.group(1) == "attn_hc" else "hc_ffn"
        return [f"mtp.{n}.{site}_{hc.group(2)}"]
    if rest.startswith("ffn.router."):
        return [f"mtp.{n}.ffn.gate.{rest[len('ffn.router.') :]}"]
    if re.fullmatch(r"ffn\.experts\.w[123]", rest):
        wn = rest.split(".")[-1]
        return [f"mtp.{n}.ffn.experts.{{e}}.{wn}.weight"]
    return [f"mtp.{n}.{rest}"]


def convert(state_dict: dict, n_layers: int):
    """Convert verl-SpeCo state_dict → released mtp.* format."""
    out: dict = {}
    skipped: list[str] = []
    n_unstacked = 0
    for k, v in state_dict.items():
        tgt = _map_key(k, n_layers)
        if tgt is None:
            skipped.append(k)
            continue
        base = tgt[0]
        if "{e}" in base:
            n_unstacked += 1
            for e in range(v.shape[0]):
                out[base.format(e=e)] = v[e].contiguous().clone()
        else:
            out[base] = v
    return out, skipped, n_unstacked


def _load_state_dict(in_dir: Path):
    from safetensors.torch import load_file

    idx = in_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        sd: dict = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(str(in_dir / shard)))
        return sd
    single = in_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    raise SystemExit(f"No model.safetensors[.index.json] in {in_dir}")


def _merge_ep_expert_slices(
    state_dict: dict,
    ep_experts_dir: Path,
    expected_expert_rows: int | None = None,
) -> dict:
    """Merge the per-rank EP expert dumps back into full stacked weights.

    ``dump_local_expert_slices`` writes ``ep_experts/rank_<r>.pt`` with
    identical keys (``layers.{i}.ffn.experts.{w1,w2,w3}``) holding each rank's
    disjoint expert slice; the leader-only ``model.safetensors`` carries just
    the leader's slice under the same keys. Rank r owns experts
    ``[r * local, (r + 1) * local)``, so rank-ordered concat reconstructs the
    full stacked weight.

    ``expected_expert_rows`` (``dsv4_n_routed_experts`` from the checkpoint
    config, when available) is cross-checked against the reconstructed row
    count so a missing or duplicated rank file fails loudly; the router weight
    in the export provides the same check independently.
    """
    import torch

    rank_files = sorted(
        ep_experts_dir.glob("rank_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not rank_files:
        raise SystemExit(f"No rank_*.pt expert slices in {ep_experts_dir}")
    ranks = [int(p.stem.split("_")[-1]) for p in rank_files]
    if ranks != list(range(len(ranks))):
        raise SystemExit(
            f"Non-contiguous EP expert ranks {ranks} in {ep_experts_dir}"
        )

    payloads = [torch.load(p, map_location="cpu", weights_only=True) for p in rank_files]
    keys = set(payloads[0])
    for rank, payload in zip(ranks, payloads):
        if set(payload) != keys:
            raise SystemExit(
                f"rank_{rank} expert keys differ from rank_{ranks[0]}: "
                f"{sorted(set(payload) ^ keys)[:5]}"
            )

    merged = dict(state_dict)
    sample_key = sorted(keys)[0]
    local_rows = int(payloads[0][sample_key].shape[0])
    for key in sorted(keys):
        slices = [payload[key] for payload in payloads]
        shape_rest = tuple(slices[0].shape[1:])
        for rank, s in zip(ranks, slices):
            if tuple(s.shape[1:]) != shape_rest:
                raise SystemExit(
                    f"rank_{rank} slice shape {tuple(s.shape)} incompatible with "
                    f"rank_{ranks[0]} {tuple(slices[0].shape)} for {key}"
                )
        merged[key] = torch.cat(slices, dim=0).contiguous()
    merged_rows = local_rows * len(ranks)

    # The router weight (``layers.{i}.ffn.router.weight`` is
    # [n_routed_experts, hidden]) independently pins the expert count; a
    # missing or duplicated rank file cannot silently produce the wrong number
    # of experts when either this or ``expected_expert_rows`` is available.
    router_rows = _routed_expert_rows(merged)
    for label, expected in (
        ("config dsv4_n_routed_experts", expected_expert_rows),
        ("router n_routed_experts", router_rows),
    ):
        if expected is not None and merged_rows != int(expected):
            raise SystemExit(
                f"Merged expert rows {merged_rows} != {label} {int(expected)}"
            )
    if expected_expert_rows is None and router_rows is None:
        print(
            "WARNING: could not validate the merged EP expert count (no router "
            "weight in the export and no dsv4_n_routed_experts in the config); "
            "verify the rank_*.pt set covers every expert."
        )
    print(
        f"Merged EP expert slices: {len(keys)} tensors x {len(ranks)} ranks "
        f"(rows {local_rows} -> {merged_rows})"
    )
    return merged


def _routed_expert_rows(state_dict: dict) -> int | None:
    """``n_routed_experts`` from a router weight, if the state dict has one."""
    for key, value in state_dict.items():
        if re.fullmatch(r"layers\.\d+\.ffn\.router\.weight", key) and value.dim() == 2:
            return int(value.shape[0])
    return None


def _n_layers(state_dict: dict) -> int:
    ns = {
        int(m.group(1))
        for k in state_dict
        for m in [re.match(r"layers\.(\d+)\.", k)]
        if m
    }
    if not ns:
        raise SystemExit("No layers.{n}.* keys found")
    return max(ns) + 1


def main():
    ap = argparse.ArgumentParser(
        description="Convert verl-SpeCo checkpoint to mtp.* format"
    )
    ap.add_argument("--in", dest="inp", required=True, help="trainer checkpoint dir")
    ap.add_argument("--out", help="output dir for mtp.* checkpoint")
    ap.add_argument("--config-from", help="config.json to copy into output")
    ap.add_argument(
        "--ep-experts-dir",
        help=(
            "ep_experts/ dir holding rank_<r>.pt slices written by "
            "dump_local_expert_slices; rank-ordered concat restores the full "
            "stacked expert weights before conversion"
        ),
    )
    ap.add_argument("--inspect", action="store_true", help="dry run: show key mapping")
    args = ap.parse_args()

    in_dir = Path(args.inp)
    # --inspect is a dry run: do not touch config.json (pre-existing behavior).
    src_cfg = Path(args.config_from) if args.config_from else (in_dir / "config.json")
    cfg = (
        json.loads(src_cfg.read_text())
        if src_cfg.exists() and not args.inspect
        else None
    )

    sd = _load_state_dict(in_dir)
    if args.ep_experts_dir:
        sd = _merge_ep_expert_slices(
            sd,
            Path(args.ep_experts_dir),
            expected_expert_rows=(
                int(cfg["dsv4_n_routed_experts"])
                if cfg and cfg.get("dsv4_n_routed_experts") is not None
                else None
            ),
        )
    n_layers = _n_layers(sd)
    out, skipped, n_unstacked = convert(sd, n_layers)

    print(f"Input: {len(sd)} tensors, {n_layers} layers")
    print(
        f"Output: {len(out)} tensors ({n_unstacked} stacked→per-expert, {len(skipped)} skipped)"
    )
    if skipped:
        print(f"  Skipped: {sorted(skipped)[:5]}")
    print("  Sample mtp.* keys:")
    for k in list(out)[:5] + [k for k in out if ".experts.0." in k][:1]:
        print(f"    {k}  {tuple(out[k].shape)}")

    if args.inspect:
        return
    if not args.out:
        raise SystemExit("--out required (or use --inspect)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(out, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"\nWrote {out_dir / 'model.safetensors'}")

    if cfg is not None:
        cfg = dict(cfg)
        tids = cfg.get("dspark_target_layer_ids")
        if tids and not cfg.get("eagle_aux_hidden_state_layer_ids"):
            cfg["eagle_aux_hidden_state_layer_ids"] = tids
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"Wrote config.json (from {src_cfg})")


if __name__ == "__main__":
    main()
