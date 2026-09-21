#!/usr/bin/env bash
# Build the pure-PyTorch padding overlay for a verl checkout.
#
# The RTX 5090 host has no flash_attn; verl imports four helper functions from
# flash_attn.bert_padding. Redirect only that import to a pure-torch equivalent, keeping
# the rest of the verl source byte-identical.
set -euo pipefail

VERL_SOURCE="$1"
DESTINATION="$2"
PADDING_SOURCE="$(cd "$(dirname "$0")" && pwd)/speco_upstream_bert_padding.py"
COMPAT_SOURCE="$(cd "$(dirname "$0")" && pwd)/speco_vllm_compat.py"

rm -rf "$DESTINATION"
mkdir -p "$DESTINATION"
cp -r "$VERL_SOURCE/verl" "$DESTINATION/verl"
cp "$PADDING_SOURCE" "$DESTINATION/speco_upstream_bert_padding.py"
cp "$COMPAT_SOURCE" "$DESTINATION/speco_vllm_compat.py"
ln -s "$DESTINATION/verl" "$DESTINATION/verl_src"

python3 - "$DESTINATION" "$VERL_SOURCE" "$PADDING_SOURCE" <<'PY'
import hashlib
import json
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
verl_source = pathlib.Path(sys.argv[2])
padding_source = pathlib.Path(sys.argv[3])
target = destination / "verl/utils/attention_utils.py"
needle = "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
original = target.read_text()
assert original.count(needle) == 1, "verl attention_utils import layout changed"
updated = original.replace(
    needle,
    "        from speco_upstream_bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n",
)
target.write_text(updated)

fp8_target = destination / "verl/utils/vllm/vllm_fp8_utils.py"
fp8_needle = (
    "try:\n"
    "    from vllm.model_executor.layers.fused_moe.layer import FusedMoE\n"
    "    from vllm.model_executor.layers.linear import LinearBase\n"
)
fp8_original = fp8_target.read_text()
assert fp8_original.count(fp8_needle) == 1, "verl vllm_fp8_utils import layout changed"
fp8_updated = fp8_original.replace(
    fp8_needle,
    "try:\n"
    "    from speco_vllm_compat import install_fused_moe_alias\n"
    "\n"
    "    install_fused_moe_alias()\n"
    "    from vllm.model_executor.layers.fused_moe.layer import FusedMoE\n"
    "    from vllm.model_executor.layers.linear import LinearBase\n",
)
fp8_target.write_text(fp8_updated)
(destination / "manifest.json").write_text(
    json.dumps(
        {
            "verl_source": str(verl_source),
            "change": "Redirect only the flash_attn.bert_padding import to a pure-torch module; "
            "no other verl file is modified",
            "attention_utils_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "vllm_fp8_utils_sha256": hashlib.sha256(fp8_updated.encode()).hexdigest(),
            "compat_module_sha256": hashlib.sha256((destination / "speco_vllm_compat.py").read_bytes()).hexdigest(),
            "padding_module_sha256": hashlib.sha256(padding_source.read_bytes()).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"overlay": str(destination), "files": len(list(destination.rglob("*.py")))}, indent=2))
PY
