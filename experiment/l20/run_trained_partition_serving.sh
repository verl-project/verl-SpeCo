#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/0z5a/speco-l20-20260919
run=$root/c5-native-latest
source=$root/variants/peagle-serving
out=$run/evidence/full-e2e/partition-serving
fixture=$run/full-e2e-head64
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$source OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
mkdir -p "$out"
trap 'echo $? > "$out/suite.exit"' EXIT
py=$run/.venv/bin/python
for arm in flat partition; do
  checkpoint=$root/evidence/l20-20260919/native-head64-$arm/draft_step_6
  "$py" -m verl_speco.convert_peagle_vllm "$checkpoint" "$fixture/converted-$arm" \
    --target-layer-ids 0 1 2 >"$out/$arm-convert.log" 2>&1
  timeout -k 15 900 "$py" "$source/experiment/l20/check_tiny_peagle_serving.py" draft-eagle3 \
    --tp 2 --attention-backend FLASH_ATTN --no-capture --models-root "$fixture" \
    --draft-model "$fixture/converted-$arm" --reference-model "$checkpoint" \
    --evidence-root "$out" --output-prefix "$arm" >"$out/$arm.log" 2>&1
done
"$py" - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reference = json.loads((root.parent / 'head64-serving-v2/native-tp2-baseline.json').read_text())
for arm in ('flat', 'partition'):
    candidate = json.loads((root / f'{arm}-draft-eagle3.json').read_text())
    assert candidate['token_ids'] == reference['token_ids'], arm
(root / 'summary.json').write_text(json.dumps({'flat_tokens_equal': True, 'partition_tokens_equal': True}) + '\n')
PY
