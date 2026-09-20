#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/0z5a/speco-l20-20260919
run=$root/c5-native-latest
source=$root/variants/peagle-serving
out=$run/evidence/full-e2e/head64-publish
fixture=$run/full-e2e-head64
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$source OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
mkdir -p "$out"
trap 'echo $? > "$out/suite.exit"' EXIT
py=$run/.venv/bin/python
"$py" -m verl_speco.convert_peagle_vllm "$root/evidence/l20-20260919/native-head64-veomni/draft_step_3" "$fixture/converted-step3" --target-layer-ids 0 1 2 >"$out/convert.log" 2>&1
for tp in 1 2; do
  for graph in eager graph; do
    flags=()
    if [[ "$graph" == graph ]]; then flags=(--graph); fi
    for mode in hot cold; do
      timeout -k 15 900 "$py" "$source/experiment/l20/check_trained_publish.py" "$mode" \
        "$fixture/target" "$fixture/converted-step3" "$fixture/converted-veomni-v2" \
        "$out/tp$tp-$graph-$mode" --tp "$tp" "${flags[@]}" >"$out/tp$tp-$graph-$mode.log" 2>&1
    done
  done
done
"$py" - "$out" <<'PY'
import json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1])
rows=[]
for tp in (1,2):
    for graph in ('eager','graph'):
        hot=json.loads((root/f'tp{tp}-{graph}-hot.json').read_text())
        cold=json.loads((root/f'tp{tp}-{graph}-cold.json').read_text())
        assert hot['tokens']==cold['tokens']
        assert hot['target_hashes']==cold['target_hashes']
        error=None
        if graph=='eager':
            a=torch.load(root/f'tp{tp}-eager-hot.rank0.pt',weights_only=True)
            b=torch.load(root/f'tp{tp}-eager-cold.rank0.pt',weights_only=True)
            torch.testing.assert_close(a,b,atol=0.01,rtol=0.02)
            error=(a-b).abs().max().item()
        rows.append({'tp':tp,'mode':graph,'tokens_equal':True,'target_equal':True,'draft_logits_max_error':error})
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
