#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/0z5a/speco-l20-20260919
run=$root/c5-native-latest
py=$run/.venv/bin/python
source=$root/variants/peagle-serving
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$source OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES=4,5
out=$run/evidence/c1
mkdir -p "$out"
trap 'echo $? > "$out/suite.exit"' EXIT
"$py" -m verl_speco.convert_peagle_vllm "$run/c1-models/draft-original" "$run/c1-models/converted" --target-layer-ids 0 1 2 > "$out/convert.log" 2>&1
for tp in 1 2; do
  backend=()
  if [[ "$tp" == 2 ]]; then backend=(--attention-backend FLASH_ATTN); fi
  for mode in baseline draft-eagle3; do
    flags=(--no-capture)
    if [[ "$tp-$mode" == 1-draft-eagle3 ]]; then flags=(); fi
    timeout -k 15 900 "$py" "$source/experiment/l20/check_tiny_peagle_serving.py" "$mode" \
      --tp "$tp" "${backend[@]}" --models-root "$run/c1-models" --draft-model "$run/c1-models/converted" \
      --reference-model "$run/c1-models/draft-original" --evidence-root "$out" \
      --output-prefix "native-tp$tp" "${flags[@]}" > "$out/tp$tp-$mode.log" 2>&1
  done
  if [[ "$tp" == 1 ]]; then
    timeout -k 15 900 "$py" "$source/experiment/l20/check_peagle_decode_logits.py" \
      --prefix native-tp1 --reference "$run/c1-models/draft-original" --evidence-root "$out" > "$out/decode-parity.log" 2>&1
  fi
done
for tp in 1 2; do
  backend=()
  if [[ "$tp" == 2 ]]; then backend=(--attention-backend FLASH_ATTN); fi
  for mode in baseline draft-eagle3; do
    timeout -k 15 900 "$py" "$source/experiment/l20/check_tiny_peagle_serving.py" "$mode" \
      --tp "$tp" "${backend[@]}" --graph --no-capture --models-root "$run/c1-models" \
      --draft-model "$run/c1-models/converted" --reference-model "$run/c1-models/draft-original" \
      --evidence-root "$out" --output-prefix "native-tp$tp-graph" > "$out/tp$tp-graph-$mode.log" 2>&1
  done
done
"$py" - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
reference=json.loads((root/'native-tp1-baseline.json').read_text())['token_ids']
for tp in (1, 2):
    for graph in ('', '-graph'):
        for mode in ('baseline', 'draft-eagle3'):
            path=root/f'native-tp{tp}{graph}-{mode}.json'
            row=json.loads(path.read_text())
            assert row['token_ids']==reference,path
(root/'matrix.json').write_text(json.dumps({'all_tokens_match':True,'tp':[1,2],'modes':['eager','graph']},indent=2)+'\n')
PY
