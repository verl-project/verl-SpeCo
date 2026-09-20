#!/usr/bin/env bash
set -euo pipefail
root=/home/kxqandccx/0z5a/speco-l20-20260919
run=$root/c5-native-latest
source=$root/variants/peagle-serving
fixture=$run/full-e2e-head64
out=$run/evidence/full-e2e/head64-publish-resume
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/dev/shm/speco-native-python-20260920:$source OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
mkdir -p "$out"
cp "$run"/evidence/full-e2e/head64-publish/tp1-* "$run"/evidence/full-e2e/head64-publish/tp2-eager-* "$out/"
trap 'echo $? > "$out/suite.exit"' EXIT
for mode in hot cold; do
  timeout -k 15 900 "$run/.venv/bin/python" "$source/experiment/l20/check_trained_publish.py" "$mode" \
    "$fixture/target" "$fixture/converted-step3" "$fixture/converted-veomni-v2" \
    "$out/tp2-graph-$mode" --tp 2 --graph >"$out/tp2-graph-$mode.log" 2>&1
done
"$run/.venv/bin/python" "$run/compare_trained_publish.py" "$out" > "$out/compare.log" 2>&1
