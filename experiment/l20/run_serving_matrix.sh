#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/experiment/online-deps:/experiment/variants/peagle-serving
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd /experiment/variants/peagle-serving
py=/experiment/.venv-clean/bin/python
out=/experiment/evidence/l20-20260919
# Regenerate the deterministic tiny fixtures with prepare_tiny_peagle.py first.
for mode in baseline draft-eagle3; do
  if timeout -k 15 240 "$py" experiment/l20/check_tiny_peagle_serving.py "$mode" --tp 2 --no-capture --output-prefix tp2-eager > "$out/tp2-eager-$mode.log" 2>&1; then
    echo 0 > "$out/tp2-eager-$mode.exit"
  else
    echo "$?" > "$out/tp2-eager-$mode.exit"
    exit 1 # Inspect and stop this arm's workers before starting another engine.
  fi
done
export CUDA_VISIBLE_DEVICES=1
for mode in baseline draft-eagle3; do
  if timeout -k 15 600 "$py" experiment/l20/check_tiny_peagle_serving.py "$mode" --graph --no-capture --output-prefix tp1-graph > "$out/tp1-graph-$mode.log" 2>&1; then
    echo 0 > "$out/tp1-graph-$mode.exit"
  else
    echo "$?" > "$out/tp1-graph-$mode.exit"
    exit 1 # Inspect and stop this arm's workers before starting another engine.
  fi
done
