#!/usr/bin/env bash
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
set -eo pipefail
source ./job.env
export ASCEND_RT_VISIBLE_DEVICES="$PRODUCER_NPU"
printf '%s\n' "$$" > "$RUN_DIR/vllm.pid"
mapfile -t cfg < <(python3 - <<'PY'
import json, os
from verl_speco.standalone_tq_training_launcher import _vllm_capture_layer_ids, _vllm_kv_transfer_config
print(json.dumps({"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":_vllm_capture_layer_ids([1,9,17,25,33],36)}}}, separators=(",",":")))
print(json.dumps(_vllm_kv_transfer_config(os.environ), separators=(",",":")))
PY
)
test "${#cfg[@]}" -eq 2
exec vllm serve "$MODEL" --host "$HOST_IP" --port 25504 \
  --gpu-memory-utilization 0.70 --max-model-len 8192 --enforce-eager --max-num-seqs 8 \
  --speculative-config "${cfg[0]}" --kv-transfer-config "${cfg[1]}" --no-enable-chunked-prefill
