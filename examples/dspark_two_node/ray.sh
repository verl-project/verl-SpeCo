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
ray_temp="/tmp/$SPECO_NATIVE_RUN_ID"
mkdir -m 700 "$ray_temp"
printf '%s\n' "$ray_temp" > ray.tmp
export RAY_TMPDIR="$ray_temp"
unset RAY_ADDRESS
args=(--node-ip-address="$HOST_IP" --num-cpus=6 --num-gpus=0 --block
  --node-manager-port=25491 --object-manager-port=25492 --runtime-env-agent-port=25493
  --dashboard-agent-listen-port=25494 --dashboard-agent-grpc-port=25495
  --min-worker-port=25520 --max-worker-port=25545 --object-store-memory=2147483648 --temp-dir="$ray_temp")
case "$ROLE" in
  producer)
    # vLLM owns the Producer NPU outside Ray. No trainer can land on this node.
    export ASCEND_RT_VISIBLE_DEVICES=""
    args+=(--address="$SPECO_EXTERNAL_RAY_ADDRESS" --resources='{"NPU":0}')
    ;;
  consumer)
    export ASCEND_RT_VISIBLE_DEVICES="$CONSUMER_NPU"
    args+=(--head --port=25490 --include-dashboard=false --ray-client-server-port=25496
      --dashboard-port=25497 --resources='{"NPU":1}'
      --system-config='{"enable_worker_prestart":true,"prestart_worker_first_driver":false}')
    ;;
  *) exit 2 ;;
esac
exec python3 -c 'from ray.scripts.scripts import main; main()' start "${args[@]}"
