# blue_zone_machine 传输与 NPU 测试顺序

Last updated: 10/29/2026

暂不接入完整 RL 时，先按 [无 RL 的 NPU 冒烟](deepseek_v4_no_rl_smoke.md) 验证独立采集、训练、恢复和导出后的推理。

如果已有完整轨迹，使用 [轨迹重放采集](deepseek_v4_trajectory_features.md)，可直接获取其 HS/pre-HC，无需重新生成回答。

## 传输内容

目标目录：`/models/z00829645/verl-SpeCo`。同步包含 `.git`、所有当前代码、未提交和未跟踪的文件、测试及文档；排除本地 `.venv`、Python 字节码、mypy/pytest/ruff 缓存和 `.DS_Store`。不传输本机 SSH 私钥。

必要训练代码已迁入本仓库，不再要求远端提供 AIRDrafter 路径。模型权重与业务数据需自行提供。10/29 已同步包内迁入及最新采集入口，380 个包内文件校验通过；备份位于 `/models/z00829645/verl-SpeCo-backup-20261029T035818Z`，服务器针对性 CPU 测试 147 项通过。真实 NPU 抓取与训练尚待验收。

## 1. 登录并进入仓库

在本地终端执行：

```bash
ssh blue_zone_machine
cd /models/z00829645/verl-SpeCo
```

## 2. 先确认运行环境

10/28 最近检查到远端 Python 3.12.13：torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、vLLM `0.26.0+empty`、vLLM-Ascend `0.26.0rc1`、DeepSpeed `0.19.8`、Ray `2.48.0`。NPU 可用，共 16 张卡。用户约定自行将 verl 安装为 `0.9.0`；本次不修改上游 verl 或放开 0.9.1。

新后端已增加该 0.26 成对版本的接口适配，同时保留原 0.25.1 配对。先确认 verl 已按约定安装为 0.9.0。源码与 CPU 测试通过不代表真实训练闭环已验收。

进入正确环境后运行：

```bash
python3 - <<'PY'
import importlib.metadata as m
for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'deepspeed', 'verl', 'ray'):
    print(name, m.version(name))
from packaging.version import Version
from verl_speco.integration.ascend_runtime_compat import require_ascend_runtime
print('runtime pair:', require_ascend_runtime(m.version('vllm'), m.version('vllm-ascend')))
assert Version(m.version('verl')).base_version == '0.9.0'
import torch
import torch_npu
assert torch.npu.is_available(), 'NPU is unavailable'
print('NPU count:', torch.npu.device_count())
PY
```

缺少包、版本不符或 NPU 不可用时先停在这一步。本轮按约定使用 verl 0.9.0。具体配套环境仍需和可运行的 DeepSeek V4 RL 配置一起核对。

## 3. 在正确环境之上创建 SpeCo 环境

使用 uv 管理环境，继承已验证的 NPU runtime：

```bash
cd /models/z00829645/verl-SpeCo
python3 -m pip install uv
python3 -m uv venv --system-site-packages --python "$(command -v python3)" .venv-npu
source .venv-npu/bin/activate
uv pip install 'hydra-core>=1.3.2,<1.4' 'omegaconf>=2.3.0,<2.4' 'packaging>=24' pytest
uv pip install --no-deps -e .
python -c 'import verl, verl_speco, deepspeed, torch_npu; print("imports OK")'
```

这里的 `--no-deps` 避免安装 SpeCo 时改动已经配套的加速框架。它不会自动补齐上游 verl 的训练依赖；应以能够正常运行 DeepSeek V4 RL 的环境为基础。

## 4. 指定真实资源路径

以下两个路径必须替换，不能直接使用占位符：

```bash
export MODEL_PATH=/实际路径/DeepSeek-V4-0731-bf16
export RL_SCRIPT=/实际路径/working_deepseek_v4_rl.sh

export SPECO_RUN_ID=$(date +%Y%m%d_%H%M%S)
export DRAFT_WORK_DIR=/models/z00829645/speco_runs/$SPECO_RUN_ID/work
export DRAFTER_CKPTS=/models/z00829645/speco_runs/$SPECO_RUN_ID/checkpoints
export SPECO_LOG_DIR=/models/z00829645/speco_runs/$SPECO_RUN_ID/logs
mkdir -p "$DRAFT_WORK_DIR" "$DRAFTER_CKPTS" "$SPECO_LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export MTP_DS4_TP_SHARD_EH_PROJECTIONS=0
export MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS=0
```

`RL_SCRIPT` 必须是已经可运行的 **verl RL** 脚本，末尾将 `"$@"` 转发给 `python -m verl_speco.main`。远端现有的 MindSpeed SFT 启动脚本不能直接充当这个脚本。模型、训练数据、actor 并行设置、设备数由该基础 RL 脚本提供。多节点时 work/checkpoints 路径必须对全部 worker 可见。

先检查路径和包内实现身份：

```bash
python - <<'PY'
import os
from pathlib import Path
from verl_speco.integration.vllm_ascend_implementation import implementation_fingerprint
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract
model = Path(os.environ['MODEL_PATH'])
assert Path(os.environ['RL_SCRIPT']).is_file(), 'RL script missing'
print('implementation fingerprint:', implementation_fingerprint())
print('model contract:', DeepSeekV4Contract.from_model(model).identity())
assert any(model.rglob('*.safetensors')), 'Model weights missing'
PY
```

## 5. 执行不依赖真实模型前反向的接口测试

```bash
python -m pytest -q \
  tests/unit/test_deepseek_v4_native_backend.py \
  tests/unit/test_deepseek_v4_hc_backward.py \
  tests/unit/test_deepseek_v4_dsa_backward.py \
  tests/unit/test_deepseek_v4_export.py \
  tests/unit/test_deepseek_v4_vocab_parallel_loss.py \
  tests/unit/test_ascend_runtime_compat.py \
  tests/integration/test_ascend_runtime_source_contract.py \
  tests/unit/test_vllm_ascend_online.py \
  tests/unit/test_rollout_feature_capture.py \
  tests/unit/test_training_backend.py \
  tests/unit/test_deepseek_v4_contract.py \
  tests/unit/test_vllm_ascend_engine.py \
  tests/unit/test_vllm_ascend_checkpoint.py \
  tests/integration/test_vllm_ascend_source_contract.py
```

## 6. 执行真实 NPU 两轮闭环

确认基础 RL 脚本本身可运行后：

```bash
set -o pipefail
PYTHONUNBUFFERED=1 bash examples/run_deepseek-v4-0731_drafter_dspark_vllm_npu.sh \
  2>&1 | tee "$SPECO_LOG_DIR/two_step_smoke.log"
```

查看训练与发布记录：

```bash
rg -n 'drafter/(collected_samples|trained|published)|optimizer_steps|committed online drafter revision|Traceback|ERROR' \
  "$SPECO_LOG_DIR/two_step_smoke.log"
```

若远端没有 `rg`，使用 `grep -nE` 执行相同匹配。两轮应均采集到有效样本并训练、发布成功，optimizer step 连续增加。再核对参数校验和及更新后的推理行为；日志通过不能代替数值一致性验收。不要在本轮同时扩大 batch、序列长度或运行时版本范围。

把完整日志及步骤 2 的版本输出提供回来，继续定位真实设备问题。运行限制与验收边界见 [在线训练说明](vllm_ascend_cotrain.md)。
