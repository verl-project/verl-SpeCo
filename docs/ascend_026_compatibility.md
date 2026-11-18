# vllm_ascend 后端适配 Ascend 0.26

Last updated: 10/28/2026

## 范围

本轮兼容适配限于新后端及其采集适配器。用户约定安装 verl 0.9.0；不修改上游 verl，不新增 0.9.1 的兼容声明，也不绕过 verl 原有 API 检查。

支持成对版本：

| vLLM | vLLM-Ascend |
| --- | --- |
| 0.25.1（允许本地构建后缀） | 0.25.1rc1 |
| 0.26.0（包括远端的 `0.26.0+empty`） | 0.26.0rc1 |

只忽略 PEP 440 本地构建后缀；不同系列混配、未知 rc/正式版和未来版本仍拒绝。训练和 rollout 共用 `integration/ascend_runtime_compat.py`，防止两边接受不同版本组合。

## 审查与变更

- 审查远端安装的 `NPUModelRunner`、`ExecuteModelState`、DeepSeek V4 target、DSpark model、sampler、worker 和 speculative metadata 源码。
- Ascend 0.26 使用包含 `positions`、attention metadata 和 `batch_desc` 的命名执行状态。只读取消费的命名字段，不依赖 tuple 下标；状态字段漂移明确拒绝。允许没有执行状态的空采样调用。
- 保持原采样结果不变；在 `sample_tokens` 清除状态之前保存请求/调度信息，在 `_sample` 返回时读取已确认 token。拒绝请求数与 draft counts 不一致。
- 验证 DSpark decoder layer ID 到 target output layer ID 的 `+1` 转换，并核对实际启用的 aux 层。pre-HC 使用原生 getter，不用普通 hidden states 替代。
- 采集路径禁用 FlashComm1，并固定 V1 model runner；不消费序列分片后的局部 aux 行。用户明确开启 FlashComm1 时拒绝该采集配置。
- 权重更新仍走现有 SpeCo 通道，传递处理后的 runtime TP 分片。测试覆盖 0.26 的 `routed_experts` 命名、BF16 MoE 和 FP32 HC 参数原地更新，不执行 checkpoint 矩阵的二次转置。
- 保留现有 PyTorch 默认训练路径。后续数值实现已迁入本仓库，来源与布局见 [实现来源说明](deepseek_v4_backend_provenance.md)。

## 执行验证

安装好用户约定的 verl 0.9.0，进入已配套的 Ascend 环境后：

```bash
cd /models/z00829645/verl-SpeCo
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

python3 - <<'PY'
from importlib.metadata import version
from packaging.version import Version
from verl_speco.integration.ascend_runtime_compat import require_ascend_runtime
from verl_speco.backends.vllm_ascend_runtime import runtime_versions
print(require_ascend_runtime(version('vllm'), version('vllm-ascend')))
assert Version(version('verl')).base_version == '0.9.0'
print(runtime_versions(require=True))
PY

python3 -m pytest -q \
  tests/unit/test_ascend_runtime_compat.py \
  tests/integration/test_ascend_runtime_source_contract.py \
  tests/unit/test_rollout_feature_capture.py \
  tests/unit/test_vllm_ascend_online.py
```

`test_ascend_runtime_source_contract.py` 自动读取已安装的 Ascend Python 源码；也可设置 `SPECO_ASCEND_RUNTIME_SOURCE_ROOT` 指向源码仓库。测试执行选取的纯接口方法与命名状态定义，不导入或初始化完整模型。

之后按 [远端操作手册](blue_zone_machine_runbook.md) 配置 0731 BF16 权重、基础 verl RL 脚本，再运行两轮 NPU 冒烟。版本和接口测试通过不代表前向、反向、TP/DP 通信、恢复或更新后推理已经通过设备验收。

## 本轮验证记录

- 本地扩展回归：505 项通过、5 项跳过、1 项因缺少 verl 未执行；17 个相关源文件 mypy 通过。
- 远端实际 0.26 环境：51 项通过、1 项跳过。3 项安装源码契约检查通过，后端依赖校验接受 `0.26.0+empty / 0.26.0rc1`。已观察到用户安装的 verl 为 `0.9.0`，本轮未修改它。
- 上述远端记录属于包内迁入前的版本，当时跳过了外部源码对照测试。当前运行已不依赖外部源码；新的包内导入和数值测试需在同步更新后重跑。真实 NPU 前反向仍未验收。
- 0.26 接口适配已同步到 `/models/z00829645/verl-SpeCo`。替换前相关文件保存在 `/models/z00829645/verl-SpeCo-backup-20261028T063944Z`。
