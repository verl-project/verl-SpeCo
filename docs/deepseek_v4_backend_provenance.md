# DeepSeek V4 后端实现来源与维护

Last updated: 10/29/2026

## 维护范围

`training.backend=vllm_ascend` 使用 SpeCo 包内的 DeepSeek V4 0731 BF16 DSpark 实现。安装和运行只需要本仓库及配套 torch-npu、vLLM-Ascend、DeepSpeed 等依赖，不需要 AIRDrafter checkout、源码 ZIP 或 `DRAFT_ENGINE_SOURCE_ROOT`。

迁入对象来自用户提供的 AIRDrafter 实现，仅提取 DSpark 前反向、必要算子、teacher 输出头、采样和权重转换依赖。未迁入其他模型训练循环、AIR CLI、数据采集服务或独立控制平面。原始相对路径、文件 SHA256 和提取范围记录在 [`deepseek_v4_backend_sources.json`](../verl_speco/config/deepseek_v4_backend_sources.json)。这些 SHA256 标识迁入时的来源文件；当前包内实现指纹由实际维护文件计算。

## 沿用 SpeCo 布局

| 目录 | 职责 |
| --- | --- |
| `models/deepseek_v4/` | DSpark 模型、Ascend runtime 包装、checkpoint 格式和 TP 导出 |
| `models/target/deepseek_v4_head.py` | pre-HC → HC head → norm → vocab teacher；复用现有 checkpoint tensor loader |
| `ops/deepseek_v4_*.py` | HC、DSA、MoE、路由及词表并行损失的自定义梯度 |
| `backends/vllm_ascend_*.py` | 模型构造、并行拓扑、DeepSpeed 调度和有限训练周期 |
| `trainer/` | DSpark 数值循环、在线适配、恢复和数据契约 |
| `data/deepseek_v4_sampler.py` | DSpark 需要的采样器和分段记录 |
| `config/` | 可打包的数值配置与来源清单 |

全部路径相对 `verl_speco/`。不设独立 AIRDrafter 子目录，不修改已安装的上游 verl。原有 PyTorch 训练循环、其他模型、launcher、Feature Store、TQ 和运行时通道保留；新后端通过集中分派接入。

## 复用与兼容边界

在线训练复用 SpeCo 的 rollout worker、collection 事务、buffer、scheduler、worker 生命周期、发布缓存、传输与回滚；离线训练复用 launcher 和 TorchShardFeatureStore。数值循环执行一次 backward 和 optimizer step，外层不重复执行。

mask 统一为 `1=监督、0=忽略`，包括 padding 和 overlap。pre-HC 保存在已有 metadata tensor tree 中；保留既有数据布局、checkpoint 字段与参数名。只识别此前已验证实现的确切旧指纹，不接受任意旧源码修订；模型、数据、数值配置和拓扑校验继续执行。

旧 `source_root`、`--source-root` 仅为调用兼容保留；默认配置来自包内 YAML。可选原始源码对照测试仍允许开发者提供 checkout，生产代码不读取它。

## 验证及设备验收

CPU 测试覆盖包内独立导入、CLI 预检、正向 mask、HC/DSA 梯度、MoE 分块、词表并行损失、TP 导出和恢复。wheel 必须包含模型、算子、数值 YAML 和来源 JSON。真实 Ascend 前反向、DeepSpeed TP/DP 恢复和更新后推理仍需设备验收；顺序见 [远端操作手册](blue_zone_machine_runbook.md)。

## 本轮本地验证记录

- 扩展回归：559 项通过，5 项因本机缺少 verl/Ray 跳过；排除的双进程词表并行测试已单独运行并通过，另一个依赖真实 verl 的旧 log-prob 测试未执行。
- 未设置外部源码路径时，包内模型、算子、导出与独立 CLI 测试共 66 项通过。
- 原始配置与包内配置经相同路径绑定后，数值配置指纹一致；旧 checkpoint 兼容仍校验其余身份字段。
- wheel 在独立临时目录使用 Python 隔离模式导入成功，同时屏蔽外部训练框架命名空间，并检查模型、数值配置与来源清单完整。
- 14 个适配和入口文件通过 mypy；45 个相关文件通过 Ruff 与语法检查；示例 shell、命名及文档链接检查通过。

10/29 已同步至服务器 `/models/z00829645/verl-SpeCo`，包内入口导入通过，服务器针对性 CPU 回归 147 项通过。旧文件保存在 `/models/z00829645/verl-SpeCo-backup-20261029T035818Z`。尚未运行真实 NPU 训练。
