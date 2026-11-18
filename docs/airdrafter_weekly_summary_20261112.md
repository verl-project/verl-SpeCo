# AIRDrafter 接入 verl-SpeCo：近一周工作汇总

统计范围：2026-11-06—2026-11-12，按截至 11 月 12 日的近七天口径整理。
现有记录主要集中在 11 月 9—12 日；更早的建设单列为前期基础，不计为本周新增。
本文汇总已有 18 份相关报告，保留原始报告和失败记录，不重新运行训练或修改代码。

## 一、当前总体进展

目标为 **DeepSeek V4 Flash 0731、BF16 DSpark、`backend=vllm_ascend`**。
目前已通过固定短数据下的真实训练、完整断点恢复、严格标准 PyTorch 导出、
独立进程冷加载及一次真实服务 IPC 热更新。

在线 actor 特征获取和目标权重同步接口已接通；**当前 actor 驱动的连续训练、
自动发布及完整 RL 尚未整链验收**。actor 与 vLLM-Ascend 的特征存在可重复差异，
已完成诊断，按要求暂搁置修复。不能将整体描述为生产稳定或全数值对齐。

## 二、前期已建立的基础

以下为此前迁入和集成的累积成果，本周在此基础上适配与验证：

1. **包内维护。** DSpark 模型、前反向、必要算子、teacher、采样及权重转换已迁入
   SpeCo 的 `models/`、`ops/`、`backends/`、`trainer/`、`data/`、`config/`。
   运行不再依赖外部 AIRDrafter checkout；来源和提取范围有清单。
2. **后端选择与复用。** 用 `training.backend` 区分 `pytorch` 和 `vllm_ascend`；
   复用 SpeCo 的启动、Feature Store、训练触发、worker 生命周期和权重发布接口。
   没有另建 AIR 控制平面，也没有修改已安装的上游 verl 源码。
3. **数据契约统一。** token、绝对位置、辅助 HS、pre-HC 和监督行严格对齐；
   mask 始终采用 SpeCo 的 `1=监督、0=忽略`，padding/overlap 不重复监督。
4. **独立采集与训练工具。** 已有轨迹重放、最小样本选择、预检、一步训练、恢复和导出入口，
   可以不介入完整 RL 进行分项验证。

## 三、本周工作分类记录

### 1. 新旧 vLLM-Ascend 版本兼容与后端隔离

- 补充 vLLM/Ascend `0.27.1/0.27.1rc1` 的模块化 DSpark/MTP、decoder 类和 MoE ABI 适配。
- 对接新 DSA KV-plan dispatch；保留旧 DeviceOperator 及旧 MoE 调用分支。
- 保留 `0.25.1/0.25.1rc1`、`0.26.0/0.26.0rc1` 的兼容路径，未接受任意混搭版本。
- 新数值分支限制在 Ascend 训练；无梯度推理继续使用原生实现，原有 PyTorch 后端保留。
- 旧版本分支保留和接口测试通过，不等于这些版本本周均重新完成了全模型 NPU 验收。

参考：[0.27 兼容报告](ascend_027_compatibility.md)、[0.26 兼容报告](ascend_026_compatibility.md)。

### 2. 训练前向审查与原生算子复用

- 检查 HC、attention、router、routed/shared experts、TP 归约及输出头的实际调用链。
- 修正 MoE 路由权重缩放和 BF16 转换顺序，恢复默认原生 TP 前向归约。
- 接入原生 routed MoE 合并，处理权重乘法、累加与舍入边界。
- 定位实际矩阵尺寸下普通 GEMM 与原生 GMM 的差异；逐专家和 streaming 路径改用
  单专家 GMM，反向重放使用同一调用选择，保留梯度公式及完整梯度存储布局。
- 在 H=4096、全局 I=2048、TP8/DP2 的合成权重组件对照中，四条执行路径、
  两类输入、16 rank 共 **128 组** shared/本地合并/TP 输出均逐位一致。
- 实际 0731 权重训练步通过，审查的梯度组存在、有限、非零，观察到有效参数更新。

**边界：** 128 组结果是相同 TP 布局的组件验证，不是 TP8 训练与 EP16 服务整块前向对齐。
vLLM-Ascend 本身没有训练反向；反向正确性使用独立参考和梯度检查，不能表述为“与原生反向对齐”。
独立 BF16 梯度精度的完整验收仍待补齐。

参考：[前向一致性审查](deepseek_v4_0731_forward_audit.md)。

### 3. 权重派生缓存与显存生命周期修复

- 补齐 router `weight_fp32` 派生缓存刷新：覆盖优化器更新、checkpoint 恢复、热更新及回滚。
  原地维护参数和 cache，保留地址、对象及 NPU 格式。
- 完善常驻 engine、Adam、scheduler 的 park/wake 状态检查，区分持久训练状态与
  经过约束的小型派生统计量，不忽略未知池外张量。
- 修复服务异步退出与训练唤醒之间的容量竞态：轮次切换等待全部卡满足容量要求；
  异常收尾只等待本卡，避免依赖可能已退出的其他 rank。
- 大容量 NPU 专项覆盖每 rank **42.084 GiB** 状态及外部进程 **24 GiB** 占用，
  单卡和全部 16 卡均通过等待、恢复值/地址及正常退出检查。
- 11 月 10 日完整两轮常驻诊断退出 0：同一 trainer/Adam/scheduler 连续 step 1→2，
  服务 revision 0→1→2，生产 park/wake 和收尾释放通过。

**历史范围：** 上述两轮使用后来撤回的 rollout HS 采集，且固定 target version=0、服务轮间重启。
它证明当时的训练/发布/生命周期组合可执行，不能代替当前 actor 路径、原生服务 sleep/wake
或真实同卡 RL 的验收。实机容器约 **1.85 TiB**，也不构成 1 TiB 配额保证。

参考：[状态生命周期](ascend_state_lifecycle.md)、[前向审查中的两轮记录](deepseek_v4_0731_forward_audit.md)。

### 4. 特征来源调整与真实 actor 采集

- 按要求撤回 **verl rollout 中直接采集 HS**；保留独立轨迹重放和在线 actor old-logprob 采集。
- actor 侧获取指定 decoder 层输出的 HC 均值，以及最终 HC head/norm 之前的完整 pre-HC。
- 适配 eager actor 的右侧 padding、不同长度样本、窗口位置和原有特征传输；
  hook 保存自有存储，避免被后续 buffer 复用覆盖。
- 完整 **284.33B** 模型在真实 verl 0.9.0、16 卡 FSDP2、BF16/CPU offload 下完成短输入验证。
  导出特征形状 `[2, 8, 28672]`，内部独立 hook/HC/logprob 对照最大误差为 0。
- 本目标辅助 HS 宽度为 `4096×3=12288`，pre-HC 宽度为 `4096×4=16384`；
  28672 是两者在传输边界的合计宽度，不是普通单层 hidden size。

**边界：** FSDP2 是此次 actor 的执行与分片方式，Drafter 新后端仍使用 DeepSpeed 与原生 Ascend 算子。
这里的零误差仅比较同一 HF actor 内部采集，不是与 vLLM-Ascend 的比较，也没有吞吐基准。

参考：[actor 特征报告](deepseek_v4_actor_features.md)、[rollout 采集回退记录](rollout_feature_capture.md)。

### 5. actor → Drafter 目标权重同步

- 补齐同一 target 版本的六项权重：最终 HC head 的 `fn/base/scale`、最终 norm、
  `lm_head.weight` 和 `embed_tokens.weight`。
- 前五项用于刷新 pre-HC→HC→norm→词表分布的训练 teacher；共享 LM-head 和 embedding
  同时刷新 Drafter 冻结层，不通过 Drafter 优化器训练这些共享层。
- FSDP2 仅聚合所需 DTensor；CPU offload 时只临时激活选中分片进行通信。
- 接入版本、字段、词表分片、缓存与原地更新检查；缺少 embedding 时拒绝沿用旧值。
- 完整 actor 六项导出通过；共享层/cache 的 NPU 分项测试 **8 passed**，覆盖 ND/NZ、
  BF16/FP32、连续更新和地址保持。

**边界：** 这是 actor→训练端的同步；Drafter→推理端发布是另一条链路。
字段与分项应用通过，不等于真实 actor 数据驱动的连续训练整链已通过。

参考：[actor 特征报告](deepseek_v4_actor_features.md)、[集成进展](airdrafter_integration_progress_20261112.md)。

### 6. 完整 checkpoint 恢复与训练确定性

- 建立复用生产保存/恢复接口的全量状态探针，按逻辑字节分块 SHA256 比较，避免只抽样参数。
- 比较模型、router cache、teacher、Adam/FP32 主权重、scheduler（若有）、训练进度、
  数据游标及 Python/NumPy/Torch/NPU RNG，不比较跨进程地址或非持久 scratch。
- 定位普通通信配置下的 TP 梯度/复制参数分叉；启用
  `HCCL_DETERMINISTIC=strict`、`LCCL_DETERMINISTIC=1`。
- 进一步发现镜像预设 `LCCL_DETERMINISTIC=0` 使 YAML 的 `setdefault` 无效；
  修正为 Ascend 子进程启动前注入，并记录实际环境。显式 PyTorch 不注入这些变量。
- 11 月 12 日 TP8/DP2 全量验收通过：新进程恢复 step 1 和再训练到 step 2，
  **16 rank 完整状态均逐字节一致**；连续/恢复训练的 rank-0 step-2 loss
  均为 `0.2737244963645935`。

另用最小复现定位首次 HCCL 与 sparse metadata 的初始化顺序问题：仅在已核对的
Ascend **0.27.1rc1、TP>1** 路径，首次 metadata 前执行一次标量 AllGather 并同步。
没有用修改注意力公式或放宽检查解决错误。

参考：[离线恢复报告](deepseek_v4_offline_recovery.md)。

### 7. 标准权重导出、冷加载与服务热更新

- 补齐所有参与导出的 TP 复制参数检查。旧 checkpoint 中的 HC/wq_a/router 等分叉被明确拒绝，
  不平均权重、不静默取 TP0，也不把旧单元素一 ULP 例外当作默认严格通过。
- 新 checkpoint 严格检查通过，导出标准 PyTorch 草稿 **2,375 个张量、约 36.97 GiB**。
  标准格式是 draft-only，不是自包含完整 target，也不是跨后端优化器恢复格式。
- 新增可选标准 artifact 初始化，复用原生 loader、后处理和 TP 安装；共享层及未导出的冻结层
  显式从匹配 target 加载。完整 Adam 恢复继续使用 DeepSpeed checkpoint。
- 16 rank 独立进程比较原生 checkpoint 和标准 PyTorch 冷加载：全部持久参数、cache、stride、
  NPU format 一致；固定训练输入的 HS、完整词表 logits 及重复前向差异均为 0。
- 11 月 12 日真实 verl HTTP/Ray server 热更新任务 **退出码 0**：TP8/DP2/EP、生产 IPC 接收、
  worker 确认、准入阻塞、target version 保持 0、drafter revision 0→1、更新前后 8 个贪心 token 一致。

**边界：** 标准 safetensors 已有格式/CPU 验证，本轮全模型 NPU 冷加载实际验证的是 PyTorch。
训练 TP 布局的冷加载零误差不等于 TP→EP 全模型前向零误差；一次服务更新也不等于 RL 自动闭环。
服务任务已取回成功退出状态，完整服务报告保存在服务器，尚未复制到本地。

参考：[权重兼容报告](deepseek_v4_weight_compatibility.md)、[离线恢复报告](deepseek_v4_offline_recovery.md)。

### 8. actor 与 vLLM-Ascend 跨运行时对齐诊断

- 使用同一 checkpoint、两条固定短输入、相同绝对位置和层号，比较真实 actor 与原生重放。
- 核对 HC 均值/pre-HC 边界和层号转换，未发现简单取层或展平错位。
- 比较各自真实前向 logits 的温度 1 分布，而非共用 teacher 重建输出。
- 确认最终 HC head 有 BF16/FP32 有效参数精度差异；精度控制实验有部分改善，但未消除差异。
- 原生同进程重复及独立重载结果均零差异，actor 重跑特征也一致，跨运行时差异可重复。

| 默认路径的比较指标 | 16-token 输入 | 12-token 输入 |
| --- | ---: | ---: |
| 辅助 HS 相对 L2 范围 | 10.71%–14.15% | 9.98%–13.26% |
| pre-HC 相对 L2 | 10.61% | 10.14% |
| 全词表分布平均 TV | 11.57% | 7.35% |
| top-1 相同位置数 | 7/8 | 7/8 |

**状态：诊断完成，对齐未通过，修复暂搁置。** 两条短样本不能量化训练效果损失，
也不能将根因直接归于 FSDP2。后续离线验收使用原生重放数据，没有以 actor HS 替代。

参考：[跨运行时对齐报告](deepseek_v4_actor_native_alignment.md)。

## 四、验证结论的统一口径

| 已有证据 | 可以支持的结论 | 不能据此宣布 |
| --- | --- | --- |
| CPU 回归、接口/版本测试 | 指定路径、契约和拒绝条件符合预期 | 全模型 NPU 数值与性能通过 |
| MoE 128 组零差异 | 已测同 TP 组件前向逐位一致 | TP8→EP16 整块前向和 BF16 反向均对齐 |
| actor 内部 hook/logprob 零误差 | HF actor 的采集位置和传输正确 | actor 与原生 vLLM 特征可互换 |
| 16 rank 恢复/续训完整状态一致 | 固定配方的两步恢复连续性通过 | 长期训练、任意数据或跨实现恢复通过 |
| 标准 PyTorch 冷加载零误差 | 同训练 TP 布局的转换及加载通过 | safetensors 本轮 NPU 冷加载或完整 target 自包含 |
| 真实服务 IPC 任务退出 0 | 本次发布、版本提交和生成检查通过 | 当前 actor 驱动的完整 RL 与性能通过 |

CPU 回归按报告分组记录，例如 0.27 适配 **433 passed**、GMM 修复 **526 passed/4 skipped**、
actor/共享层相关 **210 passed**、启动/隔离 **124 passed**、对齐探针 **7 passed**。
测试集合存在重叠，不累加成所谓“本周独立测试总数”；跳过项不计为通过。

## 五、阶段时间线

| 日期 | 本周主要工作与结果 |
| --- | --- |
| 11 月 9 日 | 0.27.1 环境适配、原生重放采集、真实 TP8/DP2 一步训练及初始前向审查；发现缓存、MoE 执行和复制参数问题。 |
| 11 月 10 日 | 缓存刷新、原生 MoE 合并/归约、单专家 GMM、唤醒/收尾时序修复；组件对照及旧采集来源的完整两轮诊断通过。 |
| 11 月 11 日 | 撤回 rollout 直接采集，接入 actor old-logprob/pre-HC 与六项权重同步；完整 284.33B FSDP2 短输入验证通过；推进恢复、导出与 metadata 排查。 |
| 11 月 12 日 | 完成跨运行时差异诊断并暂搁置；修正确定性通信启动环境；全量恢复、严格 PyTorch 导出、冷加载及一次真实服务热更新通过。 |

## 六、尚未完成与后续顺序

1. **当前在线路径验收：** 使用真实 actor 特征及配对权重，验证同一常驻训练器连续多轮、
   自动训练触发、发布和后续推理；不能复用已停用 rollout 探针的通过标签。
2. **多版本与恢复：** 验证 actor target 更新、特征/teacher/共享层配对、过期数据拒收、
   RL 与 Drafter 配对 checkpoint 恢复，以及连续多轮发布和失败恢复。
3. **进一步数值验收：** 实际 TP8 训练与 EP16 服务的整块前向、独立 BF16 梯度精度；
   actor/native 特征差异的修复按用户要求继续搁置。
4. **稳定性与性能：** 长序列、多样本、多步训练、吞吐、采集开销、发布延迟和资源峰值。
   当前没有证据承诺 1 TiB 下完整同卡常驻 RL 可用。
5. **其他能力边界：** 本轮未完成标准 safetensors 全模型 NPU 冷加载、新后端 scheduled
   TransferQueue、专用分卡资源池、其他 MTP 模型或未支持 EP/SP/EPLB 训练的扩展。
   生产在线仍有 PP/engine-DP 等限制，诊断 TP8/DP2 不代表已解除生产限制。

## 七、历史报告总索引

原报告的“尚待验证”描述可能属于较早阶段。判断当前状态时，以日期更晚且比较范围匹配的
结构化结果为准；不同数据来源或拓扑的结果不能互相替代。

| 分类 | 原报告 | 本汇总中的定位 |
| --- | --- | --- |
| 总体 | [集成进展](airdrafter_integration_progress_20261112.md) | 截至 11 月 12 日的功能状态及同步含义 |
| 设计 | [迁移计划](airdrafter_migration_plan.md) | 原始设计与复用原则；其中 rollout 采集方案已撤回 |
| 来源/布局 | [后端来源与维护](deepseek_v4_backend_provenance.md) | 前期迁入范围、包内布局和可独立安装基础 |
| 兼容 | [Ascend 0.26](ascend_026_compatibility.md) | 旧版本适配基础 |
| 兼容 | [Ascend 0.27.1](ascend_027_compatibility.md) | 本周新 ABI、导入布局、算子与早期实测 |
| 数值 | [训练/推理前向审查](deepseek_v4_0731_forward_audit.md) | 缓存、MoE、GMM、归约及历史两轮验证 |
| 资源 | [状态与显存生命周期](ascend_state_lifecycle.md) | 状态契约、CaMem、暂存和生产钩子；早期待办需结合后续记录 |
| 数据 | [actor 特征](deepseek_v4_actor_features.md) | 当前 actor 来源、完整目标模型验证和目标权重同步 |
| 数值 | [actor/native 对齐](deepseek_v4_actor_native_alignment.md) | 跨运行时误差、精度控制、重复性与暂搁置事项 |
| 恢复 | [离线恢复与标准权重](deepseek_v4_offline_recovery.md) | 最新恢复、metadata、确定性、严格导出和冷加载证据 |
| 发布 | [权重兼容与热更新](deepseek_v4_weight_compatibility.md) | 产物边界、TP→EP、IPC 和真实服务提交历史 |
| 在线 | [随 verl 自动训练与回传](vllm_ascend_cotrain.md) | 在线接入、配置及生产约束，不作为完整 RL 通过证明 |
| 数据 | [轨迹重放采集](deepseek_v4_trajectory_features.md) | 独立原生重放输入格式、mask 和特征工具 |
| 回退 | [rollout 直接采集](rollout_feature_capture.md) | 已撤回功能的历史设计，不按旧说明重新启用 |
| 验证 | [后端与冒烟指南](deepseek_v4_0731_draft_smoke.md) | 前期预检、训练、恢复和导出操作 |
| 验证 | [最小特征与一步训练](deepseek_v4_minimal_smoke.md) | 单轨迹、TP8/DP2 和历史 OOM/保存处理 |
| 验证 | [无 RL 冒烟](deepseek_v4_no_rl_smoke.md) | 独立串行分项验证；旧机器路径需替换 |
| 部署 | [旧服务器操作手册](blue_zone_machine_runbook.md) | 前期传输/环境记录；不是当前服务器命令清单 |

## 八、关键产物与交接位置

当前服务器：`SLAI-Verl-Workspace`，仓库 `/data/zzc/verl-SpeCo`。
模型：`/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16`。
原生重放数据：`/data/zzc/speco_restore_20261109_151846/features`。
环境：verl 0.9.0、vLLM 0.27.1、vLLM-Ascend 0.27.1rc1、torch 2.10.0+cpu、
torch-npu 2.10.0.post4、DeepSpeed 0.19.8、Transformers 5.14.1。

以下路径均相对于 `/data/zzc/speco_validation_archive/`：

| 位置 | 内容与用途 |
| --- | --- |
| `online_shutdown_fixed_20261110/` | 旧 rollout 来源的两轮诊断、独立复核、runtime revision 1/2 快照；不是完整优化器恢复 checkpoint |
| `shutdown_release_npu_20261110/` | 大容量收尾等待与地址/数值恢复专项 |
| `actor_fsdp_retry_20261111/results/` | 完整 actor 的采集、六项权重及内部对照报告 |
| `offline_recovery_20261111/` | 早期恢复差异、导出拒绝、metadata 和梯度故障证据；旧产物不作为本轮合格权重 |
| `actor_native_alignment_20261112/` | 跨运行时基线、精度控制、边界及重复性报告 |
| `offline_completion_20261112/status.json` | 本轮五阶段完整命令、环境、PID、退出状态 |
| `offline_completion_20261112/verification_summary.json` | 16 rank 恢复与冷加载的零差异汇总 |
| `offline_completion_20261112/reference/checkpoints/step_00000001/` | 本轮已验收的完整模型/Adam 恢复 checkpoint，约 353.7 GiB |
| `offline_completion_20261112/draft_pytorch/` | 本轮严格导出的标准草稿权重 |
| `offline_completion_20261112/server_update_status.json` | 真实服务更新命令和退出 0 记录，运行时间 14:36:49–14:42:44 |
| `offline_completion_20261112/server_update/server_update_report.json` | 服务器上的完整热更新报告；本地尚未复制 |

数值实现指纹为 `73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582`。
不能因配置形状相同或权重能导出，就放宽实现身份和运行配方检查来恢复旧 optimizer。

此前已清理两份失败诊断的冗余发布快照，约 **114.6 GiB**，保留原始故障记录及有效产物；
清单见 `validation_snapshot_cleanup_20261110.json`。本次汇总不删除文件、不创建备份目录。
此前文档同步曾受审批服务额度限制；2026-11-13 已核对实测报告并同步两端，
见 [文件一致性记录](source_consistency_20261113.md)。
