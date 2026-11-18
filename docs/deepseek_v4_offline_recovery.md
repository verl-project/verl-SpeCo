# DeepSeek V4 0731：离线恢复与标准权重验收

更新：2026-11-12。范围：`backend=vllm_ascend`、BF16 DSpark、TP8/DP2。
verl rollout 的直接 HS 采集已回退；本轮使用既有轨迹补采数据，不恢复该功能。

## 验收顺序

1. 用现有 `preflight`、`build_loader`、`build_engine` 和 `train_cycle` 训练一步。
2. 通过现有 `save_checkpoint` 保存完整模型、Adam、scheduler、进度、数据游标及各 rank RNG。
3. 原进程继续训练到 step 2，分块校验全部持久状态的字节。
4. 独立 torchrun 进程通过现有 `restore_checkpoint` 恢复 step 1，再训练到 step 2，逐 rank 比较。
5. 恢复验收后，复用该完整 checkpoint 导出标准 PyTorch/safetensors 草稿，再进行冷加载验证。
6. 离线基线完成后，验证权重发布与接收端推理；actor/native 特征对齐按要求暂搁置，旧 rollout 探针结果不替代 actor 验收。

## 2026-11-12 当前运行

新根目录为 `/data/zzc/speco_validation_archive/offline_completion_20261112`。
`status.json` 依次记录 `reference`、`resumed`、`export`、`cold_reference`、
`cold_artifact` 的完整命令、环境、PID 与 `returncode`；任一步失败即停止后续阶段。
本轮使用下文固定的原生重放数据，不依赖 actor HS 数值对齐。

旧的 strict 任务在训练前被环境检查拒绝：镜像已有 `LCCL_DETERMINISTIC=0`，
而配置采用 `setdefault`，所以 YAML 中的 `1` 未生效。现已在独立训练 launcher
创建子进程前，针对显式 `backend=vllm_ascend` 或专用配置设置
`HCCL_DETERMINISTIC=strict`、`LCCL_DETERMINISTIC=1`；专用 shell 示例也设置相同值。
显式 PyTorch 后端不注入环境，在线仍只向 Ascend Drafter worker 注入。
自定义 torchrun 命令也必须在启动前导出这两个变量；执行配方仍记录实际环境。

真实 CPU 子进程验证环境覆盖与父进程隔离；服务器启动/后端隔离回归
**124 passed**，日志 `launch_policy_tests.log`。

本轮五个阶段均退出 0；`verification_summary.json` 汇总各 rank 报告：

| 检查 | 实机结果 |
| --- | --- |
| 原进程保存 step 1 并继续到 step 2 | 16 rank 完成，完整 checkpoint 约 353.7 GiB |
| 新进程恢复 step 1 | 16 rank 的完整状态树逐字节一致，差异路径 0 |
| 恢复后训练到 step 2 | 16 rank 的模型、router cache、teacher、Adam/FP32 主权重、进度、游标及 RNG 逐字节一致；两边 rank-0 loss 均为 `0.2737244963645935` |
| 严格标准 PyTorch 导出 | 2,375 个张量，TP 复制参数检查及序列化读回通过；产物 `draft_pytorch/` |
| 原生/标准权重独立进程冷加载 | 16 rank 全部持久参数、cache、stride、NPU format 一致；训练前向 HS 与完整词表 logits 最大差值均为 0，重复前向及输入不变性检查通过 |

对应报告为 `resumed/recovery_report.json`、`resumed/restore_step1_rank_*.json`、
`cold_reference/cold_load_report.json` 和 `cold_artifact/cold_load_report.json`。
未启用探针的全局 Torch 确定性、空 metadata 改写或新建 builder 等诊断选项。
这是固定短数据、相同训练 TP 布局的验收，不代表所有输入、跨拓扑推理或长期训练。

随后 `server_update` 使用本轮 step-1 checkpoint，通过真实 verl HTTP/Ray
server、TP8/DP2/EP 和生产 IPC 接收器验证更新及版本提交。
2026-11-12 14:36:49 启动、14:42:44 结束，**退出码 0**。
探针要求全部 worker 的 prepare/IPC commit/finalize 确认，并检查 HTTP 健康状态、
更新时关闭准入且排队请求不能越过提交边界、target version 保持 0、drafter
revision 从 0 变为 1，以及更新前后 8 个目标贪心 token 一致；任一失败均非零退出。
命令、PID 与退出状态见 `server_update_status.json`，日志 `server_update.log`，
完整报告 `server_update/server_update_report.json`。2026-11-13 已核对成功退出状态及服务器完整服务报告。不采集 rollout HS，不运行完整 RL。

本轮已覆盖固定短数据的离线恢复、标准 PyTorch 交换及一次真实服务热更新。
尚未覆盖 actor 输入驱动的常驻训练两轮闭环、自动 RL 调度、多 target 版本、
长时间训练、性能或标准 safetensors 的本轮 NPU 冷加载（该格式已有 CPU 测试）。
跨运行时 HS 对齐暂搁置，不影响上述原生重放数据验收的边界。

2026-11-13 已重新核对并同步两端文档，见 [文件一致性记录](source_consistency_20261113.md)。
下文旧根目录的结果保留作问题定位记录，不作为本轮合格产物。

## 比较范围

`probe_deepseek_v4_recovery` 复用生产训练与恢复接口，额外校验模型持久 state_dict、
未注册为 buffer 的 router FP32 cache、teacher、优化器状态（含 FP32 主权重和 Adam moments）、
scheduler、训练进度、Feature Store 游标和 Python/NumPy/Torch/NPU RNG。
每次 CPU 拷贝限于约 8 MiB，计算完整逻辑字节的 SHA256；不是抽样校验。
非持久 scratch buffer 不作为可恢复权重比较，不比较不同进程的存储地址。
任何差异记录到具体路径，不自动放宽误差或修改 checkpoint。

## 固定配方与产物

- 模型：`/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16`。
- 数据：`/data/zzc/speco_restore_20261109_151846/features`。
- 运行目录：`/data/zzc/speco_validation_archive/offline_recovery_20261111`。
- 512-token segment、overlap 64、4 anchors、spacing 8、microbatch 1、accumulation 1、epochs 2。
- `reference_paired/checkpoints/step_00000001` 是当前同源验证使用的完整恢复 checkpoint；
  前一轮两版 runtime 发布快照不能代替它。
- 当前同源验证使用 `reference_paired` 与 `resumed_paired`；各自的
  `state_step1_rank_*.json` 和 `state_rank_*.json` 保存两步全量状态校验树。
- `status_paired.json` 保存完整启动 argv、进程和阶段状态；对应日志为
  `reference_paired.log`、`resumed_paired.log`。

正常探针只保存 step-1 checkpoint，step-2 对照使用校验树。
后端数值实现未更改；新探针不属于 numerical implementation fingerprint。

## 当前记录

服务器最新初始化/恢复/隔离及双进程 DTensor 导出回归：**61 passed**，
见 `/data/zzc/speco_validation_archive/cold_process_initializer_cpu_20261111.log`。
这里的测试集合与其他回归组有重合，不应累加为独立用例总数。
检查了实际 DeepSpeed 0.19.8 `state_dict` 实现：FP32 分组去 padding 使用切片视图，
不会因状态读取额外克隆整份 NPU 主权重。
首轮 `reference` 完成完整 step-1 保存和 step-2 训练；探针没有兼容
`lr_scheduler=None`，在最终状态读取时报错。该探针错误已修复，加入单测。
完整 checkpoint 约 353.7 GiB，包含 Adam 和 FP32 主权重。

为了避免重写，曾尝试以另一轮独立 step-1 作为参考（`reference_retry`）；
其模型与 Adam 状态和原 checkpoint 不同，而 RNG/进度没有报差异。
这不证明恢复损坏，也不能用于通过连续性验收，因此该对照已停止。

当前改为严格同源：`reference_paired` 于 10:53:39 启动，在同一进程完成
step-1 保存、step-1 全量校验和 step-2；11:05:17 正常退出并启动
`resumed_paired`。后者先比较恢复后的 step-1 全量状态，通过后才执行 step-2。
不跨独立训练假设 bitwise 确定性，不放宽差异阈值。

同源验证结果：16 个 rank 的恢复后 step-1 全量状态均逐字节一致。
续训到 step-2 后仍有模型及优化器差异；rank-0 loss：连续训练
`0.2736846805`，恢复后 `0.2734740973`。再次从同一 checkpoint 恢复
（`resumed_repeat`），step-1 仍一致，step-2 loss 约为 `0.272607`。
因此不能把差异仅归因于未保存连续训练热态；需核对输入和进程间前向重复性。
尚未通过严格 bitwise 续训验收，也没有放宽容差或改写 checkpoint identity。

标准 PyTorch 导出曾正常退出，分片合并及文件读回通过：
`draft_pytorch/` 共 22 个文件、约 36.97 GiB（19 个权重 shard 加 index/config/metadata）。
日志为 `export_pytorch.log`，退出状态见 `export_status.json`。
但后续实机冷加载发现旧导出器遗漏了部分 TP 复制参数检查，因此这份文件
**不能作为通过数值等价验收的权重使用**；文件读回成功不代表完整 TP 合并正确。

新增 `models/deepseek_v4/artifact_loading.py` 为独立冷加载入口：两种标准
序列化均调用原生 `load_weights` 和已有后处理、TP 投影安装函数。
共享 embedding/LM-head 及未导出的冻结 confidence 权重显式取自匹配的
原始 target；拒绝不完整参数覆盖和不支持的 confidence schema。
这仍是 draft-only artifact，不是自包含完整 target。

正式初始化入口支持可选配置：

```text
++draft_engine.overrides.model.initial_draft_artifact=/absolute/path/to/draft_pytorch
```

`actor_rollout_ref.model.path` 仍指向匹配的完整 target。设置该选项表示以草稿
权重新建优化器；完整模型/Adam/进度恢复仍使用 `draft_engine.resume_from`。
该初始化路径不计入数值配方指纹，移动 artifact 不会改变优化器恢复配方；
恢复完整 checkpoint 时可去掉此初始化选项。未设置时保留原有完整 target 加载。

`probe_deepseek_v4_artifact` 将比较原生 TP checkpoint 和标准 artifact 的
全部参数、router FP32 cache、固定输入 hidden states 与完整词表 logits。
验证范围是同一训练 TP 布局，不能替代 TP→EP 拓扑切换验收。
尚未将恢复数值连续性、标准格式实机冷加载或 actor/NPU 闭环标为通过。

## 冷加载探针中的问题与当前诊断

- `cold_pytorch` 试图同进程构造两个 wrapper，被现有 native bootstrap 保护拒绝。
  探针现已采用独立进程参考与 artifact 加载，不关闭保护，也不修改原生模型构造。
- `cold_reference` 和 `train_reference` 的重复前向均触发 AICPU
  `SparseAttnSharedkvMetadata` 参数错误（507018/0x2a）。后续 HC、Index 等算子名
  是异步报错位置，不能作为根因。
- `metadata_diagnostic` 启用同步定位，并保存 `metadata_rank_*_call_*.json`。
  rank-0 前四次 kwargs 的序列长度、dtype、shape 一致；前三次 native kernel 成功，
  第四次失败。`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_q` 都是原生
  builder 生成的空 Float32 tensor；尚未证明它们本身非法。
- 当前先做单卡、单 metadata 算子的重复调用复现，区分原生算子缓存、buffer 生命周期
  与模型调用问题。未据此替换注意力公式或修改稳定计算文件。

## 后续实测：导出副本检查缺口

`artifact_weight_check` 使用真实标准 PyTorch 文件、原生 Ascend loader 和后处理。
rank 0 的全部参数、cache、stride 和 NPU format 与 checkpoint 完全一致；其他
TP rank 的 HC、wq_a、wkv 和 router 等复制参数存在差异。直接读取原始 rank
checkpoint，确认差异在保存时已经存在：layer 43 的 wq_a 有 4,183/4,194,304
元素不同，最大差值 0.000244140625；hc_attn_fn 有 40,780/393,216 元素不同，
最大差值约 0.000190942。不能把这些差异解释为序列化读写误差。

新增 `models/deepseek_v4/export_validation.py`，在 SpeCo 正式导出入口检查
所有参与导出的复制参数，排除真实 TP shards 和重复注册别名。使用已有
topology/规范化/复制参数比较函数，不平均权重、不选择性忽略 HC/router 差异。
检查发生在输出文件写入之前。该改动不修改原生数值文件或优化器恢复指纹。

服务器导出及初始化回归 **35 passed**；真实 checkpoint 被新检查明确拒绝，见
`strict_export_validation.json`。保留原 checkpoint 和 artifact 供诊断，未改写其内容。
当前仍需定位训练更新后复制参数分叉的原因；各 TP rank 保存的 RNG 状态一致，
这只排除了状态摘要差异，不能代替实际输入及梯度比较。

## 后续实测：metadata 与 HCCL 交错复现

- 原生 metadata 单算子 5 种输入生命周期/空参数变体，共 60 次调用通过。
- 额外的输出释放及分配扰动 12 次调用通过。
- 关闭 builder 缓存、把三个空参数换成 None，完整模型仍失败，未用于生产配置。
- 无模型、无注意力的 8 卡最小测试中，先执行三次 metadata，再执行一次 HCCL
  通信，下一次 metadata 就失败。AllGather、AllReduce、ReduceScatter 均复现。
  日志为 `metadata_hccl_*.log`，启动参数及退出状态见 `metadata_hccl_status.json`。
- 先做 HCCL AllGather 再调用 metadata，8 卡连续 6 轮通过；标量预热、
  预热后再新建 HCCL group 也通过。单独设置 HOST 执行模式仍失败。
  对照日志及完整 argv 在 `metadata_hccl_control_status.json` 和
  `metadata_hccl_small_status.json`，不是完整模型验收结果。
- `backends/vllm_ascend_runtime.warmup_metadata_collectives` 已接入生产
  engine 和冷加载探针：只在 vLLM-Ascend 0.27.1rc1、TP>1 时，在首次
  metadata 前做一次标量 AllGather 并同步。旧版和 TP1 跳过，不改变模型
  参数、RNG、注意力公式或默认 HCCL 执行模式。服务器相关回归 **94 passed**。
- `metadata_warmup_status.json` 保存修复后的完整模型验证状态；该任务排在
  全量 actor 验证之后，避免并发加载大模型。重复前向尚需读取其最终结果。

新增 `recovery_probe_gradients_only=true` 诊断分支，仍调用生产 `train_cycle`。
它只记录实际采样输入、部分复制参数的完整本地梯度累计和更新前后字节，
不改变梯度，不额外保存完整 checkpoint。`replica_gradients` 与
`replica_gradients_strict` 分别验证现有通信配置与显式 HCCL 确定性配置，
完整参数记录在 `gradient_audit_status.json`。报告中的 `diagnostic_completed`
仅表示采集成功，不等于训练正确性或 TP 副本一致性通过。
