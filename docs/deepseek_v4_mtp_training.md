# DeepSeek V4 MTP 训练接入记录

## 范围

新增 `backend=vllm_ascend`、`speculative_algorithm=MTP`、`draft_engine.variant=ds4_mtp`
的独立离线训练入口。正式代码按 SpeCo 的 models、ops、trainer、integration 和 config 布局组织，
不依赖外部 AIRDrafter 源码目录。使用方法见 [MTP 离线操作说明](../offline_training/README_MTP.md)。

本次使用 `/data/share/deepseek-ai/DeepSeek-V4-Flash-BF16`，其配置包含一个原生 MTP 层，
不包含 DSpark 配置。0731 DSpark checkpoint 不是同一草稿结构，不得替换使用。

## 已迁入与复用

| 环节 | 实现 |
| --- | --- |
| 特征 | 独立 vLLM-Ascend 轨迹重放；原生 getter 获取完整 pre-HC，落入现有 Feature Store |
| 数据 | 复用 SpeCo 分段、重叠区监督归属、collator、DP sampler；MTP 单独校验 contract |
| 前向 | 原生 Ascend MTP、HC、attention、MoE 和输出头；没有另写 PyTorch MTP 架构 |
| 反向 | 复用现有 Ascend 算子适配；MTP 新入口安装既有 DSA query replay 和安全 RoPE 适配 |
| 损失 | 迁入 Eagle 词表分片 KL；按预测步的全局有效 token 数处理 DP 与梯度累积 |
| 生命周期 | 复用 SpeCo 分布式启动、DeepSpeed、teacher、优化器、调度器、训练游标和完整 checkpoint |
| 权重 | 复用原生 MTP TP 合并和键名转换；新增 MTP 标识、严格副本检查、标准权重初始化 |

mask 始终采用 SpeCo `1=监督、0=忽略`。输入为 `[token, 16384]` 完整 pre-HC，
不读取 AIR prompt mask，也不把 DSpark 的辅助 HS 当作 MTP 输入。
第 k 步递推的 token 和 teacher 偏移 k，位置偏移 k−1；尾部补零位置不参与监督。

## 验证设计与问题记录

记录目录：`/data/zzc/speco_validation_archive/mtp_integration_20261112/`。
命令、环境、PID、返回码分别记录在 `capture_status.json`、`deployment_status.json`、
`validation_status.json`；逐阶段日志与报告保存在同一目录。

- CPU 回归：覆盖 MTP 数据契约、递推偏移、独立稠密 KL/梯度/Adam 对照，以及已有 DSpark 隔离路径。
- 分布式损失：真实双进程 Gloo，非均分词表分片，与完整词表损失及梯度比较。
- NPU：先采集一条真实轨迹，再做 TP8/DP2 两步训练、完整恢复及标准权重冷加载。
  一条样本在 DP 补齐时会重复；该测试验证链路，不用于比较训练收益或吞吐。
- 恢复对照：逐 rank 比较模型、router 缓存、Adam、调度器、RNG、数据游标和特征 store 状态；
  同时比较恢复瞬间和续训一步后的状态。
- 冷加载对照：原生完整 checkpoint 与导出的标准草稿，比较参数/缓存/布局，以及固定输入的
  MTP hidden states 和完整词表 logits。该对照不等同于完整 serving engine 热更新验收。

第一次 NPU backward 检查发现 `wq_a`、`q_norm`、`wq_b` 缺失梯度，未继续保存 checkpoint。
根因为原生推理 DSA prolog 中的原地算子断开 Q 路径，MTP 外壳未安装 DSpark 已使用的适配。
修复仅复用该适配：前向保留原生输出，反向接入可微重放；没有删除或放宽梯度检查。
失败诊断归档于 `failed_query_gradient_audit/`。

首次导出发现 0.27 MoERunner 的 `_shared_experts._layer` 别名被当作复制参数检查。
修复后逐 rank 检查别名与公开 shared-expert TP 分片的值一致，再按分片合并；
真正的复制参数仍必须跨 TP 逐元素相同。早期记录归档于 `failed_export_alias_check/`；
最终 checkpoint 在修复后的完整实现身份下重新生成。

## 保留的边界

当前仅支持直接 BF16、一个原生 MTP 层、Eagle KL、非 EP 的 TP/DP ZeRO-1 训练。
MTP 在线 RL 自动训练与热回传尚未接入；不恢复 verl rollout HS 抓取。
独立 PyTorch MTP 基线尚待提供，未进行两种实现的训练收益比较。
actor/native HS 对齐仍按此前约定搁置。

MTP 使用独立数值指纹；原有 DSpark 指纹保持
`73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582`。
旧版/新版 Ascend 导入由既有兼容层负责；本次实机 MTP 验证环境为 0.27.1，
不能据此宣称 MTP 在 0.26.0 上也已实测通过。

## 最终实测结果（2026-11-12）

| 项目 | 结果与证据 |
| --- | --- |
| 最终 CPU 回归 | 134 passed；`cpu_final.log`、`cpu_final_status.json` |
| 双进程词表 KL | Gloo 两个 rank 均通过完整词表损失/梯度对照；`distributed_cpu.log` |
| 原生重放 | 1 条真实轨迹，TP8/DP2；`capture_status.json`、`features/` |
| NPU 训练 | TP8/DP2、ZeRO-1，2 个优化器步；loss 2.6906835 → 2.0570875 |
| 首次反向 | 28/28 组可训练参数有有限、非零梯度；`reference.log` |
| 完整恢复 | 16 rank 在恢复瞬间及续训一步后，模型/缓存/Adam/调度器/RNG/游标等状态逐字节相同；`resume/recovery_report.json` |
| 标准 PyTorch 导出 | 798 张量，严格副本检查及逐张量读回通过；`draft_pytorch/speco_export_report.json` |
| 标准草稿冷加载 | 16 rank 参数、缓存和布局一致；固定前 32 行的 HS/完整词表 logits 最大误差均为 0；`cold_artifact/cold_load_report.json` |
| 正式用户入口 | `offline_training/train.sh train` 从标准草稿初始化后，真实训练一步成功；`launcher/training_report.json` |

冷加载误差对照双方是**同一训练 TP 布局下，原生完整 checkpoint 与其标准 PyTorch 导出草稿**。
重复前向也逐字节一致。该结论的范围不包含完整 serving engine、热更新或另一套独立 PyTorch MTP。

实机环境：Torch 2.10.0、torch-npu 2.10.0.post4、vLLM 0.27.1、vllm-ascend 0.27.1rc1、
DeepSpeed 0.19.8、Transformers 5.14.1。完整版本、命令、时间和最终结果见
`acceptance_report.json` 及逐阶段 status 文件。

最终可复用文件均在上述记录目录下：

- 特征：`features/`。
- 可完整续训的第 1 步 checkpoint：`reference/checkpoints/step_00000001/`。
- 标准 PyTorch 草稿：`draft_pytorch/`。
- 部署源码及 SHA256：`speco_mtp_integration_20261112.tar.gz`、`source_manifest.json`。

两步连续训练的第 2 步用于恢复一致性对照，未另存一份完整 checkpoint。
正式入口验证也关闭了重复保存。失败排查中已被最终产物替代的旧 checkpoint 已删除，
释放约 122.4 GiB；保留其诊断日志和对照报告。
