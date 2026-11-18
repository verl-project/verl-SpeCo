# DeepSeek V4 0731：训练与推理前向一致性审查

> 2026-11-11 更新：按用户要求回退 verl rollout 的直接 HS 采集。本文两轮采集/训练/发布结果保留为历史证据，原探针现已停用。当前在线使用 actor old-logprob，actor/NPU 路径需要独立验收；独立轨迹补采、训练后端及热更新保留。

初始审查：2026-11-09；验证更新：2026-11-10。对象：本仓库 `backend=vllm_ascend` 的 BF16 DSpark 路径。

## 本轮最终判定

**2026-11-10 23:34:38，完整 TP8×DP2 两轮独立验证正常结束，exit=0。** 已确认 rollout 特征采集→同一常驻 trainer/Adam/scheduler 连续 step 1→2→服务 revision 0→1→2→收尾释放与正常退出；PyTorch backend 和旧版解析分支保留。当前数值实现指纹为 `73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582`。

这不是完整 RL 或整个 DSpark 数值对齐的验收。相同 TP8 布局下的 MoE 组件对照已逐位一致；实际 TP8 训练与 EP16 服务的整块前向、独立 BF16 梯度精度、最新完整 checkpoint 恢复/导出/冷加载仍需后续验证。当前容器约 1.85 TiB，不把本轮结果作为 1 TiB 容量保证。所有后续判断以本节和结构化报告为准，下文保留失败与修复过程。

## 自主验证记录（2026-11-10）

用户已授权代理在 `SLAI-Verl-Workspace:/data/zzc/verl-SpeCo` 自主执行 NPU 验证；此前仅由用户执行 NPU 的限制已撤销。下列证据均保存在 `/data/zzc/speco_validation_archive`，不覆盖旧权重、不删除原始失败日志。

| 阶段 | 当前证据与结果 | 验证范围 |
| --- | --- | --- |
| 首批接入 CPU 回归 | `native_combine_cpu_regression_20261110.log`：524 passed、4 skipped、exit=0 | 接口、存储、CPU 梯度参考和 backend 隔离；跳过的四项为 NPU cache 测试 |
| 原生合并 NPU | `moe_native_combine_npu_latest.json`：四路径×两种输入，八组全部逐位一致 | 单卡局部 routed MoE；梯度存在/有限/非零，不代表 BF16 梯度精度已对照 |
| 真实训练冒烟 | `native_combine_tp8_dp2/train_retry_status.json`：exit=0、TP8×DP2、16 ranks、optimizer step=1 | 真实 0731 权重与已有特征，完整训练步可执行；不等于同输入整块前向已对齐 |
| 共享专家单卡 | `moe_shared_tp1_retry_20261110/report.json`：exit=0，八组通过 | 原生 shared MLP/linear/activation 函数、合成权重，H=64/I=64 |
| 小尺寸 TP 对照 | `moe_shared_tp8_dp2_small_20261110/report.json`：exit=0，16 rank 报告全通过 | H=64、全局 I=512、TP8×外部 DP2；本地 shared/sum、TP 输出及原生 collective 重复性 |
| 实际尺寸首次对照 | `moe_shared_tp8_dp2_representative_20261110`：逐专家/streaming 有差异 | 原生合并仍不足以消除 GEMM/GMM 舍入差异，详见下节 |
| 单专家 GMM 修复回归 | `single_gmm_cpu_regression_20261110.log`：526 passed、4 skipped、exit=0 | 新旧调用隔离、重放选择、全参数 CPU VJP 与现有兼容用例 |
| 实际尺寸修复后对照 | `moe_shared_tp8_dp2_gmm_fixed_20261110/report.json`：16 ranks 全通过 | 每 rank 八组，shared、本地相加、TP 输出均逐位一致；128 组 max_abs=0 |
| 最新真实训练 | `single_gmm_train_20261110.log`：exit=0、TP8×DP2、step=1 | loss=0.4943084121、有效 block tokens=20、anchors=4、optimizer delta=0.0001220703125 |
| 常驻状态与分布式 loss 补查 | `single_gmm_lifecycle_cpu_20261110.log`：69 passed、exit=0 | Adam/地址/scheduler 连续性、park/wake、两进程 vocab loss 与 dense 梯度参考 |
| 完整两轮最终验收 | `online_shutdown_fixed_20261110/verified_summary.json`：exit=0 | 16 ranks、step 1→2、revision 0→1→2、生产 park/wake 与完整正常退出 |
| 大容量收尾专项 | `shutdown_release_npu_20261110/report.json`：单卡及全部 16 卡通过 | 每 rank 42.084 GiB 状态，外部进程占用 24 GiB，等待释放后值/地址保持 |
| 最新单卡反向重放 | `moe_single_gmm_backward_20261110.json`：八组通过 | 局部前向逐位一致；输入/gate/w13/w2 梯度均存在、有限、非零 |

真实训练使用现有 `/data/zzc/speco_restore_20261109_151846/features`，未重复采集。首次命令因预检/训练共用非空输出目录被正确拒绝，`failure_rank_0.json` 为 `FileExistsError`，尚未进入模型初始化。该错误源于给出的命令，不应放宽训练的目录保护。重试改用新的 `native_combine_tp8_dp2/train` 子目录，日志写在其父目录，避免 tee 提前创建文件再次触发保护。

重试命令：`RUN_DIR=/data/zzc/speco_validation_archive/native_combine_tp8_dp2/train`、`TRAIN_TP=8`、`CARD_COUNT=16`、`MAX_STEPS=1`、`PYTHON_BIN=python3`，执行示例脚本 `train`，覆盖 `save_interval_steps=100`、`save_final_checkpoint=false`。2026-11-10 13:16:03 启动，13:17:10 正常退出；loss=`0.4954467118`、有效 block tokens=20、anchors=4、optimizer probe delta=`0.0001220703125`；已审查参数组均无 missing/zero/nonfinite 梯度。`checkpoint=None` 是本次禁止写大 checkpoint 的预期结果。

原生合并单卡测得整个训练段前向峰值增量约 16.13–16.18 MiB、含反向约 40.04 MiB；这些不是与旧实现做 A/B 后得到的功能独立增量。本例新增 route 缓冲区理论大小为 12 KiB，完整分组路径为 0（复用已有输出），不能据此外推整模型峰值。

新增 `probe_deepseek_v4_moe_tp` 复用现有 routed 夹具，执行已安装的 `DeepseekV2MLP.forward`、原生 unquantized linear/activation 及 `MoERunner._maybe_reduce_final_output`；投影使用显式合成 TP 分片，未测试 native checkpoint loader。相同输入和 gate 在所有 rank 复制，专家/共享权重随 TP rank 改变，在 DP 副本间保持相同；每组另跑一次原生 collective，区分算子重复性与适配误差。初次单卡探针遗漏 TP=1 跳过归约的记录分支，已只修探针并重跑，原始错误记录保留在 `moe_shared_tp1_20261110`。

训练日志还有 AccumulateGrad stream mismatch 和退出时 shared_memory resource_tracker 警告，未将这些警告当作测试失败，也未通过关闭警告掩盖；当前 eager 配置下实际 optimizer step 完成。完整 DSpark 同输入/KV 元数据前向、BF16 显式反向精度、长程训练、W1 单 ULP 复制漂移、完整 checkpoint 恢复/导出与完整 RL 仍需分项验证；本轮 runtime 权重发布进展见下节。

### 实际尺寸诊断与单专家 GMM 修复

首轮 H=4096、全局 I=2048、TP8×DP2 对照中，共享专家、完整 grouped/budgeted 路径及 collective 重复性全部逐位一致，但逐专家/streaming 路径仍有差异：普通输入 TP 输出 max_abs=0.001953125、relative L2≈0.000278918；高幅值输入分别为 0.5、≈0.000285908。实际权重为 ND（format=2），不能将小尺寸 GEMM/GMM 一致结果外推到真实矩阵尺寸。

`moe_single_gmm_diagnostic_20261110` 进一步使用 H=4096、TP8 本地 I=256：单专家 GMM 对完整原生 GMM 的两类输入各 393216 个元素均零差异，定位为 GEMM/GMM 执行精度差异。修复将 split MoE 的逐专家及 streaming 前向改用单专家 GMM，完整分组与预算分组保留既有 GMM。反向仅将重建 gate_up / expert_output 的调用换为相同 GMM，梯度公式、完整参数/梯度存储布局保留；前向保存调用选择供 autograd 重放，避免依赖已退出的 ContextVar。CPU 和范围外旧路径保持普通 GEMM。

**当前实现指纹：`73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582`。** 本节覆盖下文历史阶段的指纹与结论；历史“backward AST 未变”仅针对 da6 原生合并阶段，本轮两个重放函数已有上述必要修改。未建立旧 optimizer 状态的自动兼容别名。

最新 16 卡实测逐一读取全部 rank 报告：四种路径×普通/高幅值输入，共 128 组；shared output、本地 routed+shared、TP final output 与 native collective 重复性均 exact=true、finite=true、max_abs=0。跨 rank 的训练输出和原生参考输出分别也完全相同。汇总为 `single_gmm_all_rank_summary_20261110.json`；这是合成权重、原生组件函数的对照，未覆盖真实 checkpoint loader、完整 DSpark、其他 layout 或其他通信设置。探针采用 `disable_custom_all_reduce=True`、外部 DP2；不把该结果扩展为所有通信配置均已验证。

并行布局必须单独区分：训练配置 `model.vllm_enable_expert_parallel=false`，路由专家按 TP8 的 intermediate 维分片；真实服务 `expert_parallel_size=tp_size*dp_size=16`，专家按 EP16 分配（现场日志每 rank 16/256 experts）。当前 128 组零误差证明的是相同 TP8 切分下的原生组件参考，不是 TP8 训练对 EP16 服务的端到端零误差。TP 归约位置、GMM 矩阵尺寸和 BF16 舍入路径可能不同；其误差尚未量化。后续完整前向验收必须使用实际 EP16 服务、相同输入/权重/KV 元数据，与 TP8 训练逐段对照；仅通过权重重分片/摘要检查不能替代该项。

最新真实训练在 21:47:38–21:48:48 完成，使用真实 0731 权重及已有特征；16 卡 TP8/DP2、global_step=1、optimizer 更新有效、审查梯度组无 missing/zero/nonfinite。未生成 checkpoint，因此还没有用新指纹验证恢复或完整权重导出。新单卡八组补测实际执行四条 NPU 反向路径，全部梯度存在、有限且非零；其判据不是独立 BF16 梯度误差上限。显式 backward 仍以 sigmoid/乘法重算 SwiGLU 中间值，而前向默认用融合 `npu_swiglu`；路由敏感度也有自身的乘法/归约精度。后续独立梯度参考必须覆盖这些中间舍入，不能把本轮“GMM 重放一致”扩展为全部 BF16 导数已经精确验收。

补充 CPU 检查本地先得到 68 passed、1 failed：唯一失败是沙箱禁止 `127.0.0.1` bind，未执行分布式 loss 运算。随后在服务器原环境运行相同三份测试，69 passed、exit=0（28.60 秒）；未为通过测试修改实现或降低判据。

### 两轮闭环：首次结果与释放时序修复

`online_gmm_fixed_20261110` 在 21:52:46–22:09:38 执行，第一轮 16 rank 完成 step=1、runtime snapshot、生产 park；revision 1 的服务权重更新和第二轮 rollout 采集均 passed，但第二轮训练前被原有 wake_capacity 检查拒绝。所需恢复状态约 41.2–42.4 GB，部分设备当时只有 20.1–39.6 GB 空闲。日志同时记录原生服务退出仍在 SIGTERM/SIGKILL worker；任务完全退出后 `npu-smi` 显示全部卡无进程、占用回到约 2.9–3.2 GB。这支持服务子进程/设备上下文异步回收与训练唤醒之间存在竞态，不支持降低容量阈值。`failure.json`、各 rank 原始 traceback 和阶段报告保留，不能把本轮报告为两轮通过。

修复限定在 `trainer/deepseek_v4_resident_probe.py`：第二轮唤醒前，使用现有 trainer 的设备查询和 CPU control group 全 rank 汇总，等待所有卡空闲容量满足原有 managed_bytes 要求，最长 120 秒；本地卡就绪不能提前退出其他卡的等待。查询异常与超时均全 rank 失败，并保存通过时的 `release_round_02.json`。不初始化额外 NPU 进程，不用固定长 sleep、不关闭生产容量检查、不修改前向/反向与 optimizer 状态。新增全 rank 等待、远端超时、查询异常三项测试，本地 resident suite 56 passed，服务器三份相关 suite 72 passed、exit=0（27.60 秒），Ruff 通过；数值实现指纹保持 73af。

重试入口为 `probe_deepseek_v4_online_rounds --production-lifecycle`，TP8/DP2、64 response tokens，输出 `online_release_fixed_20261110`，日志与状态为同名前缀 `.log` / `_status.json`；完整重试前不宣称成功。该探针重启 rollout 服务、固定 target version=0，并使用明确的 probe-only engine-DP2 许可；真实 verl 在线路径仍要求 engine-DP1、以 rollout replicas 提供训练 DP。不能将本探针等同于真实 RL 与 actor target 更新验收。

释放屏障的实机证据：`online_release_fixed_20261110/release_round_02.json` 记录等待 6.180959 秒、7 次全 rank 采样；随后全部卡满足原 managed_bytes 要求，最少空闲约 43.36 GB。第二轮完成 optimizer step=2、runtime snapshot 和生产 park。单独逐一检查两轮全部 16 个 worker：PID、engine/Adam/scheduler 身份均相同，第二轮 Adam step 全部为 2，wake_verified=true，state=parked；结果 `resident_two_step_summary.json`。第二轮 loss=0.9892123938、有效 tokens=20、optimizer delta=0.000244140625，步耗时约 1.595 秒；两轮很小的不同采样训练不构成收敛或质量评估。

最终服务的 revision 1、revision 2 两次发布均完成：每版都有 16 个唯一 `(dp_rank,tp_rank)` 的 `ipc_committed` 确认、每 worker 75 个参数，IPC、版本提交、更新期间准入阻塞和 Target greedy token 保持检查均通过，证据 `serving_final_summary.json`。该诊断路径额外读取全部 TP runtime 分片、构建回滚备份并多次核对摘要，阶段耗时包含诊断 I/O，不能直接当作生产内存权重发布延迟。

但本次主脚本最终仍 exit=1（22:13:01–22:48:34）：退出路径 `worker.finally` 直接调用 `parking.wake`，遗漏轮次间已有的容量屏障，服务设备上下文尚未完全释放时触发 `aclrtMallocPhysical ...207001 OOM`。因此“两轮训练和两版服务发布”可分项记为通过，不能将该整条命令标成成功。首次修复只覆盖轮次间交接，遗漏退出入口；已补查所有 wake 调用并新增 `wake_parked_for_shutdown`，收尾仅等待本卡容量后恢复存储，避免异常退出时依赖其他 rank 的 collective。查询失败/超时不唤醒；active 状态不重复映射。数值后端和容量要求保留，增加三项单元测试，本地 resident 59 passed，服务器相关 suite 75 passed、1 skipped（34.03 秒）。

收尾专项测试 `tests/integration/test_ascend_shutdown_release_npu.py` 使用每 rank 42 GiB 的实际 CaMem 状态池，另起进程持有 24 GiB 设备内存、显式延迟释放。它要求实测 free<managed，屏障必须等待至少两次观察，随后恢复值和地址、进程正常退出。大测试以 `SPECO_RUN_LARGE_NPU_TESTS=1` 单独启用，避免混入普通 CPU/NPU 小测试。输出与完整命令在 `shutdown_release_npu_20261110/status.json`；先单卡验证，再并发覆盖全部 16 卡。此专项结果不能伪装成此前主脚本 exit=0。

收尾专项实测通过：单卡 1 passed（36.16 秒），16 卡并发各 1 passed、总进程 exit=0。全部卡 managed=42.084 GiB，均先实测到 free<managed，等待约 3.524–4.027 秒后恢复；16 份报告均确认值和地址保持。总结果 `shutdown_release_npu_20261110/report.json`。仅测试源码发生变化，数值后端指纹保留 73af。

完整主脚本从头重跑最终通过：`online_shutdown_fixed_20261110`，22:58:54–23:34:38、TP8/DP2、生产 lifecycle、exit=0。`verified_summary.json` 独立复核当前 probe SHA256 与启动时一致、数值指纹为 73af；全部 16 rank 两轮 PID/engine/Adam/scheduler 身份一致，第二轮 Adam step 全为 2，wake 与 park 检查通过。最终服务的两次提交各覆盖 16 个唯一 `(dp_rank,tp_rank)`，每 worker 75 个参数；IPC、准入阻塞、版本标签和 Target greedy token 保持均通过。

轮次间容量等待 6.210 秒、5 次全 rank 观察；16 份收尾记录均符合本卡 free>=managed，收尾等待范围约 0.003–4.537 秒，随后全部进程正常退出。两轮 loss 分别为 0.4819715321、1.0045681000；仅用于流程与有限性验证，不是收敛证明。最新 revision 1/2 的 Markov W1 各八个 TP 分片、每份 33095680 个元素均对 TP0 零差异；仅说明本轮未复现旧漂移，不宣称旧故障根因已定位。

状态文件保存完整 argv、显式环境、依赖版本、probe SHA256 和启动前全部设备使用量。退出后再次检查：16 卡 HBM 占用回到约 2.9–3.2 GB，8 个双芯片板均无 NPU 进程。日志中原生服务退出仍可能出现强制终止 worker 的警告；本轮没有隐藏这些警告，最终训练器释放屏障、正常退出和 exit=0 均有独立证据。

最终索引：主报告 `online_rounds_report.json`，独立复核 `verified_summary.json`，全轮日志 `../online_shutdown_fixed_20261110.log`，各 rank 收尾记录 `shutdown_release_rank_*.json`。最新两版发布快照在 `snapshots/revision_1`、`snapshots/revision_2`；它们是 runtime 权重快照，不含完整 optimizer recovery checkpoint。

第二轮快照的 Markov W1 也已核对八个 TP 分片全部元素，均与 TP0 逐位相同，证据 `online_release_fixed_20261110/markov_w1_replica_revision_2.json`。仍不把两轮小样本未复现描述为旧故障根因已经查明。

首次 runtime snapshot 的 Markov W1 单独核对全部八个 TP 分片：33095680 元素对 TP0 均逐位一致，报告 `markov_w1_replica_revision_1.json`。这一轮未复现旧 checkpoint 的单 ULP 差异；没有证明旧故障根因已消除，也没有替代第二轮、其他复制参数或 Adam 状态的完整核验。

## 2026-11-10 修复进展

以下正文保留审查时的故障证据；本节记录后续修复，不能把原始问题描述当作当前实现。

- **A 已修复代码与 CPU 验证**：新增 `ops/ascend_weight_cache.py`；热更新在同一事务中复制参数、刷新 FP32 gate cache，失败恢复两者。保持参数和 cache 对象/存储地址；探针回滚、三处 checkpoint 恢复入口、optimizer boundary 后同步刷新。park/wake 状态检查现在包含未注册为 buffer 的 cache。
- **B 已修复缩放顺序**：split MoE 保留原生带 scale 的路由权重，在原生位置转换为 BF16；额外输出 scale 读取实际 runner 配置，不再无条件乘 model scale。CPU 用例分别检查 router/output scale，原有全参数梯度对照保留。
- **C 已恢复默认原生归约**：split MoE 调用 native runner 的 `_maybe_reduce_final_output`，反向采用 TP row-parallel 的 identity 连接；wo_b 和旧 MoE 包装默认也保留原生前向。确定性 FP32 累加仍可通过 `MTP_DS4_DETERMINISTIC_TP_ROW_FORWARD=1` / `MTP_DS4_DETERMINISTIC_TP_MOE_FORWARD=1` 显式开启，仅用于诊断。
- **D 窗口修复纳入本次代码更新**：保留完整 token IDs 和 prompt 边界，仅裁剪特征，保持绝对位置；不修改正式 collector 的边界检查和 SpeCo mask 语义。
- **F online guard 已修复**：读取与模型工厂相同的 `cfg.train` 字段；未适配的 confidence 训练/发布组合明确拒绝。此项不代表可选 confidence 发布链路已实现。

数值实现与指纹已改变；optimizer 恢复仅接受完全相同的实现身份，取消原始实现自动别名例外。旧 checkpoint 保留，不能直接在本次数值修复后续训。旧版运行时解析分支和原生 PyTorch backend 保留。

本次 20 个文件已按服务器基线校验并同步至 `/data/zzc/verl-SpeCo`，包括此前的窗口候选。服务器真实依赖下的回归结果为 **504 passed、4 skipped，exit=0，41.93 秒**；四个跳过项是显式关闭的 NPU cache 用例。CPU 回归覆盖旧版/新版接口、缓存事务、完整参数梯度 reference、checkpoint 恢复、actor/rollout 特征及 backend 隔离，不等于整块 NPU 前向验收。日志为 `/data/zzc/speco_validation_archive/forward_fix_cpu_validation_20261110.log`；本次涉及 Python 文件的 Ruff 检查通过。

首批修复实现指纹为 `4d0f30e4e9f3fc0458d8934be770c3c14d359f77256239480a355c2836b379f3`；正文环境表中的 `448e865...` 为修复前审查基线。后续激活修正后的当前指纹见下节。

新增轻量 NPU 测试 `tests/integration/test_ascend_weight_cache_npu.py`，覆盖 BF16/FP32 参数、ND/NZ cache、连续两次更新与恢复后的冷权重 logits 对照。默认跳过；早期由用户显式执行，当前代理已获授权执行：

首次用户执行得到 2 passed、2 failed：ND 用例通过；NZ 用例因独立进程未开启 `torch.npu.config.allow_internal_format`，申请的 NZ cache 回退为 ND，随后格式断言失败。原生 `NPUModelRunner` 初始化会开启该选项。测试现改为仅在 NZ 用例内临时开启并自动恢复，且在任何更新前检查初始格式；恢复后也检查格式。该修正只影响测试，后端代码及实现指纹不变；NZ 是否保持存储与数值仍须重跑验证，不能把首次失败报告为 cache 更新已经损坏。

```bash
cd /data/zzc/verl-SpeCo
SPECO_RUN_NPU_TESTS=1 python3 -m pytest -q -s tests/integration/test_ascend_weight_cache_npu.py
```

**尚未完成**：完整 MoE/DSpark 中间前向与梯度对照、E 中 W1 复制参数漂移的根因、真实两轮闭环及完整 RL。专家合并仍使用训练 replay，不能因 scale 和 collective 已修正就宣称与原生融合内核逐位相同。以上为早期修复阶段的待办；NPU cache 和后续自主 NPU 验证见前文。旧阶段“代理不启动 NPU”的限制已撤销，导出容差未放宽。

## 后续验证与路由专家激活修正（2026-11-10）

用户运行 cache 测试最终得到 **4 passed、17 warnings、17.99 秒**；BF16/FP32 参数、ND/NZ cache 的两次更新及恢复后 logits 对照通过，地址和格式保持。此前测试 fixture 将写入型配置误交给 monkeypatch，现通过底层选项读取与 try/finally 恢复；服务器无 NPU 初始化的真实接口检查也覆盖初始开启/关闭及异常恢复。

进一步核对 native `FusedMoEFactory`、`RoutedExperts`、输入 dataclass 与 `unquant_apply_mlp` 后确认：当前 DeepSeek V4 BF16 路由专家激活为普通 SILU，该原生分支即使收到 `swiglu_limit=10` 也不裁剪。旧 split training 无条件裁剪，实际存在不一致。执行原生函数的 CPU 控制流（仅替换底层 GMM 和 SwiGLU 为 CPU 运算）得到 400，而旧训练得到约 99.99546；该结果证明分支选择差异，不是 NPU 误差统计。

现仅修正 split training 的路由专家：普通 SILU 使用 limit=0；只有 native `swigluoai_uninterleave` 标志开启时采用配置 limit，非默认 alpha/beta 明确拒绝。共享专家原有裁剪逻辑保留，旧版 MoE 路径和 PyTorch backend 保留。新增超过阈值时的前向/全参数梯度回归及真实原生源码分支检查。当前实现指纹为 `126cae4ebac4b70ebbe58b98323498d6a547474e3c7c45180af5847d25a49852`，依然不能将旧 optimizer 状态直接视为同一数值实现。

新增 `python3 -m verl_speco.probe_deepseek_v4_moe --output <report.json>`：单卡、256 专家、top-k=6、hidden/intermediate=64、16 tokens，BF16、sqrtsoftplus、scale=1.5；对普通/超过裁剪阈值输入及逐专家/分组 replay 共四组采集。调用实际原生 router、dispatcher、MLP、token combine，分别报告路由 ID/权重、GEMM 对 GMM、相同专家结果的合并、整体 routed output 的 exact/max_abs/RMS/relative L2，并检查梯度有限且非零。报告保存运行时版本、原生源码 hash 及实现指纹。

该命令的 `measurement_completed` 仅表示采集完成；`forward_exact` 单独记录逐位一致性，`precision_acceptance=not_assessed_no_tolerance_applied`。不预设误差容差来获得“对齐通过”。当前范围**不含共享专家、TP8、完整 DSpark 或 RL**；不得把局部报告代替后续整块验收。

### 局部实测与合并精度核对

用户已完成四组 NPU 测量：路由 ID/权重、逐专家 GEMM 对原生 GMM 均逐位一致，梯度均存在、有限且非零。两种 replay 在普通输入下输出最大差为 `0.0009765625`、relative L2 为 `0.0037664161`；大幅值输入分别为 `0.25`、`0.0040644398`。差异与“相同专家值、仅更换合并”测量完全相同，已集中到 token combine。

继续读取已安装 CANN 9.1.0 的 `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/moe_token_unpermute/moe_token_unpermute.h`：概率与 token 输入转换为 FP32，`CalFirstToken` 第 297–314 行及 `CalToken` 第 317–338 行使用 FP32 Muls/Add，按 token 的 top-k 槽位依次处理，`CopyOut` 第 342–354 行最后转换回 BF16。相对地，训练旧合并先在 BF16 中乘权重，再按专家 ID 顺序逐次 BF16 累加。这是已确认的源代码执行路径差异；实测对哪种公式逐位一致仍待下一轮报告，不能仅凭读取头文件量化各因素的贡献。

本轮只扩展探针、CPU 对照测试和本文档，未改变后端、梯度或实现指纹。新增八种 CPU 算术控制（乘法 BF16/FP32 × 累加 BF16/FP32 × top-k 槽位/专家顺序），使用本次 NPU 的原始路由映射与专家输出，对照 native 合并。映射须覆盖全部 route rows，不能靠改 token/slot 顺序获得通过；同时直接调用仓库已有的 opt-in `token_unpermute` helper，检查它是否复现 native combine。该调用仅在探针内临时开启，后端默认保持不变。多个 FP32 公式也可能同时逐位一致，届时不能据此单独归因给累加顺序。

### 原生合并接入（2026-11-10 阶段记录，后续验收见前文）

用户完成算术控制后，四组测试均确认：现有训练结果与 BF16 乘加/专家顺序公式逐位一致；FP32 乘法与 FP32 累加的两种顺序均与 native 输出逐位一致；仓库已有 token-unpermute helper 也与 native 输出逐位一致。仅提升乘法或累加之一仍有误差。因此已有测量支持将主要差异归因于中间舍入，不能单独归因于顺序。

按用户要求，现已在新版 split MoE 训练边界启用原生合并，涉及逐专家、完整分组、逐专家低内存、分组预算四条前向。新增 `ops/ascend_moe_combine.py` 用局部 ContextVar 管理调用范围，退出或异常时恢复；不改进程环境开关。NPU 强制复用已验证的 token-unpermute helper，算子不可用时明确报错，不静默退回旧 BF16 合并。CPU 仅提供原生 FP32 槽位顺序公式作测试参考。旧版 MoE 调用范围之外保持原先合并，PyTorch backend 不变。

逐专家和两种低内存路径新增临时、未加权的 route-output 缓冲区，BF16 大小为 `tokens × top_k × hidden_size × 2` 字节；仍按原计划计算各专家，不额外保留全部中间激活。完整分组路径直接复用已有专家输出。缓冲区不加入 backward 保存张量，完整梯度和参数存储布局保留。两个 autograd 类及两种低内存 backward 的 AST 均与服务器基线一致。

CPU 回归新增四路径 FP32 前向/全参数 VJP 对独立 autograd reference 的比较、参数指针与梯度形状检查、四路径 BF16 舍入反例，以及局部范围/异常恢复/原生算子缺失检查。分组测试只替换底层 NPU GMM 为 CPU 矩阵乘，实际 replay、合并与显式反向均执行；这不等于验证 NPU BF16 反向精度。

当前实现指纹为 `da6a831f1478d8880a4cdfb4f9e77ac98bb02a5597cf637e68b1e6c6ddc2fc95`；上文 `126cae4...` 是接入前基线，不能作为新实现的 optimizer 恢复身份。

NPU 探针现覆盖四路径 × 两种输入共八组，记录原生合并调用次数、前向及前向+反向峰值增量和理论新增 route 缓冲区字节数。使用 `--require-forward-exact` 时，任何局部输出不逐位一致都会使命令失败，不放宽容差；通过也仅表示被测局部前向一致，不代表共享专家、TP8、完整 DSpark、NPU BF16 反向或 RL 已通过：

```bash
cd /data/zzc/verl-SpeCo
set -o pipefail
python3 -m verl_speco.probe_deepseek_v4_moe --require-forward-exact \
  --output /data/zzc/speco_validation_archive/moe_native_combine_npu_latest.json \
  2>&1 | tee /data/zzc/speco_validation_archive/moe_native_combine_npu_latest.log
```

## 初始审查结论与边界（2026-11-09 历史记录）

**目前不能认定整个训练前向与原生 vLLM-Ascend 推理前向已经对齐。**
已有结果证明了单步训练、断点恢复、参数传输和服务版本提交能够工作；它们没有证明相同输入、相同权重下两条前向的中间张量和 logits 一致。本次发现了一个可以复现的派生缓存遗漏，并确认了 MoE 缩放、BF16 转换和 TP 归约的执行路径差异。应先解决这些问题，再继续两轮闭环。

以下段落仅描述 2026-11-09 初始审查状态，后续修复和 NPU 结果以前文 2026-11-10 记录为准。该次审查没有启动新的 NPU 训练或推理。已有 NPU 结果来自用户执行的测试。窗口修复仅作为候选在本地准备，并通过临时文件用真实数据完成 CPU 复验；尚未部署到服务器仓库。没有在审查过程中修改训练数值实现或放宽默认校验。

## 1. 实际审查环境

| 项目 | 实际值 |
| --- | --- |
| vLLM / vLLM-Ascend | `0.27.1+empty` / `0.27.1rc1` |
| vLLM 提交 | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Ascend 提交 | `3b31886237ed65c435e0965b10e8002ff5769766` |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| verl / DeepSpeed / Transformers | `0.9.0` / `0.19.8` / `5.14.1` |
| 目标模型 | `/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16` |
| 当前容器内存限制 | cgroup v1：2032999997440 bytes（约 1.85 TiB）；不是 1 TiB 容量验证 |
| 模型配置 | hidden=4096，HC=4，vocab=129280，target layers=[40,41,42]，block=5 |
| 实际路由 | sqrtsoftplus，top-k=6，归一化开启，scale=1.5 |
| 其他关键配置 | SwiGLU limit=10，SWA window=128，Sinkhorn=20 次 |
| 数值实现指纹 | `448e8656462e63a66fc89e8d905611e1513fb31682071cc5f6419581cc5bf0d9` |

原生代码按服务器实际安装版本阅读；本地镜像中的 model、DSpark、MoE runner、RoutedExperts、Linear、DSA 六个文件均核对了 SHA256，一致。完整清单保存在服务器 `/data/zzc/speco_validation_archive/forward_audit_runtime_inventory_20261109.json`。

本结论不外推为 0.25/0.26 的新一轮 NPU 回归结论；旧版兼容分支仍保留。

## 2. 初始调用链核对（修复前基线）

| 环节 | 原生 / 训练实现对照 | 当前证据与结论 |
| --- | --- | --- |
| Target 特征 | 选定层按 HC 求均值；最终 teacher 保留 raw pre-HC；使用显式绝对位置 | 源码、数据契约和已采集数据相符；不等于 actor engine 的真实 NPU 路径已测 |
| Query 输入 | 原生 Triton 核首位写 target 已采样的 next token，后续写 noise token；位置从最后 context 位置加一开始 | 训练 `build_runtime_block_inputs` 的 seed=anchor+1、label=anchor+2 与默认 `sample_from_anchor` 分支相符；bonus 变体不在当前支持范围 |
| Main projection / norm | 调用原生 DSpark `combine_hidden_states` | 复用原生模块；Linear/RMSNorm 的训练包装保留原生前向值，再接反向 |
| Context KV / metadata | 调用原生 context-KV 投影和 `build_for_drafting`；训练侧构造 block table 与 slot mapping | 绝对 RoPE 位置与线性 KV 槽索引在原生输入核中也是分开的；训练布局有 CPU 测试，但尚缺整块 NPU 对照 |
| HC pre/post | 当前配置使用 `npu_hc_pre_v2` / `npu_hc_post`；显式反向由训练侧提供 | 与当前原生 decoder 使用同类前向算子；CPU 导数测试通过，尚无整层 NPU 梯度对照；关闭融合时是另一条参考前向 |
| DSA query / sparse attention | query replay 返回原生 q/qr 的前向值；sparse attention 包装返回原生算子输出 | 源码可确认这些包装保留原生前向值；不能由此证明 metadata、上下文和整个注意力输出已完全一致 |
| Attention output projection | wo_a 保留原生 BMM；wo_b 训练默认改用固定顺序 FP32 归约 | wo_b 与原生 collective 的执行路径不同，见发现 C |
| MoE gate / 选专家 | 训练从实际 gate 参数计算 FP32 logits，并调用原生 router 选择 IDs | 初始权重下计算意图相同；更新后原生 FP32 缓存可能陈旧，见发现 A |
| Routed experts | 训练复用完整专家参数，但自行执行 grouped matmul、激活和路由合并 | 不是直接复用整个原生 MoERunner 前向；缩放和类型转换顺序不同，见发现 B |
| Shared experts | 原生投影模块；非原地 clamp → cat → `npu_swiglu` | **激活算子**已有 4 个真实 NPU 对照用例通过；不把它扩大为整个 MoE 已对齐 |
| Final HC head / LM Head | 复用 DSpark HC head；训练做 vocab-local projection | 正常全词表路径复用原生调用；local 路径使用 `F.linear`，需对照；内置 projection 对照只在 strict diagnostics 开启时运行，当前默认关闭 |
| Markov Head | 训练 teacher forcing 使用真实 previous token；推理串行使用先前草稿 token | 这是训练目标与采样流程的有意区别；做算子对照时必须固定同一 previous_token_ids。复制参数已观察到单 ULP 漂移，见发现 E |
| Confidence | 当前训练关闭；推理模型自带的头冻结并保留 | 当前配置不参与 loss；可选训练功能还有配置检查/发布问题，见发现 F |

对应仓库代码：

- [DSpark 输入、metadata、logits](../verl_speco/models/deepseek_v4/modeling_dspark.py)
- [原生前向包装、HC、DSA 与 TP 归约](../verl_speco/ops/deepseek_v4_runtime.py)
- [0.27 split MoE 训练分支](../verl_speco/ops/ascend_moe_training.py)
- [0.27 DSA KV-plan 适配](../verl_speco/ops/ascend_dsa_compat.py)

## 3. 初始发现与故障证据（修复前基线）

### A — P0：热更新遗漏 FP32 gate 派生缓存

原生 `AscendUnquantizedLinearMethod.process_weights_after_loading` 根据 `gate.weight` 建立 `weight_fp32`；0.27.1 的 `AscendMoERunner` 在 internal-router 分支实际读取 `gate.weight_fp32`。真实 checkpoint 中三层 gate 参数均为 BF16 `[256,4096]`，因此该 FP32 缓存不是 BF16 参数本身。

`EmbeddedDraftLoader` 只收集 `named_parameters()`；`finish()` 对参数执行 `copy_`，没有刷新 `weight_fp32`。提交后的 `_speco_rebuild_draft_metadata_buffers` 仅处理带 `_build_fused_kv_buffers` 的模型及其三类 fused-KV/K-norm 缓存，不处理这个 gate 缓存。

使用实际 `EmbeddedDraftLoader` 的 CPU 复现：

| 检查 | 结果 |
| --- | --- |
| gate 参数是否更新 | 是 |
| 独立 FP32 缓存是否更新 | 否 |
| 使用缓存的 logits | `[1,3]` |
| 使用新参数的 logits | `[8,8]` |

这是小张量故障复现，不是大模型 NPU 误差测量；结合当前原生缓存的创建与消费代码，足以确认发布代码存在遗漏。它也会污染“更新参数后，用 no_grad 原生分支作参考”的测试；参考前向必须先保证缓存新鲜。

修复应只作用于 Ascend backend：在提交事务内同步刷新派生缓存，保留缓存对象与地址；对回滚、恢复及 park/wake 路径一并覆盖。不能重新执行整个模型的通用 postprocess，以免重复转换专家布局。验收必须读取更新后的 cache 值，并与冷加载同一新权重得到的 gate logits 对照。

代码位置：[loader 参数集合与提交](../verl_speco/integration/deepseek_v4_online.py)、[提交后 metadata 刷新](../verl_speco/integration/vllm_runtime.py)、[checkpoint probe](../verl_speco/integration/deepseek_v4_update_probe.py)。

### B — P1：MoE 缩放和 BF16 转换顺序与原生不同

当前原生 DeepSeek V4 factory 没有设置 `apply_routed_scale_to_output=True`，其默认值为 False：scale 在 router 权重中；`AscendRoutedExperts._select_experts` 再把权重转换为 hidden dtype；runner 的额外输出 scale 为 1。

当前训练代码先将 native routing weights 除以 router scale，转换为 BF16 后参与专家合并，最后对专家输出乘 `model.routed_scaling_factor=1.5`。实数运算下可等价，BF16 下不是相同的舍入路径。

CPU 示例固定专家输出，仅改变这项顺序：2048 个输出中有 1227 个不同，最大绝对差 `0.015625`。**该数值仅说明舍入顺序确实能产生差异，不能作为当前完整 NPU 模型的误差统计。**

现有 MoE CPU reference 也采用“先去 scale、最后乘 scale”的形式，能够检查该实现的内部导数，却无法排除它与原生 runner 之间的差异。需新增独立的原生基准，而不是继续用同一计算顺序作 reference。

已进一步核对 sqrtsoftplus 自定义算子：虽然 Python 传 `renorm=0`，当前 C++ `moe_gating_top_k_hash_without_group.h` 第 253–270 行仍对 sqrtsoftplus 选中分数求和归一化并乘 scale。因此**不能**把 `renorm=0` 误报为当前目标没有归一化；当前 `norm_topk_prob=True` 与这段内核语义相符。稳定性和其他 scoring 分支仍不能外推。内核使用 `sum + eps`，replay 使用 `clamp_min(eps)`；当前普通分数范围下 eps 极小，但极端输入下两者也不能直接视为相同。

优先考虑严格沿用原生 routing weights 的 dtype/scale 位置与融合合并顺序，再接显式反向；若用 reference/replay 反向，应明确其边界并验证导数，而非宣称是原生反向。

### C — P1：训练默认替换了两处 TP 归约

`_DeterministicTPRowReduce` 使用 all-gather、固定 rank 顺序的 FP32 累加、最后转换回原 dtype。wo_b 和 split MoE 都会使用它；原生 serving 使用自己的 TP/EP collective 与 finalize 路径。

这是一项真实的执行路径差异，不只是“加了梯度”。目前没有同权重、同输入的 TP8 NPU 数值对照，不能量化或直接认定其影响可以接受。应先比较 collective 前的各 rank 局部输出，再比较归约后的结果；不要用最终 Target 输出不变替代这项检查。

### D — P1（当前闭环阻塞）：窗口裁剪把绝对位置变成越界索引

故障在诊断入口 `collect_window`：token 序列被裁至 512 行，但 `hidden_positions` 仍是原序列绝对索引。正式收集接口的边界检查正确，不应放宽。

已用本次 `online_rounds_20261109_221038/serving_0` 两份真实样本在 CPU 上复验：旧函数两个 DP 均报同一越界；候选函数保留全 token 序列和 prompt 边界，只选取尾部特征，两个 DP 均通过，得到 512 行、位置 `[1433,1944]`、63 个监督位置，token/position/hidden/mask 与原始切片逐项一致。最后一个生成 token 没有对应前向行，不能额外伪造其特征。

候选仅修改测试入口，保留绝对 RoPE 位置；不要简单重编号为 0..511 来绕过错误。短序列、窗口跨 prompt/response、窗口全 response 三种 CPU 用例也已覆盖。候选尚未部署。

### E — P1：TP 复制参数尚未做到逐位一致

step-1 checkpoint 的 Markov W1 在 rank 2/5/10/13 的同一元素上与 TP0 相差 1 个 BF16 ULP，最大绝对差 `9.5367431640625e-7`，共 33095680 个元素。16 个 rank 均已核对。

严格导出因此失败；后续 PyTorch 导出使用仓库已有的显式、单元素单 ULP 选项，选择 TP0，不平均、不改原 checkpoint。导出报告已标 `strict_replica_parity=false`。不能把该结果描述为严格复制一致性通过，也不应扩大默认容差。

应检查 W1 梯度归约、embedding 梯度累积及 Adam 状态一致性。仅在 step 后广播参数而忽略 Adam 状态，不能视为完整解决。来源究竟是哪一步，当前证据不足以确定。

### F — P1（可选功能）：confidence 配置检查与发布对象不一致

模型工厂读取 `cfg.train.dspark_enable_confidence_head`，online mixin 却检查 `cfg.model` 下的同名字段。若用户只开启 train 字段，预期的 online 拒绝检查可能失效。

此外，可选训练头是 wrapper 新建的 `nn.Linear`（默认带 bias），原生 0.27 confidence projection 不带 bias；online runtime-shard 发布遍历 inner serving model 的参数，未发布 wrapper 训练头。该可选组合不能认定兼容。

当前 smoke 配置为关闭，因此不是本次默认前向的直接故障。应先纠正配置 guard，对未适配组合明确拒绝；后续若支持，应复用原生头并验证加载/训练/导出/发布全链路。

## 4. 初始验收范围与证据边界

| 已有结果 | 可以证明 | 不能证明 |
| --- | --- | --- |
| 单步 NPU 训练、梯度审计、update probe | 计算可执行，受审查参数有有限梯度，参数确实更新 | 梯度等于原生数学导数；所有中间前向与原生一致 |
| 从 step 1 恢复到 step 2 | 当前配置可恢复并继续更新 | 长期数值稳定性、完整 RL 联动 |
| SwiGLU 四个 NPU 用例 | 已测 BF16/FP32、有/无裁剪的激活前向及梯度对照通过 | 整个 Shared MLP、MoE 或 DSpark 对齐 |
| checkpoint / IPC 参数提交与 digest | 参数张量正确复制，专家转换和版本提交工作 | 未列为参数的派生缓存正确，已使用新 gate cache |
| 更新前后 Target greedy token 相同 | 被测请求的 Target 输出未改变 | 草稿 logits 新鲜、与训练一致、草稿质量或接受率正确 |
| canonical PyTorch 导出回读 | 2375 个张量、19 个 shard 与选定 merge 值一致 | 从此 artifact 原生加载并推理通过；runtime-shard 热更新不是同一入口 |
| 两轮闭环 | 首次 rollout 采集已通过 | 两轮训练/发布通过；实际结果是 0 个训练轮次完成 |

“Target 输出不变”尤其不能作为草稿前向对齐的主证据：投机解码本就应在不同草稿质量下保持 Target 的正确输出。

## 5. 初始审查新增核验与遗留测试异常

- 按当前源码镜像运行特征契约、HC/DSA 显式反向、MoE、导出、artifact、冻结策略、backend 隔离及候选窗口测试：本地 CPU **187 passed**。
- 用实际发布 loader 复现 gate-cache 不刷新的问题，并检查真实 checkpoint 中三层 gate 的 BF16 类型。
- 用真实 DP0/DP1 rollout 数据复现旧窗口错误并验证候选修复；服务器证据 `online_window_review_20261109.json`。
- 已有服务器服务类导入、NPU 服务提交测试通过；但独立的兼容层模拟测试曾在 16 个断言通过后以 native allocator 错误退出（code -6）。该异常仍未定位，不能报告为干净通过；真实服务类导入进程正常退出。保留 `server_probe_compat_isolated.log` 等记录，后续应按具体模拟用例隔离。

测试数量只说明列出的范围，不代表完整验收。

## 6. 初始建议的修复与验收顺序（进展见前文）

1. **先修 P0 缓存事务**：参数、FP32 gate cache、版本号及回滚必须一致；加入更新后和恢复后的 cache/logits 检查，只影响 Ascend backend。
2. **消除或明确量化前向差异**：以当前原生 MoERunner 为准，核对 route weights 转型和 scale 位置；明确 wo_b/MoE 的 collective。尽量保留原生前向，仅提供所需反向。
3. **补齐一次有定位能力的 NPU 对照**：同权重、同 tokens、同绝对 positions、同 context-KV/metadata、同 previous_token_ids，逐段记录 MainProj、HC、q/kv、attention、router logits/IDs/weights、routed/shared output、head/logits。先定位差异来源，再确定每段精度容限；不以放宽全局 allclose 获得通过。
4. **独立验证梯度**：小尺寸 FP32 reference 与 NPU wrapper 对照；路由检查固定 expert IDs，避开 top-k 切换边界。BF16 finite difference 不能作为唯一导数基准。
5. **验收真实发布后生效**：更新 live model 后，与冷加载同一更新权重的 runtime 比较草稿 logits、路由和 cache；覆盖 commit、rollback、恢复、park/wake，检查地址稳定。
6. **再部署窗口修复并跑两轮闭环**：先复用已采集数据通过 CPU 入口检查，再做采集→训练→发布；确认相同 engine/Adam/scheduler 身份、step 1→2、revision 0→1→2。
7. 最后验证 actor-source NPU 特征、真实 actor Target 版本变化和完整 RL 自动调度。当前独立 probe 的 Target 版本始终为 0、推理服务在轮次间重启，不能冒充完整 RL 验证。

改动继续留在现有 backend/ops/integration 布局中；保留旧版分支和原生 PyTorch backend。数值实现若因修复改变指纹，旧 checkpoint 的 optimizer resume 不能通过放宽指纹检查直接宣布兼容，需单独建立迁移/恢复证据。

## 7. 初始决策（已被前文后续验证更新）

初始审查时决定：**暂不安排新的两轮 NPU 重跑，也不进入完整 RL。**先按 A/B/C 修正并建立有独立原生基准的前向/缓存对照，再部署已在真实数据上验证的 D。当前不存在“仅凭已通过冒烟即可认定前向完全一致”的依据。

## 附录：CPU 复现 gate 缓存遗漏

在仓库环境中执行以下代码，只创建 CPU 小张量。当前未修复 loader 输出 False；修复后的验收应为 True。原生缓存的创建/消费点还需结合本报告 A 中的实际运行时源码核对，不能仅用此小模型外推完整 NPU 误差。

```python
import torch
from torch import nn
from verl_speco.integration.deepseek_v4_online import (
    EmbeddedDraftLoader, runtime_weight_name,
)

model = nn.Module()
model.model = nn.Module()
model.model.gate = nn.Linear(2, 2, bias=False, device="cpu", dtype=torch.bfloat16)
model.model.gate.weight_fp32 = model.model.gate.weight.detach().float().clone()
loader = EmbeddedDraftLoader(model)
loader.receive([(
    runtime_weight_name(0, 1, "model.gate.weight"),
    torch.full_like(model.model.gate.weight, 2),
)])
loader.finish()
print(torch.equal(model.model.gate.weight_fp32, model.model.gate.weight.float()))
```


## 本轮产物整理

成功重跑后清理本轮两个失败尝试的冗余 `snapshots` 目录：`online_gmm_fixed_20261110/snapshots` 和 `online_release_fixed_20261110/snapshots`。保留原始故障日志、特征、各 rank 报告和小型源码快照；最新通过的 `online_shutdown_fixed_20261110/snapshots/revision_1`、`revision_2` 保留供后续发布/对照使用。合计清理 123000657480 bytes（约 114.6 GiB）；清理清单见 `validation_snapshot_cleanup_20261110.json`。原始 target 权重、轨迹和既有用户 checkpoint 不在本轮清理范围。
