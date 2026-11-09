# Bubble Time 与 V1

Last updated: 2026-11-04

适配基线为上游 `release/v0.9.0`。PPO 主循环仍由上游 V1 trainer 执行；
本仓库仅提供sync 生命周期 hook、数据采集和发布协议。
V1 rollout 后端仍限于 vLLM。Bubble 仅支持 `trainer_mode=sync`；
`colocate_async`、`separate_async` 与 Bubble 同时启用时会在 worker 创建前报错。
关闭 Bubble 后，三种 V1 模式沿用原有逻辑。

## 配置

V1 sync 模式在采样结束、Actor/PPO 前以及补训后的资源归还边界强制将 drafter
训练状态卸载到 CPU，即使 `keep_hot_between_plans=true`。该开关只允许在资源
仍归训练所有的窗口内保留驻留状态；不能跨 PPO 或 serving 边界占用共享 GPU。
卸载不清空 optimizer、replay、quota 或固定 writer；严格卸载失败会停止边界推进。

CUDA Bubble 的已完成发布快照可使用 NumPy/Ray 无损传输；不支持的布局回退 Torch。
仅改变 CPU 序列化路径，不改变 quota、版本或 stage/commit ACK；Bubble 仍立即执行
原有 host 回收。与其他搬运优化共用 `gpu_transfer_optimizations` 开关。

定位首次训练与 idle 拒绝原因时可设置以下环境变量（默认均关闭，传入 Ray worker）：

```bash
export VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS=true
export VERL_SPECO_DIAGNOSTIC_IDLE_REJECTIONS=true
```

第一个开关只对 CUDA Bubble worker 最初两个 optimizer steps 的补训阶段增加
同步计时，并记录编译缓存信息与计数变化。日志的 `timing_mode=device_synchronized`
表示诊断运行，会引入同步开销，不能直接作为性能对照。idle 训练不增加同步。
第二个开关每 RL step、每拒绝原因至多输出一次 `idle_rejection`，包含缓存中的
组成员状态、显存释放状态、租约、匹配 quota 的 head ACK、deadline 与窗口预算。
缺失状态标记 unknown，不查询 worker，也不修改调度或租约。

Bubble 的 CUDA 搬运路径复用 `gpu_transfer_optimizations`（默认 true）：
FSDP2 模型批量恢复、模型和 optimizer 的 CPU 缓冲池，以及不可变 target head
的 CPU 存储复用。操作仍在原有训练激活与回收边界内执行；回收完成前等待拷贝，
不保留 GPU 缓冲。head 版本变化或内容变化会使缓存失效，不提前准备 head。
NPU 路径不变。设 `actor_rollout_ref.rollout.drafter.training.gpu_transfer_optimizations=false`
可恢复原有搬运路径。

这主要减少初始化后的恢复与首次训练结束时的卸载开销；CPU 缓冲首次仍需分配，
不会消除模型构建、编译、Adam 状态初始化或 Bubble 的 quota/发布成本，因此不能
直接保证首次训练达到普通 Sync 的相同耗时。对照首次完成完整 quota 的边界，结合
`drafter/gpu_head_restore_cache_hit`、`drafter/gpu_model_load_used_batched_load`、
`drafter/gpu_optimizer_offload_used_pool` 和原有阶段耗时判断是否启用及收益。

V1 已移除 step-end 的提前准备 head 路径。训练 preflight 仍选择并应用 quota
绑定的 head；保留控制步同步、版本检查、quota、writer、资源租约及发布规则。
`v1_head_preparation_enable` 仅为兼容旧启动命令而保留，默认 false；设置
为 true 将在 V1 Bubble 校验阶段报错；请移除旧 override 或设为 false。legacy trainer 的原有路径不受此调整影响。

默认关闭边界诊断：

```text
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.boundary_diagnostics_enable=false
```

必要时设置 true，记录回收与 Actor 边界。回收开始日志在关闭训练启动入口、
停止事件线程之后输出，避免日志等待期间启动新训练。结合已有 worker
`training_cleanup_boundary` 的 plan/rank 时间检查顺序；这些日志不是 GPU
硬件同步或带宽无竞争的证明。

Bubble V1 在创建 rollout worker 前输出一次 `[BubbleTime] v1_runtime:` JSON，
包括驱动实际模块的绝对路径及 SHA256、Git 提交号、运行模式、预热请求值、
预热实际值（固定 false）和诊断开关。未安装 Git 时提交号为 unavailable，
源码指纹仍会输出。记录使用标准输出，不依赖模块日志过滤；它只标识驱动源码，
不代替各远程 worker 的部署一致性检查。修改过但未提交的源码应以 SHA256 为准。

对照实验应从同一有效恢复点分别启动，先跑 20 步并比较实际 token、训练总步数、
发布次数、剩余 quota、接收长度与端到端耗时。保持 GPU、数据、窗口阈值、
训练额度、学习率和 hard sample 配置一致。修改文件不会更新已运行进程中的代码。

使用 `--config-name=speco_v1_trainer`，沿用以下配置：

```text
trainer.v1.trainer_mode=sync
actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=rollout_idle_worker
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable=true
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.allow_critical_path_fallback=false
```

Bubble 的 `sync` 当前要求模式配置中的
`parameter_sync_step=1`。上游 sync在每个 local batch 后更新 Actor，
因此多个 local update 的特征不能共用一个 global-step head/quota 版本。
在独立 policy snapshot 标识接入前，启动时拒绝大于 1 的配置，避免静默混用 head。
普通非 Bubble 训练不受此限制；不通过额外恢复 Actor 权重来改变 PPO 行为。

启用 ROI 时，设置
`actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze` 下的
`enabled=true`、`economics.enabled=true` 和 `probe.enabled=true`。
`mode=shadow` 只观察，`mode=active` 才实际冻结。原有窗口、覆盖率和漂移恢复阈值继续生效。

### sync 的冻结指标与 hard sample

Bubble V1 sync 复用请求级 acceptance 数据、hard sample 选样和冻结策略。
hard sample 使用 training 下的 `dspark_hard_candidate_ratio` 与
`dspark_hard_sample_ratio`，两者均大于零才启用；默认均为零。
采集计划沿用 `dspark_collect_samples_per_step` 和 `dspark_hard_shortfall_policy`，
不修改 PPO 的完整 batch。合成 padding 行不作为请求或 hard sample 证据。

请求 acceptance 分布在每个 local batch 的 old-logprob 前记录，在 Bubble
step-end 边界落盘；冻结关闭特征采集后仍记录，供分布变化排查使用。
路径通过 `VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH` 设置，默认
`/tmp/speco_vllm_request_accept_len_hist.jsonl`。选样指标使用
`drafter/selection/*`，冻结状态沿用 `drafter_convergence_freeze` 的指标。

### 计时与 acceptance 指标

`timing_s/drafter` 统计 Bubble 在 sample-end、actor 前后和发布边界增加的
driver 耗时；不包含 PPO actor 更新、上游 actor target 同步、联合 checkpoint
和已单独统计的 old-logprob 采集。后台训练工作量仍使用
`timing_s/drafter_async_training_work`，不可与重叠的关键路径时间相加。
`timing_s/actor_update_without_drafter` 和 `timing_s/drafter_v1_*` 提供分段时间，
`timing_s/oldlogprob_*` 提供准备、计算、采集和恢复耗时。
同一 global step 内多个 local batch 的 Bubble 时间累加后，在最终指标阶段输出。

sync 在每轮采样和事件处理前更新 drafter worker 的控制 RL step，
即使该轮因已有 quota 而跳过特征采集，也不能让 worker 停留在上次采集的 step。
控制 step 与 quota 的数据/head 版本独立：例如 step 11 补齐 step 8 的 quota，
计划的执行 step 为 11，训练数据和 head 仍固定为版本 8。
预检失败保留 `worker_preflight_failed`，日志包含 worker 原因及两侧 step，
不会将未执行的预检失败误报为训练结果不一致。

acceptlen 的日志名为 `drafter/spec_decode/mean_acceptance_length`，计算为
`1 + accepted_tokens / verify_steps`。优先读取原生多轮请求计数，缺失时读取
SpeCo 精确请求计数；engine 汇总 fallback 不冒充单个请求数据。
`drafter/spec_decode/num_verify_steps`、`num_accepted_tokens` 和
`request_stats_coverage` 等指标暴露计算分母与请求覆盖率。
请求数据不足且没有启用可用 sidecar 时保留覆盖率，不伪造 acceptlen 数值。

## 边界

| 模式 | Bubble 可运行阶段 | PPO/同步边界 |
| --- | --- | --- |
| sync | rollout 的已确认空闲窗口 | sample end 停线程、回收训练组，再进入上游 sleep/old-logprob |

old-logprob 特征继续使用每份采集计划独立、重试稳定的样本 ID。
sync 在 old-logprob 采集后完成不可变 head 导出。head 的 CPU 分发、版本 ACK 和 quota 保护继续使用共享协议。

ROI 事务顺序为：关闭普通请求 → 固定 actor target → before-arm →
drafter 发布并等待完整 ACK → 更新 serving revision → after-arm → 恢复请求。
两臂直接调用 serving RPC，不进入 PPO、reward 或 TransferQueue。
探测缺少 revision、混合 revision、执行失败或覆盖不足时，不产生冻结证据。

成本跨空闲窗口保留，按计划 ID 去重。组内准备和训练阶段取最大 rank 的本地耗时。
driver 上的重叠区间取并集，再加上 worker 阶段耗时。跨进程重叠采用保守计费，
因此重叠发生时该估算可能偏高。
日志分别提供更新成本、probe 耗时、暂停时间和 step-end 关键路径耗时。

未完成 quota 或未确认发布会延迟 checkpoint。发布和 ROI 状态提交后，
在关闭 admission 的边界使用实际 RL step 保存联合恢复点。
最后一步仍有 quota 且未允许关键路径补齐时明确失败，不保存不完整结果。

联合保存要求 actor 和已启用 critic 的 `checkpoint.async_save=false`，
确保模型保存 RPC 返回时组件已完成。这与 V1 的异步 rollout 模式相互独立。
上游 driver 的 latest 写入及提前清理在保存事务内暂缓；worker 仍按原配置同步保存。
模型、冻结状态、TQ、Feature Store 和联合 manifest 均完成后，才原子提交 latest，
随后各 owning worker 结合 driver 确认的步号和本机完整 manifest 执行 actor/critic
保留策略。共享目录中，同一个组件通过目录声明串行清理。
组件保存或 latest 提交失败时，已发布的旧指针和旧模型 checkpoint 均保留。
在同一输出目录续训时，下一次保存的 step 必须大于 latest 记录的 step。
如果 latest 缺失，则以该目录中最高的已提交 Bubble 联合恢复点为保护边界。
保存开始前会写入 pending 标记。latest 提交前失败的保存可以在同一步重试，
其 manifest 不会被用于续训或占用保留名额。旧版没有标记的完整 manifest 仍兼容。
从更早的 checkpoint 回退并重新保存时，请使用新的 `trainer.default_local_dir`。

generation 启动事件使用注册的真实 replica 分组，不再根据 rollout DP 数量推断 rank。
head 传输成本在同步返回或预取轮询收到完整版本 ACK 后统一记账；
同一传输重复完成不重复计费，同一版本再次分发则按新增传输计费。

## CPU 验证

专用 CI job `V1 Bubble contracts (verl 0.9)` 安装真实上游 V1 和 TransferQueue，
设置 `SPECO_REQUIRE_V1=1`。缺少 V1 导入依赖将直接失败，不能整组跳过。

初次适配在干净 Python 3.12 / CPU Torch 环境验证：真实 0.9 V1 测试 99 项通过；
legacy 基线全量 unit/integration/V1 回归 2539 项通过、16 项因可选 GPU/后端依赖跳过。
2026-10-23 的事件映射、联合提交与 head ACK 计费修复，使用真实 0.9 V1
及事件总线集成测试验证，共 168 项通过。

```bash
PYTHONPATH="$PWD:$UPSTREAM_VERL" SPECO_REQUIRE_V1=1 \
  python -m pytest tests/v1 -q -ra

PYTHONPATH="$PWD:$LEGACY_VERL" python -m pytest \
  tests/unit/test_bubble* tests/unit/test_rollout_idle_worker_scheduler.py \
  tests/unit/test_idle_window_readmission.py -q -ra
```

## V1 sync 性能实测

对比 `train_sync_freeze.log` 和 `train_sync_freeze_bubble.log` 的共同区间
step 1–95。两组均使用 V1 sync trainer，Bubble 组启用了 `rollout_idle_worker`。
数据来源见 [PR #136 的性能对照](https://github.com/verl-project/verl-SpeCo/pull/136)。

| 指标 | Sync | Bubble | 变化 |
| --- | ---: | ---: | ---: |
| `timing_s/step` 累计 | 45,136.76 秒 | 43,599.30 秒 | −3.41% |
| `timing_s/drafter` 累计 | 303.75 秒 | 194.36 秒 | −36.01% |
| 完成的 drafter optimizer steps | 460 | 460 | 持平 |
| 在 idle 窗口完成的 optimizer steps | 0 | 225 | 占 Bubble 组的 48.9% |
| 平均 acceptance length | 3.451 | 3.733 | +8.17% |

Bubble 组其余 235 步通过同步补训完成。`timing_s/drafter` 包含发布边界阻塞，
`timing_s/step` 汇总 step 计时器内的耗时。两组为单次运行，生成轨迹存在差异，
表中变化反映本次负载下的观测结果。两组均使用 `early_publish_staging=false`。


## 排查采样阶段性能差异

同卡、同代码的普通训练与 Bubble 对照均设置：

```bash
export VERL_SPECO_PHASE_TIMING=true
export VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF=true
export VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS=false
```

`rollout_perf_snapshot` 记录 drafter pre-fit 激活前、激活后、最终卸载后，以及
vLLM 首次 generate 前的元数据：进程 RSS/锁页/线程数、CPU affinity、PyTorch
线程数、GPU UUID、进程 allocator 与整卡可用显存、训练组和驻留状态。vLLM
worker 另记录可访问的 cache/scheduler 配置和 KV block 数；缺少字段明确列入
unavailable，不猜测数值。首次采样前每 replica 仅进行一次额外 worker 快照 RPC，
耗时单独记录为 `rollout_perf_snapshot_rpc`，不要把它误认为稳定 decode 开销。

`rollout_perf_execution` 只汇总每 replica 前三个请求执行 episode：提交跨度、
最大提交间隔、前端执行并发峰值/时间加权平均、执行墙钟和并行请求耗时之和。
前端并发不是 engine scheduler 的 running/waiting 数，日志明确标记后者不可用。
已有 `replica_episode` 增加全部有效请求的 decode/prefill/queue/TTFT 累计和最大值，
避免只看最慢单请求；累计并行耗时不能当作端到端耗时。

以上诊断默认关闭，不增加 token 级日志，不同步 GPU、不读取张量内容、不初始化
尚未初始化的 CUDA。快照失败只记录失败类型，不改变训练、租约或发布流程。
先跑三步比较第 2～3 步，统一 GPU、seed、参数和诊断开关，保存完整启动日志。

### V1 发布快照预取

`training.gpu_publish_cast_before_copy` 默认开启，并受
`training.gpu_transfer_optimizations` 控制。CUDA FSDP2 使用该开关在复制前转换
dtype，Bubble FSDP1 使用它合并连续 CUDA 张量的非阻塞主机复制。
`training.gpu_publish_fsdp1_copy_budget_mib` 默认 256 MiB，限制每个 leader
的单批在途复制量。设为 0 使用逐张量同步复制，超过预算的单个张量也同步复制。

一轮 Bubble FSDP1 运行中，step 10 的 leader 导出了约 2.25 GB 权重，快照生成
约 0.078 秒，其中复制等待约 0.044 秒。该数字是单次热态观测。

可通过以下配置启用发布准备与原生 target 权重同步的重叠：

```bash
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.early_publish_staging=true
```

该路径同时要求 `publish_async=true`。V1 在关闭生成准入、quota 已完成且只有一个
当前 step 的待发布计划时，在 target 同步前提交 `maybe_publish_async`。worker 只
消费已有的 CPU 快照并序列化到 Ray object store，不触发 GPU 导出或应用 rollout
权重。同步后的正常发布流程消费同一份 ObjectRef，避免再次获取、打包同一快照。
缓存未命中时，仍在同步后的原安全点执行导出；版本不匹配直接报错。

预取前复用调度器的正式发布计划检查；发布间隔未到时不消费 worker 快照。
预取结果绑定 source step 和 plan ID，防止同一步内误复用另一计划的权重。

权重应用、freeze probe 和发布 ACK 的顺序保持不变，快照准备不会越过未完成的
训练 quota。没有收益保证：CPU/内存带宽竞争也可能影响 target 同步，需要完整
训练对照验证。关注 `bubble/publish_snapshot_prefetch_submitted`、
`bubble/publish_snapshot_prefetch_consumed` 和
`timing_s/drafter_publish_snapshot_prefetch_wait`，并比较整步耗时及 rollout 耗时。

### 日志与配置精简

常规运行不再逐次打印 idle 决策、head 未就绪拒绝和快照预取 worker 的成功日志；
这些细节降为 debug，训练计划启动使用 info，未启动计划使用 debug。
OOM、训练失败、发布 ACK/版本错误仍保留。`memory_park_breakdown` 只在启用
`VERL_SPECO_PHASE_TIMING` 或 `VERL_DRAFTER_MEMORY_DIAGNOSTICS` 时输出，
阶段计时指标始终保留。诊断时仍可启用按 step/reason 去重的 idle 拒绝汇总。

Bubble 示例省略 `writer_migration_enable=false`、`full_collective_fallback=false`、
`head_reference_transport=true` 和 `calibrated_budgets=true`，沿用配置默认值。
`v1_head_preparation_enable` 虽然已无作用，仍作为旧命令的兼容键保留。


## 边界与计费维护约定

V1 session 持有共享 Bubble 运行状态，trainer 的历史属性名仅作兼容转发；
legacy owner 不迁移存储。锁顺序为训练锁 → session 锁，停止并 join 事件线程时
不持训练锁。已成功 park 的 driver receipt 可跳过重复 park RPC；新 generation、
训练完成回调与激活会使 receipt 失效。失败的 park 不会生成成功 receipt。

ROI 使用固定 epoch 加单调时钟的 CostClock。每个进程使用自己的时钟计算本地耗时，
V1 driver 只合并本进程的区间。head dispatch 的 perf_counter 起点用 driver 的
固定 epoch 转换。运行时 deadline 沿用原有调度时钟。

成本 identity 表示一次完成事件，同 identity 的重复回调采用 first-completion-wins；
真正执行额外工作的重试需要不同 attempt identity。head 传输使用版本与 dispatch
origin 区分不同传输；已完成事件的迟到重复回调不追加成本。driver 内不同 identity
的时间区间取并集，避免本地重叠重复计费。

延迟 checkpoint 后直接调用 checkpoint manager 重新应用已确认 target，不再二次
调用上游 on_step_end。sleep 会释放权重/KV，不能只解除 generation 的暂停。
retention 删除失败记录 warning，并分别返回 pruned/failed，避免假报清理成功。

V1 CPU CI 固定上游 commit 91c297dc8d9991b79c90cad5b50683a5d6d8c02b，
缓存命中后仍检查 checkout SHA。AST 方法提取测试使用导入守卫；未声明的动态
导入失败，新增依赖必须在被提取源码中显式声明。


延迟 checkpoint 保存后的 rollout 权重/KV 恢复单独记录为
`timing_s/checkpoint_rollout_restore`，从 `timing_s/drafter_v1_publish_blocking`
及 `timing_s/drafter` 中排除。它仍包含在整步与 step-end 总耗时中；该指标表示
保存后恢复 serving，不表示从磁盘恢复训练。此调整只修正计时归类，不改变同步
调用次数、顺序或实际端到端耗时。
