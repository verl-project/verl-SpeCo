# Bubble Time 与 V1

Last updated: 2026-10-27

适配基线为上游 `release/v0.9.0`。PPO 主循环仍由上游 V1 trainer 执行；
本仓库仅提供生命周期 hook、独立池 worker、数据采集和发布协议。
V1 rollout 后端仍限于 vLLM。

## 配置

V1 已移除 step-end 的提前准备 head 路径。训练 preflight 仍选择并应用 quota
绑定的 head；保留控制步同步、版本检查、quota、writer、资源租约及发布规则。
`v1_head_preparation_enable` 仅为兼容旧启动命令而保留，默认 false；即使设置
为 true，也不会启用提前准备。legacy trainer 的原有路径不受此调整影响。

默认关闭边界诊断：

```text
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.boundary_diagnostics_enable=false
```

必要时设置 true，记录回收与 Actor 边界。回收开始日志在关闭训练启动入口、
停止事件线程之后输出，避免日志等待期间启动新训练。结合已有 worker
`training_cleanup_boundary` 的 plan/rank 时间检查顺序；这些日志不是 GPU
硬件同步或带宽无竞争的证明。separate_async 的独立池训练与 Actor 可以合法重叠。

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
trainer.v1.trainer_mode=sync|colocate_async|separate_async
actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=rollout_idle_worker
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable=true
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.allow_critical_path_fallback=false
```

Bubble 的 `sync` 和 `colocate_async` 当前要求各自模式配置中的
`parameter_sync_step=1`。上游这两种模式在每个 local batch 后更新 Actor，
因此多个 local update 的特征不能共用一个 global-step head/quota 版本。
在独立 policy snapshot 标识接入前，启动时拒绝大于 1 的配置，避免静默混用 head。
`separate_async` 使用上游固定旧策略快照，可继续配置多个 local update。
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
冻结状态、ROI 和漂移恢复仍需按下文完成 GPU 验收。

### 计时与 acceptance 指标

`timing_s/drafter` 统计 Bubble 在 sample-end、actor 前后和发布边界增加的
driver 耗时；不包含 PPO actor 更新、上游 actor target 同步、联合 checkpoint
和已单独统计的 old-logprob 采集。后台训练工作量仍使用
`timing_s/drafter_async_training_work`，不可与重叠的关键路径时间相加。
`timing_s/actor_update_without_drafter` 和 `timing_s/drafter_v1_*` 提供分段时间，
`timing_s/oldlogprob_*` 提供准备、计算、采集和恢复耗时。
同一 global step 内多个 local batch 的 Bubble 时间累加后，在最终指标阶段输出。

sync/colocate_async 在每轮采样和事件处理前更新 drafter worker 的控制 RL step，
即使该轮因已有 quota 而跳过特征采集，也不能让 worker 停留在上次采集的 step。
控制 step 与 quota 的数据/head 版本独立：例如 step 11 补齐 step 8 的 quota，
计划的执行 step 为 11，训练数据和 head 仍固定为版本 8。
separate_async 不采用这个全组同步，以免干扰跨步运行的后台训练。
预检失败保留 `worker_preflight_failed`，日志包含 worker 原因及两侧 step，
不会将未执行的预检失败误报为训练结果不一致。

acceptlen 的日志名为 `drafter/spec_decode/mean_acceptance_length`，计算为
`1 + accepted_tokens / verify_steps`。优先读取原生多轮请求计数，缺失时读取
SpeCo 精确请求计数；engine 汇总 fallback 不冒充单个请求数据。
`drafter/spec_decode/num_verify_steps`、`num_accepted_tokens` 和
`request_stats_coverage` 等指标暴露计算分母与请求覆盖率。
请求数据不足且没有启用可用 sidecar 时保留覆盖率，不伪造 acceptlen 数值。

`separate_async` 固定将 drafter 训练 worker 放到独立 rollout placement groups，
不新增部署模式。分配 GPU 时复用已有资源池，不申请额外 GPU。
worker 在训练 RPC 返回、资源租约释放之前完成 offload；optimizer writer 仍保留。

## 边界

| 模式 | Bubble 可运行阶段 | PPO/同步边界 |
| --- | --- | --- |
| sync | rollout 的已确认空闲窗口 | sample end 停线程、回收训练组，再进入上游 sleep/old-logprob |
| colocate_async | 预取期间的已确认空闲窗口 | 先关闭 admission，再回收、abort/sleep；发布完成后恢复 |
| separate_async | 独立 rollout 池的已确认空闲窗口，可与 actor 更新重叠 | target 同步前回收冲突训练资源；有发布、探测、保存时进入受控暂停 |

独立池的训练 rank 与 rollout replica rank 分开处理。初始化根据 node、实际设备和
collective 元数据校验归属。每个独立 replica 的发布 ACK 使用独立命名空间，
避免不同 replica 的本地 rank 0 相互覆盖。休眠 hybrid replica 重新加入 balancer 前，
先补齐已确认的 drafter 权重及 serving revision。

old-logprob 特征继续使用每份采集计划独立、重试稳定的样本 ID。
separate_async 在上游旧策略 CPU-slot 恢复上下文内完成不可变 head 导出，
随后才恢复当前 actor。head 的 CPU 分发、版本 ACK 和 quota 保护继续使用共享协议。

ROI 事务顺序为：关闭普通请求 → 固定 actor target → before-arm →
drafter 发布并等待完整 ACK → 更新 serving revision → after-arm → 恢复请求。
两臂直接调用 serving RPC，不进入 PPO、reward 或 TransferQueue。
探测缺少 revision、混合 revision、执行失败或覆盖不足时，不产生冻结证据。

成本跨空闲窗口保留，按计划 ID 去重。组内准备和训练阶段取最大 rank 耗时，
有时间区间的阶段计算区间并集，避免 head、训练及传输重叠重复计费。
worker 时间区间使用集群 wall clock；验收机器应保持时钟同步。
日志分别提供更新成本、probe 耗时、暂停时间和 step-end 关键路径耗时。

未完成 quota 或未确认发布会延迟 checkpoint。发布和 ROI 状态提交后，
在关闭 admission 的边界使用实际 RL step 保存联合恢复点。
异步 Bubble 保存要求 TransferQueue 支持 snapshot/reissue；不支持时明确报错。
最后一步仍有 quota 且未允许关键路径补齐时明确失败，不保存不完整结果。

联合保存要求 actor 和已启用 critic 的 `checkpoint.async_save=false`，
确保模型保存 RPC 返回时组件已完成。这与 V1 的异步 rollout 模式相互独立。
上游 driver 的 latest 写入及提前清理在保存事务内暂缓；worker 仍按原配置同步保存。
模型、冻结状态、TQ、Feature Store 和联合 manifest 均完成后，才原子提交 latest，
随后在各 owning worker 上执行原有 actor/critic 保留数量策略。
组件保存或 latest 提交失败时，旧指针和旧模型 checkpoint 均保留，允许重试。

generation 启动事件使用注册的真实 replica 分组，不再根据 rollout DP 数量推断 rank。
separate_async 的事件总线只管理独立池，hybrid 请求与同号训练 rank 的租约相互隔离。
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

## GPU 验收记录

CPU 合约通过不代表 GPU 功能验收完成。当前开发环境无可用 CUDA GPU，以下项目仍须实测。

| 负载 | sync | colocate_async | separate_async |
| --- | --- | --- | --- |
| DSpark：Bubble / 普通 drafter 等工作量对照 | 待测 | 待测 | 待测 |
| DFlash：Bubble / 普通 drafter 等工作量对照 | 待测 | 待测 | 待测 |
| ROI 探测、冻结、漂移恢复、checkpoint 恢复 | 待测 | 待测 | 待测 |

每组必须记录 optimizer 工作量、发布次数、未偿 quota、probe 暂停和端到端耗时，
并核对共享 GPU 无越界重叠、完整 quota 只执行一次、发布和恢复版本一致。
separate_async 还须覆盖跨节点 CPU/Ray 传输及 hybrid replica 重新入池。
减少训练或遗留 quota 的结果不能作为等工作量加速证据。
