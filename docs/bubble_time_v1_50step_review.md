# V1 sync Bubble：50 步日志复盘

Last updated: 2026-10-26

数据来源：用户提供的 `错误日志 (3).txt`，从 step 1 至 step 50 的完整记录。
本报告区分日志确认的事实、代码中已修复的遗漏，以及尚需 GPU 对照验证的性能判断。

## 训练工作量与发布

50 步正常结束。共完成 12 个 quota，每个 quota 20 个完整训练组 optimizer step，
合计 240 步；空闲窗口完成 27 步（11.25%），关键路径补齐 213 步（88.75%），
有 12 次成功发布，最后 quota debt 为 0。不能按两个 worker 重复累加训练步数。

| quota 数据版本 | 空闲训练步数 | 补齐步数 | 发布 RL step |
| --- | ---: | ---: | ---: |
| 4 | 0 | 20 | 4 |
| 8 | 1 | 19 | 11 |
| 12 | 3 | 17 | 15 |
| 16 | 0 | 20 | 19 |
| 20 | 5 | 15 | 23 |
| 24 | 6 | 14 | 27 |
| 28 | 0 | 20 | 31 |
| 32 | 7 | 13 | 35 |
| 36 | 0 | 20 | 39 |
| 40 | 5 | 15 | 43 |
| 44 | 0 | 20 | 47 |
| 48 | 0 | 20 | 50 |

这证明实际训练和发布链路在工作，不能解释为“V1 没有训练”。
日志中 `timing_s/drafter` 累计约 218.39 秒，step 耗时累计约 8794.61 秒。
该 drafter 指标排除 old-logprob 采集等阶段，不是 Bubble 总成本或端到端加速证明。

## 隐藏比例低的主要限制

- writer 固定为 `('2', '3')`，对应 replica 1。其 50 个窗口的中位数约 1.66 秒、
  均值约 2.48 秒，范围 0.193～9.336 秒。rollout 总耗时不等于可训练空闲时间。
- 另一 replica 的长窗口不能直接替代 writer；需要完整模型、optimizer 和 replay
  所属状态迁移。当前 `writer_migration_enable=false`，不能仅根据 serving ACK 切换。
- `max_debt_age_steps=3`、`max_completion_lag_steps=3` 与允许关键路径 fallback
  使未偿 quota 在到期时被完整补齐。第 4 步另有 optimizer bootstrap，属于既有策略。
- 短窗口还要承担 head 应用、首次 forward 和 cleanup 等开销。第 10 步原计划 6 步，
  但实际窗口回收只允许完成 1 步；第 13/14 步预算只允许 1/2 步，均真实完成。

可见的调度决策包括：52 次 incomplete_training_group、28 次 quota complete、
13 次 window too small、4 次 no trainable batch，以及 9 次成功启动。
这些是打印的决策记录数，不是 RL step 数或全部后台轮询数。

## 再准入并非完全失效

第 33 步出现一次 `idle_historical_readmission`，年龄约 5.579 秒，保守剩余窗口
约 1.375 秒；该轮第一次训练完成 6 步，再准入完成 1 步，合计 7 步。

第 22 步最终观察到 8.017 秒窗口，但这在准入时不可预知。最近 16 个已完成历史
窗口在年龄 3 秒时，只有 3 个仍存活，最短剩余约 0.746 秒；年龄 4 秒时只剩 2 个。
当前门槛要求至少 3 个同 replica 历史样本，并要求最短剩余时间容纳启动、训练和尾部。
这解释了当时不能安全续训；不能用事后看到的 8 秒直接放宽未来窗口保证。

## 本次修复

### 1. 保留 sample-end drain 完成的训练指标

V1 `on_sample_end()` 之前忽略 `BubbleLifecycle.close()` 的返回值。
若训练是在该边界等待期间完成，scheduler 会正确还债，但该次 worker 的 trained、
successful steps 和耗时没有进入当步日志。第 42 步存在对应证据：累计空闲步数
从 23 增至 27，而当步 worker 训练指标缺失。

现在将 drain 返回指标保存到共享指标队列，由 actor 后的正常收取路径消费。
不改变训练执行或重复执行 optimizer step。

### 2. 最终记录时读取一致的 quota 快照

第 42 步日志仍显示 debt 19，但累计空闲训练已增加 4 步，实际 debt 已为 15。
第 43 步的 top-up 也正确只补齐了 15 步。这是旧事件指标覆盖新状态，不是 quota 少还。

现在在 V1 最终指标阶段持 Bubble 锁，读取当前计数、欠账、年龄和数据版本，
覆盖旧事件快照；不会改变 scheduler 的预算或提前清零欠账。

### 3. 接入共享 GPU 安全边界的 hot writer head 准备

legacy 已有 `_speco_prepare_hot_writer_head_at_safe_point()`，V1 之前没有调用。
日志中版本切换后的空闲训练仍支付约 0.26～0.34 秒 head 准备时间。

现在 sync/colocate_async 在 actor 完成、事件线程退出且 head ACK 完整后、
上游 target/rollout 恢复前调用已有 helper，只准备已驻留 writer 的固定 quota head。
已有 helper 保留欠账、hot residency、完整 ACK、无活动训练/发布等校验；
准备耗时进入 ROI 成本和关键路径计时，不伪装成免费工作。
separate_async 不走这个共享 GPU 边界，避免无独立池租约时操作 serving GPU。

该调整把开销移出短窗口，可能增加隐藏训练机会，但也增加边界耗时，
是否改善端到端时间必须用等工作量 GPU 对照确认。

## 配置与后续验证

日志中 `save_freq=-1`、`test_freq=-1`，因此没有周期/最终保存和验证请求；
正常完成 50 步不代表存在 step 50 的联合恢复点。若需续训，应显式启用保存，
并确认 joint manifest、latest 指针和配套组件完整。

本次不调整 quota 20 步、不放宽 freshness、再准入历史样本门槛、租约或回收检查，
也不启用 writer 迁移。验证应固定模型、样本、quota 和发布语义，比较普通训练、
修复前 Bubble 和修复后 Bubble 的完整训练工作量、隐藏步数、补齐次数、发布次数、
head 准备耗时、采集耗时与端到端时间。CPU 回归不能替代该 GPU 对照。

## CPU 验证记录

真实 verl `release/v0.9.0` V1 类及 scheduler 回归共 609 项通过，覆盖三模式、
历史再准入、writer 保留、quota、发布和恢复；缺少 V1 依赖时按 CI 合约直接失败。
本次新增回归验证 head 准备的 close/ACK/target-resume 顺序、同版本不重复准备、
separate 池隔离、sample-end drain 指标收取和第 42 步式陈旧 quota 快照覆盖。
Ruff、格式、Mypy 与文档日期检查通过。尚未在 GPU 上重跑修复后的 50 步对照。
