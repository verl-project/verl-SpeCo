# .44 V1 Draft-OPD / SpecOPD 调试台账

2026-11-16：用户要求两种方法使用 `trainer.use_v1=true`，先在
`7.208.150.44` / `verl-speco-090` 短步调试，再完成 `.242` 部署。
既有旧版 trainer 的成功结果不能作为 V1 验收结果。

## 阶段与验收

- [x] 配置阶段：V1 sync + TransferQueue；两种方法能解析配置并通过方法校验。
- [x] 合约阶段：V1 变长数据与 proposal wire 对齐；正常 old-logprob 与完整分支联合评分；一次消费/更新；同步发布版本确认。硬件覆盖两步调试组合，padding 由 CPU 合约覆盖。
- [x] 短测阶段：每种方法两步、使用现有 17k/AIME 小样本；loss 有限、optimizer 与发布版本前进；退出码为 0。
- [x] 部署阶段：`.242` 的真实 `specopd` 仓目录与两个 V1 脚本对应，语法及配置核验；源码和脚本哈希核对。

## 执行记录

### V1-00：环境与配置检查（进行中）

- 容器运行中；已有模型、数据与旧版成功日志保留。
- 使用空闲 die；die 3 上其他容器的 VLLMWorker PID 1799413 保持不动。
- 当前 SpecOPD 校验明确拒绝 V1。需要接入 V1 生命周期与 TransferQueue，不能只删除校验。
- 原 `.242` 参考脚本使用 V1 sync；模型、数据、16 die、TP=2 与训练长度保留。SpecOPD 的 dense tree 与一次更新要求需相应参数。
- 尚未启动 V1 训练，尚无 V1 通过结论。

### V1-01/02：配置与 Draft-OPD 首次硬件尝试

- `v1-draft-01` 在 Hydra argparse 阶段退出（2）：`--config-name` 不能插在原有 overrides 中间。改为启动命令首部选择配置，新增 V1 wrapper 设置 `SPECO_CONFIG_NAME=speco_v1_trainer`。
- `v1-draft-02` 使用 die 4–7、两步、900 秒上限，日志 `/home/c00954340/specopd-debug-logs/v1-draft-02{.log,-config.log,.exit}`；配置阶段通过，正在初始化 V1/TQ/actor。
- SpecOPD 新增 V1 sync adapter：rollout 绑定 revision、TQ proposal wire 读取、联合 old-logprob 写回、actor 更新前一次草稿训练、actor 权重同步后发布与 ACK。异步 V1、多次本地更新、group-filter refill 仍拒绝。
- 先前 focused 合约 57 passed / 2 skipped；新增生命周期测试 2 passed。
- 扩大到整个 `tests/v1` 的本地运行暴露环境问题：默认 pytest 临时目录无权限及本地没有 Ray。改用 workspace 内临时目录；不将这些环境失败报告为算法失败，也不据此宣称 V1 硬件通过。

### V1-03/04：同步链路与 TensorDict 兼容修复

- V1-02 停在资源组初始化；Ray 16 CPU 中 TQ/driver 已占 10 CPU。结束本次任务及其已核实的子进程，未触及其他容器；wrapper 改用参考脚本的 64 CPU。
- V1-03 两种方法均完成模型、vLLM 与草稿初始化，但在 agent 请求阶段退出（1）：诊断访问 `batch.meta_info`，V1 输入实际为 TensorDict。改为可选读取。
- 显式 V1 worker 子类入口和原地 patch 入口各自携带/恢复 proposal revision ContextVar，避免只修其中一条入口；增加异常下 context 恢复与 TensorDict 回归检查。
- 容器合约第一组 49 passed；V1 mixin + 新适配器组 92 passed。新增本地 TQ 对齐与 padding 检查通过；agent + adapter focused 27 passed / 2 skipped。
- V1-04 仍使用 die 4–7 / 8–11，各两步、900 秒；源码快照 `/home/c00954340/verl-SpeCo-v1-debug`，与旧版成功快照隔离。结果待定。
- `.242` 已以 Git bundle 克隆 `specopd` 到 `/efs_rl/z00886395/codes/test_speco/verl-SpeCo-specopd`，基线为 `7bca89f`；两个脚本暂存为 `.pending`，bash 语法通过，等待已验证的 V1 代码覆盖与最终落地。原参考脚本保持不动。

### V1-05：冻结请求上下文修复后重跑（进行中）

- V1-04 初始化期间主动结束，未作算法判定：补充发现内外两层包装器若没有显式 kwarg，需要继承已有 ContextVar，而不是清空为 None。新增继承上下文及异常恢复用例，agent targeted 26 passed / 2 skipped。
- V1-05 按相同短预算重新启动，源码修改前后一致；日志 `v1-draft-05` / `v1-spec-05`，结果待定。
- V1 source adapter、配置校验、driver、mixin、agent runtime 及变更测试的 Ruff 和 `git diff --check` 通过。
- `.242` 两个实际 bash 命令通过无模型执行的 argv 捕获；同一批 argv 在 `.44` 的真实 verl 配置环境成功 compose，并通过 SpecOPD config 校验（两个方法的 V1 sync 开关均正确）。`.242` 宿主机没有 torch 且缺少完整 verl 配置包，未安装依赖；实际训练仍应在已有训练容器内启动。
- `.242` 基线 checkout 已覆盖 V1 工作树源码，receipt 位于 `/efs_rl/z00886395/codes/test_speco/specopd-v1-source-manifest.json`；还未发布最终脚本名，也未启动 `.242` 训练。

### V1-06：保留原生 TQ worker（进行中）

- V1-05 两种方法均在首轮 rollout 退出（1）。完整栈确认并非诊断语句残留，而是父类 manager 初始化用通用 AgentLoopWorker 覆盖了原生 TQ worker，随后 upstream 旧 `generate_sequences` 对 TensorDict 访问 `meta_info`。
- `_configure_speco_agent_loop_manager_instance` 对原生 `AgentLoopManagerTQ` 保留已安装的序列化 TQ worker；原生 V1 方法与 TQ 数据生命周期保持一致。新增 worker 选择回归，agent + adapter 30 passed / 2 skipped。
- V1-06 重跑相同双方法两步预算，日志 `v1-draft-06` / `v1-spec-06`；结果待定。

### V1-06 硬件结果：两种方法通过

| 项目 | Draft-OPD | SpecOPD |
|---|---|---|
| trainer | V1 sync / TQ | V1 sync / TQ |
| die / TP | 4–7 / TP=2 | 8–11 / TP=2 |
| 退出码 / RL 步数 | 0 / 2 | 0 / 2 |
| 每步采集 / 评分 | old-logprob 采集 8 样本 | 8 轨迹、8 次联合 forward、112 草稿位置 |
| optimizer step | 1 → 2 | 1 → 2 |
| rank 0 日志 loss | 0.9165 / 0.7001 | 0.6689 / 0.7989 |
| 更新 / 发布 | 每步 trained=1、published=1 | 每步 trained=1、published=1 |
| 来源 actor/drafter → 发布 ACK | 基线保留原协议 | 0/0 → 1/1；1/1 → 2/2 |
| AIME 小验证 | 完成，acc@1=0 | 完成，acc@1=0 |

SpecOPD 真实 wire 包含拒绝后尾部。两步逻辑行数 2334 / 2302、独立分支计数 4804 / 4711；不是加速比。正常回答 old-logprob 写回 TQ，正常 actor 更新路径执行；两步 actor grad_norm 都为 0，短回答预算下不代表有效 actor 学习。
草稿 loss 有限且 optimizer 推进；SpecOPD 通用分类 accuracy 为 NaN（未提供该诊断统计），不是 E2E-TV loss 的 NaN。

日志和完整配置/exit 保存在 `/home/c00954340/specopd-debug-logs/v1-{draft,spec}-06*`；本地副本在 workspace `tmp/debug44-artifacts`。
原有旧版 trainer 测试记录保持独立。V1 异步、长序列、16-die 正式训练、恢复 checkpoint、性能和质量仍未验收。

### 最终回归与部署核验

- 容器最终 focused 回归 **159 passed**，包含 V1 mixin、SpecOPD adapter、配置/版本/分支合约与 agent transport。日志 `v1-final-contracts-06.log`，退出 0；40 个依赖警告没有使检查失败。
- 本地最终变更 Ruff 和 diff whitespace 检查通过；本地缺 Ray 的广泛 V1 测试失败没有计为通过，完整依赖下的上述 159 项作为最终证据。
- 两个正式脚本已从 `.pending` 转为 `.sh`，具有执行权限，bash 语法通过。模块实际解析到 `/efs_rl/z00886395/codes/test_speco/verl-SpeCo-specopd/verl_speco/__init__.py`。
- 参考原脚本 SHA256 `ed9bf6fef29ada5676503eb958f76bd440bb8dcc63ad264ae1463373d19c78bf`，与下载时一致；未覆盖原文件。
- `.44` 活跃 V1 调试 driver 列表为空，NPU 进程核验仅余其他容器原有 PID 1799413；本次使用的 die 4–11 全部释放。容器保持运行。
- `.242` 未启动训练；两个完整生产 argv 的 compose/校验与 `.44` 两步硬件结果分别记录，不宣称 16 die / 8192 response 的完整生产运行已验证。
