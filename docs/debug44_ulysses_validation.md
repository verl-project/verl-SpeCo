# SpecOPD Ulysses 接入与验收

2026-11-16，用户要求适配 Ulysses，并以 `.242` 指定的
`/efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_drafter_with_train.sh`
（`train_sp=4`）作为配置参照。此前 V1 sync 短测通过只覆盖 SpecOPD SP=1。

## 阶段

- [x] SP-00：确认 upstream remove-padding/SP forward 与通信接口；给出作用域隔离的联合评分实现。
- [x] SP-01：CPU 多进程 SP=2/4 与完整树前向对照，覆盖非整除长度、GQA、逻辑 RoPE、不同分支位置、hidden/logprob 与隔离；该完整树参考实现已有独立因果 oracle。
- [x] SP-02：真实 NPU 小模型/通信检查，SP=4、DP=2/SP=2、leader 和 dummy 组无重复/遗漏。
- [x] SP-03：`.44` V1 sync + actor/ref SP=4、10 个 RL step 短测，覆盖第 5/10 轮，完成正常 actor/ref 与草稿更新、评分/发布 ACK。
- [x] SP-04：参数、方案和 `.242` 代码/脚本同步；短测范围和生产长序列范围分别说明。

## SP-00 进行中

- `.44` 实际 verl FSDP engine 明确要求 SP>1 时 `use_remove_padding=true`；当前联合评分既拒绝 remove-padding，也拒绝 SP。
- 不能删校验后把树输入交给原 causal FlashAttention。需保留普通 PPO 的 upstream SP 路径，并在联合评分调用中使用支持树 mask 的真实 Ulysses QKV 通信。
- 未启动新的 NPU 训练。

## SP-00/01 初步通过

- 实现独立作用域的 Ulysses QKV all-to-all 与逆向交换；按 SP 组分配 DP 轨迹，只有组 leader 回传一份结果。普通 PPO/ref 仍使用 upstream unpadding + FlashAttention，不修改其全局 attention 实现。
- CPU Gloo 的 SP=2/4、GQA（KV heads 少于 SP）、Qwen3/Qwen2/Llama、非整除长度、分支/future-trunk 隔离、logprob/entropy/hidden/readout 对照，4 项多进程测试通过。其他相关回归 87 passed / 2 skipped。
- 用户明确调整调度：**每 5 个 RL step 采集一次，固定该组数据训练 20 个 optimizer step，结束后统一发布一次**。这替代此前单次 optimizer 更新的保守约束；20 步间复用采样分支属于近似的多步训练，不声称每步都有重新从当前 q 采样。
- 新增调度合约：非更新轮保留正常 PPO old-logprob 和 actor 更新、不采集分支；第 5/10 轮分别训练 20 步，逻辑草稿发布 revision 各增加 1，actor ACK 每轮更新。相关 39 项检查通过。
- 短测新脚本计划使用 SP=4、KL/ref 启用、10 个 RL step（覆盖第 5/10 轮），两种方法都用 5×20 调度；尚未作为 NPU 通过结果。

## SP-02：NPU 数值与分组通过

- 首次远端 CPU Gloo / NPU rendezvous 因主机名解析失败而未进入数学检查；Gloo 设 `GLOO_SOCKET_IFNAME=lo`，torchrun 使用显式 `127.0.0.1` 静态 rendezvous 后通过。未改变公共网络/认证配置。
- `ulysses-cpu-02`：4 个 CPU 多进程对照通过，补充 DP=2/SP=2 和 dummy/leader 检查在本地通过。
- `ulysses-npu-02`：4 die、真实 HCCL，随机 Qwen3 两层模型。SP=4 的 FP32/BF16 response logprob、branch logits、branch hidden 最大差异均为 0；退出 0。
- `ulysses-npu-dp-03`：同一 NPU oracle 再覆盖 DP=2/SP=2，仅 rank 0 回传真实轨迹，其余 rank（含 dummy 组）不重复回传；FP32 LP 最大差异 4.77e-7，BF16 为 0；退出 0。
- 阈值预设为 FP32 5e-5 / BF16 0.03，仅针对该小探针，不能当作所有大模型/长上下文的通用误差界。
- 暂拒绝 adaptive/dynamic/longrope 的 SP 评分，避免各序列 shard 按各自 position 最大值生成不同 RoPE 频率；部署用 Qwen3-8B 为静态 RoPE。普通 LoRA、滑窗、混合注意力等原有范围限制不放开。

## SP-03：真实 Qwen3-8B V1 / SP=4 / 5×20 调度通过

源码快照 `/home/c00954340/verl-SpeCo-ulysses-debug`，与旧 V1 SP=1 验收隔离。
两方法各使用 4 die，actor/ref SP=4、rollout TP=2、KL/ref 开启；16 个 17k 来源训练 prompt、4 个 AIME 测试 prompt，batch=4、n=2、response 上限 128、10 个 RL step、最多 3 epoch、1800 秒上限。没有运行 200 步正式训练。

| 项目 | Draft-OPD | SpecOPD |
|---|---|---|
| die | 8–11 | 4–7 |
| 日志前缀 | `ulysses-draft_opd-01` | `ulysses-spec_opd-01` |
| 退出码 | 0 | 0 |
| 草稿训练轮次 | 仅 5、10 | 仅 5、10 |
| optimizer 累计 | 20 → 40 | 20 → 40 |
| 草稿发布 | 第 5/10 轮各一次 | 第 5/10 轮各一次；逻辑 revision 1 → 2 |
| 其余 8 轮 | 正常 RL、无草稿训练 | 正常 old-logprob/actor/ref；无 proposal 采集、无草稿训练/发布 |
| 实际 actor SP | 配置 SP=4，普通 upstream 路径 | 联合前向 RPC 直接报告 `actor_sp_size=4` |
| 每个采集轮 | 基线旧 trace 采集 | 8 次联合 forward、112 个 proposal 位置 |
| 来源 actor/drafter → ACK | 原基线版本协议 | 4/0 → 5/1；9/1 → 10/2 |
| AIME 小验证 | 完成，acc@1=0 | 完成，acc@1=0 |

两方法 actor grad_norm 均为 0；本次验证控制流和数值/更新/发布，不证明 RL 学习收益。SpecOPD 同批数据内 20 步复用的损失降低也不能作为泛化质量提升证据。
SpecOPD 两个采集轮 logical/independent 行数分别 2526/5388、2426/5092，仅为结构计数，不是加速比。

结束清理中出现一条 `EngineDeadError`。完整日志显示：最终 validation 与 100% 进度已经完成；对应 Ray core-worker 日志在 11:49:53 记录 owner `INTENDED_SYSTEM_EXIT`，随后主动杀死子进程，11:49:54 vLLM 才打印 shutdown/output-handler 错误。该条为退出时的 worker 回收日志，不是训练过程中的 EngineCore 失败；原始日志保留，不屏蔽。

本地日志在 `tmp/debug44-artifacts/ulysses-*`，远端在 `/home/c00954340/specopd-debug-logs/ulysses-*`。多 optimizer 测试 1/20 两种配置均在真实容器依赖下通过；每组只在全部更新完成后生成一次发布快照。


## SP-04：最终回归与部署完成

2026-11-16：最终源码回归 `ulysses-final-contracts-04.log` 为 **187 passed**、退出 0；`ulysses-npu-final-04.log` HCCL oracle 退出 0，SP=4 FP32/BF16 三类 readout 最大差异均为 0，DP=2/SP=2 FP32 LP 最大差异 4.77e-7、BF16 为 0，仅真实组 leader 导出。原日志已取回 `tmp/debug44-artifacts/`。

- 代码 `0f7645a` 已推送 `crp0128/specopd`，补丁 52,160 字节；方案更新 `ed0cac9` 补丁 33,019 字节，均小于 100 KB。
- `.242` 代码仓 `/efs_rl/z00886395/codes/test_speco/verl-SpeCo-specopd` 使用干净的 specopd 分支，通过 bundle 快进同步。
- 两个 `*_draft_opd_v1_with_train.sh` / `*_spec_opd_v1_with_train.sh` 已发布至 `/efs_rl/z00886395/shell/speco-runping/`。覆盖前核对旧 release SHA，备份为 `.before-ulysses-89f2915`。原参考脚本 SHA256 保持 `ed9bf6fef29ada5676503eb958f76bd440bb8dcc63ad264ae1463373d19c78bf`。
- 使用 shell function 替代启动入口捕获真实 argv；在 `.44` 完整依赖环境通过 Hydra compose、SpecOPD 校验及 vLLM runtime 配置。实际为 V1 sync、16 die、TP=2、actor/ref SP=4、remove-padding=true、step=20、采集/训练间隔均为 5；SpecOPD async_scheduling=false、联合评分 budget=9216。只替换模型路径读取本地元数据，没有加载模型或启动 `.242` 训练。可选 MSC 配置源缺失打印依赖日志，未影响校验，退出 0。
- import 定位到上述真实代码仓，发布 SHA 记录在 `specopd-v1-source-manifest.json`；最终文档提交同步后 receipt commit 相应更新。
- `.44` 本次 Ulysses 测试训练 driver 为零，NPU 仅保留其他任务原有的 PID 1799413，未操作该进程。

生产配置仍保留 16 die、8192 response token、200 RL step；该完整负载未运行。本次验收证明短测控制流和数值一致性，不证明完整负载容量、吞吐或训练收益。


## PERF-05：耗时优化进行中

2026-11-16：保留 V1 sync/SP4、每5轮20步和固定分支训练语义，使用 phased-test-efficiency 分阶段验证。原始10步日志：DraftOPD 累计600.76秒，SpecOPD 676.03秒；第5/10轮 SpecOPD 联合评分31.69/31.89秒。计划消除逐轨迹RPC/actor offload上下文重复、合并分支readout通信、有限缓存固定训练输入，并增加训练/搬运独立计时。先做数值/生命周期检查，再在 .44 独立短测，不以减少训练量换取加速。

- [x] PERF-05A：小模型数值与输入缓存/单次engine上下文回归。
- [x] PERF-05B：.44 HCCL oracle及真实10轮优化后短测，记录可比范围。
- [x] PERF-05C：推送小于100KB的提交，同步 .242 代码及文档。


PERF-05A 通过：本地34 passed/2缺Ray跳过；`.44` 最终相关回归201 passed、真实草稿前后向补充12 passed；HCCL FP32/BF16 oracle退出0。新增20步DP2/SP1与DP2/SP2梯度/参数等价检查覆盖非均匀token和dummy。6个运行模块哈希已与baseline b09287c/本地优化源码分别核对，证据 `perf05-source-provenance.json`。

PERF-05B：新增独立baseline-01完成到第7步后driver SIGKILL，退出137；第5步联合评分31.60秒、step114.37秒。容器OOM计数/内存failcnt均0、无限制，主机内核仅有10月及更早OOM记录，未发现此次OOM证据；不能将未知SIGKILL归因于代码或内存不足。保留日志，不计为完整10步成功。依赖baseline退出0的优化版排队未启动，确认无本次残留训练进程后单独启动优化版10步，另用 `/tmp/speco44-perf05-opt` Ray目录；未改训练参数。仍需优化版完成后再决定是否补跑完整独立baseline。


PERF-05B 中间结果：优化版第5轮联合评分6.3649秒（同设备新baseline第5轮31.6004秒），old-logprob总阶段6.7929秒，actor+草稿更新38.8420秒，step总79.3867秒（baseline114.3687秒）。仍为112个proposal位置、20个optimizer更新；实际草稿DP=2，每副本准备4条轨迹。新增独立计时：训练RPC18.7893秒、训练loop最大16.1734秒、输入准备0.0821秒、激活0.6017秒、快照1.0498秒、释放1.5934秒。各rank阶段最大值不可直接相加视为单个rank串行trace。尚待第10轮/最终退出，不能作为完整10步成功。


PERF-05B 最终通过：优化版10步退出0，第二采集轮联合评分6.3450秒、草稿训练RPC18.4836秒、训练loop16.1261秒、整轮79.5878秒。累计optimizer20→40，发布revision1→2，仅5/10轮训练且每次20步；8次联合forward/112个proposal位置、actor SP4、草稿DP2与缓存上限已逐条检查，10轮ACK顺序通过。总step552.07秒，201+12项测试通过。新独立基线只有前7轮完整数据，比较前5轮319.13→280.32秒（减少12.2%）、前7轮415.12→376.36秒（减少9.3%），不计算独立10轮加速率。第5轮联合评分减少79.9%、整轮减少30.6%。不再重复昂贵基线，保留未知SIGKILL及范围说明；详见 [性能记录](specopd_performance_optimization.md)。本次driver为零，NPU仅留其他任务PID1799413。


PERF-05C 完成：性能代码 `0caec0b`（31,721字节）及方案/结果 `161464a`（12,241字节）已推送 `crp0128/specopd` 并快进同步 `.242:/efs_rl/z00886395/codes/test_speco/verl-SpeCo-specopd`。部署时工作区干净，486个源码/测试/文档文件形成SHA回执；两份现有V1启动脚本SHA保持不变、bash语法和真实import路径通过，原参考脚本SHA也未改变。最后台账提交同步后receipt commit随之推进。`.242` 正式训练未启动。


## PERF-06：草稿轨迹微批合并进行中

用户要求合并微批，并优先测试4条一批。增加独立的training_microbatch_trajectories=4（可设1/2/4）、training_microbatch_token_budget=16384。按全局DP槽位最大长度选择一致分组，长序列超预算拆小；单条原样保留。先检查ragged/padded/dummy数值及梯度，再在.44短测第5/10轮的20步训练与显存。PERF-05的联合评分配置不变。

- [x] PERF-06A：批内轨迹隔离、ragged与dummy、loss/梯度等价及配置回归。
- [x] PERF-06B：.44实际4条合并短测，20→40更新、一次发布与阶段耗时/显存。
- [x] PERF-06C：推送并同步 .242，记录实际选择及范围。


PERF-06A通过：221 passed，真实NPU DSpark FP32/BF16四条合并oracle均退出0；loss差0，最大梯度差分别1.86e-9/1.22e-4。最后dummy形状选取修正后，CPU DP20步梯度与分组检查8 passed（本地缺Ray的10个真实trainer控制用例跳过，这些用例已在远端221项中通过）。PERF-06B使用`perf06-spec-b4-01`，独立die4–7，HBM只读采样有上限并随测试退出停止；仍待实际5/10轮计时与退出。


PERF-06B通过：`perf06-spec-b4-01` 完整10轮退出0，采集轮仅5/10，每轮20步、一次发布；实际训练批大小4、每DP副本1个微批，actor SP4、草稿DP2、112个proposal位置/8个联合forward、全部ACK顺序和64MiB输入缓存上限均已核对。第10轮train_loop16.1261→5.0967秒，actor+草稿42.0785→29.3069秒，step79.5878→69.7278秒；10轮累计552.07→541.04秒，不能将采集轮收益外推为总收益。221项回归及NPU FP32/BF16梯度oracle通过，HBM采样最大50,891/65,536 MB。训练driver为零，其他PID1799413未操作。原始文件 `tmp/debug44-artifacts/perf06-*`，详细边界和计时见性能记录。


PERF-06C完成：核心 `6b0143a`（22,162字节）与性能文档 `8ae25f1`（11,042字节）已推送 `crp0128/specopd` 并快进部署到 `.242` 真实仓库；487个源码/测试/文档文件形成SHA回执，两个V1脚本路径/语法/原SHA、实际import路径均通过。原参考脚本未改，生产训练未启动。最后台账提交随后同步并更新receipt commit。
