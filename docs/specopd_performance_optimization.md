# SpecOPD 短测耗时分析与优化

2026-11-16。目标：在 `.44` 保持 V1 sync、actor/ref Ulysses SP=4、每5个RL step采集一次、每次20个optimizer step及一次发布，不减少训练数据和更新次数；验证后同步 `.242`。

## 原短测观察

两方法原先并行运行、各4 die；16条17k来源训练prompt、4条AIME验证prompt，train batch=4、rollout n=2、response cap=128。日志 `ulysses-{draft_opd,spec_opd}-01.log`。

| 指标（秒） | DraftOPD | SpecOPD |
|---|---:|---:|
| 10轮step累计 | 600.76 | 676.03 |
| 非采集轮平均 | 51.89 | 50.02 |
| 第5轮old-logprob/采集评分 | 9.14 | 32.12 |
| 第10轮old-logprob/采集评分 | 6.45 | 32.34 |
| 第5轮actor+草稿更新阶段 | 56.68 | 89.89 |
| 第10轮actor+草稿更新阶段 | 33.36 | 50.34 |

第10轮含最终验证；step累计排除进程启动，不能解释为进程总wall time。actor+草稿更新不是纯草稿计算时间：包含target head同步、模型激活、训练、快照及内存释放。旧SpecOPD没有独立训练阶段计时，不能从差值精确推断纯训练耗时。

## 已定位开销与修改

1. 原driver每条轨迹一个RPC，8条轨迹重复8次actor engine eval/offload上下文。改为一个RPC携带微批划分，单次engine驻留期间依次执行原微批。每条轨迹的主干和分支仍合成一次forward；8条轨迹仍执行8次模型forward，保留microbatch=1和原mask/token预算，不把全部轨迹堆成一个大mask。完整请求/微批划分在分布式rank间做一致性检查，异常时恢复engine与module状态。
2. 原草稿每个DP副本都拿到完整组，8条轨迹乘20次更新，副本间重复。改为按drafter实际DP rank分摊完整轨迹、SP peers仍共享同一条。每轮微批数为ceil(全组轨迹数/DP)，不足处零loss dummy保证各rank collective次数相同。保持全组有效token-sum归一化和20次optimizer更新；dummy不计真实样本或有效token。不修改DraftOPD基线。
3. 原20次更新反复组装同一固定输入、做摘要与CPU→设备传输。现在每组每个本地槽位只构造一次；最多缓存64MiB输入tensor在设备，超出部分保留CPU准备结果、逐微批搬运。缓存仅活到本次组训练结束，不缓存学习得到的context投影、draft logits或梯度，不跨revision复用。
4. 原每个分支分别all-reduce logZ/hidden/logits，且为完整本地序列重复计算全词表logZ。现在只计算分支读出行的归一化，合并同轨迹所有分支readout通信，再按原分支边界还原。
5. 增加 `timing_s/specopd_{activation,prepare,train_loop,snapshot,release,train_rpc}`，driver汇总各训练rank阶段最大值。RPC总耗时含传输/等待，阶段计时不是所有rank时间之和；train_loop含微批搬运、前后向、loss统计和optimizer，不能当作纯设备kernel时间。同步head及发布沿用已有指标。

## 验证计划与边界

- 多进程CPU对照：Qwen3/Qwen2/Llama，SP2/4，含树分支隔离、读出、单次engine驻留与异常恢复。
- 分布式梯度对照：20次更新逐步对比完整组参考梯度/参数，覆盖DP2/SP1、DP2/SP2、非均匀有效token和样本少于DP产生dummy的情况。
- `.44` HCCL小模型FP32/BF16 readout oracle；真实10轮SpecOPD，检查仅第5/10轮训练、每轮20步、一次发布与版本ACK。
- 为减少并行任务干扰，新增 `perf05-spec-baseline-01` 与 `perf05-spec-optimized-01`，按顺序在同一die4–7运行；其余原有任务不操作。两次模型启动分别发生，单次成对短测仍存在缓存/系统负载波动。
- 当前树attention仍为显式QK→mask→softmax→V，普通old-logprob是FlashAttention；此优化不宣称已实现高效树attention kernel。完整16die、8192 response的生产容量与吞吐另行验证。

## 结果

优化候选代码 `0caec0b`，补丁31,721字节；优化版 `perf05-spec-optimized-01` 完成10步及最终验证，退出0。201项相关回归加12项真实网络前后向测试通过，HCCL小模型oracle退出0。`.44` 本次测试driver已全部退出，其他任务PID1799413未操作。

| 同设备第5轮，单位秒 | 新独立基线 | 优化后 | 耗时减少 |
|---|---:|---:|---:|
| 联合评分RPC | 31.60 | 6.36 | 79.9% |
| old-logprob总阶段 | 32.05 | 6.79 | 78.8% |
| actor+草稿更新阶段 | 50.27 | 38.84 | 22.7% |
| 采集/训练/发布所在整轮 | 114.37 | 79.39 | 30.6% |
| 前5轮step累计 | 319.13 | 280.32 | 12.2% |
| 前7轮step累计 | 415.12 | 376.36 | 9.3% |

新独立基线在第7轮之后收到未知SIGKILL、退出137，未完成10步。容器OOM/内存failcnt计数均0，主机无同期OOM记录，不能断言OOM或代码导致。上表只比较双方已经完成的共同轮次，单次运行仍有系统负载波动；没有拿7步累计冒充10步。优化版单次完整10步累计552.07秒，只报告绝对值，不与这条不完整基线计算10步加速率。此前并行运行的旧10步676.03秒也不作为严格独立对照。因此本阶段不重复昂贵完整基线。

优化后两个采集轮明细：

| 指标（秒，rank阶段取最大值） | 第5轮 | 第10轮 |
|---|---:|---:|
| 联合评分 | 6.3649 | 6.3450 |
| 草稿训练RPC（含传输/等待/释放） | 18.7893 | 18.4836 |
| 训练循环（含微批搬运、前后向、optimizer） | 16.1734 | 16.1261 |
| 固定输入准备 | 0.0821 | 0.0894 |
| 模型激活 | 0.6017 | 3.5179 |
| 发布快照准备 | 1.0498 | 0.5435 |
| 模型/optimizer内存释放 | 1.5934 | 1.2527 |
| target LM head同步 | 2.6895 | 4.5681 |
| 整轮step | 79.3867 | 79.5878 |

不同rank的阶段最大值不能直接加总为单rank串行trace，训练循环与RPC也不是独立相加项。旧版没有对应纯训练计时，不能给出纯训练循环的可靠前后百分比。

两轮均为实际actor SP=4、8次联合forward、112个proposal位置；草稿实际DP=2、每副本4个轨迹微批，缓存分别53,471,912/49,948,664字节，均低于64MiB。optimizer累计20→40，revision1→2，actor ACK每轮推进；第5轮来源4/0、第10轮9/1。branch位置数及logical rows（2526/2426）与旧短测一致。actor grad_norm仍均为0，小AIME验证acc@1=0；此处只证明控制流、数值和短测耗时，不证明RL学习质量。

原始日志在 `.44:/home/c00954340/specopd-debug-logs/perf05-*`，本地副本在 `tmp/debug44-artifacts/perf05-*`，解析结果 `tmp/perf05-timing-results.json`。源码哈希证据 `perf05-source-provenance.json`。最终部署与提交见 [Ulysses台账](debug44_ulysses_validation.md) PERF-05。


## PERF-06：轨迹微批合并

2026-11-16，用户要求优先测试4条一批。独立于actor评分微批，新增：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        specopd:
          training_microbatch_trajectories: 4
          training_microbatch_token_budget: 16384
```

设为1复现PERF-05的逐条累积；设2/4控制每个DP副本的训练批大小。token预算限制合并后的 `批大小×最大主干长度`；长序列拆成更小批，单条仍按原路径处理，不把超出合并预算的单条直接丢弃。默认16384不保证8192-response生产配置始终能四条合并，实际值需要看长度分布与记录的微批数。

实现使用独立batch行保存不同轨迹，分别pad主干、anchor与proposal；padding/dummy的proposal mask为false，query有安全self-attention行，不能访问其他轨迹。E2E-TV前将batch/anchor轴摊平，按全部真实proposal token之和归一化；20个optimizer step与每5轮一次发布均不变。CUDA/Flex与其他设备SDPA沿用已有mask构造，不把不同轨迹拼到同一因果主干中。

DP间分组边界从全局槽位的最大主干长度计算，保证不齐/长短数据下每个rank都有相同的前后向及collective次数。缺少轨迹的dummy使用同槽位真实轨迹的形状，避免从全组第一条长轨迹引入意外padding。输入缓存仍有64MiB设备上限。

验证：221项相关回归通过。真实DFlash/DSpark/DFlash2的4条ragged（主干长短、分支数、proposal长度）+dummy合并loss/梯度，与逐条累积一致；更新计数检查覆盖批大小1/4及DP不齐。真实NPU DSpark小模型oracle：FP32 loss差0、最大梯度差1.8626e-9；BF16 loss差0、最大梯度差1.2207e-4。预设容差FP32 atol1e-4/rtol1e-3，BF16 atol0.01/rtol0.05，限于该小模型检查，不能视作任意模型/长度的误差保证。日志 `perf06-contracts-01`、`perf06-npu-grad-01`。

实际Qwen3-8B检查 `perf06-spec-b4-01` 在 .44 die4–7，保留V1 sync/actor-ref SP4、5×20、16条训练/4条验证prompt及response cap128，另用Ray目录 `/tmp/speco44-perf06-b4`。以已通过的PERF-05独立batch1十步结果为参照；HBM采样文件 `perf06-memory-01.jsonl` 只观察专用die4–7，每2秒采样，不能当作精确allocator瞬时峰值。完整10步已退出0，结果如下。


### PERF-06 实测验收

对照PERF-05的独立逐条累积运行 `perf05-spec-optimized-01`，新运行 `perf06-spec-b4-01` 仍为同一die4–7和同一短测配置；运行之间存在系统负载/数据采样波动，以下为单次实测。

| 秒 | 原逐条，第5轮 | 4条合并，第5轮 | 原逐条，第10轮 | 4条合并，第10轮 |
|---|---:|---:|---:|---:|
| 草稿训练循环，20步 | 16.1734 | 9.3327 | 16.1261 | 5.0967 |
| 草稿训练RPC | 18.7893 | 12.8622 | 18.4836 | 7.8949 |
| actor+草稿更新阶段 | 38.8420 | 34.4033 | 42.0785 | 29.3069 |
| 整个RL step | 79.3867 | 76.1441 | 79.5878 | 69.7278 |

第10轮草稿训练loop耗时减少68.4%（约3.16倍速度），actor+草稿阶段减少30.4%，整轮减少12.4%。第5轮仍有较高的首次阶段开销，不将连续optimizer日志间隔误当整个训练循环计时。十轮step累计552.07→541.04秒，减少约2.0%；草稿每5轮更新一次，加上其他阶段波动，不能把采集轮加速率直接当作整体训练加速率。没有另外跑batch2，不能声称4是所有硬件/长度下的最优批大小。

两次采集轮均实际为每个DP副本1个训练微批、批内4条轨迹，20次前后向对应20个optimizer step；原来每副本4个微批累积对应80次前后向。每轮仍为112个真实proposal位置，actor联合评分8次、SP4，草稿DP2，20→40累计更新和revision1→2，10轮ACK校验通过。输入缓存56,339,672/53,553,848字节，小于64MiB。独立轨迹batch行隔离、ragged padding、dummy及全组token归一化通过221项回归与NPU数值oracle。

HBM采样196次，die4/5/6/7最大工具读数分别49,974/50,616/50,891/50,582 MB，总容量各65,536 MB；这是包含初始化/actor/rollout/草稿的设备整体观测值，不是精确allocator或草稿独占瞬时峰值。无本次NPU溢出，所有训练driver已退出，其他原有PID1799413保留。采样程序在该短测退出后自动结束。actor梯度仍为0，验证准确率仍为0，未作学习质量结论；16-die长序列生产负载未测。

验收源码候选 `6b0143a`，核心补丁22,162字节。两个生产脚本保持原参数/路径，新默认配置从代码仓读取；最终部署回执见Ulysses台账PERF-06。
