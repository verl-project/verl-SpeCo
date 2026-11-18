# DeepSeek V4 0731：actor 与 vLLM-Ascend 对齐记录

日期：2026-11-12。范围：独立诊断；不运行 RL 或 Drafter 训练，不恢复 rollout HS 采集。

## 比较对象

- actor：verl 0.9.0 FSDP2、完整 284.33B HF 模型、BF16、eager、16 卡，使用现有 old-logprob 采集接口。
- 推理端：vLLM/vLLM-Ascend 0.27.1、TP8/DP2/EP、BF16、eager、关闭 prefix cache 和 chunked prefill 的独立轨迹重放。
- 共同源 checkpoint：`/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16`。
  同一 checkpoint 不意味着两端加载后的有效参数精度相同，见下文。
- 两条固定输入分别为 16、12 个 token；actor 右侧 padding 成一个 batch，推理端逐条 prefill。
  不重新生成输入；统一比较绝对位置 3–10，所选 decoder 层为 40、41、42。

辅助 HS 按相同层号取 decoder 输出并对 HC 维度求均值；最终 pre-HC 保留四路完整值。
不对特征做额外归一化或重排来人为减小误差。相对 L2 定义为
`||native − actor||₂ / ||actor||₂`，余弦相似度单独记录。

## 首轮特征结果

以下对比真实 actor 与原生重放。本轮完整 actor 重跑已正常退出，特征与此前保留结果逐字节一致；
源 checkpoint 文件大小及修改时间未变化，特征差异可重复。

| 输入 | 特征 | 相对 L2 误差 | 余弦相似度 |
| --- | --- | ---: | ---: |
| 16 token | layer 40 | 12.72% | 0.991916 |
| 16 token | layer 41 | 14.15% | 0.990006 |
| 16 token | layer 42 | 10.71% | 0.994277 |
| 16 token | 最终 pre-HC | 10.61% | 0.994374 |
| 12 token | layer 40 | 13.01% | 0.991736 |
| 12 token | layer 41 | 13.26% | 0.991362 |
| 12 token | layer 42 | 9.98% | 0.995182 |
| 12 token | 最终 pre-HC | 10.14% | 0.994984 |

这不是先前“actor 内部采集对照误差为 0”的重复结论，而是真实跨运行时比较。
首轮已经证明这两条默认路径的特征不逐字节一致，也不能因为余弦接近 0.99 就认为可直接互换。
尚未据此确定训练效果损失或生产容差。

已核对原生 `layer_idx + 1` 的内部约定与采集器转换一致。两条输入在两端均满足
“最后一层辅助 HS = 同位置 pre-HC 的 HC 均值”，最大误差为 0；记录在
`boundary_invariants.json`。没有发现最后一层取错边界或 HC 展平错位。

## 已确认的参数精度差异

实际导出的最终 HC head：actor 为 BF16，原生推理为 FP32。

| 参数 | 最大绝对差值 | 相对 L2 误差 |
| --- | ---: | ---: |
| `hc_head_fn` | 0.000239268 | 0.165% |
| `hc_head_base` | 0.00679755 | 0.179% |
| `hc_head_scale` | 0.000733733 | 0.093% |
| 最终 norm | 0 | 0 |

这是已测得的精度差异，不能单凭它归因全部层间误差。FSDP2 是 actor 的分片执行方式，
目前没有证据表明分片机制本身就是差异根因。
另外，三项 actor HC head 的值均逐元素等于原生权重经 BF16 舍入后的值，确认这部分差异来自精度转换。

## 输出分布与精度对照

原生侧已保存实际 `compute_logits` 输出；HF 探针新增实际 LM-head 输出 hook，已通过小模型 NPU 验证。
完整 actor 的新一轮运行已完成，比较相同位置的全词表 logits、总变差距离（TV）、KL 和 top-1 一致率。
这些输出来自各自实际前向，不使用一个公共 teacher 重建两边 logits 来代替原生输出。

| 输入 | logits 相对 L2 | 平均 TV | 最大 TV | 平均 KL（actor → native） | top-1 一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 token | 9.03% | 11.57% | 21.84% | 0.052930 | 7/8 |
| 12 token | 7.73% | 7.35% | 17.89% | 0.027301 | 7/8 |

概率均由实际原始 logits 转为 FP32 后，以温度 1 做 softmax；不比较采样后文本或 greedy 的 one-hot 分布。
另将保存的 actor logits 计算 log-softmax 并按原存储精度 BF16 舍入，与实际 old-logprob 输出逐字节一致，
核实预测位置与输出边界。记录见 `actor_logit_boundary.json`。
跨运行时 TV/KL 不把 actor 的 BF16 logprob 存储舍入误差混入模型差异。

这两条短输入共 16 个位置，2 个位置的 top-1 不同；结论是当前默认路径存在实质数值差异，
不是完整真实数据集上的准确率、训练效果或接收率评估。

另准备独立精度实验：仅在专用 vLLM 测试实例中，将 FP32 参数值经 BF16 舍入后恢复为 FP32 存储，
保持计算实现不变，检验误差是否缩小。测试后恢复内存参数，不写入源 checkpoint。
这个实验不等同于完整复刻 actor 精度策略，也不自动构成生产修复方案。

该实验已完成，16 个 worker 各处理 406 个 FP32 参数；计算实现和参数存储 dtype 不变。
控制后的最终 HC head 三项参数值与 actor 导出值一致。结果如下：

| 输入 | 辅助 HS 相对 L2 范围 | pre-HC 相对 L2 | 平均 TV | top-1 一致 |
| --- | ---: | ---: | ---: | ---: |
| 16 token | 10.34%–13.27% | 10.18% | 9.37% | 6/8 |
| 12 token | 7.11%–9.22% | 7.17% | 5.57% | 8/8 |

部分指标有所改善，但差异仍未消失，top-1 也没有一致改善。
因此仅把这些 FP32 参数值经 BF16 舍入，不足以实现对齐；尚不能把剩余差异归因于某个具体算子或 FSDP2。
后续若继续定位，需要逐层检查实际有效参数、精度策略、HC/attention/MoE 计算与批处理/padding 边界。
源 checkpoint 未修改；实验结束前执行内存参数恢复。

## 重复性校准

默认精度的原生实例再次加载后，在同一实例中将两条固定输入各重放两次：
辅助 HS、pre-HC 和 logits 的最大误差均为 0；与首轮独立进程采集相比也均为 0。
actor 本次与此前保留的特征同样逐字节一致。
因此本轮跨运行时差异是可重复的，不能用这几次运行中的随机波动解释。
记录见 `native_repeatability.json`、`actor_repeatability.json`。

## 实现及记录

- 探针：`verl_speco/probe_deepseek_v4_feature_alignment.py`。
- actor logits 导出：现有 `probe_deepseek_v4_actor_features.py` 的独立验证路径。
- 首次额外 logits RPC 因通用通信层将 Tensor 转为列表而失败；改用 safetensors 二进制传输后，
  真实 vLLM UtilityResult 序列化检查通过，原生采集正常退出。未修改模型计算来规避探针错误。
- 服务器统一目录：`/data/zzc/speco_validation_archive/actor_native_alignment_20261112/`。
  `alignment_features_initial.json` 保存首轮数值，`output_head_precision.json` 保存实际 head 差异；
  `pipeline_status.json` 保存完整 actor 重跑及比较的 argv、时间和退出状态。
  `native/alignment_report.json` 为默认精度报告；
  `native_bf16_rounded/alignment_report.json` 为参数精度控制报告；
  `precision_status.json`、`repeatability_status.json` 保存其余启动及退出记录。

本轮检查：指标/恢复参数的 CPU 单测与真实 vLLM 序列化测试共 **7 passed**；
小模型 NPU hook 检查、完整 actor、默认原生采集、精度控制及重复性采集均正常退出。
这里的正常退出表示测量完成，不代表两端数值等价通过。

**当前结论：默认路径和参数精度控制实验均已测得未消除的差异；不能将当前 actor HS 默认视为与原生推理等价的训练输入。**
本轮只完成对齐诊断，没有修改生产训练前向，也没有重新开启 rollout 采集。
