# GPU VAT 修改说明

Last updated: 2026-11-04

本次只修改 verl-SpeCo。目标是给 DFlash 和标准 anchor-as-first DSpark 增加训练端 VAT，不修改 vLLM，不实现 Draft-OPD、SPARK、NPU 适配或 bonus-anchor 训练。

## 修改与理由

| 位置 | 修改 | 理由 |
| --- | --- | --- |
| `backends/verification_aware.py` | 新增两种 backend 共用的 VATObjective | 两者共用接受前缀、衰减、KD、self-coverage 计算；本期不引入 trace planner 或通用 BlockPlan |
| `backends/dflash_trainer_backend.py` | 增加 `vat` 权重模式，CE 使用重锚定权重，叠加 VAT 辅助 loss | 保留已有前向和 anchor 采样路径；复用已有 student logits |
| `backends/dspark_trainer_backend.py` | 模拟使用包含 Markov bias 的 logits；重加权 CE/l1/LK；复用 confidence head | Markov bias 会改变 draft argmax；无需新增第二个接受 head |
| `integration/oldlogprob_layer_ids.py` | VAT 自动选择 `dflash_aux_plus_last` | 训练需要 target 最后一层 hidden 重算 teacher 分布，普通 DFlash CE 默认不采集它 |
| `trainer/base_trainer.py` | 热发布过滤 DFlash VAT accept head；透传并汇总 VAT 指标 | 推理模型不认识训练专用 head；已有发布路径共用该过滤函数，无需改推理端参数翻译 |
| `config/speco_base.yaml` | 增加 `training.vat` 配置 | 显式表达新增 loss 权重与温度；默认不开启 VAT |
| 测试与两个示例脚本 | 数值、梯度、backend、恢复、发布过滤测试；复用已有 GPU 示例 | 验证算法和接线；避免复制已有启动配置 |

## 算法与取舍

设 `j` 为 draft 索引，DFlash 的 slot 为 `j+1`，DSpark 的 slot 为 `j`。比较 draft 与 teacher 的 argmax，连续匹配数为 `a`；无效位置中断接受链。位置权重为 `mask * exp(-max(0,j-a)/gamma)`。权重不对接受数反传梯度。

这是 **argmax 接受代理目标**。当 target 随机采样时，它不等于真实验证结果。这里不实现 probabilistic 模拟；配置该模式会报错。teacher 使用原始 lm_head 分布，不重放 rollout 的温度、top-p 或 penalties。KD 温度独立于 rollout 温度。

DFlash 总 loss 为既有 `ce_alpha*CE + lk_alpha*LK` 加 `kd_weight*KD + accept_head_weight*BCE + self_cov_weight*self_cov`。KD 为 `KL(teacher || draft)`，使用温度 T 并乘 T²；CE 与 KD 使用重锚定权重。独立 accept head 挂在训练 wrapper 的 `vat_objective` 下，标签为 `j<a`，BCE 使用未衰减的有效 mask。已有 front-position 配置继续作用于 CE/LK。

DSpark 保留既有 CE+l1/LK，并增加同样的 VAT 辅助项。默认不加 KD，避免默认 CE+l1 上再叠一项分布蒸馏。confidence head 的标签为当前 slot 的 argmax 是否匹配，只监督 `j<=a` 的有效位置；它替代已有 confidence loss，不重复叠加。此处保留条件预测形式，但二值代理标签仍不同于原来的解析接受率，这是 VAT 实验中的明确改动。BCE 使用未衰减的前缀 mask，CE/l1/LK/KD 使用重锚定权重。

self-coverage 默认关闭。开启时，对有效连续 draft 前缀计算 `P_j = product(q_i[data_label_i])`，对 `j>=a` 累加 `-log(max(P_j,floor))`，除以最大 draft 数，再对有有效首位置的 block 求平均。全接受块贡献为零。DFlash 按本地 VAT 参考公式实现；DSpark 将同一公式用于不含 anchor 的 slots，是适配方案，不是已验证的论文结果。

teacher logits 按 `vat.chunk_size` 分块，并使用 activation checkpoint；student logits 复用 backend 已有全词表 CE 路径。本次没有融合已有 LK/l1 与 VAT 的 teacher 计算，因此同时开启时会有重复计算；实际显存和耗时需 GPU 测量。

## 配置和运行

在仓库根目录执行，模型与数据路径沿用基础示例的环境变量：

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export DRAFTER_PATH=/path/to/drafter
export TRAIN_FILE=/path/to/train.parquet
export TEST_FILE=/path/to/test.parquet
export CKPTS_DIR=/path/to/checkpoints

bash examples/run_qwen3-8b-vat_drafter_dflash_vllm.sh
# DSpark 使用对应的 DSpark checkpoint：
bash examples/run_qwen3-8b-vat_drafter_dspark_vllm.sh
```

两个脚本设置 GPU MRv2 环境变量，调用基础 GPU 示例并传入 VAT 参数。GPU 数、TP 等沿用基础示例；请按机器资源调整基础示例已有参数。DSpark 脚本设置 `dspark_confidence_head_alpha=1`，让现有模型构建逻辑启用 confidence head；其 VAT loss 系数由下表控制。

| `actor_rollout_ref.rollout.drafter.training.vat` 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `kd_weight` | null | null 时 DFlash=1、DSpark=0；可显式指定 |
| `kd_temperature` | 2.0 | KD 温度 |
| `accept_head_weight` | 1.0 | DFlash accept / DSpark confidence BCE 系数 |
| `self_cov_weight` | 0.0 | 可设 0.6 做 self-coverage 消融 |
| `self_cov_floor` | 1e-6 | self-coverage 前缀概率下限 |
| `accept_mode` | greedy | 当前仅支持该代理模式 |
| `chunk_size` | 1024 | teacher 词表投影的行分块数 |

开启条件是对应算法的 `dflash_per_position_loss_weight=vat` 或 `dspark_per_position_loss_weight=vat`。该单值配置使 VAT 与 D-PACE 互斥；VAT 要求 full_vocab、有限正 gamma，以及 target final hidden。DSpark 要求 confidence head，且拒绝 bonus-anchor checkpoint。

例如开启 self-coverage：

```bash
bash examples/run_qwen3-8b-vat_drafter_dflash_vllm.sh \
  actor_rollout_ref.rollout.drafter.training.vat.self_cov_weight=0.6
```

指标位于原 backend 的指标前缀下：`vat/ce_loss`、`vat/kd_loss`、`vat/accept_loss`、`vat/self_cov_loss`、`vat/mean_accepted`。它们按 rank/microbatch 诊断次数求平均；`mean_accepted` 不含 target 额外提交 token，不应直接与含 bonus 的 rollout AL 混为同一指标。

DFlash 的 accept head 包含在训练 wrapper state dict 中，可随完整训练 checkpoint 恢复；HF draft 导出不包含它，热发布也过滤它。HF 导出不是包含训练 head 的恢复 checkpoint。

## 验收与限制

CPU 数值测试将分块 VAT loss 和梯度与独立的稠密公式比较，覆盖首拒、部分接受、全接受、mask 中断、空有效行及配置错误。backend 测试覆盖两种算法的实际前向/反向、空有效块、state dict 恢复，并用仅 Markov bias 改变 argmax 的样例验证 DSpark 接受模拟。发布测试调用真实过滤方法，检查训练 head 被排除。

本机为 PyTorch CPU 环境。测试导入只读使用本地 `Draft-OPD/verl`，未修改该目录。这不能替代项目指定 verl release 与 GPU 环境的验收。

尚需在 GPU 环境执行：FSDP2 多 rank 空有效块、完整训练 checkpoint 恢复、vLLM 热发布与 rollout，以及 B0/V1/V2 的真实 AL、耗时和显存比较。CPU 空行梯度测试不能证明 FSDP2 不挂起。当前没有 GPU 收益结论。


本机验证结果：VAT/D-PACE 单测与 backend 回归共 107 passed、1 skipped；发布过滤测试另有 1 passed。Ruff、Python 编译和 git diff --check 通过。示例完成 shell token 解析与 OmegaConf 参数合并检查；本机 Bash 不可用，未执行 bash -n 或实际训练启动。

复现 CPU 测试时需先安装项目指定的 verl 依赖，再运行：

```bash
python -m pytest tests/unit/test_vat_loss.py tests/unit/test_dpace_loss.py tests/integration/test_dspark_trainer_backend.py -q
python -m pytest tests/integration/test_selective_publish_state.py -k vat -q
```
