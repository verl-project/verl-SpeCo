# .242 V1 OPD 启动脚本

目标主机 `7.150.15.242`。参考原脚本：
`/efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_drafter_with_train.sh`。
原文件保持不变。

真实代码仓：`/efs_rl/z00886395/codes/test_speco/verl-SpeCo-specopd`，分支 `specopd`。
两个脚本都先 `cd` 到该目录，并将其放到 `PYTHONPATH` 首位，避免导入旧 v5 仓。

最终脚本已落地。启动命令（在具有 `/vllm`、`/vllm-ascend` 和完整 verl/torch 环境、挂载同一 EFS 的训练容器内执行）：

```bash
bash /efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_draft_opd_v1_with_train.sh
bash /efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_spec_opd_v1_with_train.sh
```

两者各自默认使用原脚本的全部 16 die；上面两条命令用于分别启动实验。
模型、17k/AIME 数据、rollout TP=2、batch=64、n=5、prompt=512、response=8192、200 个训练步与保存/测试频率沿用原脚本。
title、日志和 checkpoint 输出名按方法区分。用户额外 Hydra 参数仍放在最后。

| 配置 | Draft-OPD | SpecOPD |
|---|---|---|
| trainer | V1 sync，`use_v1=true` | V1 sync，`use_v1=true` |
| 方法选择 | `drafter_opd.method=draft_opd` | `drafter_opd.method=spec_opd` |
| 草稿损失 | PR 基线 E2E-TV | 完整 rollout 分支的 E2E-TV |
| draft sampling | probabilistic，T=1 | probabilistic，T=1 |
| DSpark loss mode | full_vocab / fixed_exp_decay | full_vocab / fixed_exp_decay |
| actor/ref SP | 4 | 4（Ulysses 联合评分） |
| actor attention | 保留原路径，remove_padding=true | 普通 PPO/ref 用 FlashAttention；联合评分用 SP 树 attention，remove_padding=true，fused/compile=false |
| 草稿更新调度 | step=20、采集/训练间隔=5 | step=20、采集/训练间隔=5；20 步后同步发布一次 |
| SpecOPD 采集 | 不使用该通道 | front 512-token 窗口，每请求最多 8 个完整 K=7 分支（56 proposal tokens） |

SpecOPD 联合评分 token budget=9216、dense mask budget=9216²，覆盖原脚本 512+8192 主干加 8 个分支的物理行数；没有因默认 4096² mask 上限而静默截断主干或分支。
该设置只通过配置校验，完整长序列/16-die 内存与吞吐尚需正式实验验证。

`.242` 宿主机当前没有 torch 和完整 verl 配置，未安装或改变公共依赖；它用于接收代码和脚本。
两脚本已做 bash 语法检查；捕获实际 argv 后，在 `.44` 的完整训练环境完成 Hydra compose 和方法校验。
早期 V1 SP=1 的验收见 [V1 台账](debug44_v1_validation.md)；最新 actor/ref SP=4、每 5 轮训练 20 步的 10-step 结果及数值/DP-SP 检查见 [Ulysses 台账](debug44_ulysses_validation.md)。

源码部署 receipt：`/efs_rl/z00886395/codes/test_speco/specopd-v1-source-manifest.json`。
脚本与代码最终哈希已核对；原参考脚本哈希未改变。两个方法最新的 V1 sync/SP=4 十步硬件测试均退出 0，optimizer 累计均为 20→40；完整回归结果见 Ulysses 台账。`.242` 的完整生产训练未启动。


PERF-06 草稿训练批配置：SpecOPD新增 `training.specopd.training_microbatch_trajectories=4` 和 `training_microbatch_token_budget=16384`，从代码基础配置读取，两个现有启动脚本无需修改。实际会按主干padding预算拆小，长序列生产负载不能据短测断言始终四条一批；设1可恢复逐条累积。DraftOPD沿用原训练微批路径。


## ModelArts 启动入口填写

2026-11-16日志 `modelarts-job-9e0449e3-2d3f-4d67-ac5e-491137feef04-worker-0.log` 第120行实际命令为 `bash /efs_rl/z00886395/shell/speco-runping/bash run_qwen3-8b_drafter_dspark_draft_opd_v1_with_train.sh`，第196行报目录下的 `bash` 不存在，随后退出127，尚未进入Python训练。

同目录原 `run_qwen3-8b_drafter_dspark_drafter_with_train.sh`、base和两份OPD脚本都是 `#!/usr/bin/env bash`，内部使用 `PYTHONUNBUFFERED=1 python3 -m verl_speco.main --config-name=speco_v1_trainer`。两份OPD脚本语法检查通过，训练脚本内容不需要为此次127错误修改。

如果填写的是完整自定义启动命令，两个方法分别复制以下一条：

DraftOPD：
```bash
bash /efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_draft_opd_v1_with_train.sh
```

SpecOPD：
```bash
bash /efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_spec_opd_v1_with_train.sh
```

如果平台使用“代码目录＋启动文件”模式，代码目录使用 `/efs_rl/z00886395/shell/speco-runping`，启动文件使用对应的 `.sh` 文件名（不加 `bash` 前缀），例如 `run_qwen3-8b_drafter_dspark_draft_opd_v1_with_train.sh`。不要在启动文件中填写 `bash run_….sh`，也不要把目录拼成 `/speco-runping/bash`。

以上修正的是入口填写方式，不能据此确认已提交ModelArts任务配置被修改。尚未重新提交或启动ModelArts训练。自动诊断引用的11月12日AIV事件不是本次20:38入口失败的直接根因证据。


2026-11-17 bootstrap OOM修复：SpecOPD NPU/FSDP2启用既有选择性发布，先过滤冻结embedding/共享LM head，再逐项汇总到CPU，避免原先全量state汇总的2.32GiB临时分配。每项汇总后显式释放设备临时tensor；bootstrap恢复/版本校验、完整checkpoint及DraftOPD NPU路由保留。保持脚本gpu_memory_utilization=0.7，不靠降低该参数修复。针对性回归及1GiB余量/2.32GiB冻结embedding的真实HCCL检查已通过；额外完整五步大模型测试按用户加快交付要求提前结束，正式ModelArts配置待重试。
