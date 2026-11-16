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
| actor/ref SP | 保留 4 | 均为 1，适配 dense actor/ref 路径 |
| actor attention | 保留原路径 | eager，remove_padding=false，fused/compile=false |
| 草稿更新调度 | 保留 step=20、间隔=5 | 每轮单次更新，采集/训练间隔=1、同步发布 |
| SpecOPD 采集 | 不使用该通道 | front 512-token 窗口，每请求最多 8 个完整 K=7 分支（56 proposal tokens） |

SpecOPD 联合评分 token budget=9216、dense mask budget=9216²，覆盖原脚本 512+8192 主干加 8 个分支的物理行数；没有因默认 4096² mask 上限而静默截断主干或分支。
该设置只通过配置校验，完整长序列/16-die 内存与吞吐尚需正式实验验证。

`.242` 宿主机当前没有 torch 和完整 verl 配置，未安装或改变公共依赖；它用于接收代码和脚本。
两脚本已做 bash 语法检查；捕获实际 argv 后，在 `.44` 的完整训练环境完成 Hydra compose 和方法校验。
V1 的短步硬件验收及失败历史见 [V1 调试台账](debug44_v1_validation.md)。

源码部署 receipt：`/efs_rl/z00886395/codes/test_speco/specopd-v1-source-manifest.json`。
脚本与代码最终哈希已核对；原参考脚本哈希未改变。两个方法的 V1 两步硬件测试均退出 0，最终 focused 回归 159 passed。`.242` 的完整生产训练未启动。
