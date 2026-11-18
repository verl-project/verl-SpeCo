# 本地与服务器文件一致性核对

核对日期：2026-11-13。
本地：`/Users/steven/Desktop/blue_zone/verl-SpeCo`。
服务器：`SLAI-Verl-Workspace:/data/zzc/verl-SpeCo`。

## 核对范围与差异

对仓库文件逐项计算 SHA256，排除 Git 数据、虚拟环境、构建产物、缓存、outputs 和 logs。
初次比较发现 1 个 Python 文件、4 份文档内容不同，服务器还缺少本地的周报汇总。
服务器另有历史 NPU dump 和 `fusion_result.json`，它们属于运行产物。

| 差异 | 核对和处理 |
| --- | --- |
| `trainer/deepseek_v4_resident_probe.py` | 本地有 actor 输入，服务器有旧 store 控制器对应的 teacher 初始化；合并两者，保留显式输入校验 |
| `airdrafter_integration_progress_20261112.md` | 以实测报告核对后的完成状态同步，移除已过时的同步失败说明 |
| `deepseek_v4_actor_features.md` | 核对 16 rank 完整 actor 报告，保留已完成的内部采集验证及其范围 |
| `deepseek_v4_offline_recovery.md` | 核对五阶段状态、恢复/冷加载汇总与真实 server 热更新报告，保留最新结论 |
| `deepseek_v4_weight_compatibility.md` | 保留最新验收状态和 rollout HS 抓取已撤回的说明 |
| `airdrafter_weekly_summary_20261112.md` | 补传服务器缺失的周报，并更新同步状态 |

## 核心脚本合并的边界

此前“本地版本正确”的判断不够完整：现有 `probe_deepseek_v4_online_rounds.py` 控制器发送
`store`，本地 worker 却只读取 `actor_features` 和 `actor_target_output`，直接覆盖会不兼容。

合并后，验证 worker 支持成对 actor 导出文件或已有的录制 store，并拒绝缺失、混用的输入。
actor 路径保留目标版本、contract 和采集 scope 检查，并先同步配对 target 权重；
旧 store 路径保留按 drafter revision 的检查，以及训练激活后初始化 teacher 的逻辑。
报告记录实际采用的来源。

本次调整限于诊断 worker，不重新安装 rollout HS 采集，不修改训练前向、反向或优化器算法。
新的 actor/store 兼容分派通过 CPU 回归检查；未重新运行完整 NPU 两轮常驻闭环。

## 验证与记录

针对 resident worker、训练周期、actor 输入、后端隔离和 MTP 的相关测试执行回归。
新增用例覆盖真实 store 分段及 SpeCo mask、成对 actor 文件加载顺序、旧版本输入拒绝、
缺失和混合来源拒绝。

DSpark 实现指纹保持 `73af47a5bafa08220e4417a74ba312d049f53a9e0561b359806f09d6b9111582`。
MTP 实现指纹保持 `bfd508c7d5b7c3ac42920153b523ca62914cfeb73b9ca5fb32e9ca49ddc4ee92`。
此前通过的 MTP 训练/恢复/导出验收报告保持原样。

服务器核对记录位于 `/data/zzc/speco_validation_archive/source_consistency_20261113/`，
包含变更前的小型源码快照、同步文件哈希、测试日志及最终文件比较结果。

## 最终结果

- 本地回归：166 passed、3 skipped；服务器同范围回归：169 passed。
- Ruff 检查通过；本次不涉及训练数值代码，因此未重跑 NPU 训练。
- 最终 SHA256 核对：506 个仓库文件内容一致，无本地独有的源码或文档。
- 服务器独有的 59 个文件均为 `extra-info/data-dump/` 历史诊断产物及 `fusion_result.json`，未删除。
- 详细比较结果为 `comparison_final.json`；服务器回归命令和结果为 `test_status.json`、`tests.log`。
