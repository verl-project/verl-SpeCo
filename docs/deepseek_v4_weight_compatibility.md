# DeepSeek V4 DSpark 权重与热更新验证

更新：2026-11-12。

当前服务器为 `/data/zzc/verl-SpeCo`。本轮 TP8/DP2 的完整恢复、标准 PyTorch
导出及独立进程冷加载已通过，详见 [离线恢复验收记录](deepseek_v4_offline_recovery.md)。
标准权重冷加载与原生 checkpoint 的全部持久参数和 cache 一致，固定训练输入的
HS/完整词表 logits 差值为 0。随后真实 HTTP/Ray server 的 TP8/DP2/EP IPC 发布
探针也退出 0，包含权重接收、准入边界、版本提交和更新后的贪心生成检查。
这些结果不代表完整 RL 自动训练发布闭环已完成。
下文 `/models/...` 命令及历史结果保留作记录，请勿把旧 checkpoint 当成本轮产物。

## 格式边界

DeepSpeed checkpoint 保留优化器、调度器、数据游标及 RNG，负责同 backend、同 TP/DP 拓扑恢复。标准草稿权重只用于参数交换，不替代恢复 checkpoint。原生 PyTorch 路径和默认 `--format patched` 完整模型导出保持原行为。

新增 `--format draft-safetensors` 和 `--format draft-pytorch`：复用已有 TP 合并、融合矩阵拆分和方向转换，选择一个 DP 副本，输出标准分片、索引、config.json 及 SpeCo metadata.json。写入临时目录，逐张量读回验证名称、形状、dtype 和数值，通过后才发布最终目录。不会复制完整 target。

权重键采用 DeepSeek V4 的 `mtp.*` checkpoint 布局，config 标记 `speco_draft_only=true`。这表示标准 Tensor 文件格式，不代表 Qwen/EAGLE 模型类或通用 AutoModel 能直接加载该草稿；实际推理仍须支持 DeepSeek V4 DSpark。PyTorch 训练循环跨 backend 恢复优化器不在此格式的承诺范围内。

```bash
python3 -m verl_speco.export_deepseek_v4_draft \
  --checkpoint /models/z00829645/speco_training_cycle_20261102_143322/train/checkpoints/step_00000001 \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --output /models/z00829645/speco_draft_step1_safetensors \
  --dp-rank 0 --format draft-safetensors
```

使用 `--format draft-pytorch` 可输出 PyTorch `.bin` 文件。目的目录必须不存在。脚本验证权重读回，尚不证明 NPU 推理正确。

标准参数读取复用分片索引，每次读取一个分片：

```python
from verl_speco.models.deepseek_v4.artifact import iter_draft_weights
for name, tensor in iter_draft_weights(path):
    # 使用支持 mtp.* 的模型加载器消费；避免默认一次汇集全部权重。
    pass
```

## 热更新边界

在线发布复用 SpeCo 现有 worker、快照、IPC、调度和版本提交，仅 DP0 提供 runtime TP 分片，要求训练/推理 TP 一致。runtime 参数已被 Ascend 后处理，不可直接用标准 checkpoint 矩阵覆盖。

发布 CPU 拷贝通过共享文件锁限流；leader 使用 mmap 读取 ZIP 格式的发布分片，降低一次读取全部分片的驻留内存。在线 DeepSpeed 保存同样使用受限流的同步保存接口。

接收端先校验所有 TP lane 的参数集合、形状、dtype、有限性；本 rank 数据复制到自有 CPU 暂存，避免 IPC bucket 复用。完整校验后原地写入，保留参数地址；写入异常尝试恢复本 rank 原值。这不是跨 rank 两阶段原子提交的替代：各 worker 完成、版本确认及异常后的暂停/恢复仍由既有发布流程负责，需真实硬件验证。

## 验收顺序

1. 保留当前成功 checkpoint，从 TP8/DP2 恢复到 global_step=2，使用新 RUN_DIR，保持其余训练配方一致。
2. 导出标准草稿，核对严格 TP replica 检查和全量读回。
3. 独立推理 worker 加载完整模型，发布一次真实训练权重，核对每个 worker 的版本和参数；不要求完整 RL。
4. 注入缺失、重复、非有限张量和错误 TP，确认拒绝更新、无部分写入；设备写入失败还要检查全局失败恢复。
5. 完成上述后再验证在线训练触发、发布与 target 版本绑定。rollout 直接采集 HS 已撤回；actor 特征的跨运行时数值对齐暂搁置。

CPU 测试覆盖格式、TP/DP 选择和接收端校验；跨 rank 热更新、真实恢复连续性和导出后 NPU 推理必须单独验收，不应以文件生成成功代替。

## TP8/DP2 恢复一步的 NPU 命令

进入同一 Ascend 环境后执行，保持原配方，仅将总步数上限改为 2。不得修改 TP/DP 或从标准导出目录恢复。

```bash
cd /models/z00829645/verl-SpeCo
set -euo pipefail
export PYTHON_BIN=python3 CARD_COUNT=16 TRAIN_TP=8
export MODEL_PATH=/models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16
export FEATURE_STORE=/models/z00829645/speco_training_cycle_20261102_143322/capture/features/target_step_00000000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RESUME_FROM=/models/z00829645/speco_training_cycle_20261102_143322/train/checkpoints/step_00000001
export RUN_DIR=/models/z00829645/speco_resume_dp2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
MAX_STEPS=2 bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh resume \
  draft_engine.target_runtime_version=0 \
  2>&1 | tee "$RUN_DIR/resume.log"
```

预期 training_report.json 中 global_step=2，且新的 step_00000002/speco_ds4.json 存在。恢复成功只证明状态可用；与不中断训练的数值连续性还应使用相同数据、随机状态比较。

目前在线路径仍要求 rollout engine DP1、PP1、无 EP 且训练/推理 TP 一致。独立采集的 TP8/DP2/EP 支持不等同于在线发布已经支持该拓扑。接入该 BF16 target 的在线热更新前，需要验证 target/proposer 的实际 EP 布局并处理 expert 重分片，不能删除现有限制来假装兼容。

## 本轮验证结果

10/30 服务器 CPU 回归覆盖 55 个不同测试，包含两种标准序列化、TP 合并/DP 选择、导出入口、恢复状态和热更新校验。Ruff 与四个源文件的 mypy 检查通过。

实际 TP8/DP2 step_00000001 已选 DP0 合并导出，2,375 个张量逐一验证数值、形状及 dtype，通过后发布到 `/models/z00829645/speco_weight_validation_20261030T074831Z/draft_step_1`，约 37 GiB。日志为同目录上级的 export.log。尚未执行导出后的 NPU 推理、真实恢复 step2 或在线热更新；inference_validated=false。

## TP 训练分片到 EP 推理专家的独立验证

新增 `probe_deepseek_v4_draft_update`。生产接收器中的专家组装器读取实际 `expert_map`，仅暂存本 rank 专家，并分别拼回 w13 的 gate/up 两半及 w2 中间维度。仅支持 BF16 input-major、静态一对一专家映射；EPLB、冗余/融合共享专家、已缓存的 expert weight lists 仍拒绝，不能绕过校验。普通 attention/共享层仍要求训练与推理 TP 一致。

独立入口复用已有 TP8/DP2/EP 模型启动与调度，各 DP 前端均发送更新 RPC；原特征抓取 RPC 仍仅由 DP0 返回。每个 worker 从只读 checkpoint 选择 DP0 训练分片，处理参数别名，暂存并核验完整集合。全部准备成功后提交，逐张量 SHA256 比对准备值与参数读回；全部提交且同一轨迹的前后 8 个贪心 token 一致后才释放回滚快照。失败时协调所有 worker 尝试恢复，进程或设备死亡时不承诺回滚成功。

```bash
cd /models/z00829645/verl-SpeCo
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PROBE_ROOT=/models/z00829645/speco_ep_update_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROBE_ROOT"
python3 -m verl_speco.probe_deepseek_v4_draft_update \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --checkpoint /models/z00829645/speco_training_cycle_20261102_143322/train/checkpoints/step_00000001 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$PROBE_ROOT/report" \
  --tensor-parallel-size 8 --data-parallel-size 2 --max-model-len 4096 \
  2>&1 | tee "$PROBE_ROOT/update.log"
```

目标为 report/update_report.json 的 status=passed，每个 worker 有 prepared/committed 确认，expert_tensors 非零，target_tokens_unchanged=true。这个入口使用只读训练分片和 RPC 控制，不测试 SpeCo 的 IPC 权重运输，也不运行 RL，所以报告中 ipc_validated 与 rl_integration_validated 均保持 false。单条轨迹的贪心一致性不证明全部推理数值等价。

先验证真实 EP 参数和热更后生成，再验证完整 IPC/版本提交及 verl 在线 DP 调度。生产在线的原有 no-EP/engine-DP1 检查暂不解除；独立验证命令不需要修改这些限制。没有替换训练反向，也没有开启训练 EP。

本轮服务器 CPU 回归 96 项通过；真实 TP8 checkpoint 的六个专家矩阵已完成 256 全局专家到 16 本地专家的非连续映射转换，8 个 TP lane 分别核验通过，并验证回滚恢复零初始值。结果为 `/models/z00829645/verl-SpeCo-backup-20261030T083615Z/cpu_real_expert_conversion.json`。另新增复制中途失败回滚、EPLB 和非 BF16 拒绝测试。真实 NPU 热更仍待执行上述命令。

## 真实 IPC 分桶传输和 runtime revision 验证

10/31 独立 checkpoint-RPC 冒烟已在 TP8/DP2/EP 下通过：16 worker 各准备 6 个专家矩阵并更新 74 个参数，前后 8 个贪心 token 相同。该结果不含 IPC。

下一项使用相同命令增加 `--transport ipc`。此分支选择可选 worker extension，继承并调用 SpeCo 的生产 `update_draft_weights_from_ipc`；发送端使用已安装 verl 的 BucketedWeightSender，不修改上游源码。各 DP/TP worker 使用独立通道，模型更新期间不发起生成请求，所有接收 RPC 完成后才继续。

最大的 BF16 runtime 分片为 1 GiB，发送 bucket 按真实最大张量容量设置，16 通道约需 16 GiB 空闲 /dev/shm。生产 SHM 发布也增加按最大 Tensor 自动扩容的策略，小张量保留原 bucket 配置。新冒烟为每个发送/接收 socket 配置收发超时，默认 600 秒；接收器的新增 timeout 参数默认 None，保留旧调用行为。

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PROBE_ROOT=/models/z00829645/speco_ipc_update_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROBE_ROOT"
python3 -m verl_speco.probe_deepseek_v4_draft_update \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --checkpoint /models/z00829645/speco_training_cycle_20261102_143322/train/checkpoints/step_00000001 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$PROBE_ROOT/report" --transport ipc \
  --tensor-parallel-size 8 --data-parallel-size 2 --max-model-len 4096 \
  2>&1 | tee "$PROBE_ROOT/update.log"
```

验收 update_report.json：status=passed、ipc_validated=true、runtime_revision_validated=true，16 worker 的 runtime_revision 均恰好增加 1，参数 SHA256 与 checkpoint 重分片参考值一致，target_tokens_unchanged=true。失败时尝试恢复所有 worker 的权重及原 revision，保存 failure.json；设备/进程死亡时不能保证恢复成功。

该入口不经过 Ray 发布快照、HTTP server 的 set_global_steps 或完整 RL 调度，因此 server_version_commit_validated 与 rl_integration_validated 仍为 false。普通 checkpoint-RPC 模式和生产在线拓扑限制保持原样。需先验收真实 IPC，再处理 server 版本提交和 verl engine-DP2 在线采集。

10/31 服务器 CPU 回归 157 项通过，包含已安装 verl 0.9.0 的真实跨进程 ZMQ/SHM 传输、多 bucket 复用和回调失败超时退出。日志为 `/models/z00829645/verl-SpeCo-backup-20261031T075536Z/cpu_ipc_validation.log`。另将真实传输测试接入完整生产接收方法，以检查实际 runtime revision 和原地参数更新；NPU IPC 分支仍待用户执行。

完整生产接收组合测试的两项 CPU 用例也已通过：跨进程真实 SHM/ZMQ -> 生产 update_draft_weights_from_ipc -> EP 组装与原地写入 -> revision=1；注入回调失败后参数及 revision 保持原值，发送端超时退出。仅 CPU 设备与 TP 查询使用替身，运输和生产接收方法使用安装版本/仓库原实现。日志为 `/models/z00829645/verl-SpeCo-backup-20261031T080031Z/cpu_production_ipc_validation.log`。同时只读检查了真实 NPUWorker，新的可选 extension 无方法名冲突。两项组合用例是前述 157 项测试的加强复验，不额外累计测试数量。

## 真实 HTTP/Ray server 的版本提交与准入边界

10/31 TP8/DP2/EP 的真实 IPC NPU 冒烟已通过，报告为 `/models/z00829645/speco_ep_update_20261031_164141/report/update_report.json`：16 通道完成，每 worker 更新 74 个参数，runtime_revision=1，8 个贪心 token 不变。

新增 `probe_deepseek_v4_server_update`，复用安装版本 verl 0.9.0 的完整 standalone HTTP server 构造、启动和 generate，以及 SpeCo 的 Ray 启动兼容钩子、统计桥、准入 gate 和生产 IPC 接收器。不创建 RL trainer、reward 或训练 Actor，不重新训练和抓取特征。通过 Ray 为 server 预留 16 个 NPU，OpenAI HTTP 服务仅监听本机回环地址。

新提交接口要求入口关闭、在途请求为零、target 版本未变且旧 drafter revision 匹配，才能将 drafter revision 增加一次。target 的 global_steps 保持 0，不因草稿更新变成 1。任何提交阶段失败都不恢复普通请求入口，由调用方处理回滚/关闭服务；旧版本 setter 保留原行为。

vLLM 0.26 异步 DP utility 默认广播所有引擎但只返回 DP0 结果。新入口对已核对的私有 client 能力做结构检查，显式收集所有 DP 结果；不支持该结构时拒绝成功，不退回只检查 DP0。

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export SERVER_ROOT=/models/z00829645/speco_server_update_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SERVER_ROOT"
python3 -m verl_speco.probe_deepseek_v4_server_update \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --checkpoint /models/z00829645/speco_training_cycle_20261102_143322/train/checkpoints/step_00000001 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$SERVER_ROOT/report" \
  --tensor-parallel-size 8 --data-parallel-size 2 --max-model-len 4096 \
  2>&1 | tee "$SERVER_ROOT/server.log"
```

验收 report/server_update_report.json：status=passed、http_health_validated=true、server_version_commit_validated=true、admission_blocked_during_update=true。更新时排入的普通请求必须实际到达 gate 并等待，恢复后输出 target/global_steps=0、drafter revision min/max=1，且与更新前贪心 token 一致。worker 仍要求全部 DP/TP 确认和逐张量读回。

这个实验包含真实 HTTP health、Ray generate 和控制 RPC，不验证所有 OpenAI 文本路由。完整 RL 的自动采集/训练/发布仍为 rl_integration_validated=false；生产在线 engine-DP2/EP 的采集限制尚未解除。

本轮服务器已通过 143 项 CPU 回归和原生配置/Ray 方法注册预检。预检发现 native tokenizer 包含 129,283 个条目，模型容量为 129,280；server 探针用代理保留 native tokenizer 的编码行为，但向原生权重掩码报告模型容量，避免把 tokenizer 额外条目当作可生成 logits。该适配只作用于新探针的 HFModelConfig，不修改模型、tokenizer 文件或训练数据。额外固定 enable_fused_mc2=false，保持本轮已验证的 BF16 静态专家张量布局。NPU server 启动和提交尚待上述命令验收。

## 2026-11-02：真实 server 热更新已通过

服务器 `/models/z00829645/speco_server_update_20261102_101723/report/server_update_report.json`
为 `passed`。TP8/DP2 的 16 个 worker 均确认 IPC 提交，每个更新 74 个张量；
server drafter revision 从 0 变为 1，target version 保持 0。更新期间普通请求到达
准入 gate 并等待，恢复后版本标签正确，8 个贪心 token 与更新前一致，HTTP health 通过。
退出阶段仍有 worker 超时清理警告；报告不代表完整 RL 或常驻在线 trainer 已验收。

## 自动采集、训练一步、回传的分阶段闭环

`probe_deepseek_v4_training_cycle` 自动串联现有实现：

1. 启动真实 verl standalone rollout server，在每个 DP 引擎各执行一次普通 `generate`；
   直接在该次前向中采集 HS/pre-HC，不 replay。使用准备文件第一条记录的首个
   `loss_mask=1` 之前的 token 作为 prompt，生成 64 token，原轨迹答案不参与监督。
2. 使用现有 SpeCo mask（prompt=0、已前向的 response=1），写入版本为 target 0 的
   TorchShardFeatureStore。终止采样 token 没有前向特征，因此不进入样本。
3. 释放推理进程，用原有 separate-training 脚本完成 CPU preflight 和 TP8/DP2
   训练一步（512-token segment、4 anchors、microbatch=1）。从实际训练日志验证
   loss/指标有限、梯度覆盖无缺失/非有限值、optimizer probe delta > 0。
4. 自动把本轮新 checkpoint 传给已通过的 server IPC 探针，验证准入、版本提交和
   更新后生成。不会引用之前的 smoke checkpoint。

DP 采集复用 server 的共享 begin/finish/cancel 逻辑；RPC 收齐每个 DP 引擎的回复，
避免 vLLM DP utility 只返回 DP0 结果导致 DP1 的样本丢失。未执行请求的 DP leader
返回 None，仍只允许一个非空 payload。PyTorch/单 DP 默认路径保持原有行为。

在当前 16 卡机器上执行（不启动完整 RL）：

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CYCLE_ROOT=/models/z00829645/speco_training_cycle_$(date +%Y%m%d_%H%M%S)
python3 -m verl_speco.probe_deepseek_v4_training_cycle \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$CYCLE_ROOT" \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --max-model-len 4096 --response-tokens 64 \
  2>&1 | tee "${CYCLE_ROOT}.log"
```

加 `--dry-run` 仅检查本地模型/轨迹契约并输出命令计划，不启动 Ray/NPU，不创建输出目录。
运行目录必须不存在。各阶段日志为 capture.log、preflight.log、train.log、update.log；
失败即停止并写 failure.json，保留全部产物供定位。成功标志是 cycle_report.json
的 status=passed、rollout_capture_validated=true、hardware_training_validated=true、
server_version_commit_validated=true。

本轮只采两条短回复并训练一个 optimizer step，但完整 DeepSpeed checkpoint 仍约
350 GiB，保存可能耗时十分钟以上。推理和训练分阶段共享 16 卡，服务会在训练前关闭并
在训练后重启。它验证自动的数据/权重闭环，**不验证**常驻在线 trainer 的调度/优化器
跨轮复用、服务全程驻留、完整 RL 或接受率提升。在线 trainer 的 DP=1 限制仍保留，
不会根据此分阶段探针直接解除。


## 2026-11-02：分阶段闭环通过与产物清理

`/models/z00829645/speco_training_cycle_20261102_143322/cycle_report.json` 为 passed。
两路 DP 各生成 64 token 并原位采集；训练 loss=0.83054185，71 个已审计参数的
梯度均有限且非零，optimizer probe delta=0.0001220703125。16 个 worker 均提交
本轮新权重，每个更新 74 个张量，版本 0→1，生成 token 一致。服务在训练前关闭并
在训练后重启；常驻在线 trainer 和完整 RL 仍未验收。

旧 TP8/DP1、TP8/DP2 smoke checkpoint 与 10/30 的导出权重已由最新闭环替代并清理。
历史日志/报告及旧增量代码备份归档于服务器 `speco_validation_archive/`；旧日志中的
绝对路径仅用于追溯。上文命令已改用最新保留的 checkpoint；恢复必须同时使用其
原始 capture/features/target_step_00000000，并声明 target_runtime_version=0。
旧标准导出目录已移除，需要时可用本文导出命令从最新 checkpoint 重新生成。

## 两轮常驻 trainer 验证（新增，NPU 尚待验收）

`probe_deepseek_v4_online_rounds` 在一个 torchrun 作业中构造完整的
`build_online_trainer`，显式建立 TP8/DP2 进程组，跨两轮保留同一 DeepSpeed engine、
Adam 优化器及调度器对象。第一轮使用 revision 0 的真实 rollout 特征；发布 revision 1
后，再从更新后的服务直接采集第二轮特征。target 固定为版本 0，第二轮 optimizer
累计步数必须到 2。每轮训练窗口取样本末尾 512 行，保留绝对位置及 SpeCo mask。

16 卡分时使用：推理服务在训练前关闭；trainer 使用独占的 Ascend CaMem 内存池，
训练后将管理的分配暂存到 CPU，唤醒时映射回原 NPU 地址。trainer 进程和优化器对象
不重建。此原生内存池路径首次用于本项目训练，**CPU 测试不构成 NPU 验收**。
代码检查参数/优化器存储是否受内存池管理、容器可用内存、暂存后每卡至少 50 GiB
空闲，以及唤醒后的地址、参数/Adam 状态采样摘要、步数及调度器状态。摘要是有界
采样，不代表逐字节校验整个优化器。检查失败即停止，不启动下一轮服务。

复用既有 runtime 分片 writer 和生产 IPC 接收器。仅保存两份 DP0 runtime 权重快照，
预计合计约 74 GiB；不保存新的完整 DeepSpeed 恢复 checkpoint。快照 metadata 明确
标记 recovery_checkpoint=false，不能用于恢复优化器。作业结束后内存中的优化器
会释放，最新原有成功恢复 checkpoint 保持不变。

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ONLINE_ROOT=/models/z00829645/speco_online_rounds_$(date +%Y%m%d_%H%M%S)
python3 -m verl_speco.probe_deepseek_v4_online_rounds \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$ONLINE_ROOT" \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --max-model-len 4096 --response-tokens 64 \
  2>&1 | tee "${ONLINE_ROOT}.log"
```

无需全局修改 PYTORCH_NPU_ALLOC_CONF：控制器仅为 trainer 子进程设置
expandable_segments:False，适配 CaMem 内存池；推理服务沿用原环境。加 --dry-run
只校验模型/轨迹并输出计划，不启动 Ray/NPU，也不创建输出目录。

验收 `online_rounds_report.json`：status=passed、optimizer_steps=[1,2]、
drafter_revisions=[0,1,2]、engine_optimizer_scheduler_reused=true、
address_preserving_park_wake_validated=true。`result_01.json` 与 result_02.json
保存全部 16 个 worker 的 PID、对象标识、步数、暂存容量及训练指标；trainer.log
记录梯度覆盖和实际执行。最终服务顺序装载 revision 1、2，并验证每次更新期间的
准入阻塞和生成一致性。失败查看根目录 failure.json、trainer_failure_rank_*.json
以及对应 serving_N.log。所有新产物集中在 ONLINE_ROOT 一个目录。

默认生产在线 trainer 的 engine DP=1 限制不变；DP2 仅由此诊断入口显式启用并验证
训练组拓扑，不将尚未验证的生产 worker 路由一起解禁。仍不涉及 RL scheduler、
actor target 更新或服务始终驻留，也不承诺接受率提升。

## 2026-11-03：第一轮暂存通过，隔离训练与推理 HCCL 端口

`speco_online_rounds_20261103_140340/result_01.json` 记录 16 个 worker 全部完成
optimizer step=1、结构/数据状态检查及 CaMem 暂存；唤醒与第二轮尚未验证。
随后 serving_1 启动失败，HCCL 报 EI0020：NPU 网络端口 16666 已被常驻 trainer
绑定。暂停/暂存不会释放该进程的 HCCL 监听器。

两轮入口现为不同角色设置独立范围（不改变其他入口的默认环境）：

| 角色 | HCCL_NPU_SOCKET_PORT_RANGE | HCCL_HOST_SOCKET_PORT_RANGE |
| --- | --- | --- |
| trainer | 61000-61099 | 61100-61199 |
| serving | 62000-62099 | 62100-62199 |

环境写入子进程、Ray actor runtime_env 及 actor 构造参数。服务启动后使用现有
all-DP RPC 检查全部 worker 的实际环境，保留原 Ray 导入兼容 setup hook。
`--trainer-hccl-port-base`、`--serving-hccl-port-base` 可调整起始端口；每个角色
需要连续 200 个端口，两角色区间不可重叠，错误配置在启动前拒绝。
原启动命令不变。新的端口隔离路径需继续做 NPU 验证。
参见[昇腾 HCCL 端口配置说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclenvref_07_0021.html)。


## 2026-11-03：第二轮 CaMem 回调失效修复（待 NPU 重测）

`speco_online_rounds_20261103_143956` 已完成第一轮 optimizer step=1、暂存、
serving_1 的 16-worker IPC 更新（revision 0→1）、固定 target token 校验及新特征采集。
这次 HCCL 角色端口隔离通过。第二轮唤醒后，日志先出现
`TypeError: 'tuple' object is not callable`，随后 DeepSpeed 梯度缓冲区分配报 OOM。

核对服务器 `vllm-ascend/csrc/camem_allocator.cpp`：`init_module` 仅保存 Python
回调的借用指针，不增加引用；malloc 回调调用失败会返回 nullptr，表现为分配失败。
仅保留 CaMem 单例不能保留每次属性访问生成的绑定方法；首次 `use_memory_pool`
上下文结束后，这些临时方法会失效。

修复仅在 `trainer/ascend_state_parking.py` 的常驻 Ascend 诊断路径生效：在首次
注册前，将 malloc/free 的确切绑定方法保存在 allocator 实例属性上，跨上下文及
parking helper 生命周期持续持有。无需修改或重编译 vLLM-Ascend，不调整 TP8/DP2、
模型、mask、优化器算法或显存比例，也不影响原生 PyTorch 路径。

CPU 回归复现未持有时方法在上下文退出后被回收，并验证修复后经历 GC、暂存、唤醒、
第二轮分配和 optimizer step=2 后回调仍存活。修复后的真实第二轮须用新的
`ONLINE_ROOT` 重跑上方命令，不能将 CPU 回归通过视为 NPU 两轮闭环通过。


后续 `speco_online_rounds_20261103_160030` 已完成第二轮优化器更新和 revision 2
保存，暂存检查因池外梯度范数失败。状态分类修正、结构性限制及低成本验证顺序
见 [Ascend 状态生命周期](ascend_state_lifecycle.md)。先跑小模型分配器诊断，
通过后再运行完整模型；目前不能宣称两轮闭环或生产同卡 RL 已通过。
