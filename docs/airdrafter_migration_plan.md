# AIRDrafter DeepSeek V4 0731 训练方案迁移计划

Last updated: 10/20/2026

## 1. 目标与当前状态

在 verl-SpeCo 中通过 `actor_rollout_ref.rollout.drafter.training.backend` 选择训练实现：默认 `pytorch` 保留现有路径，新增 `vllm_ascend` 复用 AIR 已验证的模型、前向、反向、损失和 DeepSpeed 更新逻辑。SpeCo 继续管理启动、资源、数据、调度、日志、恢复流程与权重发布。

本计划基于两份本地源码阅读。当前已实现离线训练、rollout 特征回传、随 RL 自动训练及权重发布，并将所需数值实现迁入 SpeCo，按现有目录维护；无需外部 AIRDrafter 路径。原有 PyTorch、离线与其他运行时功能保留。真实 DeepSeek V4 0731 权重、Ascend 前反向及在线热更新仍待 NPU 验收。当前入口见 [在线闭环说明](vllm_ascend_cotrain.md) 和 [冒烟测试指南](deepseek_v4_0731_draft_smoke.md)；迁入范围见 [实现来源说明](deepseek_v4_backend_provenance.md)。下文保留原始设计依据，实际落地以这些文档和代码为准。

采集入口现已明确为 **verl 训练中的 rollout worker**：直接复用 target 前向，新增公共收集器和 sampler 适配，普通 PyTorch 路径接回原有 scheduler，DeepSeek V4 离线路径按 target 版本写 Feature Store；在线路径复用现有 collection、worker、scheduler 和发布通道自动训练、回传权重。新增持续 DeepSpeed 实例、完整 HC/norm/LM-head teacher 同步和运行时 TP 分片发布。CPU 契约与 worker 生命周期测试已加入，真实 NPU 闭环尚未验收；当前资源池仍沿用 SpeCo 原有布局，专用分卡资源池尚未接入。自动闭环配置见 [在线训练说明](vllm_ascend_cotrain.md)。详见 [rollout 内采集说明](rollout_feature_capture.md)。独立 Producer 的二次 prefill 优化不在本次范围。

- SpeCo 源码：本仓库。
- AIR 源码：`/Users/steven/Desktop/blue_zone/AIRDrafter-master`。
- 目标模型：DeepSeek V4 0731，使用与捕获对应的准确 checkpoint 和运行时。
- 交付顺序：离线 DeepSeek V4 0731 DSpark → 在线分卡闭环 → DS4 MTP → 其他 MTP 模型 → 按需支持同卡。
- 在线分卡首期采用同步训练周期与同步发布；持续异步执行另行评估。
- 保留当前 native backend，保证原有 Qwen DSpark 等训练路径继续可用。

在线与分卡是不同维度：在线描述消费当前推理产生的数据，分卡描述训练与 rollout 的设备分配。

## 2. 源码结论

| 能力 | AIR 当前实现 | 迁移意义 |
| --- | --- | --- |
| DSpark 训练 | `offline_training/train_mtp.py::run_dspark_training()` | 已有可调用入口，可封装完整训练周期 |
| MTP 训练 | 主要训练循环位于同一脚本的 `__main__` | 需要先提取可调用的训练执行函数 |
| 常驻训练 | `co_training/trainer.py::DSparkResidentPlanRunner` | 可更换 dataloader，保留模型、优化器和累计步数 |
| 特征采集 | 离线 DS4 DSpark capture 与在线 sampler hook | 复用模型相关提取、拼接和验证逻辑 |
| 教师头 | `training_common/modeling/lm_head.py::DS4TargetLogitsHead` | 包含 HC、norm、vocabulary head，不能仅同步普通 lm_head |
| 参数导出 | `loading_trained_mtp/loading_ds4.py` | 已有 TP 合并、参数映射和目标形状校验 |
| 在线快照 | `co_training/drafter_snapshot.py` | 可复用不可变、按 TP rank 捕获的快照逻辑 |

AIR 的 DSpark 是 DeepSeek V4 专用实现。当前 SpeCo 的 DSpark 训练模型是 Qwen/DFlash 风格，两者不能仅通过切换 backward 函数互换。

AIR 已有在线采集与发布协议接口，但其 `co_training/README.md` 仍列出真实在线 hook、远程传输和完整闭环待验证项。已有接口与文档中的验证记录不能替代本次集成验收。

## 3. 架构与复用原则

```text
数据来源
├─ AIR 现有离线捕获文件
└─ SpeCo 管理的 vLLM-Ascend 在线 rollout
         │ 模型相关采集适配
         ▼
SpeCo DraftFeatureSample / Feature Store
         │
SpeCo 文件读取 / TransferQueue / 背压 / 预取
         │
SpeCo Scheduler → 独立 Drafter Ray WorkerGroup
         │
训练引擎接口
├─ pytorch：委托现有 SpeCo 实现
└─ vllm_ascend：AIR 模型、loss、前向、反向、DeepSpeed
         │
         ├─ DeepSpeed 恢复 checkpoint
         ├─ 离线推理权重导出
         └─ 在线发布快照
                    │
SpeCo 发布调度 / 传输 / rollout 准入控制
                    ▼
vLLM-Ascend Drafter 加载与更新
```

| 模块 | 处理方式 |
| --- | --- |
| Hydra 配置与分布式启动 | 复用 `draft_train_launcher.py`、`draft_train.py` |
| Ray 资源与 WorkerGroup | 复用 `standalone_ray_runtime.py`、`workers/standalone_drafter_worker.py`；扩展引擎选择与独立资源绑定 |
| Feature Store | 扩展 `trainer/feature_store.py`，不新增另一套存储系统 |
| TQ、重试、预取、背压 | 复用现有组件，增加 TP 感知分配与消费确认 |
| 训练触发、步数预算、发布时机 | 复用 `trainer/scheduler/`，保持唯一决策入口 |
| DS4 模型、损失、自定义反向 | 复用 AIR，不改写数学逻辑 |
| DS4 特征提取与 verifier 过滤 | 复用 AIR 模型相关逻辑，提交为 SpeCo 样本 |
| checkpoint 实际内容 | AIR/DeepSpeed 保存，接入 SpeCo 的目录、进度与恢复管理 |
| 推理参数名称和布局转换 | 复用 AIR 导出规则 |
| 在线参数传输与更新时机 | 扩展 SpeCo 现有发布 executor、transport 和 admission |
| 指标与日志 | AIR 指标接入 SpeCo tracking |

不再运行另一套 AIR Feature Store、Scheduler、Coordinator 和 publication controller。其中必要的数据验证、快照与一致性规则接入 SpeCo 对应组件。

复用不意味着保持公共组件完全不变：应扩展现有组件承载缺失能力，避免建立平行系统。调度边界遵循 [Drafter Scheduler 指南](drafter_scheduler.md)。

## 4. 统一数据契约

### 4.1 字段与语义

以 SpeCo 样本作为系统内部表示，将 AIR 文件格式作为一种输入格式。

| AIR 数据 | SpeCo 内部表示 | 规则 |
| --- | --- | --- |
| `input_ids` | `input_ids` | 原样保留 token ID，禁止重新分词 |
| `positions` | `position_ids` | 保留绝对位置，分段后不得从零重置 |
| 选定层 `hidden_states` | `hidden_states` | 保存层号、拼接顺序和采集边界 |
| `target_head_hidden_states` | 首批使用 `metadata.ds4_draft.target_head_hidden_states` | 明确为 pre-HC 状态，不复用已有末层特征字段 |
| SpeCo `loss_mask` | `loss_mask` | 全链路 1=监督、0=忽略，不反转；导入拒绝旧 AIR mask 字段 |
| 请求、模型、版本身份 | `metadata` 与计划约束 | 持久化、传输、训练前均校验 |

字段必须贯穿样本验证、序列化、文件存储、分段、collate、TQ codec 和消费端，不能仅在训练函数内增加参数。首批实现复用现有 metadata tensor-tree 编码，避免改动稳定协议文件；已增加 round-trip 测试。后续如提升为通用字段，再评估协议兼容。

mask 处理使用 SpeCo 规则重写最小相关逻辑：分段筛选监督位置，overlap 和 padding 填 0，模型输入准备直接向原生 `build_block_targets` 传入正向监督 mask。原生训练调用的 `raw_mask` 仅保留参数名，值仍为 1=监督。其余采样器、目标构造、损失和前反向继续复用；相关逻辑维护在 SpeCo 包内，不要求修改或部署外部 AIR 源码。

### 4.2 DS4 DSpark 必须保持的约束

- 选定层特征宽度为 `hidden_size × 选定层数`，层顺序与模型配置一致。
- 教师 pre-HC 特征宽度为 `hidden_size × hc_mult`。
- token 行、位置、两类 hidden states 和 mask 对齐。
- 保留 AIR 的 `anchor+2` 标签起点和 seed 规则；当前 SpeCo Qwen 路径的 `anchor+1` 规则保持独立。
- 分段 overlap 只提供上下文，不重复监督。
- padding、prompt 和无效尾部不能参与 loss。
- 缺少 pre-HC 或选定层特征时拒收，不能补零冒充有效特征。
- 终止 token、最后一个 predictor row 与完整请求序列的关系沿用对应采集契约，并单独测试。

### 4.3 身份与版本

至少保存并校验：

```text
模型架构 / tokenizer 标识 / 特征布局版本
target 权重版本 / Drafter 初始权重标识
request_id / sample_id / rollout step
target_layer_ids / 采集边界 / 标签偏移规则
```

同一个训练计划只消费兼容的 target 版本。模型目录相同不能作为权重版本相同的充分依据。

## 5. 离线与在线采集适配

### 5.1 离线导入

新增 AIR 输入适配器，读取 `request_*/step_0_*.cpu.pt`、mask 文件及相关 manifest，输出 SpeCo 样本。

- 复用 SpeCo Feature Store 和 loader。
- 训练热路径不再物化第二套 AIR 数据目录。
- 迁移验证可保留临时参考数据视图，与 AIR 原始 loader 比较结果。
- 保留来源身份，避免将 DS4 数据误送入 Qwen backend。

### 5.2 在线采集

扩展 SpeCo 的 vLLM runtime adapter，在现有推理实例中安装版本受控的采集 hook：

1. 读取 DS4 指定层特征和教师 pre-HC 状态。
2. 复用 AIR confirmed-token 选择与 chunked-prefill 拼接规则。
3. 复用安全张量交接，避免 vLLM 重用底层 buffer 导致数据变化。
4. 完整请求验证通过后提交 SpeCo Feature Store/TQ。
5. 复用 SpeCo 容量、预取和消费调度；采集失败的请求明确丢弃，不污染训练。

拒绝的草稿状态不得进入训练。replacement、bonus 和 terminal token 必须按 predictor/label 关系处理。大张量不进入普通生成响应。

### 5.3 必须先核对的采集差异

AIR 离线 DSpark 采集从 backbone 的辅助输出取得选定层特征，通过 `get_mtp_target_hidden_states()` 取得教师状态；在线通用 hook 从 `get_mtp_target_hidden_states()` 和 `execute_model_state.hidden_states` 读取两个字段。

需在目标 DS4 runtime 上验证这些对象的实际语义、维度和采样时机，不能直接假设离线和在线 hook 等价。

在线 hook 当前对 vLLM `0.25.1` / vLLM-Ascend `0.25.1rc1` 和 vLLM `0.26.0` / vLLM-Ascend `0.26.0rc1` 有明确版本门控。这是源码中的已审阅边界，并非对其他版本兼容性的结论。

首期覆盖能够完整关联请求的部署。P/D 分离涉及跨进程拼接，独立评估。

## 6. 训练引擎与初始化

### 6.1 引擎接口

建议增加以下能力边界，最终签名在实现时确定：

```python
initialize(config, topology)
train_cycle(plan, batch_source)
snapshot_for_publish()
save_checkpoint(path, progress)
load_checkpoint(path)
close()
```

一个优化器更新可能包含多个数据 micro-batch、anchor chunk 和 backward，因此接口以有限训练周期为边界。

- `pytorch` 分派委托现有 SpeCo 实现，避免先全面重写 `DrafterBaseTrainer`。
- AIR 适配器接收 SpeCo 计划和数据源，执行原有训练数学逻辑。
- 公共入口与 Worker 必须在构造具体模型前选择引擎。
- AIR、DeepSpeed、vLLM-Ascend 依赖按需导入，不能影响未启用 AIR 的用户。
- 固定 AIR 源码修订和依赖版本；正式配置不绑定本机绝对路径。

### 6.2 保留 AIR 的更新语义

AIR 负责 loss、token 权重、梯度累积边界、anchor 分块、combined-context 梯度回放、裁剪与 optimizer/scheduler 更新。SpeCo 不对该结果再次执行自己的归一化、backward 或 optimizer step。

SpeCo 的计划预算以完成的优化器更新计数；micro-batch、anchor chunk 与 backward 调用数作为独立指标。学习率和累计步数不能随每个计划重置。

### 6.3 初始化

从 AIR CLI 中提取可调用组件，保留初始化顺序：配置与环境开关 → 设备/分布式初始化 → TP/DP 拓扑 → 模型加载与冻结 → teacher → DeepSpeed 与恢复。

复用 SpeCo 启动器和 WorkerGroup 提供的 rank/rendezvous 环境，避免在 Ray Worker 中再次启动 DeepSpeed launcher，也避免重复创建默认进程组。

拟议配置示例：

```yaml
training:
  backend: vllm_ascend  # 缺省为 pytorch；位于 Drafter 的 training 配置中。
```

Actor 策略、Drafter backend、Drafter 训练拓扑和 rollout TP 分别表达。现有 `training.strategy` 的 native 含义继续兼容，新后端不隐式继承 Actor 策略。后端专用配置保留在 `draft_engine`，不再设置第二个引擎选择开关。

DS4 首期保留 AIR 已审阅的 ZeRO-1、`overlap_comm=false`、非 EP 训练，以及 eager/禁用训练图捕获等约束；设备和算子支持按 AIR 的实际能力检查。

## 7. TP/DP 数据分配

SpeCo 当前文件 loader/TQ assignment 主要按 rank 分配不同样本，AIR 的 TP 训练要求组内数据一致，因此必须扩展现有分配组件：

- 同一 TP 组读取相同样本、位置、mask，并保持相同 anchor 选择。
- 不同 DP 副本读取不同样本。
- 全局 batch 大小按 DP 副本数计算，不能误乘 TP rank 数。
- 保留 AIR 的 DP 步数对齐、空 anchor 和 padding chunk 处理，使各 rank 进入兼容的 runtime collectives。
- 复用 SpeCo queue plan、预取和重试，增加 DP owner/TP group 信息。
- 所有依赖该样本的训练 rank 完成后，由指定 owner 统一确认消费和清理。
- 任一 rank 预检失败时，整组不进入分布式训练。

不能只开启 AIR 的 TP 配置，继续沿用 SpeCo 现有逐 rank 分片。

## 8. 教师与冻结权重生命周期

训练使用产生特征的 target 版本。DS4 teacher 包含完整 HC、norm 和 vocabulary head，扩展 SpeCo 当前目标 head 同步能力，支持模型专用 teacher bundle。

- teacher 按 target 版本缓存和校验，保持 frozen/eval。
- PPO 更新后，旧样本使用对应旧 teacher，或按明确策略淘汰。
- 计划开始前冻结所需版本，在训练期间不得被后续更新覆盖。
- 优先传递必要教师参数与版本信息，不要求每轮保存整个大模型 checkpoint。
- 若同一请求跨越 target 更新，必须通过已有 rollout 安全边界或明确取消策略防止混合特征。

AIR Drafter 的 embedding/head 可能来自专用 MTP 权重，也可能是初始化时从 target 加载的冻结副本。记录来源与 base 身份，不默认与在线 Actor 保持别名关系。

## 9. checkpoint、导出和在线发布

### 9.1 三种产物

| 产物 | 内容 | 用途 |
| --- | --- | --- |
| 恢复 checkpoint | 模型、优化器、scheduler、拓扑、训练和消费进度 | 断点续训 |
| 离线导出权重 | 合并并转换后的推理参数 | 新推理实例加载 |
| 在线快照 | 不可变参数及版本、base、分片信息 | 更新运行中的 Drafter |

三者独立验收，不能用“checkpoint 保存成功”代替“推理兼容”。

### 9.2 恢复

复用 AIR/DeepSpeed 保存加载，接入 SpeCo 目录、训练进度、消费记录与 tracking。

- 扩展 checkpoint 格式分派，不把 DeepSpeed optimizer state 当作 FSDP DCP 读取。
- 恢复后核对 engine global step、优化器、scheduler、拓扑和数据消费位置。
- 训练状态与消费记录保持一致，避免恢复后漏样本或重复更新。
- 不支持的拓扑变更应明确拒绝。

### 9.3 离线导出

复用 AIR 的 TP 合并、复制参数一致性检查、MoE/attention/HC/Markov 参数映射和目标 shape 校验。

保留原配置、量化及冻结权重来源的兼容检查。训练用反量化参数与推理 checkpoint 的量化布局需要显式转换和验证。目标运行时不使用的 confidence head 必须明确记录处理方式。

### 9.4 在线发布

扩展 SpeCo 发布 executor 和 transport：

1. 完整训练成功后，捕获与后续更新隔离的不可变快照。
2. 记录模型、base 身份、训练步、target 版本、TP 布局和发布 generation。
3. 通过现有 Scheduler，在 rollout 安全点暂停新请求并等待相关在途请求结束。
4. 由 DS4 适配器处理参数名称、融合布局及推理分片。
5. 所有必要 rollout rank 确认完整加载后，提交活动版本并恢复请求。
6. 失败时保持暂停并恢复上一完整版本，禁止部分新参数继续服务。

首期固定兼容的训练/推理 TP 布局；不同 TP 或 EP 布局单独验收。rank 编号一致不能证明参数布局相同，也要防止对已有本地 shard 再次执行切片。

复用 SpeCo 分桶传输、ObjectRef/已有后端能力，增加分片清单与路由；不在 driver 聚合完整 Drafter，也不每轮导出完整目标模型。

离线全量导出作为正确性参照。在线加载需与离线加载逐参数、逐 logits 对照。冻结参数和 persistent buffers 不在高频快照中时，新建或恢复的 rollout worker 必须先加载匹配 base。

## 10. 建议改动位置

以下路径除标注“新增”外均为现有 SpeCo 模块，具体接口在实施时细化。

新增代码按职责命名：后端执行、配置与恢复使用 `vllm_ascend_*`；DeepSeek V4 专属数据契约、导入与导出使用 `deepseek_v4_*`。当前已实现范围仍为 DeepSeek V4 0731 DSpark 离线训练；数据和 checkpoint 的既有序列化标识保持兼容。

| 位置 | 计划改动 |
| --- | --- |
| `verl_speco/backends/vllm_ascend_engine.py`、`vllm_ascend_runtime.py`（新增） | AIR 有限训练周期与 Ascend 构造，沿用现有目录布局 |
| `verl_speco/config/` | 引擎与模型变体配置、能力验证、可复用示例 |
| `verl_speco/draft_train.py`、`backends/training_backend.py` | 公共入口按 backend 集中分派，各实现延迟导入 |
| `verl_speco/trainer/vllm_ascend_training_loop.py`（新增） | 新后端执行流程，共用现有 launcher，不新增专用训练入口 |
| `verl_speco/integration/vllm_ascend_config.py`、`trainer/vllm_ascend_checkpoint.py`（新增） | 外部后端配置约束和 DeepSpeed 恢复 |
| `verl_speco/trainer/deepseek_v4_contract.py`、`deepseek_v4_dataset.py`、`export_deepseek_v4_draft.py`（新增） | DeepSeek V4 数据契约、导入与推理权重导出 |
| `verl_speco/workers/standalone_drafter_worker.py` | 引擎无关执行入口及状态/RPC 适配 |
| `verl_speco/standalone_ray_runtime.py` | 复用 WorkerGroup，扩展分卡绑定和 TP/DP 分配 |
| `verl_speco/trainer/feature_store.py`、`draft_dataset.py` | pre-HC 字段、契约验证、AIR 输入适配、分段/collate |
| `verl_speco/transport/drafter_sample_protocol.py` | 新字段和身份信息的无损编解码、兼容验证 |
| `verl_speco/trainer/tq_sample_source.py` | TP 组内相同样本、DP 分配及消费确认 |
| `verl_speco/integration/vllm_runtime.py` | 版本受控 DS4 capture 和权重加载适配 |
| `verl_speco/trainer/scheduler/` | 复用决策，扩展独立 Worker executor 与分片发布协议 |
| `verl_speco/integration/rollout_publish.py` | teacher bundle、快照载荷与传输适配 |
| `verl_speco/trainer/checkpoint.py`、`standalone_resume.py` | 引擎 checkpoint 格式分派和进度协调 |
| `tests/` | 数据、分布式、权重、恢复、原有路径回归 |

不修改安装的 upstream verl 源码。所需数值实现按来源清单迁入 SpeCo；复用 SpeCo launcher、store、scheduler、worker、发布及回滚，不复制 AIR CLI 或控制平面。

## 11. 实施阶段与验收

| 阶段 | 交付内容 | 验收门槛 |
| --- | --- | --- |
| P0 基线冻结 | 模型、运行时、源码修订、捕获数据、训练配置及预期结果 | AIR 原始训练、恢复、导出可复现 |
| P1 数据适配 | AIR importer、SpeCo schema/存储/TQ 扩展 | round-trip 后 tensor、位置、mask 和身份无损 |
| P2 离线训练 | AIR 引擎、初始化、TP/DP 数据分配 | 同输入、同随机状态下，loss、梯度、参数更新与 AIR 基线一致 |
| P3 恢复与导出 | DeepSpeed 恢复接口、推理导出 | 连续训练与保存恢复后训练一致；导出权重可推理 |
| P4 在线采集 | DS4 hook、请求拼接和提交 | accepted/rejected/bonus token、chunked prefill、取消均正确 |
| P5 在线分卡 | 调度、teacher 版本、训练与发布闭环 | 连续多轮更新成功；推理使用新版本；版本不混用 |
| P6 回归与性能 | 回归报告、资源及吞吐指标 | native 路径通过；无持续内存增长；开销可解释 |

### 11.1 数值与功能验证

- token、位置、mask、层顺序、参数名称和分片关系精确比较。
- 先测 AIR 重复运行的数值误差，再设 BF16 容差；不能任意放宽阈值。
- 固定 seed、anchor、噪声和数据次序，比较各 loss、有效 token 数、梯度和单步参数增量。
- 覆盖 TP>1、DP>1、长序列、尾部不足、空监督、不同 anchor 数和累积尾批。
- 连续多个训练计划验证优化器、scheduler 和累计步数不重置。
- 导出、热更新分别验证参数及 logits，随后运行真实生成与接受长度评估。
- 训练 loss、teacher-forced 指标不能替代在线接受率与吞吐结果。

### 11.2 故障与回归验证

覆盖缺失特征、错误层号、target 切换、混合版本、Worker 重启、分片缺失、重复消息、消费确认中断、发布中断及恢复。验证失败不应允许部分 rank 单独进入训练或使用半更新权重。

复用现有 SpeCo scheduler、TQ、checkpoint、native draft update 测试；AIR 的模型、loss、TP 导出、DSA/HC 反向测试作为参考与模型层回归。设备测试在实际目标运行时执行，不以 CPU mock 代替硬件验收。

## 12. MTP 与同卡扩展

DSpark 通路通过后，MTP 复用同一套数据、引擎、调度、恢复和发布接口。

MTP 仍需完成：

- 从 AIR `__main__` 提取训练循环。
- 保留多步 hidden-state 递推、图分离、token 加权与 loss 规则。
- 按模型定义特征边界、冻结权重和导出映射。
- 扩展 SpeCo 算法识别、工厂和推理配置。
- 分别验证 DS4、Qwen3.5、GLM5，不能用单模型结果代表全部 MTP。

首个 MTP 选择 DS4，优先复用已适配 runtime 和 teacher。Qwen3.5、GLM5 追加各自验收。

同卡作为后续能力，另行验证设备驻留、显存让出与恢复、runtime cache 和调度。分卡成功不等于同卡支持完成。

## 13. 工作量与依赖

以下为单人有效开发人日，假设 AIR 原环境可复现、模型和样本齐备、Ascend 资源可用，不包含排队和上游修复等待。

| 阶段 | 预计人日 |
| --- | ---: |
| P0 基线与审计 | 2–3 |
| P1 数据适配 | 3–5 |
| P2 引擎与离线训练 | 4–7 |
| P3 恢复与导出 | 3–5 |
| P4 在线采集 | 3–6 |
| P5 在线分卡与发布 | 4–7 |
| P6 回归及性能 | 2–4 |
| **DSpark 总计** | **21–37** |

离线 DSpark 里程碑 P0–P3 约 12–20 人日。首个 DS4 MTP 预计额外 5–10 人日；其他模型、异构 TP/EP、P/D 分离、同卡与持续异步执行另行估算。

P0 前需要：

1. 目标模型的配置、tokenizer 和准确权重版本。
2. 实际 AIR 捕获样本、mask、manifest，以及对应训练配置。
3. 完整训练 checkpoint 和离线导出参考结果。
4. Python、PyTorch、torch_npu、DeepSpeed、vLLM、vLLM-Ascend、Transformers 版本。
5. 可运行基线与迁移验收的 Ascend 环境。
6. 在线阶段所需的 verl Actor 训练和 rollout 可运行配置；目标 Actor 支持性单独核实。

## 14. 完成定义

- SpeCo 能通过配置选择 native 或 AIR 引擎。
- AIR 数据经过 SpeCo 存储与传输后语义和数值保持正确。
- DS4 DSpark 前向、反向和更新与 AIR 基线通过一致性验收。
- checkpoint 恢复包含完整训练状态和一致的数据进度。
- 离线导出可推理，在线更新能完整提交或恢复。
- 在线分卡连续多轮运行无 target/teacher/Drafter 版本混用。
- 原有 native 路径继续通过相关回归。
- 文档明确支持矩阵、限制、验证命令与实际结果。

## 15. 主要源码索引

### SpeCo

- [统一调度边界](drafter_scheduler.md)
- [现有训练循环](../verl_speco/trainer/base_trainer.py)
- [独立训练入口](../verl_speco/trainer/draft_training_loop.py)
- [Feature Store 与样本结构](../verl_speco/trainer/feature_store.py)
- [TQ 编解码](../verl_speco/transport/drafter_sample_protocol.py)
- [TQ 分配与消费](../verl_speco/trainer/tq_sample_source.py)
- [Ray 运行时](../verl_speco/standalone_ray_runtime.py)
- [vLLM 适配与权重加载](../verl_speco/integration/vllm_runtime.py)
- [权重发布与目标 head 同步](../verl_speco/integration/rollout_publish.py)
- [已有模型与算法工厂](../verl_speco/backends/factory.py)

### AIRDrafter

以下路径相对 AIRDrafter 源码根目录，需与实施时固定的修订一起记录：

- `offline_training/train_mtp.py`：DSpark 训练函数、MTP 主循环、DeepSpeed 初始化。
- `training_common/modeling/ds4_dspark.py`：DS4 DSpark 契约、前向与训练适配。
- `training_common/modeling/ds4_runtime.py`：vLLM-Ascend runtime 与自定义反向。
- `training_common/modeling/lm_head.py`：完整 DS4 teacher。
- `training_common/utils/data.py`、`samplers.py`：分段、mask、collate 和采样。
- `training_common/utils/parallel.py`、`ds_utils.py`：TP/DP 与恢复。
- `data_preparation/vllm_ascend_hs_dataset_creator.py`：离线真实推理特征捕获。
- `co_training/vllm_ascend_runtime.py`、`verifier_selection.py`：在线 hook 与确认状态选择。
- `co_training/runtime_capture_bridge.py`、`npu_handoff.py`：异步捕获与安全交接。
- `co_training/trainer.py`、`weight_lifecycle.py`：常驻执行与 teacher 生命周期参考。
- `co_training/drafter_snapshot.py`：不可变分片快照与验证。
- `loading_trained_mtp/loading_ds4.py`：TP 合并及推理导出。
- `co_training/README.md`、`docs/DS4_DSPARK_TRAINING_SUPPORT.md`：已报告验证范围与运行时限制。
