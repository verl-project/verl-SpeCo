# verl-SpeCo TransferQueue 落地方案

> 目标:在**不修改上游 verl**的前提下,把 SpeCo online 训练里的逐样本特征流
> 从「`SpecoRayPPOTrainer` driver 中转 + Ray object store」改为「TransferQueue
> 直传」,干掉 driver 这个数据瓶颈,并解锁流式消费与跨副本负载均衡。
>
> 约束:仅 hook,与 SpeCo 现有 hook 模式一致;TQ 作为独立库使用,**不复用** verl
> 的 `main_ppo_sync` TQ 集成。

---

## 0. 现状:SpeCo online 特征流的 controller 瓶颈

SpeCo online 路径以 `SpecoRayPPOTrainer` 为 hub,所有跨进程大张量都被 driver 串行
中转,介质是 Ray object store(`ray.put`/`ray.get`/`parallel_put`)。这与 verl 引入
TQ 想解决的痛点 1:1 对应,只是 verl 干掉的是 `RayPPOTrainer`,我们要干掉的是
SpeCo 在它之上加的 drafter 管线中转。

| # | 流向 | 当前机制 | 是否逐样本 | hook 位置(SpeCo 侧) |
|---|---|---|---|---|
| **a1** | target hidden states(SGLang 采集)-> drafter | `drafter_sample` 塞进 `DataProto.non_tensor_batch` -> driver pop/bucket -> `parallel_put` -> drafter `ray.get` | ✅ | `speco_ray_trainer.py` `generate_sequences_with_speco`;`sglang_adapter.py` `pop_drafter_samples`/`bucket_drafter_samples_by_replica`;`sglang_runtime.py` 组装 `drafter_sample` |
| **a2** | target hidden states(old-logprob hook)-> drafter | actor 前向 hook 截行 -> `ray.put` chunk -> driver 重打包 -> 分发 | ✅ | `oldlogprob_runtime.py` `_install_oldlogprob_hidden_hooks`/`_put_oldlogprob_hidden_refs`;`speco_ray_trainer.py` `_speco_collect_oldlogprob_features` |
| **b2** | target top-logprobs -> drafter(`use_logits=true`) | 随 a1 同一 side-channel | ✅ | `sglang_runtime.py` `target_logprobs`/`hidden_raw_target_logprobs` |
| **d** | rollout tokens -> drafter 训练集 | 随 a1 同一 side-channel(online)/`torch.save` 分片(offline) | ✅ | `sglang_runtime.py`;`speco_worker.py` `_store_rollout_sample` |
| b1 | target **lm_head 权重**(行)-> drafter `TargetHead` | ONE_TO_ALL Ray 分发 | ❌ 参数广播 | `rollout_publish.py` `export_actor_lm_head_weight`/`get_actor_lm_head_weight`;`speco_ray_trainer.py` `_speco_sync_target_lm_head_weight` |
| c | drafter 权重 -> rollout 引擎 | `ray.put` -> driver -> actor;vLLM 末段 ZMQ+SHM,SGLang 进程内 | ❌ 参数广播 | `speco_worker.py` `maybe_publish`;`rollout_publish.py` `update_draft_weights`;`vllm_runtime.py` `BucketedWeightSender` |

**关键事实**:hidden states 跨进程前一律 CPU 物化(`oldlogprob_runtime.py`、
`sglang_runtime.py`、`feature_store.py` 均 `.cpu()`),a1 路径下 driver 进程的
host memory 会真正承载整批 hidden states 并做一次 Ray store 往返。这正是 TQ 要
消除的往返。

---

## 1. 为什么不把"替换 feature_store"作为第一刀

`TorchShardFeatureStore`(`feature_store.py`)是 `torch.save` 分片 + JSONL manifest,
**只服务于 `collect_only`/`offline`**,不参与 online 热路径。替换它能统一离线存储
抽象、换更快的分布式后端,但**不解决 controller 瓶颈**,性能收益有限。降级为
可选尾项(见 §5 P3)。

---

## 2. 目标方案:TQ 直传逐样本特征流(a1 / a2 / b2 / d)

### 2.1 角色映射

| TQ 角色 | SpeCo 对应 |
|---|---|
| Producer(写) | rollout worker(SGLang 路径,a1/b2/d)/ actor worker(old-logprob 路径,a2)——均在 SpeCo 既有 hook 内 |
| Consumer(读) | drafter worker `collect_rollout_features`(SpeCo 侧) |
| TransferQueueController(control plane) | SpeCo launcher 启动一个 Ray actor;drafter 经 `Sampler`/`StreamingDataLoader` 拉取 |
| Storage backend | `SimpleStorage`(ZMQ,跨节点 CPU 内存);进阶可切 `MooncakeStore`(RDMA,GPU-DRAM) |

### 2.2 partition / key / 字段设计

- `partition_id`:`speco_train`(验证集用 `speco_val`)。
- `key`:`{uid}_{session_id}_{index}`,与 verl TQ 一致;`uid` SpeCo 已有。
- `tags`:`global_steps`、`source`∈{`rollout`,`oldlogprob`}、`replica_rank`/`owner_rank`、`status`、`prompt_len`/`response_len`/`seq_len`。ReplayBuffer/负载均衡按 tag 匹配。
- `fields`(列):`input_ids`、`loss_mask`、`position_ids`、`hidden_states`、`last_hidden_states`/`target`、`target_logprobs`、`hidden_positions`、`prompts`、`responses`。与 `DraftFeatureSample`(`feature_store.py`)字段对齐,便于 online/offline 复用。

### 2.3 数据流(目标)

```
rollout/actor worker (SpeCo hook)
  │  生成/截取 hidden states 后,就地 tq.kv_batch_put(samples)
  ▼
TransferQueue (SimpleStorage, 跨节点 CPU 内存;可选 MooncakeStore RDMA)
  │  control plane 按 sample 粒度追踪 ready 状态,Sampler 跨 drafter 副本均衡
  ▼
drafter worker
  │  tq.kv_batch_get / StreamingDataLoader 消费 → 喂入既有 DataBuffer / collect_online_data
  ▼
drafter 训练 (不变)
```

driver 只下发触发与轻量 key/meta,**不再承载 hidden states**。

---

## 3. 落地改动点(全部在 SpeCo 侧,hook-only)

### 3.1 启动与配置
- `draft_train_launcher.py` / `main.py`:`tq.init(config.transfer_queue)`;起 `TransferQueueController.remote(Sampler)`。
- `config/speco_base.yaml`:新增 `drafter.transfer_queue` 块(backend、partition、enable 开关)。参考 verl `ppo_trainer.yaml` 的 `transfer_queue:` 结构,但**独立配置**,不复用 verl 的。

### 3.2 Producer 侧
- **a1/b2/d(SGLang)**:`sglang_runtime.py` 组装 `drafter_sample` 处(~1594-1648),增加 `tq.kv_batch_put`;返回给 driver 的 `drafter_sample` 只保留 key/meta(或整段不再走 DataProto side-channel,driver 仅触发)。
- **a2(old-logprob)**:`oldlogprob_runtime.py` `_put_oldlogprob_hidden_refs`(~216),把 `ray.put(hidden_chunk)` 换成 `tq.kv_batch_put`;`OLD_LOGPROB_HIDDEN_CHUNK_REFS_KEY` 改为 TQ key 列表。

### 3.3 Consumer 侧
- `speco_worker.py` `collect_rollout_features`(~665):把 `_resolve_ray_object_ref`/`_resolve_hidden_state_chunks`(`ray.get`)换成 `tq.kv_batch_get`;`_dispatch_nd_compute`(~159)的 `parallel_put` 退化为只传 key(或 drafter 直接从 TQ Sampler 拉,driver 不参与分发)。
- drafter 内部 `DataBuffer`/`collect_online_data`(`base_trainer.py`)保持不变,只是数据来源由 `ray.get` 改为 TQ get。

### 3.4 Driver 侧
- `speco_ray_trainer.py`:`_speco_collect_rollout_features_rpc`/`speco_collect_rollout_features`(~351)、`_speco_collect_oldlogprob_features`(~1114)不再搬数据,只做触发/传 key;`bucket_drafter_samples_by_replica` 可由 TQ `Sampler` 替代(逐步迁移,先保留作回退)。

### 3.5 不改动
- **b1(lm_head 权重)、c(drafter 权重)**:保持现状。与 verl 上游一致(权重不走 TQ),且 c 的 vLLM 末段已有专用 ZMQ+SHM 通道。
- verl 本体:零改动。

---

## 4. 收益与边界(诚实评估)

### 收益
1. **去掉 driver 对 hidden states 的 host-memory 中转 + Ray store 往返**:producer 直存 TQ,consumer 直取,driver 不再承载整批特征。
2. **流式消费**:drafter 在样本 ready 时即可消费,不必等整批 `generate_sequences` 返回,采集与训练可重叠。
3. **跨 drafter 副本负载均衡**:TQ `Sampler`/`RankAwareSampler` 替代手写 `bucket_drafter_samples_by_replica`/`owner_rank` 分配。
4. **(若采纳 P3)统一 online/collect_only/offline 存储**:同一 TQ partition,`collect_only` 写、`offline` 读,消掉 on-disk 分片层。

### 边界 / 不解决的事
- 只优化**特征采集**这一子阶段,**不加速** rollout 本身、actor update、reward;e2e 增益取决于该子阶段在 step 中的占比。 SpeCo README 的 20% rollout / 11% e2e 提升来自 acceptance length,与本方案是不同机制,不要混为一谈。
- **权重同步(b1/c)不放进 TQ**,与 verl 上游保持一致。
- hidden states 跨进程前**仍需 CPU 物化**(现状如此);要避免物化需切 `MooncakeStore` RDMA,属进阶项。
- 引入 TQ 依赖与一个 control-plane Ray actor,增加少量运维面。

### 风险
- TQ 与 SpeCo 现有 `owner_rank`/`replica_rank` 路由语义需对齐(Sampler 要复刻「按 owner 分桶」语义,否则样本会错配 drafter 副本)。
- old-logprob 的 chunk 拆分(`hidden_states_ref_chunks`)映射到 TQ 列式存储时,需保证 chunk meta 与 key 的一致性。
- 回退路径:保留 `enable_transfer_queue=False` 时走原 Ray 路径,渐进切换。

---

## 5. 分阶段实施

| 阶段 | 范围 | 产出 |
|---|---|---|
| **P0** | a1(SGLang hidden states)走 TQ 直传;drafter `kv_batch_get` 消费;driver 仅触发 | 验证 controller-bypass 闭环 + 正确性 |
| **P1** | a2(old-logprob hidden states)走 TQ;chunk 拆分映射 TQ 列 | 覆盖第二条采集路径 |
| **P2** | b2(top-logprobs)+ d(tokens)随 a1 同 partition 传输;Sampler 替代手写 bucket | 完整特征流 + 跨副本均衡 |
| **P3(可选)** | `TorchShardFeatureStore` → TQ partition,统一 online/collect_only/offline | 离线工作流统一 |

每个阶段保留 `enable_transfer_queue` 开关与原 Ray 路径回退。

---

## 6. 待确认决策

1. **TQ backend**:`SimpleStorage`(CPU 内存,默认)起步,还是直接上 `MooncakeStore`(RDMA,省 CPU 物化)?后者依赖 RDMA 网络,建议 P0 用 SimpleStorage。
2. **drafter 消费模式**:`kv_batch_get`(主动拉,改动小)还是 `StreamingDataLoader`(全自动流式,改动大、收益高)?建议 P0 用前者,P2 再考虑后者。
3. **driver 角色**:P0 先保留 driver 传 key(最小改动),还是直接让 drafter 从 TQ Sampler 自取(driver 彻底退出数据路径)?前者风险低,建议 P0 用前者。
4. **是否做 P3**:离线统一是否在本次范围内,还是单独立项。
