# GLM-5.3 DSpark MLA 草稿模型适配计划

## 1. 结论

可行。PR [vllm-project/speculators#1053](https://github.com/vllm-project/speculators/pull/1053)（`feat(dspark): add GLM-5 Dense MLA draft via --draft-arch glm5`）的核心是把 GLM-5.x 的 **DeepSeek 风格 dense MLA** 作为 DFlash/DSpark 草稿主干：

- Q/KV 低秩投影（`q_a_proj`/`q_b_proj`、`kv_a_proj_with_mqa`/`kv_b_proj`）
- 双源 KV 注入（context feature + draft block）
- `rope_interleave`（DeepSeek 逐对旋转）+ YaRN mscale
- `v_head_dim != qk_head_dim` 时对 V 做 pad、输出再切片
- 新增 `glm5_dspark` decoder 配置并导出 `model_type=glm5_dspark`、`architectures=["DSparkDraftModel"]`

本仓库训练侧与 speculators 的实现栈不同（自有 `DFlashAttention` + flex block mask / Ascend dense mask、扁平 `DSparkConfig`、自有 rotary），因此**不能直接 cherry-pick PR 代码，需要按其数学语义在本仓库的 DFlash/DSpark 骨架上重写 MLA 注意力层**。

本次范围：**训练 + 训练检查点导出配置**。vLLM / vLLM-Ascend 服务端的 `glm5_dspark` 模型实现不在本仓库内，默认其已支持 PR 约定的配置契约。

### 1.1 分支与执行约束

- **先建分支再改动**：执行前从当前分支新建 `feat/glm53-dspark-mla`，所有改动只落在该分支，不直接提交到主分支。
  - 执行 `git status --short` 确认工作区干净；如有未提交改动，先告知用户再决定 stash/commit。
  - `git switch -c feat/glm53-dspark-mla`
- **本机无加速卡**：不做真实训练/推理验证，只做 CPU 单元测试与静态配置/导出测试。
- **测试解释器**：base 环境没有 torch；使用 `D:\appinstall\miniconda\envs\speculators_test\python.exe`（torch 2.13.0+cpu、transformers 5.16.1、pytest 可用）。训练侧目标 transformers 为 5.20.1，实现只使用两版本都存在的 API（`ROPE_INIT_FUNCTIONS` 等）。
- **flex attention 路径**：CPU 单测默认走 dense mask + SDPA 路径（与 Ascend 训练一致）；flex 分支只在 `torch.cuda.is_available()` 时测试，其余环境 skip。

## 2. 现状分析

### 2.1 当前 GLM-5.3 DSpark 草稿
- 主干：`DFlashDraftModel` + `DFlashDecoderLayer` + `DFlashAttention`（GQA、NeoX RoPE、QK RMSNorm、SwiGLU）。
  - [modeling_dflash.py](file:///d:/coding/verl-SpeCo-sunny/verl_speco/models/dflash/modeling_dflash.py#L114-L220)
  - [modeling_dflash.py](file:///d:/coding/verl-SpeCo-sunny/verl_speco/models/dflash/modeling_dflash.py#L240-L292)
  - [modeling_dflash.py](file:///d:/coding/verl-SpeCo-sunny/verl_speco/models/dflash/modeling_dflash.py#L294-L400)
- 结构参数生成：`DSparkTrainerBackend._build_fallback_config` 从目标 config 继承 hidden/heads/intermediate/head_dim，始终构造 GQA 配置。
  - [dspark_trainer_backend.py](file:///d:/coding/verl-SpeCo-sunny/verl_speco/backends/dspark_trainer_backend.py#L1600-L1704)
- GLM-5.3 脚本当前只设置层数/block/markov 等，无 MLA 开关。
  - [run_glm53_drafter_dspark_separate_training.sh](file:///d:/coding/verl-SpeCo-sunny/examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh#L46-L63)

### 2.2 GLM-5.3 目标配置（transformers 5.20.1，`GlmMoeDsaConfig`）
- MLA 字段：`q_lora_rank=2048`、`kv_lora_rank=512`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`、`head_dim = qk_rope_head_dim`、`rope_interleave=True`、`num_attention_heads=num_key_value_heads=64`。
- 注意力实现（`GlmMoeDsaAttention`）与 DeepSeek-V3 一致：`kv_b_proj` 展开到全部 head、RoPE 只作用于 rope 通道、`yarn_apply_mscale` 缩放。

### 2.3 缺口清单
1. `DFlashAttention` 只有 GQA 投影，无 MLA 低秩投影和 latent norm。
2. `DFlashRotaryEmbedding` 只有基础频率 + NeoX `rotate_half`，不支持 interleaved、YaRN 和 `attention_scaling`。
3. `DSparkConfig`/`_build_fallback_config` 没有 MLA 字段，无法表达 `qk_head_dim != head_dim`。
4. `DFlashDraftModel.__init__` 硬编码 `DFlashDecoderLayer`，无按配置分派。
5. 导出重写只把 DSpark 映射为 `model_type=qwen3`/`Qwen3DSparkModel`，MLA 草稿需要改成 `glm5_dspark`/`DSparkDraftModel`。
6. GLM53 脚本和 `speco_base.yaml` 没有 MLA 配置项。

## 3. 方案与关键决策

| 决策点 | 选择 |
| --- | --- |
| 启用方式 | 新增 `dspark_draft_attn: gqa \| mla`，默认 `gqa`，GLM53 脚本显式设为 `mla`；不改其他示例行为 |
| 代码复用 | 不复用 `dsv4_dspark` 的 `LatentAttention`（单共享 KV + sink + o_lora，结构不同），在 dflash 内新增 MLA 类 |
| 配置形态 | 保持扁平 `DSparkConfig`，新增 MLA 字段；不新增独立 Config 类 |
| RoPE | 复制目标 `rope_parameters`；非 `default` 时用 `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS` 计算 inv_freq + attention_scaling，保证与上游/服务端一致；不支持的 rope_type 直接报错 |
| Attention scaling | 复刻 `yarn_apply_mscale(rope_parameters, qk_head_dim ** -0.5)` |
| 掩码 | 沿用后端现有 `block_mask` / `dense_attention_mask` 协议；MLA 的 `num_kv_heads == num_heads`，flex 用 `enable_gqa=False` |
| 导出契约 | MLA 草稿导出 `model_type="glm5_dspark"`、`architectures=["DSparkDraftModel"]`（与 PR 一致） |
| 权重兼容 | 参数名与 PR 一致（`q_a_proj`/`q_a_layernorm`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_a_layernorm`/`kv_b_proj`/`o_proj`），为后续加载 speculators 检查点留出可能；本次不承诺外部 checkpoint 加载 |
| 滑动窗口 | MLA 路径不支持 SWA；GLM53 不启用。检测到 MLA + SWA 时明确报错 |

## 4. 具体改动

### 4.1 `verl_speco/models/dspark/configuration_dspark.py`

**做什么**
- `_SPECULATOR_TRANSFORMER_KEYS` 增加：`q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`rope_interleave`、`draft_attn`。
- `DSparkConfig.__init__` 新增参数并持久化为属性：
  - `draft_attn: str = "gqa"`（仅允许 `gqa`/`mla`）
  - `q_lora_rank: Optional[int]`、`kv_lora_rank: Optional[int]`、`qk_nope_head_dim: Optional[int]`、`qk_rope_head_dim: Optional[int]`、`v_head_dim: Optional[int]`
  - `rope_interleave: bool = True`
  - MLA 时校验：`kv_lora_rank >= 1`、`qk_nope_head_dim + qk_rope_head_dim` 合法、`qk_rope_head_dim` 为偶数
- `architectures` 默认值：MLA 用 `["DSparkDraftModel"]`，GQA 保持 `["Qwen3DSparkModel"]`。
- `from_dspark_dict`：若 `model_type == "glm5_dspark"` 或同时存在 `kv_lora_rank`/`qk_rope_head_dim`，则 `draft_attn = "mla"`；保留 MLA 字段（当前会被 `internal_config["model_type"] = "dspark"` 覆盖，需要按原值保留语义）。

**为什么**：让训练配置能完整表达并往返序列化 MLA 拓扑。

### 4.2 `verl_speco/models/dflash/modeling_dflash.py`

**做什么**
1. 扩展 `DFlashRotaryEmbedding.__init__(..., rope_parameters: Optional[dict] = None)`：
   - `rope_type = rope_parameters.get("rope_type", "default")`
   - `default`：保持现有 `1/base^(2i/dim)`，`attention_scaling = 1.0`
   - 其他：`inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config_like)`，其中 config-like 暴露 `rope_parameters`/`hidden_size`/`num_attention_heads`/`head_dim`/`max_position_embeddings`
   - 缓存 `cos/sin` 后乘以 `attention_scaling`
   - 不支持的 rope_type 抛 `ValueError`（fail-closed）
2. 新增 RoPE 作用函数：
   - `_apply_rotary_neox(x, cos, sin)`：沿用 `rotate_half`
   - `_apply_rotary_interleaved(x, cos, sin)`：复刻上游 `apply_rotary_pos_emb_interleave`（`cos/sin` 取前一半，偶/奇配对旋转）
3. 新增 `DFlashMLAAttention`，forward 签名与 `DFlashAttention` 完全一致：
   ```
   forward(draft_hidden, context_hidden, draft_position_ids, context_position_ids,
           block_mask=None, dense_attention_mask=None) -> Tensor
   ```
   - 初始化：`q_a_proj/q_a_layernorm/q_b_proj`（`q_lora_rank>0`）或 `q_proj`（`q_lora_rank` 为空）；`kv_a_proj_with_mqa`、`kv_a_layernorm`、`kv_b_proj`、`o_proj`；`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`；`head_dim` 取 `qk_rope_head_dim` 用于 rotary
   - `scaling = yarn_apply_mscale(rope_parameters, qk_head_dim ** -0.5)`（本地复制 8 行 helper）
   - 前向：
     1. Q 由 `draft_hidden` 投影 → `[B, H, S_d, qk_head_dim]`，拆 `q_nope/q_rot`
     2. `kv_input = cat([context_hidden, draft_hidden], dim=1)` → `kv_a_proj_with_mqa` → `kv_lora` + `k_rot`
     3. `kv_lora` 过 `kv_a_layernorm` + `kv_b_proj` → 拆 `k_nope`/`v`；`k_rot` 复制到全部 head
     4. RoPE 只作用 rope 通道：Q 用 `draft_position_ids`、K 用 `cat(context_position_ids, draft_position_ids)`；按 `rope_interleave` 选择 interleaved/neox
     5. `k = cat(k_nope, k_rot)`；若 `v_head_dim != qk_head_dim`，`v = pad(v, (0, qk_head_dim - v_head_dim))`
     6. 注意力：`block_mask` 走 `compile_friendly_flex_attention(..., block_mask=..., enable_gqa=False, scale=scaling)`；否则 dense 路径 `F.scaled_dot_product_attention(..., attn_mask=dense_attention_mask, scale=scaling)`
     7. 输出切回 `v_head_dim` → reshape → `o_proj`
4. `DFlashDecoderLayer` 增加类属性 `attention_class = DFlashAttention`，`__init__` 改为 `self.self_attn = self.attention_class(config)`；新增 `DFlashMLADecoderLayer(DFlashDecoderLayer)`，只覆盖 `attention_class = DFlashMLAAttention`。
5. `DFlashDraftModel`：
   - 抽出 `_build_decoder_layers(config)`，默认返回 `DFlashDecoderLayer` 列表
   - `_no_split_modules` 加入 `"DFlashMLADecoderLayer"`
6. `verl_speco/models/dflash/__init__.py` 导出新类。

**为什么**：MLA 的数学与 GQA 差异集中在投影和 RoPE 布局，复用 DFlash 的双源 KV、残差、MLP、掩码协议可把改动限制在注意力层。

### 4.3 `verl_speco/models/dspark/modeling_dspark.py`

- `DSparkDraftModel` 覆盖 `_build_decoder_layers`：`config.draft_attn == "mla"` 时返回 `DFlashMLADecoderLayer`，否则走父类默认。
- 其余 Markov/confidence head 不变（它们只依赖 `hidden_size`）。

### 4.4 `verl_speco/backends/dspark_trainer_backend.py`

`_build_fallback_config` 增加 MLA 分支：
- 读取 `training_cfg["dspark_draft_attn"]`（默认 `gqa`）。
- MLA 时从 `target_text_config` 读取 `q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`rope_interleave`、`rope_parameters`：
  - 缺字段则抛清晰错误（例如目标不是 GLM-5/DeepSeek MLA）
  - 若 `qk_nope + qk_rope != qk_head_dim`，按 `qk_rope = qk_head_dim - qk_nope` 修正（兼容部分 transformers 版本）
  - `num_key_value_heads = num_attention_heads`、`head_dim = qk_rope_head_dim`
  - `architectures=["DSparkDraftModel"]`
  - 传递 `rope_parameters`（复制 dict）供 rotary 与导出使用
- 保留现有 `dspark_hidden_size` / `dspark_intermediate_size` 覆盖能力；GLM-5.3 默认 `intermediate_size=12288` 会使 5 层草稿约 2B 参数，README 中提示可按显存下调。

### 4.5 `verl_speco/trainer/draft_training_loop.py`

- `_VARIANT_RUNTIME_ALIASES["dspark"]` 增加 MLA 字段：`q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`rope_interleave`、`rope_parameters`、`draft_attn`，写入 `dspark_config` 子配置。
- `_rewrite_standalone_block_runtime_config`：
  - 仅当 `draft_attn != "mla"` 时保留当前 `dspark -> qwen3/Qwen3DSparkModel` 重写
  - MLA 时设置 `runtime_config["model_type"] = "glm5_dspark"`、`architectures = ["DSparkDraftModel"]`
- 不修改 `speco_training_config.json` 的既有字段语义。

### 4.6 配置与脚本

- `verl_speco/config/speco_base.yaml`：在 dspark 训练块新增 `dspark_draft_attn: gqa`（其余 MLA 字段由目标自动推导，不设默认值）。
- `examples/run_qwen3-8b_drafter_dspark_separate_training.sh`：新增 `DSPARK_DRAFT_ATTN=${DSPARK_DRAFT_ATTN:-gqa}` 并传 `...training.dspark_draft_attn=${DSPARK_DRAFT_ATTN}`。
- `examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh`：`export DSPARK_DRAFT_ATTN=mla`。
- `examples/glm53-train/README.md`：补充 MLA 草稿结构说明、开关、参数规模提示、导出契约（`model_type=glm5_dspark`）。

### 4.7 测试

新增 `tests/unit/test_dspark_mla_attention.py`：
- 配置往返：MLA 字段进入 `to_dict`，`from_dspark_dict` 保持 `draft_attn="mla"`，speculators 嵌套 `transformer_layer_config` 能被提升。
- 结构：MLA 模型存在 `q_a_proj/kv_a_proj_with_mqa/kv_b_proj`，不存在 `k_proj`；`fc`/`hidden_norm` 保持。
- RoPE：interleaved 与 neox 输出不同；interleaved 结果与 `transformers.models.deepseek_v3` 的 `apply_rotary_pos_emb_interleave` 一致；mscale 时 `scaling == qk_head_dim**-0.5 * mscale^2`。
- 数值：tiny config（若干 head、ctx=4、draft=2）下，MLA 输出与显式 `einsum + mask + softmax` 参考实现一致（fp32）。
- V pad：令 `v_head_dim < qk_head_dim`，输出 shape 仍为 `[B, S, hidden]`。
- 导出：模拟 scratch 训练配置，断言 runtime config 为 `model_type=glm5_dspark`、`architectures=["DSparkDraftModel"]`，且 `dspark_config` 带上 MLA 字段。
- 设备约束：默认只用 dense mask + SDPA；flex 分支加 `@pytest.mark.skipif(not torch.cuda.is_available())`。

回归：现有 `tests/unit/test_dspark_confidence_detach.py`、`tests/unit/test_dspark_hidden_layout.py`、`tests/unit/test_dflash_flex_attention.py`、`tests/unit/test_draft_training_loop.py` 保持通过（GQA 路径零行为变化）。

## 5. 假设与风险

1. **目标配置字段**：假设 GLM-5.3 实际 `config.json` 暴露 `q_lora_rank/kv_lora_rank/qk_*_head_dim/v_head_dim/rope_parameters`。缺失时启动即报错，不会静默退化为 GQA。
2. **服务端契约**：导出改为 `glm5_dspark`/`DSparkDraftModel`，需要 vLLM-Ascend 侧已注册该模型；本仓库不实现服务端。若服务端未支持，训练仍可进行，但部署需另行适配。
3. **YaRN 一致性**：通过 transformers `ROPE_INIT_FUNCTIONS` 复用上游频率与 `attention_scaling`，并复刻 `yarn_apply_mscale`，训练与导出配置一致；不再沿用当前“忽略 rope scaling 仅告警”的行为（仅 MLA 路径）。
4. **FFN 宽度**：默认继承目标 `intermediate_size=12288`，草稿参数约 2B；可按需用 `DSPARK_INTERMEDIATE_SIZE` 下调，不影响结构正确性。
5. **外部检查点**：本次不验证加载 speculators 已训练的 `glm5_dspark` 权重；参数命名与 config 推断已对齐，后续可作为独立任务验证。
6. **SWA**：MLA 路径不支持滑动窗口；检测到 `layer_types` 含 `sliding_attention` 时抛错。
7. **验证边界**：本机无加速卡，不跑真实训练；只保证 CPU 单测、配置序列化和导出重写正确。真实 GLM-5.3 训练需在 Ascend 环境另行冒烟。
8. **版本兼容**：训练环境 transformers 5.20.1、CPU 测试环境 5.16.1；实现只依赖两版本共有的 `ROPE_INIT_FUNCTIONS` / `GlmMoeDsaConfig` API，避免 5.20 独有接口。

## 6. 实施顺序

0. 检查工作区并新建分支 `feat/glm53-dspark-mla`。
1. `DSparkConfig` 新增 MLA 字段与推断。
2. `DFlashRotaryEmbedding` 支持 `rope_parameters`/attention_scaling，新增两种 RoPE 作用函数。
3. 新增 `DFlashMLAAttention`、`DFlashMLADecoderLayer`，并让 `DFlashDecoderLayer` 可替换 attention 类。
4. `DFlashDraftModel._build_decoder_layers` + `DSparkDraftModel` 分派。
5. `_build_fallback_config` MLA 分支与校验。
6. 导出配置重写与 alias 字段。
7. YAML / 启动脚本 / README。
8. 单元测试与回归。

## 7. 验证方式（仅 CPU 单测）

使用带 torch 的测试环境：

```powershell
$PY = "D:\appinstall\miniconda\envs\speculators_test\python.exe"

# 新增 MLA 单测
& $PY -m pytest tests/unit/test_dspark_mla_attention.py -q

# 现有 DSpark/DFlash 回归
& $PY -m pytest tests/unit/test_dspark_confidence_detach.py `
                  tests/unit/test_dspark_hidden_layout.py `
                  tests/unit/test_dflash_flex_attention.py `
                  tests/unit/test_draft_training_loop.py -q
```

验证点（全部为静态/CPU 测试，不做真实训练）：
- 配置：`DSparkConfig` 往返后 `draft_attn=="mla"` 且 MLA 字段完整；GQA 配置保持不变。
- 结构：MLA 模型参数名与预期一致；`fc`/`hidden_norm`/Markov/confidence head 不受影响。
- 数值：tiny config 下 MLA 输出与显式参考实现一致；V pad、interleaved RoPE、mscale 分别有断言。
- 导出：对训练配置调用导出重写函数后，`model_type == "glm5_dspark"`、`architectures == ["DSparkDraftModel"]`，`dspark_config` 含 MLA 字段；GQA 仍为 `qwen3`/`Qwen3DSparkModel`。
- 分支：确认所有改动均在 `feat/glm53-dspark-mla` 分支，主分支无改动。
