# GLM-5.3 回复生成与 DSpark 独立训练

Last updated: 2026-11-20

本目录包含回复合成、双节点 hidden-state 服务和单机 DSpark drafter 训练脚本。
下面的命令均从仓库根目录执行。Shell 脚本在 Linux/Ascend 环境运行。

## 脚本与执行顺序

| 顺序 | 脚本 | 用途 |
| --- | --- | --- |
| 1 | [generate_responses.py](generate_responses.py) | 请求已有 GLM 服务，生成第一组 human + 新 assistant 的训练 JSONL |
| 2a | [glm53_hidden_serve_node0.sh](glm53_hidden_serve_node0.sh) | 在节点 0 启动带 API 的 hidden-state 服务 |
| 2b | [glm53_hidden_serve_node1.sh](glm53_hidden_serve_node1.sh) | 在节点 1 启动 headless 计算进程，加入节点 0 的服务 |
| 3 | [run_glm53_drafter_dspark_separate_training.sh](run_glm53_drafter_dspark_separate_training.sh) | 在训练机启动 producer 和单机 drafter 训练，访问上述服务获取 hidden states |

步骤 2a、2b 在两个节点分别执行，两个进程都需要保持运行；不要等节点 0 完全就绪才启动节点 1，因为初始化需要两个节点参与。
等节点 0 的 API 就绪后，再执行步骤 3。有现成的 human/assistant 训练数据时，可以跳过步骤 1。

数据流为：原始 JSONL → 合成训练 JSONL → producer 请求 hidden states → 单机训练 drafter → 保存 checkpoint。
回复合成的服务与步骤 2 的 hidden-state 服务可以是不同部署。合成脚本的 `MODEL_NAME` 与训练服务的模型名称分别配置。

## 1. 生成回复 JSONL

使用 Python 3.10+，在当前 Python 环境安装依赖：

```bash
uv pip install httpx tqdm
```

编辑 `generate_responses.py` 顶部配置：

| 配置 | 含义 |
| --- | --- |
| `INPUT_FILE` / `OUTPUT_FILE` | 输入和输出 JSONL；相对路径以 Python 脚本所在目录为基准 |
| `MODEL_NAME` | 合成服务接受的模型名称，默认 `glm-53` |
| `INSTANCES` | 每个实例的完整 `/v1/chat/completions` 地址、独立并发、运行时段及启用开关 |
| `REQUEST_PARAMETERS` | 所有实例共用的 token 上限、temperature、reasoning effort 和 thinking 参数 |
| `STREAM_READ_TIMEOUT_SECONDS` | 等待下一段网络数据的超时，不是总生成时限 |
| `STREAM_INCLUDE_USAGE` | 请求流末尾的 usage；服务不支持 `stream_options` 时设为 `False` |

实例时段使用北京时间。`[(0, 24)]` 表示全天，`[(0, 7)]` 表示每天 00:00～07:00。
第四个实例默认禁用，填写地址后设置 `enabled=True`。所有实例共享任务池，每条数据只分配给一个工作协程。

输入每行一个 JSON 对象，例如：

```json
{"conversations":[{"from":"human","value":"请介绍一下你自己"},{"from":"assistant","value":"原有回答"}]}
```

每行只请求第一条 human，后面的对话不参与请求；输入可以没有 assistant。
第一条 human 是空字符串或全空白时，整行跳过，不取后面的 human，也不计入总任务数。
没有 human、JSON 损坏或 value 不是字符串时，会在发请求前报出行号。

运行：

```bash
python -u examples/glm53-train/generate_responses.py
```

脚本接收 SSE 流，将回复写为：

```json
{"request_key":[1,0],"conversations":[{"from":"human","value":"请介绍一下你自己"},{"from":"assistant","value":"思考过程</think>正式回答"}]}
```

实际输出还包含 model、instance、finish_reason 和 usage。没有 reasoning 时只保存正文。
输出按完成顺序写入，不保证与输入顺序相同。`request_key` 的第一项为原文件行号（从 1 开始），第二项为第一条 human 的索引（从 0 开始）。

### 停止与续跑

不同实例按自己的时间段运行；到点会取消该实例的本地流式请求，并把未完成任务交回任务池。
仍有全天实例时，任务会继续处理；全部实例不在时段内时，程序等待。全部任务成功或丢弃后退出。
服务端已收到的请求可能在本地取消后继续计算。

按 `Ctrl+C` 停止，重新执行相同命令即可续跑。机器需保持运行，才能自动等待下一个时段。

| 输出文件 | 用途 |
| --- | --- |
| `OUTPUT_FILE` | 成功结果；续跑时跳过 |
| `OUTPUT_FILE.discarded.jsonl` | `finish_reason=length` 的超长任务；丢弃且不再请求 |
| `OUTPUT_FILE.errors.jsonl` | 历史失败日志；补跑成功后不删除旧日志 |
| `OUTPUT_FILE.meta.json` | 输入和生成配置签名 |
| `OUTPUT_FILE.lock` | 防止多个进程同时写同一份结果 |

续跑保留成功结果、discarded 和 meta 文件。普通失败先即时重试，耗尽后延迟重新分配；进度条的 `failed` 是本次运行累计失败次数，补跑成功不会扣减。
强制结束进程后可能残留锁文件，确认原进程已停止后才可删除对应 `.lock`。不要同时启动两个程序写相同输出。
调整实例地址、并发和运行时段不影响续跑；修改输入、模型名称或生成参数会触发签名校验，应换输出文件名。

## 2. 启动双节点 hidden-state 服务

当前 hidden-state 服务使用 **vLLM `0.30.0rc2`**。启动节点 0 和节点 1 时，均使用该版本，并准备好 Ascend/CANN 和 `ExampleHiddenStatesConnector`。
这两个脚本只启动服务，不安装运行环境。模型与训练环境准备可参考仓库的[使用文档](../../docs/usage.md)。

在两个节点分别编辑脚本顶部，直接填写以下变量；也可以通过同名环境变量传入，没有自动网卡探测：

```bash
NIC_NAME="实际通信网卡名"
LOCAL_IP="本节点IP"
NODE0_IP="节点0IP"
MODEL_PATH="/path/to/GLM-5.3"
HIDDEN_STATES_PATH="/shared/glm53_hidden_states"
```

- `LOCAL_IP` 在两个节点上分别填写各自 IP；`NODE0_IP` 保持一致。节点 0 默认将 `NODE0_IP` 取为自己的 `LOCAL_IP`。
- 两个服务节点使用相同模型，hidden states 写入同一个共享存储目录。
- 训练机必须能以相同绝对路径读取返回的 hidden-state 文件；仅 HTTP 地址可通还不够。
- 当前并行配置为总 DP=8、每节点 DP=4、TP=4。节点 1 使用起始 rank 4 和 `--headless`，不单独提供 API。

### 先对齐服务模型名称

当前服务脚本使用 `--served-model-name glm`，而训练 launcher 使用训练侧 `MODEL_PATH` 字符串作为请求里的模型名称。
在**两个服务脚本**中将该行改为：

```bash
    --served-model-name glm "${MODEL_PATH}" \
```

并让训练脚本的 `MODEL_PATH` 与这个服务别名字符串一致。
如果训练机上的本地目录不同，需要在两个服务脚本的别名列表中额外加入训练侧路径字符串；训练机本地仍必须存在该目录。
不要只给训练脚本传 `speco.standalone_tq_producer.vllm_model=glm`，当前 launcher 会自行计算并覆盖这个字段。

在节点 0 的终端执行：

```bash
bash examples/glm53-train/glm53_hidden_serve_node0.sh
```

随后在节点 1 的另一个终端执行：

```bash
bash examples/glm53-train/glm53_hidden_serve_node1.sh
```

在训练机等待服务就绪：

```bash
python tools/wait_for_vllm_endpoints.py \
    --endpoints '[http://NODE0_IP:9000/v1]' \
    --timeout-seconds 3000
```

将 `NODE0_IP` 替换为实际地址。该检查确认 `/v1/models` 可访问；还需确认服务日志无异常，且支持 hidden-state 提取协议。
训练 producer 使用 `/v1/completions` 提交 token IDs 获取特征，与回复合成脚本的流式 chat 接口不同。

## 3. 启动 DSpark 单机独立训练

使用已配置好本仓库训练依赖、Ascend PyTorch 和 Ray 的训练环境。
编辑 `run_glm53_drafter_dspark_separate_training.sh` 顶部：

| 配置 | 含义 |
| --- | --- |
| `MODEL_PATH` | 训练机可读取的目标模型目录，同时要被服务接受为模型名称 |
| `TRAIN_FILE` | 步骤 1 的成功输出 JSONL；不要填写 errors/discarded 文件 |
| `DRAFT_CKPTS_DIR` | Drafter checkpoint 输出目录 |
| `DRAFTER_PATH` | 留空初始化新 drafter；非空加载已有 drafter/checkpoint |
| `TRAIN_DEVICES` / `TRAIN_GPUS` | 本机训练卡号及数量，两者保持一致 |
| `SPECO_VLLM_ENDPOINTS` | 节点 0 的 `/v1` 基地址，使用 `[http://地址:9000/v1]` 列表格式 |

同机部署时，训练卡不能与服务占用的卡重叠。两个 hidden-state 服务节点和训练机可以分开部署；“单机训练”指 drafter 训练的节点数。

脚本还提供训练步数、每卡 batch、学习率及调度、checkpoint 保存间隔、drafter 深度、block size、anchors、Markov rank、loss 权重与 shuffle 等配置。
这些值是基于现有示例的初始设置，需要结合实际训练效果调整。

服务与训练的层配置必须对应：

```text
服务抓取层：    [4,22,41,59,77,78]
训练辅助层：    [4,22,41,59,77]
辅助层数量：    5
额外最终层：    78
```

以上配置按当前服务脚本约定，使用其他模型配置时需重新核对最终层编号。
不显式覆盖 mask token ID 与 FFN 宽度：新 drafter 初始化时，代码默认取目标模型 `vocab_size - 1` 作为 mask ID，FFN 取目标 `intermediate_size`，缺失时用 `hidden_size * 4`。
这不是从 tokenizer 自动查找 mask token，也不保证目标 FFN 宽度适合作为小 drafter 的宽度。

启动：

```bash
bash examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh
```

该脚本封装仓库现有的 `examples/run_qwen3-8b_drafter_dspark_separate_training.sh`，覆盖 GLM 层配置，并显式将 drafter 训练策略从示例的 DDP 改为 FSDP2；复用其 producer、packing 和 Ray 调度逻辑。
它不会启动目标 vLLM；两个服务进程需要在训练期间持续运行。
训练入口的 `online` 数据源表示向外部服务申请特征，不是再次执行回复合成脚本。

需要临时覆盖底层 Hydra 参数时，追加在命令后，例如：

```bash
bash examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh \
    actor_rollout_ref.rollout.drafter.training.max_steps=100 \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=100
```

Shell 配置使用直接赋值的 `export`，因此修改配置应编辑脚本；不依赖同名外部环境变量覆盖它们。
当前训练日志后端为 console 和 TensorBoard，checkpoint 保存到 `DRAFT_CKPTS_DIR`。
使用量化目标权重目录时，需要确认训练端能正确读取所需配置、embedding 和 lm_head；本目录没有完成 GLM 大模型的实际端到端训练验证。
