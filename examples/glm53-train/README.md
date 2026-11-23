# GLM-5.3 Response Generation and Standalone DSpark Training

Last updated: 2026-11-23

This directory contains streaming response generation, two-node hidden-state serving, and single-node DSpark drafter training scripts. Run the commands below from the repository root. The shell scripts target Linux/Ascend environments.

## Scripts and execution order

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | [generate_responses.py](generate_responses.py) | Request existing GLM services and generate human/assistant training JSONL |
| 2a | [glm53_hidden_serve_node0.sh](glm53_hidden_serve_node0.sh) | Start the hidden-state service with an API server on node 0 |
| 2b | [glm53_hidden_serve_node1.sh](glm53_hidden_serve_node1.sh) | Start the headless compute process on node 1 and join node 0 |
| 3 | [run_glm53_drafter_dspark_separate_training.sh](run_glm53_drafter_dspark_separate_training.sh) | Start the producer and single-node drafter training using the hidden-state service |

Run steps 2a and 2b on their respective nodes and keep both processes running. Do not wait for node 0 to become fully ready before starting node 1: initialization requires both nodes. Start training after node 0's API is ready. Skip step 1 if you already have human/assistant training data.

Data flows from source JSONL to generated training JSONL, then through producer hidden-state requests to drafter training and saved checkpoints. Response generation and hidden-state serving can use different deployments. Configure the generation script's `MODEL_NAME` independently of the training service's model name.

## 1. Generate response JSONL

Use Python 3.10+ and install dependencies in your active Python environment:

```bash
uv pip install httpx tqdm
```

Edit the configuration section at the top of `generate_responses.py`:

| Setting | Description |
| --- | --- |
| `INPUT_FILE` / `OUTPUT_FILE` | Input/output JSONL; relative paths resolve against the Python script's directory |
| `MODEL_NAME` | Model name accepted by the generation service; defaults to `glm-53` |
| `INSTANCES` | Each instance's full `/v1/chat/completions` URL, concurrency, operating windows, and enable flag |
| `REQUEST_PARAMETERS` | Shared token limit, temperature, reasoning effort, and thinking settings |
| `STREAM_READ_TIMEOUT_SECONDS` | Timeout waiting for the next network data chunk, not a total generation deadline |
| `STREAM_INCLUDE_USAGE` | Request usage at the end of the stream; set to `False` if the service does not support `stream_options` |

Operating windows use Beijing time (UTC+8). `[(0, 24)]` runs all day; `[(0, 7)]` runs daily from 00:00 to 07:00. Replace the generic `REPLACE_INSTANCE_N_HOST` placeholders with your service hosts and ports, or disable unused instances. The fourth instance is disabled by default; configure its URL and set `enabled=True` to enable it. All instances share a task pool, with each record assigned to one worker at a time.

Input contains one JSON object per line:

```json
{"conversations":[{"from":"human","value":"Please introduce yourself."},{"from":"assistant","value":"Original response"}]}
```

Only the first human message in each row is sent to the service. Later turns are ignored, and an existing assistant message is not required. If the first human message is empty or whitespace-only, the entire row is skipped and excluded from the task count; no later human message is selected. Missing human messages, malformed JSON, or non-string values produce an error with the input line number before requests are sent.

Run:

```bash
python -u examples/glm53-train/generate_responses.py
```

The script receives SSE streams and writes responses in this form:

```json
{"request_key":[1,0],"conversations":[{"from":"human","value":"Please introduce yourself."},{"from":"assistant","value":"Reasoning process</think>Final answer"}]}
```

Actual output also includes model, instance, finish_reason, and usage. Without reasoning, only the answer content is saved. Records are written in completion order, which can differ from input order. The first element of `request_key` is the original input line number (starting at 1); the second is the first human message's index (starting at 0).

### Stopping and resuming

Each instance follows its own operating windows. When a window closes, its local streaming requests are cancelled and unfinished tasks return to the shared pool. All-day instances continue processing; if every instance is outside its window, the program waits. It exits after all tasks have succeeded or been discarded. Requests already received by the server may continue computing after local cancellation.

Press `Ctrl+C` to stop, then rerun the same command to resume. Keep the machine and process running to wait automatically for the next operating window.

| Output file | Purpose |
| --- | --- |
| `OUTPUT_FILE` | Successful results, skipped on resume |
| `OUTPUT_FILE.discarded.jsonl` | Length-truncated tasks (`finish_reason=length`), permanently discarded |
| `OUTPUT_FILE.errors.jsonl` | Historical failures; entries remain after a successful retry |
| `OUTPUT_FILE.meta.json` | Input and generation configuration signature |
| `OUTPUT_FILE.lock` | Prevents multiple processes from writing the same output |

Keep the successful results, discarded records, and metadata files when resuming. Ordinary failures receive immediate retries, then are reassigned after a delay if retries are exhausted. The progress bar's `failed` counter counts failures during the current run and does not decrease after successful retries.

Forced termination can leave a lock file. Delete the corresponding `.lock` only after confirming the previous process has stopped. Do not run two processes against the same output. Changing instance URLs, concurrency, or operating windows does not invalidate resume. Changing the input, model name, or generation parameters triggers signature validation; use a new output filename.

## 2. Start the two-node hidden-state service

The hidden-state service uses **vLLM `0.30.0rc2`** on both node 0 and node 1, with Ascend/CANN and `ExampleHiddenStatesConnector` available. The scripts start services but do not install the runtime. See the repository's [usage documentation](../../docs/usage.md) for environment setup.

Edit the configuration on each node, or supply environment variables with the same names. Network interfaces are not detected automatically:

```bash
NIC_NAME="YOUR_COMMUNICATION_INTERFACE"
LOCAL_IP="THIS_NODE_IP"
NODE0_IP="PRIMARY_NODE_IP"
MODEL_PATH="/path/to/GLM-5.3"
HIDDEN_STATES_PATH="/shared/glm53_hidden_states"
```

- Set `LOCAL_IP` to each node's own address and use the same `NODE0_IP` on both nodes. Node 0 defaults `NODE0_IP` to its `LOCAL_IP`.
- Both service nodes use the same model and write hidden states to the same shared storage directory.
- The training machine must read returned hidden-state files at the same absolute paths. HTTP connectivity alone is insufficient.
- The topology is total DP=8, local DP=4 per node, and TP=4. Node 1 uses start rank 4 and `--headless`; it does not expose its own API.

### Align the served model name first

The service scripts currently use `--served-model-name glm`, while the training launcher uses the training-side `MODEL_PATH` string as the requested model name. In **both service scripts**, change that line to:

```bash
    --served-model-name glm "${MODEL_PATH}" \
```

Use the same path string for the training script's `MODEL_PATH` and the service alias. If the training machine uses a different local model directory, add that training-side path string as another alias in both service scripts. That directory must still exist locally on the training machine. Passing only `speco.standalone_tq_producer.vllm_model=glm` will not work: the current launcher computes and overrides that field.

On node 0, run:

```bash
bash examples/glm53-train/glm53_hidden_serve_node0.sh
```

Then, in a separate terminal on node 1, run:

```bash
bash examples/glm53-train/glm53_hidden_serve_node1.sh
```

On the training machine, wait for the service:

```bash
python tools/wait_for_vllm_endpoints.py \
    --endpoints '[http://NODE0_IP:9000/v1]' \
    --timeout-seconds 3000
```

Replace `NODE0_IP` with the actual address. This confirms `/v1/models` is reachable; also check service logs and hidden-state extraction support. The training producer submits token IDs through `/v1/completions` to obtain features, unlike the generation script's streaming chat requests.

## 3. Start single-node standalone DSpark training

Use an environment with this repository's training dependencies, Ascend PyTorch, and Ray installed. Edit `run_glm53_drafter_dspark_separate_training.sh`:

| Setting | Description |
| --- | --- |
| `MODEL_PATH` | Target model directory readable by the training machine; the service must accept this string as a model name |
| `TRAIN_FILE` | Successful output JSONL from step 1, not the errors or discarded file |
| `DRAFT_CKPTS_DIR` | Drafter checkpoint output directory |
| `DRAFTER_PATH` | Leave empty to initialize a new drafter, or specify an existing drafter/checkpoint |
| `TRAIN_DEVICES` / `TRAIN_GPUS` | Local training device IDs and count; keep them consistent |
| `SPECO_VLLM_ENDPOINTS` | Node 0's `/v1` base URL in list format: `[http://HOST:9000/v1]` |

If serving and training share a machine, their devices must not overlap. The service nodes and training machine can be separate hosts; single-node training refers to the drafter's training topology.

The script also exposes training steps, per-device batch size, learning rate and schedule, checkpoint intervals, drafter depth, block size, anchors, Markov rank, loss weights, and shuffling. These are initial settings based on an existing example; tune them for your workload.

Keep serving and training layer settings aligned:

```text
Captured service layers: [4,22,41,59,77,78]
Training auxiliary IDs:  [4,22,41,59,77]
Auxiliary layer count:  5
Additional final layer: 78
```

These values follow the current service configuration. Recheck the final layer index for a different model configuration. The wrapper does not override the mask token ID or FFN width. For a newly initialized drafter, the code defaults to target `vocab_size - 1` as the mask ID and target `intermediate_size` as the FFN width, falling back to `hidden_size * 4` if absent. This does not identify a mask token from the tokenizer, and the target FFN width may not suit a small drafter.

Run:

```bash
bash examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh
```

The wrapper calls `examples/run_qwen3-8b_drafter_dspark_separate_training.sh`, overrides the GLM layer configuration, and explicitly selects FSDP2 instead of the example's DDP drafter strategy. It reuses the producer, packing, and Ray scheduling logic. It does not start target vLLM servers; keep both service processes running throughout training. The training entry point's `online` data source means requesting features from the external service, not rerunning response generation.

Append Hydra arguments to override underlying settings temporarily:

```bash
bash examples/glm53-train/run_glm53_drafter_dspark_separate_training.sh \
    actor_rollout_ref.rollout.drafter.training.max_steps=100 \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=100
```

The wrapper assigns its exported settings directly, so edit the script to change them; external environment variables with the same names do not override those assignments. Training logs use console and TensorBoard, and checkpoints are saved under `DRAFT_CKPTS_DIR`.

For quantized target weights, verify that training can read the required configuration, embeddings, and lm_head. End-to-end GLM training has not been validated for these examples.
