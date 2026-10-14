# CI layers

The repository uses nine workflow layers:

- `cpu_unit_tests.yml`: required PR checks without installing this repository or using accelerator runtimes.
- `gpu_vllm_unit_tests.yml`: manual-only vLLM example-script runs on GPU.
- `gpu_sglang_unit_tests.yml`: manual-only SGLang example-script runs on GPU.
- `gpu_drafter_training_smoke.yml`: manual-only standalone drafter training smoke tests on GPU.
- `npu_vllm_unit_tests.yml`: PR, push, and manual NPU examples plus standalone/HCCL tests.
- `examples_e2e_manual.yml`: select any of the 21 example scripts, automatically routed to its GPU/NPU runner.
- `npu_standalone_training_manual.yml`: manual-only EAGLE1/EAGLE2 standalone training; requires a cached target model path.
- `npu_megatron_example_manual.yml`: manual-only Megatron + EAGLE3 example with the same strict training/result checks.
- `npu_sglang_unit_tests.yml`: manual-only SGLang example-script runs on NPU.

The hardware workflows require self-hosted runner labels:

- GPU: `self-hosted`, `linux`, `x64`, `gpu`
- NPU: `self-hosted`, `linux-aarch64-a2-8`

The NPU vLLM workflow targets an 8-card NPU runner by default and sets
`SPECO_ACCELERATOR_COUNT=8` unless the GitHub environment variable or manual
workflow input overrides it.

NPU vLLM pull requests and pushes to `main` run three example lanes:

- FSDP2 + EAGLE3
- FSDP2 + DFlash
- FSDP2 + DSpark

Examples wait for the lightweight two-card HCCL smoke. It needs neither a
model cache nor upstream verl, and checks synchronized gradients, parameter
updates, and model/AdamW checkpoint resume with a small linear model.
EAGLE1/EAGLE2 standalone training is manual-only in its own workflow, so PR
examples no longer wait for its repeated target-model loads.

NPU trigger paths list the example runner, result checker, and HCCL script
explicitly; changes only to CPU CI requirements do not trigger NPU tests.
Online GPU and NPU examples upload result JSON, combined stdout/stderr logs, and elapsed-time/
exit-code JSON even on failure. `pipefail` preserves example failure through
`tee`. These timings cover the example and its result check; container startup
is outside this measurement. Existing per-step metrics provide training and
publication timings. Initialization substages are not separately instrumented.

NPU SGLang (EAGLE3 and DFlash) remains manual-only until the runner runtime
and model cache have been verified. Neither NPU workflow currently schedules runs.

Online GPU and NPU example runs clear their result file before launch and write versioned JSON
via `SPECO_CI_RESULT_PATH`. Training lanes require a completed run, at least one
successful optimizer step and publication, and no activation, training, or
worker-consistency failures. Worker errors retain their type and message across
retries. Generation-only lanes require a completed run with a logged training
iteration and do not require drafter training or publication. Missing or malformed
results fail closed. Result artifacts upload even on failure; initialization
failures before `fit()` produce no result and cannot pass. Publication runs
synchronously by default; enable async publication only with an appropriate
completion check. The contract does not inspect arbitrary log text.

Online drafter training is enabled for all NPU vLLM drafters. The EAGLE3 job
sets `SPECO_EAGLE3_DISABLE_TORCH_COMPILE=true` to avoid the EAGLE3 NPU
`torch.compile` training path that can fail with Ascend vector-core exceptions,
while still exercising online drafter collect/train/publish in CI.

Like verl's CI, the hardware workflows assume the runner image has the runtime
stack and a small default model cache. Most workflows default to:

- `/home/runner/models`
- `/home/runner/models/hf_data`

The default paths are intentionally tiny-runner friendly and can be overridden
with GitHub environment variables or manual workflow inputs.

The NPU vLLM workflow follows verl's Ascend CI layout. Its image provides the
accelerator runtime plus pre-cached model assets under
`/root/.cache/huggingface/hub`. It checks out the pull-request revision and runs
`pip install --no-deps -e .`, then verifies that `verl_speco` imports from the
current GitHub workspace rather than the image's preinstalled copy.

Before launching the example script, the NPU vLLM workflow checks that the
selected target model directory and selected draft model directory exist. Its
default datasets are local parquet files inside the pre-cached Hugging Face
snapshot directories. The workflow uses
`HF_ENDPOINT=https://hf-mirror.com` and disables `hf_transfer` for more stable
downloads on the NPU runner. Override the defaults with GitHub environment
variables or manual inputs when the image uses different paths or dataset ids.

The CPU workflow runs on pull requests, pushes to `main`, and manual dispatch.
It keeps three jobs:

- Lightweight contracts: collect all of `tests/compat`, `tests/config`,
  `tests/examples`, and `tests/integration` with the minimal CI dependencies.
- CPU unit suite: collect the entire `tests/unit` directory plus the existing
  CPU/Gloo distributed regression. New unit test files are included automatically.
  CPU PyTorch and `ci/requirements-unit.txt` supply runtime, dataset, and
  checkpoint dependencies. `-ra` reports skipped tests in the CI log; directory
  collection does not override dependency or platform skips inside tests.
- Model/checkpoint integrations: keep the existing targeted integration tests
  with their runtime dependencies and existing explicit deselection.

Jobs that use upstream verl set `VERL_SPECO_UPSTREAM_ROOT` and add the
checkout from `REQUIRED_VERL.txt` to `PYTHONPATH`. The full unit suite has a
40-minute timeout. GPU/NPU hardware smoke scripts remain separate workflows;
there are no hardware markers to migrate in the current unit suite.

The vLLM and SGLang hardware workflows call `ci/run_example_test.sh`, which
selects one of the repository-owned scripts in `examples/` and passes CI
variables as Hydra overrides.

The GPU drafter training workflow directly runs the standalone tests under
`tests/special_standalone/` for EAGLE-1, EAGLE-2, Domino, and P-EAGLE. Its
default target model is `/home/runner/models/Qwen/Qwen3-4B`, and it runs three
optimizer steps per matrix entry. Configure these defaults with
`SPECO_TRAINING_SMOKE_TARGET_MODEL`, `SPECO_TRAINING_SMOKE_STEPS`, and
`SPECO_TRAINING_SMOKE_LR`.

Configure CI variables in the `speco-gpu-ci` and `speco-npu-ci` GitHub
environments, or pass them as manual workflow inputs where available:

- `SPECO_MODEL_ROOT`
- `SPECO_DATA_ROOT`
- `SPECO_TARGET_MODEL`
- `SPECO_TRAINING_SMOKE_TARGET_MODEL`
- `SPECO_TRAINING_SMOKE_STEPS`
- `SPECO_TRAINING_SMOKE_LR`
- `SPECO_EAGLE3_DRAFT_MODEL`
- `SPECO_DFLASH_DRAFT_MODEL`
- `SPECO_DSPARK_DRAFT_MODEL`
- `SPECO_TRAIN_FILE`
- `SPECO_TEST_FILE`
- `SPECO_CKPT_DIR`
- `SPECO_ACCELERATOR_COUNT`
- `SPECO_TENSOR_PARALLEL_SIZE`
- `SPECO_SEQUENCE_PARALLEL_SIZE`
- `SPECO_ENABLE_TRAINING`
- `SPECO_EAGLE3_DISABLE_TORCH_COMPILE`
- `SPECO_SPEC_STEPS`
- `SPECO_SPEC_TOPK`
- `SPECO_SPEC_VERIFY_TOKENS`
- `SPECO_DFLASH_NUM_ANCHORS`
- `SPECO_DFLASH_MAX_WINDOW`
- `SPECO_DSPARK_BLOCK_SIZE`
- `SPECO_DSPARK_SPEC_STEPS`
- `SPECO_DSPARK_SPEC_VERIFY_TOKENS`
- `SPECO_DSPARK_NUM_ANCHORS`
- `SPECO_DSPARK_MAX_WINDOW`
- `SPECO_TOTAL_TRAINING_STEPS`
- `SPECO_TRAIN_MAX_SAMPLES`
- `SPECO_VAL_MAX_SAMPLES`
- `SPECO_DATALOADER_NUM_WORKERS`
- `SPECO_EXTRA_HYDRA_ARGS`

For NPU runs, `ci/run_example_test.sh` generates
`ASCEND_RT_VISIBLE_DEVICES=0,...,N-1` from `SPECO_ACCELERATOR_COUNT` when the
caller has not already set `ASCEND_RT_VISIBLE_DEVICES`. If the caller provides
`ASCEND_RT_VISIBLE_DEVICES`, the script preserves it and checks that
`SPECO_ACCELERATOR_COUNT` does not exceed the visible device count.

NPU vLLM smoke jobs use lightweight fallback settings for pull requests, pushes,
and manual runs:

- `SPECO_TOTAL_TRAINING_STEPS=2`
- `SPECO_MAX_RESPONSE_LENGTH=512`
- `SPECO_TRAIN_BATCH_SIZE=64`
- `SPECO_TRAIN_MAX_SAMPLES=128`
- `SPECO_VAL_MAX_SAMPLES=32`
- `SPECO_PPO_MINI_BATCH_SIZE=8`
- `SPECO_DRAFTER_TRAINING_STEPS=1`
- `SPECO_DRAFTER_BATCH_SIZE_PER_GPU=1`
- `SPECO_TRAIN_BATCHES_PER_CYCLE=1`
- `SPECO_COLLECT_INTERVAL_STEPS=1`
- `SPECO_TRAINING_INTERVAL_STEPS=1`
- `SPECO_PUBLISH_INTERVAL_STEPS=1`
- `SPECO_DATALOADER_NUM_WORKERS=0`

The NPU vLLM smoke batch uses sixty-four samples so the generation batch can be
split evenly across the eight NPU agent-loop workers while giving the drafter
owner buckets enough candidates to form at least one training batch even when
only a subset of rollout samples pass the hidden-state collection filters. The
PPO mini-batch uses eight samples so actor updates split evenly across the
eight data-parallel workers. These training-oriented defaults apply to the NPU
vLLM jobs that enable online drafter training.

The runner image is responsible for providing the matching verl, vLLM/SGLang,
PyTorch accelerator runtime, and model files. Hardware workflows deliberately
fail closed when required models or datasets are absent.

Fork pull requests execute repository code on the NPU self-hosted runner.
Keep the `speco-npu-ci` environment free of privileged secrets, require review
before first-time contributors can run workflows, and use isolated or
ephemeral runners where possible.

## Testing CI locally

Run the CPU layer checks from the repository root:

```bash
python -m compileall verl_speco
bash -n examples/*.sh
bash -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

On Windows, prefer Git for Windows bash when WSL is not installed:

```powershell
D:\git\bin\bash.exe -n examples/*.sh
D:\git\bin\bash.exe -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

You can inspect the selected script and Hydra overrides without launching a
model by setting `SPECO_DRY_RUN=true`:

```bash
SPECO_DRY_RUN=true \
SPECO_TARGET_MODEL=/models/target \
SPECO_DSPARK_DRAFT_MODEL=/models/dspark \
SPECO_TRAIN_FILE=/data/train.parquet \
SPECO_TEST_FILE=/data/test.parquet \
SPECO_CKPT_DIR=/tmp/speco \
bash ci/run_example_test.sh npu vllm dspark
```

To test the hardware workflows on GitHub, open Actions and choose one of:

- `gpu_drafter_training_smoke`
- `gpu_vllm_unit_tests`
- `gpu_sglang_unit_tests`
- `npu_vllm_unit_tests`
- `npu_sglang_unit_tests`

Run the selected workflow without inputs after preparing the default paths
above. Fill the manual inputs only when you want to override the defaults for
one run.

## Manual end-to-end examples

Use Actions -> `examples_e2e_manual` -> Run workflow. Select the exact script;
`ci/manual_examples.json` determines the hardware. This does not add PR jobs.
`environment_json` accepts string-valued environment variables. Existing
placeholder defaults are now overridable without editing scripts. Missing
model/data/drafter paths are reported before launch. Checkpoint and feature-store
placeholder directories default to a job-local directory under `runner.temp`.
Only logs and result/status JSON are uploaded, not large checkpoint files.

For an online example, supply e.g.:

```json
{"MODEL_PATH":"/models/Qwen3-8B","DRAFTER_PATH":"/models/matching-drafter","TRAIN_FILE":"/data/train.parquet","TEST_FILE":"/data/test.parquet"}
```

`hydra_args` takes one argument per line (no shell evaluation). Online examples
use the strict training/publication result check; disable drafter training
explicitly via the corresponding Hydra override for generation-only checks.
Separate-training examples keep their script's stage behavior and are checked
for process failure, not online weight publication. Supply stage settings such
as `RUN_STAGE`, `DRAFT_ALGO` (Domino/P-EAGLE), `MAX_STEPS`, `TRAIN_GPUS`,
`TRAIN_DEVICES`, and `SAVE_INTERVAL_STEPS` via `environment_json` as applicable.
Two-stage examples may have different Hydra trees; use environment variables
for shared paths rather than passing incompatible overrides to both stages.

EAGLE3 separate training requires an existing compatible hidden-state vLLM
endpoint (`SPECO_VLLM_ENDPOINTS`); DSpark scripts start or reuse a target
service depending on their configuration. Target and training devices must be
disjoint for simultaneous service/training. Runner environments still need
VeOmni, Megatron, SGLang, or native MRV2 as required by the selected script;
this workflow installs the source checkouts, not those accelerator stacks.
Select `npu_image` and `verl_ref` to match the runtime when needed; GPU examples
use the preconfigured GPU runner environment.
The 35B case needs sufficient hardware and a cached matching model.

Online GPU vLLM/SGLang and NPU SGLang workflows now install both source
checkouts using uv and assert import locations and the upstream release before
launch. Manual E2E uses the same verification for the selected supported
release. The shared runner enforces training/publication or generation-only
results and uploads logs/timings on both GPU and NPU. Separate-training runner
branches, scripts, result protocols, and checkpoint tests are unchanged by this
online-workflow update.

Compatible manual online Qwen3-8B examples default to the same target model,
EAGLE3/DFlash/DSpark drafters, DAPO training data and AIME validation data as
NPU PR CI, including the corresponding SPECO_* repository/environment variable
overrides. Explicit environment_json values take precedence. GPU runners must
have these assets at the configured paths, or callers must override their paths.
The 4B Megatron target and DFlash2 drafter require explicitly matching assets;
no incompatible 8B target or DFlash drafter is substituted. Separate-training
defaults are unaffected.

## Run a selected example for a PR

After this workflow is available on the default branch, a user with Actions
write permission can open Actions -> examples_e2e_manual -> Run workflow,
choose a case, and fill in `pr_number`. Use a trusted workflow branch (normally
main). The resolver accepts only an open PR in this repository and pins its
head repository and SHA, including forks. Blank PR number preserves selected-
branch testing. PR updates need a new manual run; an old run reports only to
its original SHA. The target checkout must contain the CI result instrumentation
and runner files introduced by this CI update.

Pending and final commit statuses named `manual-e2e/<example path>` link back
to the workflow on the PR head commit. A separate GitHub-hosted reporting job
has status-write permission; hardware jobs retain contents-read permission and
checkout the target without persisted credentials. Existing speco-gpu-ci and
speco-npu-ci environments apply. Configure required reviewers in repository
settings if hardware execution must await maintainer approval; this workflow
does not configure those settings. External PR authors without repository
Actions permission ask a maintainer to dispatch with their PR number. PR comment commands are described below.

## PR author requests and maintainer approval

After `pr_e2e_comments.yml` and the manual workflow are on the default branch,
the PR author comments `/ci run eagle3` (or `dflash` / `dspark`). The bot replies
with `/ci approve <request-comment-id>`. A maintainer with write, maintain, or
admin permission posts that approval command. This dispatches the selected
manual E2E case using trusted default-branch workflow definitions. Request
comments alone never start hardware tests. Fork authors do not need Actions
write permission. The controller runs only on GitHub-hosted runners and never
checks out PR code or executes comment text as shell commands.

`ci/comment_cases.json` lists aliases for all 21 scripts. An unknown alias
returns the available names. Other scripts use descriptive GPU/NPU aliases.
The approved SHA is checked again by the dispatched workflow: new commits,
edited requests, closed PRs, and duplicate approvals cannot silently start a
new run. A failed dispatch requires a new request. Results appear on the pinned
PR commit as in manual dispatch. If environment reviewers are configured,
the existing hardware environment approval still applies in addition.

Maintainers may configure `SPECO_CI_CASE_SETTINGS_JSON` as a repository variable:

```json
{"eagle3":{"environment":{"MODEL_PATH":"/models/Qwen3-8B"}}}
```

Each alias can specify `environment` (string-valued variables), `hydra_args`,
`npu_image`, and `verl_ref`. These settings are trusted maintainer configuration;
PR comments select only allowlisted aliases, never arbitrary images or commands.
Specialized cases still require compatible runtime/model/service settings;
separate-training behavior is not changed by this request mechanism.

## Shared online smoke parameters

Automatic NPU online examples and the unified manual/comment entry use
`ci/online_smoke_profile.json`: 2 total training steps, 1 epoch, collection /
drafter training / publication every step, 1 drafter optimizer step per cycle,
64 training samples per batch, 128/32 train/validation sample limits, PPO mini
batch 8, prompt/response limits 256/512 and hidden-state windows 32/32.
Publication is synchronous; periodic checkpoint saving and evaluation are off.
Explicit environment overrides win over repository/environment variables, then
the shared defaults apply. Explicit manual Hydra overrides are appended last.
Algorithm-specific scripts, device topology and runtime selection are preserved.
Separate-training profiles and checkpoint tests remain unchanged.
