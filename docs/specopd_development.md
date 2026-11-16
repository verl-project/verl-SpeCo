# Draft-OPD / SpecOPD implementation and validation

Last updated: 11/16/2026

Current hardware verdict: **Draft-OPD and SpecOPD short co-training passed** on
`.44` / `verl-speco-090`, Qwen3-8B + DSpark, Ascend NPU, native vLLM MRV2 and
legacy synchronous verl. Baseline attempt 10 and strict-ACK SpecOPD attempt 15
both exited 0 after two updates/publications and four-prompt AIME validation.
Real-checkpoint joint/independent causal oracles and a native-serving q probe
also passed their explicitly recorded numerical checks. This is a small debug
validation, not a quality, throughput or other-backend certification.
See [results and reproduction](debug44_results.md) and
[per-attempt ledger](debug44_validation.md).

## Integration source

Reuse source: [vx120/verl-SpeCo PR #1](https://github.com/vx120/verl-SpeCo/pull/1),
commit `7611fd559f3c5a8d89034683eacb436bf0a377b7`.
Only that PR's diff is being integrated; unrelated changes on its base branch
are not imported. Original workspace files are backed up outside the checkout
under `D:/CodexProject/specopd/tmp/pr-1-integration-original`.

## One selector, three behaviors

Public configuration is under
`actor_rollout_ref.rollout.drafter.training`:

```yaml
drafter_opd:
  enable: true
  method: spec_opd # draft_opd | spec_opd; default draft_opd
  capture_verify_trace: true
specopd:
  enable: false # deprecated compatibility flag, not needed with the selector
  proposal_width: 7
  max_branches_per_request: 8
  max_proposal_tokens_per_request: 56
```

- `enable: false` with the legacy flag false selects original SpeCo.
  For capture-free original SpeCo, also leave `capture_verify_trace: false`;
  PR #1's independent capture-only diagnostic setting remains compatible.
- `enable: true, method: draft_opd` keeps the PR baseline: visible accepted
  prefix plus first rejection, optional CE or weighted E2E-TV, and random/trace/
  mixed anchor planning. It uses ordinary trajectory feature collection and
  the existing training scheduler. It does not save complete proposal tokens.
- `enable: true, method: spec_opd` collects complete original proposals and
  uses joint actor tree scoring plus pure branch E2E-TV. Baseline trace capture,
  random-anchor ratio, first-rejection weighting and baseline loss selection
  do not alter this arm.
- Old `specopd.enable: true` remains compatible. Combining it with an enabled
  `method: draft_opd` raises an error before launch. Unknown methods also fail.

`verl_speco/opd_method.py` resolves the selection without mutating config.
No step executes both methods. Turning on SpecOPD does not automatically make
the baseline fully validated on NPU. After real small-model NPU backward tests
passed on .44, the CUDA-only guard was extended to accept NPU; actual short
serving/RL validation is tracked in [the .44 ledger](debug44_validation.md).

## Reused and extended code

| Area | Reuse from PR #1 | SpecOPD-specific implementation |
|---|---|---|
| Entry | `drafter_opd` config and disabled-by-default baseline | Method resolver; conflict checks and effective SpecOPD config |
| vLLM | Embedded config restoration and existing request-stat transport machinery | Pre-schedule full proposal snapshot, preplanned window, revision/digest, bounded JSON sidecar |
| Agent | Remote worker patch binding, extra-field propagation, turn trace helpers | Exact single-turn full-proposal channel and drafter revision hint |
| Actor | Existing engine, RPC registration, layer-ID helpers and LM-head synchronization | Joint tree compiler, logical RoPE, one forward, independent RL/branch readouts |
| Drafter | Existing DFlash/DSpark wrappers, context projection, block masks and optimization/publication infrastructure | Fixed full proposals, DSpark sampled predecessor, DFlash2 candidate selector, branch-only objective |
| Objective | PR's CE/weighted E2E-TV baseline, TV and prefix-survival calculation | Shared `opd_math.py`; SpecOPD keeps position normalization and candidate-outside target mass |
| Tests | PR trace/planner/loss/transport/runtime-control regressions | Mode isolation, complete tails, joint equivalence, real drafter backward and lifecycle |

The objective's block mean and first-rejection weights remain in Draft-OPD.
SpecOPD divides the accumulated failure sum by valid proposal positions over
the update group, without random trunk loss or rejection weighting. Chunked
teacher projection/checkpointing is retained in the PR baseline; SpecOPD's
teacher projection is currently bounded by the per-trajectory branch budget,
not claimed to have the same chunked implementation.

## Implemented SpecOPD path

1. Bootstrap from the actual loaded/restored drafter, synchronize the full
   target head and publish the drafter snapshot with acknowledgement.
2. Before generation, propagate the acknowledged drafter revision. The vLLM
   server plans a front/random response window and request/branch budgets.
   It forces synchronous scheduling; it captures raw `request.spec_token_ids`
   before scheduler consumption, including rejected tails, before output frees
   requests. Unsupported partial/invalid proposals fail rather than being
   presented as complete data. A first branch EOS ends that valid proposal.
3. Only tokens, prefix digests, coordinates and versions cross the rollout
   channel. Even an unselected/no-proposal request returns an explicit empty
   bundle. No rollout target hidden/logits are saved in this mode.
4. `SpecOPDDriver` compiles each exact trunk and all its collected branches.
   Whole trees are assigned to microbatches and FSDP ranks. Empty ranks have a
   dummy forward; real trajectories are not split or repeated.
5. `compute_joint_old_log_prob` enters the actual actor engine eval/autocast
   context and calls its model once per local microbatch. A tree mask and
   logical positions prevent future-trunk/sibling leakage. Normal response
   logprobs/optional entropy are restored to the original DataProto layout;
   selected trunk features and branch predecessor final hidden go separately
   to the drafter. The original normal old-logprob path is not called again.
6. Drafter training reuses all captured anchors without resampling. DFlash,
   DSpark and DFlash2 reconstruct their probabilities using the actual sampled
   histories. Microbatches accumulate to exactly one optimizer update, before
   actor update, using the same target head as scoring.
7. After actor weight publication, publish the updated drafter synchronously;
   do not start another rollout with an unacknowledged revision. Scored groups
   are consumed once. Failed/partial groups stop the run rather than replay.
   Resume discards in-flight data and bootstraps from restored model state.

## Launch examples

Run from the repository root with `MODEL_PATH`, `DRAFTER_PATH`, `TRAIN_FILE`,
`TEST_FILE` and `CKPTS_DIR` supplied. These scripts do not provision the user's
runtime or download checkpoints.

```bash
# PR baseline defaults, retained for DFlash and DSpark:
bash examples/run_qwen3-8b-draftopd_drafter_dflash_vllm.sh
bash examples/run_qwen3-8b-draftopd_drafter_dspark_vllm.sh

# Small synchronous DFlash SpecOPD run; N_GPUS defaults to 8, ROLLOUT_TP to 1:
OPD_METHOD=spec_opd N_GPUS=8 bash examples/run_qwen3-8b-specopd_drafter_dflash_vllm.sh

# Same small launch template with Draft-OPD selected:
OPD_METHOD=draft_opd N_GPUS=8 bash examples/run_qwen3-8b-specopd_drafter_dflash_vllm.sh
```

The short template shares proposal width 7, temperature 0.6 and synchronous
publication, with target full-vocabulary head synchronization. Baseline PR
scripts keep their original defaults; they are not budget-matched to this
short template. A quality/performance experiment must additionally match
data, parameter snapshots, sampling, update budget and run duration, and
report the different valid training-position counts.

SpecOPD requires full-attention Qwen2/Qwen3/Llama eager or SDPA actor, FSDP/
FSDP2, no actor sequence parallelism, no remove-padding/fused forward/LoRA,
single-turn exact token histories, probabilistic drafter sampling, and equal
rollout/actor/proposal/target temperature. Only unmodified sampling distributions
are supported: no top-k/top-p filtering, custom processors or penalties.
The short example uses eager explicitly. PPO/GAE and GRPO are supported by the
output adapter; external teacher distillation and old-logprob bypass are rejected.

`training.specopd` owns collection/scoring limits; original
`hidden_state_window_*` options continue to belong to baseline feature capture.
Dense tree masks are limited by both token and mask-element budgets. Oversized
complete trees fail before scoring; they are never silently split or truncated.

## Remaining environment validation

- Actual vLLM/Ascend proposal sampling, token identity, EOS and final-output
  transport on the installed runtime; strict serving-q/training-q parity probe.
- Real Ray actor RPC and FSDP/FSDP2 collectives, uneven/empty-rank participation,
  offload/autocast, failure and restore behavior.
- CUDA and NPU BF16 correctness, memory peak, latency and kernel support.
- Full rollout → reward → joint old-logprob → drafter update → actor update →
  publication/checkpoint multi-step run, with original RL output contracts.
- Controlled quality, acceptance length and wall-clock comparisons. One model
  call is a verified local property, not evidence of hardware speedup.

The local machine lacks installed upstream verl, vLLM, Ray and accelerators.
Transport/lifecycle tests use fakes; numeric tests use real small HF actors and
actual drafter models. Passing those tests is not an end-to-end training verdict.

Remote .44 validation is now underway in the user-provided `verl-speco-090`
container: 94 runtime tests and 12 actual small-model NPU numeric tests passed.
The first complete two-step baseline RL run did not update the drafter because
its short-test window was too large; SpecOPD's first rollout revealed a missing
LLM-output to agent-trajectory bridge. `specopd_agent.py` now carries the opaque
single-request proposal bundle through a trajectory-local ContextVar, mirroring
the PR's trace transport while retaining a separate payload. Full drafter
training and publication are not yet accepted. See [the live test ledger](debug44_validation.md).

## Test ledger

| Date | Stage / configuration | Result | Evidence / remaining checks |
|---|---|---|---|
| 2026-11-16 | Before PR integration: dense joint scoring, actor adapter, actual CPU DFlash/DSpark/DFlash2 backward and lifecycle contracts | 41 passed | Prior local run; later changes and integration require focused rerun. Real vLLM/Ray/FSDP/NPU untested. |
| 2026-11-16 | Before PR integration: existing agent/vLLM/scheduler/buffer contracts | 207 passed, 4 skipped across initial run and failed-only rerun | Windows temporary-directory errors resolved using workspace `--basetemp`; not a hardware verdict. |
| 2026-11-16 | Integrate pinned PR; unified method dispatch and compatibility checks | Started | Test artifacts will be stored under `D:/CodexProject/specopd/tmp/pr-1-validation`. No hardware jobs are launched. |
| 2026-11-16 | Selector + PR objective + joint scorer/actor + lifecycle + real drafter backward | 81 passed | `focused-01.log`; actual CPU model math, mocked cluster/transport. |
| 2026-11-16 | Integrated agent/vLLM/verify-trace/runtime-control/scheduler/buffer regressions | 239 passed, 6 skipped | `regression-01.log`; skipped imports/hardware do not verify distributed execution. |
| 2026-11-16 | Shared prefix-survival math, real Draft-OPD CE/E2E-TV DFlash/DSpark backward, SpecOPD DFlash/DSpark/DFlash2 backward and selection | 67 passed | `numerics-02.log`; real CPU tensor/model computation; unused verl device/scheduler imports shimmed in this model test only. |
| 2026-11-16 | Bash syntax and actual shell argv/config resolution, both methods and both baseline scripts | 4 passed; three `bash -n` checks passed | `launch-01.log`; training command intercepted, no model/Ray job launched. Example naming check passed for all 39 scripts. |
| 2026-11-16 | Strict wire versions/sampler identity, EOS tails, lifecycle and launch guards | 30 passed | `guards-01.log`; rejects invalid metadata and checks EOS-bounded raw proposal retention. |
| 2026-11-16 | Final combined numeric, protocol, selector, launch, lifecycle and regression run after formatting | **337 passed, 6 skipped** | `final-01.log`, 8.78 s. Skips: missing upstream verl/trainer stack and vllm_ascend; these checks remain pending. |
| 2026-11-16 | Static checks: Ruff, compileall, license, diff whitespace | Passed | `ruff.log`, `license.log`, `diff-check.log`. |
| 2026-11-16 | Device/DataProto boundary checks | Initial checks identified missing adapter registrations; documented entries added | Runtime CUDA selection and the legacy old-logprob DataProto return are intentionally confined to adapters. Rerun recorded below. |
| 2026-11-16 | Boundary checks rerun, document dates and test structure | Passed | `device-check-02.log`, `dataproto-check-02.log`, `docs-check.log`, `structure-check.log`. Hardware stages remain pending; no remote runs or jobs were started. |

Reproduce the combined local run from the repository root:

```bash
python -m pytest -q -rs \
  tests/unit/test_opd_method.py tests/unit/test_drafter_opd.py \
  tests/unit/test_specopd_contracts.py tests/unit/test_joint_scoring.py \
  tests/unit/test_joint_actor.py tests/integration/test_specopd_lifecycle.py \
  tests/integration/test_specopd_forward.py tests/integration/test_opd_launch_config.py \
  tests/integration/test_agent_loop_runtime_contract.py \
  tests/integration/test_vllm_runtime_contract.py tests/integration/test_verify_trace_runtime.py \
  tests/integration/test_drafter_runtime_control_contract.py \
  tests/integration/test_dspark_trainer_backend.py \
  tests/integration/test_vllm_native_acceptance_boundary.py \
  tests/unit/test_drafter_scheduler.py tests/unit/test_data_buffer.py tests/unit/test_opd_cotrain.py
```

Use the uv-managed virtual environment; on Windows add a fresh workspace
`--basetemp` directory if the default user temporary directory is restricted.
The six skips represent two whole dependency-gated modules and four individual
cases; the run does not claim to execute those modules' hardware/runtime tests.

### Remote transport configuration follow-up (2026-11-16)

On `.44`, native generation and the HTTP wrapper execute, but attempt 08 proved
that the HTTP process had no drafter training configuration. A bridge installed
before trainer configuration kept an empty class snapshot on reinstall. The
bridge now refreshes that snapshot and explicitly carries it in the HTTP
actor's runtime environment even when upstream supplies its own environment.
A regression reproduced the empty-snapshot failure and passed after the fix;
163 HTTP/runtime contracts passed. Attempt 09 is running; actual collection,
drafter update, publication and joint old-logprob remain unverified. See
`debug44_validation.md` for per-attempt evidence and short-run limitations.
