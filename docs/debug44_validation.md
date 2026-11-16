# .44 Draft-OPD / SpecOPD short-run validation

Last updated: 11/16/2026

Goal: debug both methods inside `verl-speco-090` on `7.208.150.44`, with code
under `/home/c00954340` and 17k/AIME datasets under `/data/c00954340`.
No long training run is authorized or needed.

## Final short-run verdict (2026-11-16)

| Gate | Verified result | Authoritative artifact |
|---|---|---|
| Draft-OPD real co-training | Two steps; 8 collected samples/step; optimizer 1→2; finite losses 0.6675/0.8236; trained/published=1; exit 0 | `draft_opd-10.log`, config and exit |
| Strict-ACK SpecOPD real co-training | Two steps; 8 trajectory forwards/step; 112 scored positions/step; optimizer 1→2; finite losses 0.8023/0.7994; actor/draft source 0/0 then 1/1; publication ACK 1/1 then 2/2; exit 0 | `spec_opd-15.log`, config, exit and runtime source manifest |
| Complete proposals | Two K=7 branches/request, including rejected-tail tokens; wire reaches agent and joint scorer | SpecOPD serving transport lines and scored-position counts |
| AIME smoke validation | Both methods ran four distinct AIME prompts; acc@1=0 under max 128 generated tokens | Step-2 validation metrics in both full logs |
| Native NPU draft transport | 2 native-handler tests passed | `npu-transport-01.log/.exit` |
| Actual Qwen3-8B causal oracle | BF16 + FP32: 2 passed with physical layout controlled; branch logits/hidden and trunk auxiliary error 0; normal response LP max <2e-6 | `checkpoint-oracle-03.log/.exit` |
| Native DSpark q reconstruction | Same captured context/history and T=1, 7 positions; max TV 0.0243364, KL 0.0020278, sampled LP difference 0.101965; within predeclared 0.03/0.125 bounds | `q-native-probe-04`, `q-training-probe-06`, comparison JSON |
| Target dependencies / TP | Both TP ranks' sampled IDs identical and conditional logits exactly equal; checkpoint embedding/head equal in effective BF16 | `q-probe-04/dependency-verification.json` |
| Final local regression | 368 passed, 15 skipped; hardware tests are separately evidenced above | `final-local-02.xml` |

Full logs, configs, numerical captures and results are retained under local
`D:/CodexProject/specopd/tmp/debug44-artifacts` and remote
`/home/c00954340/specopd-debug-logs`. See [concise results](debug44_results.md).
The chronological entries below intentionally retain failed attempts; later
verdicts supersede earlier statements that hardware execution was pending.

| Date | Stage | Result / evidence | Remaining |
|---|---|---|---|
| 2026-11-16 | Environment, dataset/model inventory, current device occupancy | Started; local SSH evidence under workspace `tmp/ssh-result-*` | Exact versions, model paths and data formats |
| 2026-11-16 | Inventory/deployment | Passed; deployed 504 source files to `/home/c00954340/verl-SpeCo-debug`; torch 2.10/torch-npu 2.10, verl 0.9.1, vLLM 0.30.1 dev, Transformers 5.10.4 | Runtime-specific API compatibility and sampler parity |
| 2026-11-16 | Data | 17k parquet: 1,791,700 rows; AIME parquet: 960 rows; both have prompt/reward_model fields. Model paths `/data/c00954340/Qwen3-8B`, `/data/c00954340/qwen3_8b_dspark` | Use bounded debug subsets, retain original files |
| 2026-11-16 | Container focused tests and NPU availability | 94 passed in 15.03s; `runtime-tests-01.log`; NPU count 16 and sum probe 28.0 | Actual model NPU math and short RL runs |
| 2026-11-16 | Actual draft/actor model numerics on NPU | Started, timeout 240s; `npu-numerics-01.log` | Check gradients and method isolation before enabling NPU baseline launch |
| 2026-11-16 | NPU numeric attempt 1 | 9 passed, 3 failed: test built proposal batch explicitly on CPU while model was on NPU | Test placement corrected to actual model device; production code unchanged |
| 2026-11-16 | NPU numeric attempt 2 | **12 passed**, 6.34s; `npu-numerics-02.log`, exit 0 | Includes joint actor → DFlash/DSpark/DFlash2 backward and Draft-OPD CE/E2E-TV backward; actual serving/RL still pending |
| 2026-11-16 | Installed APIs | vLLM scheduler/output signatures present; verl compatibility reports no missing API, but 0.9.1 is outside declared exact-version matrix | Record experimental runtime, do not claim supported release until short runs pass |

Stages: inventory → isolated source deployment → focused runtime/numeric checks
→ proposal-distribution probe → one/two-step Draft-OPD → one/two-step SpecOPD
→ small AIME generation/validation and publication checks. Save separate logs
for each attempt. Existing source/data are not overwritten blindly. All launched
training processes must have bounded steps and recorded exit status.

Parallel run stage (2026-11-16): two isolated four-chip groups, Draft-OPD
`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, SpecOPD `4,5,6,7`. Both use 16 distinct
17k-derived training prompts, four distinct AIME prompts, max 128 generated
tokens, two optimizer/RL steps and 900s wall timeout. Results go to
`/home/c00954340/specopd-debug-logs/{draft_opd,spec_opd}-01*`; separate Ray
clusters have method-specific `/tmp/speco44-*` roots. Eight chips remain unused
by these jobs. NPU baseline launch guard was extended after real small-model
NPU backward tests passed; full serving behavior is still experimental.

Attempt 01: both tasks terminated before model loading in Gloo initialization
with `DistBackendError: File name too long`. No training success is claimed.
A standalone single-process probe reproduced the default failure and passed
with `GLOO_SOCKET_IFNAME=lo`; the single-host launch template now passes this
explicitly through Ray runtime env. Attempt 02 uses new logs/output directories.
The CUDA/NPU baseline config capacity tests passed locally (8 cases).

Attempt 02 progress: both independent Ray clusters are live, four-rank Gloo
groups connected successfully, actor weights loaded, and each group launched
two TP=2 vLLM engine cores. No rollout/update success is claimed yet. Current
host PIDs observed: drivers 1336164 (Draft-OPD), 1336097 (SpecOPD); respective
timeout supervisors 1336152/1336092. A later poll must revalidate these handles
and exit files before any restart.

Attempt 02 verdict: both jobs exited 1 during actor → vLLM weight update/wake-up,
before rollout. Ascend rejected default `FRACTAL_NZ` mode for RL hot updates.
Actor and native DSpark weights had loaded, and four-rank communication passed.
Attempt 03 explicitly sets `engine_kwargs.vllm.additional_config.weight_nz_mode=0`
as required by the installed runtime, with fresh logs/output paths. This is a
launch configuration change; loss/data semantics remain unchanged.

Attempt 03: Draft-OPD completed two RL steps and AIME generation (exit 0), but
**drafter trained=0 and collected=0**, so this is not a Draft-OPD success.
The baseline requires `train_window + 1` hidden rows; a 128-row window with
128 generated tokens made every sample too short. Short template window is
reduced to 64 (original PR semantics unchanged).
SpecOPD generated successfully but stopped on an empty proposal wire at
old-logprob (exit 1). The server-output → agent-trajectory boundary discarded
the custom field; a per-trajectory ContextVar bridge now forwards the single
complete wire, with multi-request trajectories rejected. Agent regression:
21 passed / 2 missing-verl skips locally. Attempt 04 uses fresh paths and must
prove actual drafter collection, gradient update and publication.

Bridge regression follow-up: local protocol/lifecycle/agent suite passed
47 tests with 2 missing-verl skips. Additional asynchronous trajectory isolation
and multi-request rejection tests passed in the five-case lifecycle suite.
Attempt 04 is live in separate device/Ray groups (timeout supervisor host PIDs
2511287 and 2511372 observed), currently loading actor/worker state. Do not
count startup or baseline RL-only progress as successful drafter training.

Attempt 04 verdict: both exited 1. Draft-OPD reached real feature collection
and rejected `use_remove_padding=false`; its original old-logprob collector
requires unpadded actor execution. The launch now selects original
remove-padding/FlashAttention behavior for the baseline and dense eager tree
execution for SpecOPD. These are debugging runs, not matched performance runs.
SpecOPD still had an empty wire despite the agent bridge. Saved logs showed
the HTTP `generate` bridge never ran. A fresh-base regression reproduced a Ray
serialization bug: the dynamic subclass inherited driver-patched upstream
methods, which disappeared when that base was imported afresh in a worker.
The subclass now explicitly owns its launch/generate/helper methods. Red run:
1 failed / 158 passed; after fix: 185 passed / 2 missing-verl skips including
agent/lifecycle contracts. Attempt 05 adds transport boundary diagnostics.

Attempt 05: both exited 1 before drafter updates. Baseline padding is now
accepted, but its trace was missing; SpecOPD still had no proposal wire. A
no-model native Ray/cloudpickle probe proved copying identical parent methods
is insufficient: cloudpickle removes those entries from subclass state (the
actual class pickle was only 369 bytes and contained no bridge methods).
Distinct async forwarding closures plus the stable helper mixin now preserve
the bridge (native probe pickle 30,350 bytes). A separate Python subprocess
regression verifies pristine base returns `native` while the deserialized
server returns `bridge:native`. HTTP regression 159 cases passed before this
additional subprocess case; source semantics unchanged for the two losses.
Attempt 06 keeps method-specific padding, explicitly propagates project
PYTHONPATH to Ray/spawned processes and uses new bounded run/log paths.

Final bridge follow-up suite: **186 passed, 2 skipped** locally, including the
real fresh-interpreter cloudpickle test. Shared DFlash short launcher also now
selects method-specific padding; four shell/config resolution cases passed.
Attempt 06 live supervisors observed on host: 309983 and 310011, both bounded
to 900 seconds; no completion or drafter update claimed yet.

Attempt 06 verdict: both exit 1; trace/proposal transport remains empty.
A real Ray actor (no model) confirmed the serialized HTTP method bridge is
present in a fresh worker; the earlier cloudpickle fix alone therefore does
not resolve this run. Attempt 07 adds opt-in stdout diagnostics at HTTP and
client/worker boundaries because module loggers may be disabled by native
logging configuration. Focused agent/lifecycle regressions: 26 passed,
2 missing-verl skips. No drafter optimizer update verified yet.
Attempt 07 stopped before model loading: diagnostic env value was parsed as
an integer by Hydra and rejected by Ray (env_vars requires strings). Quoted
the value explicitly; attempt 08 launched on the same disjoint chip groups
with fresh log/output paths. No model or loss change in this retry.

Attempt 08 verdict: both exit 1; stdout diagnostics decisively show the HTTP
bridge executes, but `config_present=False` on both methods. SpecOPD revision
0 reaches the server correctly; collection is disabled because the server has
no training config. Earlier absence of logger messages did not prove the
bridge was absent. Found and reproduced late-config reinstall bug: initial
bridge class retains None after configuration becomes available (red test).
Fix: refresh HTTP class snapshot before the replica idempotence return and
explicitly carry the snapshot in per-actor runtime_env, which upstream replaces.
HTTP regression suite: 163 passed, including late-config reinstall and class/
environment/absent payload paths. Attempt 09 launched with fresh bounded logs.
A read-only Ray attach probe failed authentication; automatic review rejected
reading process auth environment, so that action was not executed. Continue
using opt-in task-owned stdout diagnostics instead; no auth changes made.

Attempt 09: both exit 1 with HTTP config still empty. The late-refresh fix is
valid in isolation but insufficient for the actual startup path. A no-model
native Ray replica probe confirms its config snapshot survives serialization
(payload/env/snapshot all true), so do not claim another serialization fix is
proven to solve the RL job. Entry point now explicitly puts the resolved JSON
in the Ray job runtime_env and TaskRunner runtime_env before child imports.
Local entrypoint checks: 7 passed, 7 dependency-gated skips; HTTP suite 163
passed; targeted Ruff passed. A fresh native Ray job-env/actor-env merge probe
is running before the next heavyweight attempt.

Native job-env probe passed after clearing the driver's canonical env: the
child actor received the config from Ray runtime_env even when actor-specific
env_vars were supplied (payload/env/snapshot all true). Attempt 10 launched
with this entrypoint fix, fresh bounded logs, startup restore diagnostics and
proposal-count/rejected-tail-count diagnostics. HTTP suite remains 163 passed.
No short-run drafter update/publication claimed until actual run evidence.

Attempt 10 transition: both HTTP servers now have environment/embedded
configuration. Draft-OPD receives real verification traces, collected 8 samples
and reports trained=1, published=1 at RL step 1; job still live, step 2/AIME
pending. SpecOPD enables capture at revision 0, but exits 1 in the scheduler's
complete-proposal guard. Guard kept strict; added width/type/negative-token
and offset details to distinguish truncation, placeholder and scalar types.
SpecOPD-only attempt 11 uses cards 4–7 with fresh paths; baseline 10 continues
untouched. Device occupancy confirmed separate WorkerDict groups on 0–3 and
4–7. Actual checkpoint is vanilla Markov rank 256, full vocab 151936, K=7.

Draft-OPD attempt 10 completed with exit 0: two RL steps, each collected 8
samples and reported trained=1/published=1. AIME four-prompt validation ran
at step 2; acc@1=0 under the 128-token debugging cap, not a quality result.
Authoritative full baseline log/config/exit and SpecOPD-10 failure log/exit
copied to local `tmp/debug44-artifacts/`; baseline does not need another run
for SpecOPD-only capture changes. Remaining: inspect baseline finite loss/
optimizer evidence; finish SpecOPD capture, joint score, update and publication.

Baseline finite objective evidence: `[drafter loss]` reports E2E-TV total
loss/ploss 0.6675 and 0.8236 at steps 1 and 2; optimizer_step_total advances
1 → 2. The separate `dspark/ce_loss` metric is a CE diagnostic, not the selected
training objective. Full log includes both optimizer updates and publication
success; normal actor/RL outputs and final validation remain present.

SpecOPD attempt 11 exit 1; decisive guard details: raw_width=7,
scheduled_width=7, negative_tokens=7, offset=1. Native MRV2
`DraftTokensHandler` skips real-token D2H for ordinary batches and returns
all -1 placeholders, even with synchronous scheduling. This is not an empty
prefill to skip: real verification rounds also use the placeholders.
Added a SpecOPD-only token transport hook. CUDA reuses native async copying
through a read-only setter input proxy; the real batch/grammar flag is unchanged.
Ascend constructs the native helper under torch_cuda_wrapper but runs its setter
outside that scope, so the SpecOPD NPU path explicitly copies only the small ID
rows to CPU and returns them through the original native DraftTokenIds result.
No model forward, resampling or hidden/probability supervision is added.
Focused HTTP/lifecycle run 170 passed, then 9 lifecycle cases passed after the
NPU-specific transport refinement. Actual native NPU copy probe on device 8
passed with IDs [11,12,13], original structured flag still false. First probe
had CUDA-context constructor failure; corrected to match native initialization.
Attempt 12 launched SpecOPD-only on cards 4–7, two steps/900s. Baseline 10
remains a completed separate artifact; no unnecessary repeat.

SpecOPD attempt 12: exit 1 on second-step actor-revision regression, after a
real first update. Captured two complete K=7 branches per request, including
rejected tails (e.g. 8 tail tokens), and transported the wire to the agent.
Step 1: model_forward_calls=8 for 8 trajectories; logical_rows=2334 versus
independent_rows=4805; scored_positions=112; trained=1; actual loss=0.8023;
optimizer_step_total=1. These are execution/count evidence, not speedup claims.
Root cause for step 2: draft-only compat publication called set_global_steps,
relabeling the native actor snapshot during bootstrap. Cycle advancement then
correctly rejected the next actor label as regressed. SpecOPD now leaves the
native actor global_steps untouched on draft-only publication; separate drafter
ACK revisions remain in use. Baseline/ordinary behavior is preserved.
Red adapter regression: 1 failed / 2 passed; after fix + two-cycle lifecycle
and publication-ACK metrics: 174 passed. Publication metrics now reach the
same actor_output that upstream logs after update_weights, rather than lagging.
Native NPU transport regression is now a repository test: 2 passed in 15.54s,
logs npu-transport-01, copied locally. Full SpecOPD-12 log/config/exit also saved
locally. Attempt 13 launched SpecOPD-only on cards 4–7, fresh bounded paths.

Strict serving-revision audit while attempt 13 runs: the legacy path had
acknowledged weight publication but only carried the expected drafter revision
in request metadata. Added explicit post-commit ACKs from every rollout frontend
for (actual native actor global_steps, installed drafter revision), and require
consensus. Requests now reject a missing/stale installed drafter revision before
generation; actor bundles must match the acknowledged target snapshot. Added
negative replica/request tests; HTTP/lifecycle suite 179 passed. These changes
are not deployed into the live attempt 13, avoiding mixed code versions; a clean
final run will validate the stricter boundary after attempt 13 terminates.

SpecOPD attempt 13 completed two RL steps and AIME validation, exit 0; no owned
processes remained at the terminal poll. Its version-separation fix is now
verified across the second rollout/update. Full log/config/exit saved locally.
A clean attempt 14 now validates the additional explicit serving ACK boundary
and request-side checks. It uses fresh output/log paths, source SHA256 manifest
(505 .py/.yaml/.sh files) and the same bounded 2-step/900s configuration.
No source will be hot-swapped into this final candidate while it is live.

Attempt 14 stopped at initial serving ACK handle lookup (exit 1): verl 0.9.1
keeps LLMServerManager separate from AgentLoopManager. A real-layout regression
reproduced the missing-handles error (1 failed/1 passed); ACK lookup now uses
trainer.llm_server_manager.server_handles, with legacy manager fallback. Local
lifecycle: 13 passed. Attempt 15 (clean runtime manifest, no runtime hot swaps)
completed both RL steps, all joint scores, actual updates and explicit serving
ACKs, plus AIME validation; exit 0. Logs/config/exit/manifest saved locally.
Step 1: actor/draft source revisions 0/0, ACK 1/1 after publication; step 2:
source 1/1, ACK 2/2. Each step: 8 joint forwards, 112 scored proposal positions,
trained=1 and published=1. Losses 0.8023 and 0.7994, optimizer steps 1 and 2.

Final local combined pre-oracle regression: 367 passed / 14 skipped (Junit
final-local-01.xml). The skipped modules/cases do not verify hardware.

Real Qwen3-8B NPU oracle stage: initial compact BF16 oracle exposed physical
layout sensitivity (LP max 0.352724, branch TV 0.028218 on the synthetic fixture).
FP32 errors were small. Equal-length padding alone was insufficient for branch
hidden values because it did not preserve the physical KV reduction order.
Independent ordinary-causal oracles now use masked padding holes and explicit
logical positions, containing neither sibling nor future-trunk tokens, to align
physical layouts without changing the original tolerance. Oracle 03: 2 passed
(BF16 and FP32), branch logits distributions/hidden and auxiliary hidden max
error 0; response LP max 1.43e-6 / 1.91e-6. Compact-layout sensitivity remains
reported as a diagnostic, not silently relaxed away. Logs saved locally.

Remaining numerical gate: native DSpark serving q versus training reconstruction
under identical captured context/state/head/temperature. A one-request inference
probe uses native worker collective_rpc to capture initial aux features and all
seven conditional logits/IDs; this is validation-only, never training input.
First standalone startup hit the container's OpenMP invalid thread-pool failure;
retry 02 uses spawn and OMP_NUM_THREADS=2, matching the working launcher style.
It is inference-only on dies 0–1, 360s timeout; both training jobs are finished.

Final numerical gate: native q probe 04 exited 0 after capturing an actual
96-token/7-position DSpark proposal. Probe setup fixes were limited to external
validation scripts: spawn/OMP threads, named RPC (no insecure pickle opt-out),
Transformers 5 integer token extraction, native request-pool slot indexing, and
backend.build_model's tuple return. Reconstruction 06 exited 0, max TV 0.0243364,
KL 0.0020278, sampled LP delta 0.101965. The complete distributions and fixed
sampled IDs were compared; no resampling or teacher-label training data was used.
Both TP ranks' drafts and logits agree exactly; effective BF16 checkpoint head
and embedding equal target checkpoint values. Existing DSpark target-update
hook copies Actor LM head without replacing online-owned drafter weights;
current checkpoint owns a frozen embedding. Raw captures/results are archived
locally under q-probe-04. The bounds are single-probe tolerances, not universal
or bitwise-equivalence claims.

Final regression: **368 passed, 15 skipped**, 9.80s; final-local-02.xml.
Remote final process audit: no q probes, EngineCore/TP or training processes from
this task remained in verl-speco-090. All task NPU allocations released. A new
VLLM::Worker PID 1799413 on physical die 3 is outside this container and started
at 16:35, after these probes; it was not touched. Container left running.
Final scope: short-run functional/numerical validation completed for the named
DSpark/NPU/MRV2/legacy-sync combination. Quality, long contexts, other backends,
CUDA production serving, host/Ray peak profiling and fair performance studies
remain future work, not claimed by this task. See debug44_results.md.

Final file checks: targeted Ruff and git diff --check passed (only Git's
line-ending notices). Runtime package SHA256s match the successful SpecOPD-15
candidate. A final deployment audit found one outdated launch-config test file;
it was synchronized without changing runtime code or rerunning completed
hardware stages. Final matching source/test/example/document manifest is
final-source-verification-02.json. No further accelerator job was launched.
