# OPD collection-step capture gate

Last updated: 11/17/2026

Scope: Draft-OPD per-request verification trace opt-in, using the pre-rollout
collection planner; SpecOPD retains its existing collection-step proposal plan
and now gates MRV2 real draft-ID transfer on runner-local opted-in requests.
Normal acceptance counters, native structured-output transport and RL generation
remain enabled on non-collection steps. No rollout teacher tensors are collected.

Isolation: worktree `verl-SpeCo-opd-collection-gate`, branch
`fix/opd-collection-step-gate`. Do not overwrite the other session's modified
`docs/debug44_ulysses_validation.md` or its remote source/run directories.

| Stage | Evidence | Result |
|---|---|---|
| Local request/scheduler/lifecycle regressions | workspace `tmp/opd-gate-local03` | 233 passed |
| Local native SamplingParams decision tests | `tmp/opd-gate-local04` | 4 passed; 2 NPU tests skipped |
| .44 inventory | `tmp/ssh-result-7.208.150.44-20261117-023437-175283.json` | Other session running on logical devices 4–7; no task stopped. Native API inspected read-only. |
| .44 installed-runtime and NPU gate validation | `/home/c00954340/opd-collection-gate-20261117-0240/capture-gate-tests-01.log` | 239 passed, 31.24s; includes actual NPU tensor and native MRV2 handler. |
| .44 final collection policy | `capture-gate-tests-02.log`, exit 0 | 54 passed (48 trace cases rerun plus 6 new trainer policy cases), 24.51s; collection/training intervals, data buffer, validation and freeze gates passed. |
| .44 native install probe | `tmp/ssh-result-7.208.150.44-20261117-024353-596656.json` | `NATIVE_MRV2_INSTALL_OK`; installed runner and handler patch successfully. |
| Merge into shared local checkout | `git apply --check` then apply task-only diff | Other session changed only its Ulysses validation document; no overlap or overwrite. |

Acceptance: no verification trace on unselected requests; trace remains complete
on selected requests; normal acceptance statistics retained; skipped SpecOPD
requests have no proposal plan and no extra MRV2 ID copy. Mixed batches may copy
a whole ID tensor if any request is selected, using the native transport layout.
No throughput claim is made without a matched serving benchmark.


Remote evidence: `tmp/ssh-result-7.208.150.44-20261117-024148-024476.json`
and `tmp/ssh-result-7.208.150.44-20261117-024353-596656.json`. The final trainer
helper and its test were uploaded only to this task's isolated source root.
Changed Python files pass Ruff; diff whitespace checks pass. This task does
not claim a complete RL run or a matched end-to-end throughput result.
