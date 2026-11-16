# Draft-OPD training for DFlash and DSpark

Last updated: 11/16/2026

Integrated from [vx120/verl-SpeCo PR #1](https://github.com/vx120/verl-SpeCo/pull/1),
commit `7611fd559f3c5a8d89034683eacb436bf0a377b7`.
The baseline objective and anchor semantics are retained. The shared selector
now also supports SpecOPD; see [implementation and validation](specopd_development.md)
for the differences, restrictions and local evidence.

Draft-OPD trains on real speculative verification anchors. This implementation supports GPU DFlash and standard anchor-as-first DSpark with full-vocabulary CE or weighted E2E-TV. It is disabled by default.

## Configuration

Set `actor_rollout_ref.rollout.drafter.training.drafter_opd`:

```yaml
enable: true
method: draft_opd # spec_opd selects the separate full-proposal joint-scoring path
capture_verify_trace: true
anchor_planner: mixed
random_anchor_ratio: 0.25
loss: e2e_tv
first_reject_weight: 2.0
chunk_size: 1024
```

The planner uses the existing `num_anchors` budget. `random` retains random anchors, `trace` samples captured verification anchors, and `mixed` reserves `ceil(num_anchors * random_anchor_ratio)` random slots. Missing trace anchors leave invalid slots rather than adding random replacements.

Trace blocks supervise the visible accepted prefix and first rejection. Bonus tokens are excluded; EOS and window truncation do not invent rejection labels. DFlash draft positions start at slot 1; DSpark starts at slot 0. Random blocks retain fixed exponential position decay. Visible first rejections receive `first_reject_weight`.

For each valid contiguous prefix:

```text
TV_j = 0.5 * sum_v |softmax(target_logits / T) - softmax(draft_logits / T)|
survival_j = product_{i<=j}(1 - TV_i)
loss_block = sum_j w_j * (1 - survival_j) / num_valid_positions
```

Temperature uses `lk_temperature`. Weights apply to cumulative prefix loss rather than survival probabilities. The denominator counts valid positions, so a single rejection retains its multiplier. Zero-weight positions remain in the survival chain. Valid blocks are averaged; empty blocks do not contribute. DSpark includes Markov bias in draft logits. Vocabulary scoring is chunked with activation checkpointing.

## Trace transport

The scheduler records anchor, draft count, accepted count, raw generated count and committed count before/after stop processing. Traces travel through request statistics, agent trajectories, old-logprob collection and feature metadata into padded training batches. Multi-turn token prefixes are checked by digest. Prompt padding and window offsets are accounted for. Missing or inconsistent traces raise errors.

The vLLM runtime restores private training configuration from the embedded launcher snapshot in synchronous mode and binds the remote agent worker explicitly so capture survives worker imports. Teacher scoring requests are excluded from student traces.

## Examples

Set model/data environment variables required by the base scripts, then run from the repository root:

```bash
bash examples/run_qwen3-8b-draftopd_drafter_dflash_vllm.sh
bash examples/run_qwen3-8b-draftopd_drafter_dspark_vllm.sh
```

Both examples use synchronous publication, mixed anchors, E2E-TV, full vocabulary and no packing/data buffer. Override `drafter_opd.loss=ce` or `drafter_opd.anchor_planner=trace` for ablations.

## Limits and validation

Packing, bonus-anchor DSpark checkpoints, non-label DSpark CE targets and the original DSpark confidence loss are rejected. This change does not include verification-aware training objectives. L2 rejected-token scoring, density-based window selection and staleness filtering are outside this implementation.

CPU tests cover trace coordinates and transport, planner behavior, weighted dense loss/gradient agreement, empty blocks, first-rejection weighting and zero-weight prefixes. GPU training, asynchronous scheduling, preemption, distributed empty-rank behavior and performance still require hardware validation. TV uses temperature-normalized raw logits and does not replay sampling processors; its survival chain is not a measured rollout acceptance guarantee.
