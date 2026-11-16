# Joint trajectory and branch scoring

Last updated: 11/16/2026

## Implemented scope

`verl_speco.joint_scoring.score_joint_batch` computes normal response logprobs
and every supplied draft branch's target supervision in **one model call**.
Each trunk occurs once; branch ancestry is enforced by a 4D additive mask and
logical RoPE positions. This is a dense correctness implementation, not a
FlashAttention performance claim.

The exact original prompt plus final rollout response is the trunk. A branch
contains its anchor's zero-based trunk index and the complete valid proposal,
including rejected tail tokens. Invalid padding, bonus tokens and tokens after
the first branch EOS must already have been removed by the caller. No decoding,
sampling, re-tokenization or rejection filtering occurs in this executor.

```text
trunk:       A B C D E F
branch at C:       u v w
branch at E:           x y z
physical:    A B C D E F | u v | x y
RoPE:        0 1 2 3 4 5 | 3 4 | 5 6
readout u,v,w: C,u,v
readout x,y,z: E,x,y
```

Terminal proposal tokens need no input rows. Normal trunk queries cannot read
branches; branch queries cannot read future trunk tokens or sibling branches.
Different requests occupy separate batch items and padding is masked.

## API example

Use an existing **unpatched** HF full-attention Qwen3/Qwen2/Llama causal LM with
eager or SDPA attention. For a local CPU experiment, instantiate a tiny random
Qwen3 model from config; no checkpoint download is required.

```python
from transformers import Qwen3Config, Qwen3ForCausalLM
from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, score_joint_batch

config = Qwen3Config(
    vocab_size=48, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
)
config._attn_implementation = "eager"
model = Qwen3ForCausalLM(config)
request = JointScoringRequest(
    sample_id="request-0",
    trunk_ids=(1, 2, 3, 4, 5, 6),
    response_start=3,
    branches=(
        DraftBranch("branch-0", anchor=2, token_ids=(20, 21, 22)),
        DraftBranch("branch-1", anchor=4, token_ids=(30, 31, 32)),
    ),
)
results, metrics = score_joint_batch(
    model, [request], calculate_entropy=True, hidden_state_indices=(1,),
)
assert metrics["model_forward_calls"] == 1
assert results[0].response_log_probs.shape == (3,)
assert results[0].branches[0].target_logits.shape == (3, 48)
```

`response_start` is the index of the first response token. The first response
logprob uses the preceding prompt row. `response_log_probs` and optional entropy
contain only real response positions, in original order; branch/padding rows
never enter this output.

Branch outputs optionally include raw full-vocabulary predecessor logits, final normalized
hidden states, and FP32 `logsumexp(logits / target_temperature)`. Actor and target
temperatures are separately specified. Requested trunk features use **HF tuple
indices**, where index 0 means embeddings; these are not automatically converted
from SpeCo auxiliary-layer IDs. A caller must perform that explicit conversion.

The function uses no gradients, temporarily evaluates the model and restores
every module's original training flag even on failure. It does not update model
parameters, optimizer state or the sampling policy.

## Actor entry point

The real RL actor's existing `DraftWeightPublishMixin` now exposes an explicit
`compute_joint_old_log_prob(requests, **options)` RPC. It delegates through
`verl_speco.joint_actor.score_actor_joint`, enters the actor engine's `eval_mode`
context, respects its autocast dtype, and returns CPU scoring snapshots. It is
not attached to the SFT-only `SpecoTrainingWorker`.

The adapter assigns complete trajectories to FSDP ranks using one shared
schedule. Empty ranks execute a one-token dummy, and rank outputs are restored
to request order. Preflight and schedule consensus occur before the forward.
Real multi-rank collectives and NPU kernels still require environment validation.
Remove-padding, sequence parallelism, fused forward and LoRA are rejected;
the core also rejects FlashAttention, sliding-window and hybrid attention.

The automatic production entry now exists behind the unified selector:

```yaml
drafter_opd:
  enable: true
  method: spec_opd
```

`specopd_driver.py` replaces normal old-logprob rather than stacking another
forward. It restores response logprobs/entropy into DataProto and routes trunk
features and branch predecessor hidden into a separate fixed-proposal payload.
The driver requests `return_target_logits=False` and
`capture_empty_trunk_features=False` to avoid unnecessary host transfers.
vLLM complete-proposal collection, revision checks, one-use drafter training
and acknowledged publication are wired through their own hooks. Selecting
`draft_opd` retains PR #1's ordinary trajectory path. See
[implementation and validation](specopd_development.md) for launch constraints,
reuse attribution and the distinction between local tests and hardware testing.

## Cost limits

`max_packed_tokens` limits the sum of real tree rows. `max_mask_elements` limits
the **padded** batch mask size (`batch * max_length**2`) before allocation. The
executor rejects over-budget inputs rather than silently dropping branches or
truncating prefixes. The caller chooses microbatches before entering distributed
execution.

Dense masks and full logits/hidden-state materialization can consume substantial
memory. The reported `logical_rows` versus `independent_rows` counts projection
rows, not measured runtime, FLOPs or peak-memory reduction. Tree masking alone
does not reduce the dense attention kernel's padded quadratic work.

## Validation

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_joint_scoring.py tests/unit/test_joint_actor.py -q
```

Tests instantiate real tiny random Qwen2, Qwen3 and Llama models with eager and SDPA attention.
They compare trunk logprobs/entropy, auxiliary features, branch logits, final
hidden states and partition functions against independent causal forwards.
They also verify a single model invocation, isolation from future trunk/sibling
tokens, mixed training-state/RNG preservation, variable batch lengths,
single-token branches, empty responses, budget rejection and actor lifecycle.
The actor context tests use a small in-process engine adapter; they are not a
substitute for multi-rank/Ray/NPU integration tests.
