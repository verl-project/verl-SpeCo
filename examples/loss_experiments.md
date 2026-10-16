# LK and D-PACE experiment examples

These Qwen3-8B examples adapt the loss and decoding configurations used for the
results reported in [PR #105](https://github.com/verl-project/verl-SpeCo/pull/105#issuecomment-6206864821)
to four GPUs with sequence parallelism of four. The reported experiments used
Qwen3-1.7B for EAGLE3 and Qwen3-4B for DSpark and DFlash.
Paths and device selection are configurable for other machines.

| Configuration | Script |
| --- | --- |
| Qwen3-8B EAGLE3 + LK | [EAGLE3 script](run_qwen3-8b-lk-loss_drafter_eagle3_vllm.sh) |
| Qwen3-8B DSpark + adaptive hybrid LK | [DSpark script](run_qwen3-8b-adaptive-hybrid-lk_drafter_dspark_vllm.sh) |
| Qwen3-8B DFlash + D-PACE | [DFlash script](run_qwen3-8b-dpace_drafter_dflash_vllm.sh) |

Use a Linux GPU environment with the backend-specific dependencies described in
the repository README. Run from the repository root and provide the target model,
a compatible pretrained drafter, DAPO-Math-17k training data filtered to 512 prompt
tokens, and AIME 2024 validation data:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export DRAFTER_PATH=/path/to/vllm-compatible-qwen3-8b-dflash-drafter
export TRAIN_FILE=/path/to/dapo-math-17k.filtered_512.parquet
export TEST_FILE=/path/to/aime-2024.parquet
export CKPTS_DIR=./checkpoints/dflash-dpace-train
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash examples/run_qwen3-8b-dpace_drafter_dflash_vllm.sh
```

Select the corresponding script and model/drafter paths for EAGLE3 or DSpark.
All scripts default to four GPUs, rollout tensor parallelism of two, sequence
parallelism of four, and 16 Ray CPUs. Override these with
`SPECO_N_GPUS_PER_NODE`, `SPECO_GEN_TP`, `SPECO_TRAIN_SP`, and
`SPECO_RAY_NUM_CPUS`. Each script has its own default checkpoint directory.

To run the fixed-drafter baseline, use the same script and inputs with these
overrides (the actor still trains):

```bash
CKPTS_DIR=./checkpoints/dflash-dpace-baseline \
bash examples/run_qwen3-8b-dpace_drafter_dflash_vllm.sh \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=False \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=False \
    trainer.experiment_name=qwen3_8b_dflash_dpace_vllm_baseline
```

Trailing arguments override Hydra settings, including `trainer.total_training_steps`
and `trainer.logger`. All scripts default to 100 total steps; the DSpark result
uses the first 48 matched steps. Compare `drafter/spec_decode/mean_acceptance_length`,
`perf/time_per_step`, and `timing_s/gen` over the same step window in both runs.

The DFlash example follows the updated 100-step experiment: four verification
tokens and synchronous weight publication (`publish_async=False`). EAGLE3 and
DSpark retain their experimental verification lengths of four and seven tokens,
respectively, and asynchronous publication.
