#!/bin/bash
# ─── Cleanup: kill all stale processes and wait for NPU memory release ───
echo "=== Cleaning up stale processes ==="
ray stop --force 2>/dev/null
sleep 2
kill -9 $(ps aux | grep -E "python3|ray|EngineCore|vllm" | grep -v grep | grep -v "$$" | awk '{print $2}') 2>/dev/null
sleep 5
rm -rf /tmp/ray/session_* /home/model/tmp/ray_tmpdir/ray /tmp/torchinductor_root /tmp/speco_precision_ckpt_* 2>/dev/null

echo "=== Waiting for NPU memory release ==="
for i in $(seq 1 12); do
    _free=$(python3 -c "import torch_npu; print(int(torch_npu.npu.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    if [ "$_free" -ge 60 ]; then
        echo "NPU0: ${_free}G free — OK"
        break
    fi
    echo "  NPU0: ${_free}G free — waiting... (${i}/12)"
    sleep 10
done

echo "=== NPU status ==="
python3 -c "import torch_npu; [print(f'  NPU{i}: {torch_npu.npu.mem_get_info(i)[0]/1e9:.1f}G free') for i in range(8)]" 2>/dev/null
echo "=== Disk status ==="
df -h / | tail -1
echo "=== Starting training ==="

set -x
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHDYNAMO_DISABLE=1
case "${LD_PRELOAD:-}" in
    *libjemalloc*) ;;
    *)
        if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        elif [ -f /usr/lib64/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
        ;;
esac
export MALLOC_CONF="${MALLOC_CONF:-narenas:8,thp:never,metadata_thp:disabled,dirty_decay_ms:0,muzzy_decay_ms:0}"
export SPECO_JEMALLOC_RECLAIM_MODE="${SPECO_JEMALLOC_RECLAIM_MODE:-purge}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

# NOTE: do NOT set PYTORCH_NPU_ALLOC_CONF=expandable_segments:True — vLLM-Ascend
# CaMemAllocator asserts it is incompatible with the memory pool.
export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MOD=AIV

project_name='verl_grpo_example_dapo_drafter_megatron'

gen_tp=2
actor_tp=8
actor_pp=1

exp_name="qwen3_8b_drafter_eagle3_vllm_npu_megatron_tp${actor_tp}pp${actor_pp}_$(date +%Y%m%d_%H%M%S)"

ppo_gpus_per_node=${SPECO_ACCELERATOR_COUNT:-8}
ray_num_cpus=${SPECO_RAY_NUM_CPUS:-64}
ray_worker_soft_limit=${SPECO_RAY_WORKER_SOFT_LIMIT:-8}

MODEL_PATH=/home/model/Qwen3-8B
CKPTS_DIR=/home/model/ckpts/qwen3_8b_eagle3_speco_megatron
TRAIN_FILE=/home/dataset/DAPO-Math-17k/data/dapo-math-17k-train-10k.parquet
TEST_FILE=/home/dataset/DAPO-Math-17k/data/dapo-math-17k-val.parquet
DRAFTER_PATH=/home/model/Qwen3-8B_eagle3_AngelSlim

mkdir -p /home/model/tmp/ray_tmpdir
PYTHONUNBUFFERED=1 VERL_DRAFTER_ALIGNMENT_DEBUG=1 VERL_DRAFTER_ALIGNMENT_DEBUG_EVERY_N_STEPS=1 VERL_DRAFTER_LAST_HIDDEN_LOGPROB_CHECK=1 RAY_TMPDIR=/home/model/tmp/ray_tmpdir python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    transfer_queue.enable=False \
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus} \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=256 \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=NONE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm="EAGLE3" \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=4 \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=True \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.load_format="auto" \
    model_engine=megatron \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${ppo_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ 2>&1 | tee /verl-SpeCo/logs/${exp_name}.log &

# --- Kill after step 1 completes ---
TRAIN_LOG=/verl-SpeCo/logs/${exp_name}.log
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID, log: $TRAIN_LOG"
echo "Will kill after step 1..."

for i in $(seq 1 180); do
    sleep 30
    if grep -q "step:1 " "$TRAIN_LOG" 2>/dev/null; then
        echo "Step 1 found! Waiting 10s for logs to flush..."
        sleep 10
        echo "Killing training..."
        kill -9 $TRAIN_PID 2>/dev/null
        ray stop --force 2>/dev/null
        sleep 5
        kill -9 $(ps aux | grep -E "python3|ray" | grep -v grep | awk '{print $2}') 2>/dev/null
        break
    fi
    echo "  [$((i*30))s] waiting... ($(grep -c 'step:' "$TRAIN_LOG" 2>/dev/null || echo 0) steps, $(ps aux | grep python3 | grep -v grep | wc -l) procs)"
done

echo ""
echo "========================================"
echo "STEP 1 ANALYSIS"
echo "========================================"
echo "--- TP ranks calling forward_step ---"
grep "\[SPECO RANK\]" "$TRAIN_LOG" 2>/dev/null | grep -o "tp_rank=[0-9]*" | sort | uniq -c | sort -rn
echo "--- sp_size in hooks ---"
grep "megatron hooks installed" "$TRAIN_LOG" 2>/dev/null | grep -o "sp_size=[0-9]*" | sort | uniq -c | sort -rn
echo "--- local_positions ---"
grep "megatron hooks installed" "$TRAIN_LOG" 2>/dev/null | grep -o "local_positions=[0-9]*" | sort | uniq -c | sort -rn | head -5
echo "--- collected_samples ---"
grep -o "drafter/collected_samples:[0-9.]*" "$TRAIN_LOG" 2>/dev/null | head -3
echo "--- trained ---"
grep -o "drafter/trained:[0-9.]*" "$TRAIN_LOG" 2>/dev/null | head -3
echo "--- eval probe ---"
grep "eval probe" "$TRAIN_LOG" 2>/dev/null | head -5
echo "--- acceptance_length ---"
grep -o "mean_acceptance_length:[0-9.]*" "$TRAIN_LOG" 2>/dev/null | head -5
echo "--- sequence_parallel in config ---"
grep "sequence_parallel" "$TRAIN_LOG" 2>/dev/null | head -5 | cut -c1-150
echo "========================================"
