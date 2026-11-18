<p align="center">
  <a href="https://verl-project.github.io/posts/2026-11-16-verl-speco-v0-2-0/">
    <img src="docs/assets/speco-logo.png" alt="verl-SpeCo" width="70%">
  </a>
</p>

<h3 align="center">verl-SpeCo: A Unified Training System for Speculative Drafters</h3>

[Documentation](docs/usage.md) · [Releases](https://github.com/verl-project/verl-SpeCo/releases)

SpeCo keeps speculative drafters aligned with evolving RL policies. It reuses
training features, adapts the drafter online, and hot-updates rollout engines
without restarting the RL job. Built as an import-only extension of
[verl](https://github.com/verl-project/verl), it leaves upstream verl unmodified.
Standalone training can also produce a deployable drafter or initialize co-training.

SpeCo brings drafter training into practical RL and deployment workflows:

- **Keep the drafter aligned with the actor.** Adapt from responses and target features produced by the evolving policy, then hot-publish weights at safe points without restarting RL. The actor retains its original RL optimization objective.
- **Reuse existing target computation.** Capture hidden states during old-policy log-probability computation instead of adding a full-batch target forward pass. A shared scheduler coordinates collection, optimization, and publication.
- **Control updates using adaptation feedback.** Mix low-acceptance hard samples with uniform samples, pause collection and updates when gains plateau, and resume on sustained degradation while continuing to serve the adapted drafter.
- **Use idle capacity and support async pipelines.** Bubble Time fits optimizer work into rollout-idle windows and completes remaining quotas through fallback. Async co-training coordinates publication and acceptance statistics for requests spanning drafter versions.
- **Connect standalone training to online adaptation.** Build a drafter for a fixed target, deploy its checkpoint, or use it to initialize co-training. Both workflows share backends and checkpointing and can also run independently.
- **Reuse hardware and algorithm infrastructure.** CUDA GPU and Huawei Ascend NPU paths integrate multiple drafter training backends and serving engines. Consult [runtime compatibility](docs/usage.md#runtime-compatibility) for supported combinations.

Current Qwen3-8B + DSpark evaluation reaches **1.50× GPU and 1.58× NPU
end-to-end step speedup**. These single-run async Train+Freeze results use each
platform's synchronous non-speculative baseline, include adaptation, exclude
checkpoint I/O and validation, and combine async scheduling and speculation
with adaptation. [Full evaluation](docs/evaluation.md).

## Architecture

![SpeCo unified training architecture](docs/assets/paper/framework.png)

The scheduler coordinates **collect → train → publish**. Online co-training
captures target features, performs distributed drafter updates, and publishes
successful weights at rollout-safe points. Standalone and online workflows
share training backends and checkpointing.

[Scheduler guide](docs/drafter_scheduler.md) · [Bubble Time guide](docs/bubble_time_drafter_training.md)

## News

- [2026/11] [v0.2.0](https://github.com/verl-project/verl-SpeCo/releases/tag/v0.2.0) released: V1 synchronous/asynchronous DSpark co-training, Bubble Time, acceptance-guided freezing and hard sampling, standalone training, and expanded runtime integration.
- [2026/08] [v0.1.0](https://github.com/verl-project/verl-SpeCo/releases/tag/v0.1.0) released: end-to-end DSpark support, decoupled feature collection/training, and more drafter backends.
- [2026/06] [v0.1.0-pre](https://github.com/verl-project/verl-SpeCo/releases/tag/v0.1.0-pre) pre-released: speculative drafter collection, training, and hot-updates on verl.

## Key Features

- **Online adaptation:** reuse rollout responses and old-policy features; train from current data or replay, then safely publish drafter weights.
- **Acceptance-guided control:** mix hard and uniform samples, and pause/resume updates as adaptation gains change.
- **Flexible execution:** synchronous and asynchronous co-training; optional Bubble Time moves optimizer work into rollout-idle windows. Bubble Time currently requires the legacy trainer.
- **Extensible infrastructure:** EAGLE-1/2/3, DFlash/DFlash2, DSpark, Domino, and P-EAGLE training backends, with backend-specific rollout support on GPU/NPU.
- **Standalone training:** independently generate features and train deployable drafters using shared training infrastructure.
- **Hardware portability:** CUDA GPU and Ascend NPU paths for co-training and standalone training; availability depends on the drafter and serving runtime.

## What Can You Do with SpeCo?

| Goal | Workflow | Start here |
| --- | --- | --- |
| Accelerate generation-heavy RL, such as long-reasoning GRPO | Adapt the drafter alongside the actor and evaluate generation savings against adaptation costs | [Co-training quickstart](docs/usage.md#quickstart) |
| Address declining fixed-drafter acceptance | Train from current or replayed features with hard-sample mixing and freeze/resume control | [Scheduler guide](docs/drafter_scheduler.md) |
| Use early-finishing workers in synchronous rollout | Run part of the configured optimizer quota through Bubble Time | [Bubble Time guide](docs/bubble_time_drafter_training.md) |
| Add online adaptation to asynchronous RL | Coordinate feature capture, updates, and publication with V1 execution modes | [V1 support](docs/usage.md#verl-v1-runner-phase-1) |
| Build a deployable drafter for a fixed target | Decouple feature generation and optimization, or reuse saved features, then deploy on a compatible engine | [Standalone training](docs/usage.md#separate-draft-model-training) |
| Integrate drafter algorithms or study scheduling | Reuse training infrastructure while implementing algorithm-specific models, feature alignment, objectives, or scheduling policies | [Scheduling and extension](docs/drafter_scheduler.md) |

The paper evaluates DSpark, Qwen3-8B, GRPO, and vLLM; implemented backends do
not imply equivalent speedups. Bubble Time currently requires the legacy
trainer. Check compatibility and measure E2E gains for new configurations.

## Getting Started

[Full documentation](docs/usage.md)

Quickstart:

1. [Install SpeCo and upstream verl](docs/usage.md#installation), then check [runtime compatibility](docs/usage.md#runtime-compatibility).
2. Set model, drafter checkpoint, and dataset paths, then follow the [co-training quickstart](docs/usage.md#quickstart).
3. Select a [scheduling strategy](docs/drafter_scheduler.md) or optional [Bubble Time execution](docs/bubble_time_drafter_training.md).

Launch examples:

- [GPU DSpark co-training](examples/run_qwen3-8b_drafter_dspark_vllm.sh)
- [NPU DSpark co-training](examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh)
- [Standalone drafter training](examples/run_qwen3-8b_drafter_dspark_separate_training.sh)

Advanced usage:

- [Configuration](docs/usage.md#configuration) and [RL-Insight monitoring](docs/usage.md#online-monitoring-with-rl-insight)
- [Paper evaluation results](docs/evaluation.md) and [standalone results](docs/evaluation.md#standalone-training)
- [Testing](docs/usage.md#testing)

Support varies by drafter, engine, and hardware; P-EAGLE currently supports
training only. For V1 and native MRV2 deployments, follow the verl 0.9 and
runtime requirements in the detailed guide.

## Community & Contributing

Join the verl-SpeCo Lark user group:

<p align="center">
  <img src="docs/assets/Lark_QR_Code.png" alt="verl-SpeCo Lark user group" width="220">
</p>

Before contributing, follow [AGENTS.md](AGENTS.md), including duplicate-work
checks and test reporting. See the usage guide for test commands.

## Acknowledgements

Built on [verl](https://github.com/verl-project/verl),
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang), and
[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).
