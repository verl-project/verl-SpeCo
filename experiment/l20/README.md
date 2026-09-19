# Dense P-EAGLE on VeOmni

Set `actor_rollout_ref.rollout.drafter.training.engine=veomni` in the
standalone launcher. The default remains `fsdp`. This implementation is
validated with VeOmni 0.1.11 and verl 0.9.0, using a full-world FSDP2 mesh
and dedicated drafter processes. Other algorithms, Ulysses, hybrid sharding,
and colocated online actors are rejected.

The algorithm still owns loss, optimizer, scheduler, and publish names. VeOmni
owns wrapping and materialization. A temporary per-rank checkpoint transfers
the initialized wrapper into VeOmni's meta loader; buffers, including rotary
frequencies absent from state_dict, survive that transition. Temporary weights
are removed after materialization. Persistent trainer checkpoints still use
SpeCo's existing DCP and export flow.

Two prerequisite P-EAGLE fixes preserve trained embeddings when resuming and
unwrap the training module for pretrained export. They apply to both engines.

`check_veomni_drafter.py` checks three FP32 AdamW steps against native FSDP2,
including loss, gradients, clipping, optimizer moments, parameters and buffers.
`prepare_features.py` captures eight target forwards from the deterministic
4-layer/64-hidden fixture. `run_veomni_ab.sh` runs six BF16 standalone steps
and checkpoints in four fresh processes, followed by a checkpoint resume.
`summarize_veomni.py` checks completion and reports the raw timing comparison.

These tests establish standalone drafter execution. They do not establish
online RL integration, training quality, RNG/data-cursor exact resume, or
large-model throughput. SpeCo's existing non-TQ checkpoint does not restore
the feature-store cursor or COD RNG; resume verifies optimizer step restoration
and further successful updates, not bitwise equality with uninterrupted training.
