# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4-Flash DSpark draft model.

Subclasses :class:`DSparkDraftModel` and swaps the DFlash decoder stack for the
DeepSeek-V4-Flash sparse backbone (latent attention + sink + MoE + mHC), while
reusing the DSpark Markov/confidence heads and the target-context projection.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from verl_speco.models.dspark import DSparkDraftModel

from .backbone.block import MhcDecoderBlock
from .backbone.hyper import HyperHead
from .backbone.rotary import precompute_freqs_cis
from .configuration_dsv4_dspark import DSV4DSparkConfig


class DSV4DSparkDraftModel(DSparkDraftModel):
    config_class = DSV4DSparkConfig
    _no_split_modules = ["MhcDecoderBlock"]

    def __init__(self, config: DSV4DSparkConfig) -> None:
        super().__init__(config)
        bb = _BackboneConfig(config)
        self.backbone_cfg = bb
        self.grad_checkpoint = bool(int(os.environ.get("DSPARK_RECOMPUTE", "0")))

        self.layers = nn.ModuleList(
            MhcDecoderBlock(bb) for _ in range(bb.n_draft_layers)
        )
        self.hc_head = HyperHead(bb)

        self._rope_dim = bb.rope_head_dim
        self.register_buffer(
            "freqs_cis",
            torch.view_as_real(
                precompute_freqs_cis(
                    bb.rope_head_dim,
                    bb.original_seq_len or 1,
                    bb.original_seq_len,
                    bb.rope_theta,
                    bb.rope_factor,
                    bb.beta_fast,
                    bb.beta_slow,
                )
            ).contiguous(),
            persistent=False,
        )
        self._init_backbone_params()

    def _init_backbone_params(self) -> None:
        if any(p.is_meta for p in self.parameters()):
            return
        std = 0.02
        for m in [*self.layers, self.hc_head]:
            for name, p in m.named_parameters():
                if (
                    p.dim() >= 2
                    and (".fn" in name or "weight" in name or "hc_fn" in name)
                    and (
                        torch.isnan(p).any()
                        or not p.abs().sum().isfinite()
                        or p.abs().sum() == 0
                    )
                ):
                    nn.init.normal_(p, std=std)

    def _rebuild_freqs(self, seqlen: int) -> None:
        bb = self.backbone_cfg
        self.freqs_cis = torch.view_as_real(
            precompute_freqs_cis(
                self._rope_dim,
                seqlen,
                bb.original_seq_len,
                bb.rope_theta,
                bb.rope_factor,
                bb.beta_fast,
                bb.beta_slow,
            )
        ).contiguous()

    def _ep_active(self) -> bool:
        from verl_speco.models.dsv4_dspark.backbone import moe_ep

        return moe_ep._EP is not None and moe_ep._EP.size > 1

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        import glob as glob_mod
        import json

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        from verl_speco.models.eagle.base import resolve_embedding_key

        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path)

        index_json_path = glob_mod.glob(os.path.join(model_path, "*.index.json"))
        if len(index_json_path) == 0:
            super().load_embedding(model_path, embedding_key)
            return

        # Resolve nested/multimodal embedding keys (e.g. ``model.language_model``)
        # before the multi-index selection, matching the DFlash parent behavior.
        embedding_key = resolve_embedding_key(model_path, embedding_key)
        weight_maps = []
        for ip in index_json_path:
            try:
                with open(ip, "r") as f:
                    weight_maps.append(json.load(f).get("weight_map", {}))
            except (OSError, json.JSONDecodeError):
                weight_maps.append({})

        if not any(embedding_key in weight_map for weight_map in weight_maps):
            # Quantized checkpoints nest the embedding under a different prefix;
            # accept it only when exactly one ``*.embed_tokens.weight`` exists.
            candidates = sorted(
                {
                    key
                    for weight_map in weight_maps
                    for key in weight_map
                    if key.endswith("embed_tokens.weight")
                }
            )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"{embedding_key!r} not found in any index.json under "
                    f"{model_path} (embedding candidates: {candidates})"
                )
            embedding_key = candidates[0]

        ckpt_file = None
        for weight_map in weight_maps:
            if embedding_key in weight_map:
                ckpt_file = weight_map[embedding_key]
                break
        if ckpt_file is None:
            raise FileNotFoundError(
                f"{embedding_key!r} not found in any index.json under {model_path}"
            )

        if ckpt_file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                self.embed_tokens.weight.copy_(f.get_tensor(embedding_key))
        else:
            state_dict = torch.load(
                os.path.join(model_path, ckpt_file),
                map_location="cpu",
                weights_only=True,
            )
            self.embed_tokens.weight.copy_(state_dict[embedding_key])

    def fsdp_wrap_plan(self) -> list:
        plan = []
        for block in self.layers:
            if not self._ep_active():
                plan.append(block.ffn.experts)
            plan.append(block.ffn.shared_experts)
            plan.append(block.attn)
            plan.append(block)
        return plan

    def fsdp_ignored_params(self) -> set:
        if not self._ep_active():
            return set()
        params = set()
        for block in self.layers:
            params.update(block.ffn.experts.parameters())
        # Also ignore frozen params that shouldn't be FSDP2-cast
        if hasattr(self, "embed_tokens"):
            params.update(self.embed_tokens.parameters())
        return params

    def shard_experts_as_dtensor(self, mesh) -> None:
        from torch.distributed.tensor import DTensor, Replicate, Shard

        # Experts are sharded over the (expert-parallel) leading mesh dim and
        # replicated over the remaining dims. FSDP2's standard flow needs every
        # parameter to be a DTensor, otherwise the gradient-norm reduction mixes
        # plain tensors with DTensors and fails.
        placements = [Shard(0)] + [Replicate() for _ in range(int(mesh.ndim) - 1)]

        for module in self.modules():
            if type(module).__name__ != "GroupedExperts":
                continue
            for name in ("w1", "w2", "w3"):
                p = getattr(module, name, None)
                if p is None or isinstance(p.data, DTensor):
                    continue
                dt = DTensor.from_local(p.data, mesh, placements, run_check=False)
                setattr(
                    module, name, torch.nn.Parameter(dt, requires_grad=p.requires_grad)
                )

    def _rope_at(self, positions: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_freqs_ok", False):
            self._freqs_ok = True
            self._rebuild_freqs(max(self.freqs_cis.shape[0], 1))
        if positions.numel() and int(positions.max()) >= self.freqs_cis.shape[0]:
            self._rebuild_freqs(int(positions.max()) + 1)
        return self.freqs_cis.to(positions.device)[positions]

    @staticmethod
    def _mask_to_bias(dense_attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Convert the dense bool mask into an additive attention bias.

        ``dense_attention_mask`` is ``[bsz, 1, draft_len, ctx+draft]`` from the
        shared DFlash mask builder; masked positions become ``-inf``.
        """
        if dense_attention_mask is None:
            return None
        if isinstance(dense_attention_mask, (list, tuple)):
            # Mixed sliding-window layers yield per-layer mask lists; DSV4 only
            # supports a single dense mask today, so fail closed rather than
            # silently indexing a list.
            raise NotImplementedError(
                "DSV4 sink attention does not support per-layer mask lists "
                "(mixed sliding-window layouts)"
            )
        mask = dense_attention_mask
        while mask.dim() > 3:
            mask = mask.squeeze(1)
        neg_inf = torch.finfo(mask.float().dtype).min
        return torch.where(mask.bool(), 0.0, neg_inf).to(torch.float32)

    def forward(
        self,
        draft_input_ids: torch.Tensor | None,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        dense_attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the sparse backbone over the draft block.

        The signature matches :meth:`DFlashDraftModel.forward` so the shared
        ``DSparkTrainingModel`` can drive it unchanged. Returns the final
        draft hidden states ``[bsz, block_len, hidden_size]``.
        """
        if noise_embedding is not None:
            draft_hidden = noise_embedding.to(context_feature.dtype)
        else:
            draft_hidden = self.embed_tokens(draft_input_ids).to(context_feature.dtype)

        block_freqs = self._rope_at(draft_position_ids.reshape(-1).long())
        context_freqs = self._rope_at(context_position_ids.reshape(-1).long())
        if dense_attention_mask is None and block_mask is not None:
            # The DSV4 sink attention consumes a dense bias only. The DFlash mask
            # builder returns a flex ``BlockMask`` on CUDA, which this model does
            # not implement; fail closed instead of silently dropping the mask.
            raise NotImplementedError(
                "DSV4 DSpark sink attention only supports dense_attention_mask; "
                "a flex/block mask was provided (CUDA path). Run on a device that "
                "uses the dense mask path (CPU/NPU) or add block-mask support."
            )
        attn_bias = self._mask_to_bias(dense_attention_mask)

        hc = self.backbone_cfg.hc_mult
        streams = draft_hidden.unsqueeze(2).repeat(1, 1, hc, 1)

        from torch.utils.checkpoint import checkpoint

        if self.training:
            for layer in self.layers:
                if hasattr(layer.ffn.router, "update_load_balance_bias"):
                    layer.ffn.router.update_load_balance_bias()

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                streams = checkpoint(
                    layer,
                    streams,
                    context_feature,
                    block_freqs,
                    context_freqs,
                    attn_bias,
                    use_reentrant=False,
                )
            else:
                streams = layer(
                    streams,
                    context_feature,
                    block_freqs,
                    context_freqs,
                    attn_bias,
                )

        hidden = self.norm(self.hc_head(streams))
        return hidden.contiguous()


class _BackboneConfig:
    """Flatten :class:`DSV4DSparkConfig` into the plain attributes the backbone
    modules read, so the backbone stays decoupled from transformers configs.
    """

    def __init__(self, config: DSV4DSparkConfig) -> None:
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.n_draft_layers = int(config.num_hidden_layers)
        self.num_heads = int(config.dsv4_num_heads)
        self.head_dim = int(config.dsv4_head_dim)
        self.rope_head_dim = int(config.dsv4_rope_head_dim)
        self.q_lora_rank = int(config.dsv4_q_lora_rank)
        self.o_lora_rank = int(config.dsv4_o_lora_rank)
        self.o_groups = int(config.dsv4_o_groups)
        self.rope_theta = float(config.dsv4_rope_theta)
        self.rope_factor = float(config.dsv4_rope_factor)
        self.original_seq_len = int(config.dsv4_original_seq_len)
        self.beta_fast = float(config.dsv4_beta_fast)
        self.beta_slow = float(config.dsv4_beta_slow)
        self.n_routed_experts = int(config.dsv4_n_routed_experts)
        self.n_shared_experts = int(config.dsv4_n_shared_experts)
        self.n_activated_experts = int(config.dsv4_n_activated_experts)
        self.moe_inter_dim = int(config.dsv4_moe_inter_dim)
        self.score_func = str(config.dsv4_score_func)
        self.route_scale = float(config.dsv4_route_scale)
        self.swiglu_limit = float(config.dsv4_swiglu_limit)
        self.hc_mult = int(config.dsv4_hc_mult)
        self.hc_sinkhorn_iters = int(config.dsv4_hc_sinkhorn_iters)
        self.hc_eps = float(config.dsv4_hc_eps)
