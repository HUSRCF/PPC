"""Sequence-only residue contact-site model using ESM embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import torch
from torch import nn


def _sanitize(x: torch.Tensor, clip: float = 1.0e4) -> torch.Tensor:
    if x.is_floating_point():
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        x = torch.clamp(x, min=-clip, max=clip)
    return x


@dataclass
class ESMSiteConfig:
    d_esm: int = 1280
    d_seq: int = 28
    d_prottrans: int = 1024
    d_model: int = 256
    d_hidden: int = 512
    n_classes: int = 2
    dropout: float = 0.10
    classifier_dropout: float = 0.20
    max_chains: int = 128
    use_chain_embedding: bool = True
    use_seq_features: bool = True
    use_prottrans_embeddings: bool = False
    prottrans_fusion_mode: str = "gated_residual"
    prottrans_gate_input_mode: str = "full"
    prottrans_gate_bias: float = -2.0
    prottrans_residual_alpha_init: float = 1.0
    prottrans_residual_alpha_trainable: bool = False
    use_position_features: bool = True
    use_global_context: bool = True
    use_contact_graph: bool = False
    contact_graph_layers: int = 0
    contact_score_clip: float = 1.0
    use_sparse_graph_transformer: bool = False
    sparse_graph_layers: int = 0
    sparse_graph_heads: int = 8
    sparse_graph_seq_neighbor_k: int = 0
    sparse_graph_use_contact_edges: bool = True
    sparse_graph_make_contact_bidirectional: bool = True
    sparse_graph_use_seq_edges: bool = False
    sparse_graph_use_global_node: bool = True
    sparse_graph_use_chain_edge_type: bool = True
    sparse_graph_edge_hidden: int = 64
    sparse_graph_score_clip: float = 1.0
    sparse_graph_adaptive_residual_gate: bool = False
    use_contact_profile_features: bool = False
    contact_profile_hidden: int = 64
    contact_profile_include_aux: bool = True
    contact_profile_score_clip: float = 1.0
    use_aux_contact_graph: bool = False
    aux_contact_mode: str = "none"
    dual_contact_fusion_mode: str = "residual_mlp"
    use_mamba_context: bool = False
    mamba_layers: int = 0
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_bidirectional: bool = True
    mamba_ff_mult: int = 2
    n_transformer_layers: int = 0
    n_heads: int = 8
    transformer_ff_mult: int = 4
    use_tcn_context: bool = False
    tcn_layers: int = 0
    tcn_kernel_size: int = 7
    tcn_dilations: tuple[int, ...] | list[int] = field(default_factory=lambda: (1, 2, 4, 8))
    tcn_block_type: str = "standard"
    esm_layer_fusion: str = "concat"
    esm_layer_count: int = 1
    input_fusion_mode: str = "add"
    extra: dict[str, Any] = field(default_factory=dict)


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _sanitize(x + self.net(self.norm(x)))


class ESMScalarMix(nn.Module):
    """Learn a residue-wise source mix over concatenated ESM layer embeddings."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = max(1, int(n_layers))
        self.logits = nn.Parameter(torch.zeros(self.n_layers))
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers <= 1:
            return x
        if x.shape[-1] % self.n_layers != 0:
            raise ValueError(f"ESM embedding dim {x.shape[-1]} is not divisible by esm_layer_count={self.n_layers}")
        layer_dim = x.shape[-1] // self.n_layers
        y = x.view(*x.shape[:-1], self.n_layers, layer_dim)
        weights = torch.softmax(self.logits.float(), dim=0).to(dtype=x.dtype)
        return _sanitize(self.scale.to(dtype=x.dtype) * (y * weights.view(*((1,) * (y.ndim - 2)), self.n_layers, 1)).sum(dim=-2))


class TCNContextBlock(nn.Module):
    """Masked residual 1D convolution context over residue order.

    Sequence-neighbor context is applied over the concatenated protein residue
    order. Padding positions are zeroed before and after the convolution so they
    cannot inject values into valid residues.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float,
        kernel_size: int,
        dilation: int,
        block_type: str = "standard",
    ) -> None:
        super().__init__()
        kernel_size = max(1, int(kernel_size))
        dilation = max(1, int(dilation))
        padding = dilation * (kernel_size - 1) // 2
        block_type = str(block_type or "standard").lower()
        if block_type not in {"standard", "gated_depthwise"}:
            raise ValueError(f"Unsupported tcn_block_type={block_type!r}; use standard or gated_depthwise")
        self.norm = nn.LayerNorm(d_model)
        self.block_type = block_type
        if block_type == "gated_depthwise":
            self.conv = nn.Sequential(
                nn.Conv1d(
                    d_model,
                    d_model * 2,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=d_model,
                ),
                nn.GLU(dim=1),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_hidden, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_hidden, d_model, kernel_size=1),
                nn.Dropout(dropout),
            )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(h).masked_fill(~mask.unsqueeze(-1), 0.0).transpose(1, 2)
        update = self.conv(x)
        if update.shape[-1] != h.shape[1]:
            update = update[..., : h.shape[1]]
        update = update.transpose(1, 2).to(dtype=h.dtype)
        h = _sanitize(h + update).masked_fill(~mask.unsqueeze(-1), 0.0)
        h = _sanitize(h + self.ffn(self.ffn_norm(h))).masked_fill(~mask.unsqueeze(-1), 0.0)
        return h


class ContactGraphBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float, score_clip: float) -> None:
        super().__init__()
        self.score_clip = float(score_clip)
        self.message = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        contact_edge_index: torch.Tensor | None,
        contact_edge_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        if contact_edge_index is None or contact_edge_index.numel() == 0:
            return h
        batch, length, d_model = h.shape
        n_nodes = batch * length
        edge_index = contact_edge_index.to(device=h.device, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return h
        scores = (
            torch.ones(edge_index.shape[1], dtype=h.dtype, device=h.device)
            if contact_edge_scores is None
            else contact_edge_scores.to(device=h.device, dtype=h.dtype).flatten()
        )
        if scores.shape[0] != edge_index.shape[1]:
            return h
        valid = (
            torch.isfinite(scores)
            & (edge_index[0] >= 0)
            & (edge_index[1] >= 0)
            & (edge_index[0] < n_nodes)
            & (edge_index[1] < n_nodes)
            & (edge_index[0] != edge_index[1])
        )
        if not bool(valid.any()):
            return h
        src = edge_index[0, valid]
        dst = edge_index[1, valid]
        weights = torch.clamp(scores[valid], min=0.0, max=max(self.score_clip, 1.0e-6))
        flat = h.reshape(n_nodes, d_model)
        messages = self.message(flat[src]) * weights.unsqueeze(-1)
        agg = torch.zeros_like(flat)
        denom = torch.zeros(n_nodes, dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, messages)
        denom.index_add_(0, dst, weights)
        agg = agg / denom.clamp_min(1.0e-6).unsqueeze(-1)
        out = flat + self.update(agg)
        return _sanitize(out.view(batch, length, d_model)).masked_fill(~mask.unsqueeze(-1), 0.0)


class ContactProfileEncoder(nn.Module):
    """Project sparse contact statistics into residue node features.

    This keeps the input sequence-only: profiles are computed from predicted
    contact priors already present in the ESM or predicted-structure feature
    payloads, never from the ground-truth interface labels.
    """

    stats_per_graph = 8

    def __init__(
        self,
        d_model: int,
        hidden: int,
        dropout: float,
        include_aux: bool,
        score_clip: float,
    ) -> None:
        super().__init__()
        self.include_aux = bool(include_aux)
        self.score_clip = float(score_clip)
        n_graphs = 2 if self.include_aux else 1
        self.proj = nn.Sequential(
            nn.LayerNorm(self.stats_per_graph * n_graphs),
            nn.Linear(self.stats_per_graph * n_graphs, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def _graph_stats(
        self,
        mask: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_scores: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, length = mask.shape
        n_nodes = batch * length
        device = mask.device
        stats = torch.zeros((n_nodes, self.stats_per_graph), device=device, dtype=dtype)
        if edge_index is None or edge_index.numel() == 0:
            return stats
        edges = edge_index.to(device=device, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2:
            return stats
        scores = (
            torch.ones(edges.shape[1], device=device, dtype=dtype)
            if edge_scores is None
            else edge_scores.to(device=device, dtype=dtype).flatten()
        )
        if scores.shape[0] != edges.shape[1]:
            return stats
        flat_valid = mask.reshape(-1)
        valid = (
            torch.isfinite(scores)
            & (edges[0] >= 0)
            & (edges[1] >= 0)
            & (edges[0] < n_nodes)
            & (edges[1] < n_nodes)
            & (edges[0] != edges[1])
            & flat_valid[edges[0].clamp(0, max(0, n_nodes - 1))]
            & flat_valid[edges[1].clamp(0, max(0, n_nodes - 1))]
        )
        if not bool(valid.any()):
            return stats

        src = edges[0, valid]
        dst = edges[1, valid]
        score = torch.nan_to_num(scores[valid], nan=0.0, posinf=self.score_clip, neginf=0.0)
        score = torch.clamp(score, min=0.0, max=max(self.score_clip, 1.0e-6))
        one = torch.ones_like(score)
        out_sum = torch.zeros(n_nodes, device=device, dtype=dtype)
        in_sum = torch.zeros(n_nodes, device=device, dtype=dtype)
        out_count = torch.zeros(n_nodes, device=device, dtype=dtype)
        in_count = torch.zeros(n_nodes, device=device, dtype=dtype)
        out_max = torch.zeros(n_nodes, device=device, dtype=dtype)
        in_max = torch.zeros(n_nodes, device=device, dtype=dtype)
        out_sum.index_add_(0, src, score)
        in_sum.index_add_(0, dst, score)
        out_count.index_add_(0, src, one)
        in_count.index_add_(0, dst, one)
        out_max.scatter_reduce_(0, src, score, reduce="amax", include_self=True)
        in_max.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)

        norm = math.sqrt(max(1, length))
        stats[:, 0] = out_sum
        stats[:, 1] = out_sum / out_count.clamp_min(1.0)
        stats[:, 2] = out_max
        stats[:, 3] = out_count / norm
        stats[:, 4] = in_sum
        stats[:, 5] = in_sum / in_count.clamp_min(1.0)
        stats[:, 6] = in_max
        stats[:, 7] = in_count / norm
        return stats

    def forward(
        self,
        mask: torch.Tensor,
        contact_edge_index: torch.Tensor | None,
        contact_edge_scores: torch.Tensor | None,
        aux_contact_edge_index: torch.Tensor | None = None,
        aux_contact_edge_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = mask.shape
        primary = self._graph_stats(mask, contact_edge_index, contact_edge_scores, torch.float32)
        feats = [primary]
        if self.include_aux:
            feats.append(self._graph_stats(mask, aux_contact_edge_index, aux_contact_edge_scores, torch.float32))
        x = torch.cat(feats, dim=-1).view(batch, length, -1)
        return _sanitize(self.proj(x)).masked_fill(~mask.unsqueeze(-1), 0.0)


def _edge_softmax(logits: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Softmax over incoming sparse edges for each destination node and head."""
    if logits.numel() == 0:
        return logits
    n_heads = int(logits.shape[1])
    head_ids = torch.arange(n_heads, device=logits.device).view(1, n_heads)
    flat_index = (dst.view(-1, 1) * n_heads + head_ids).reshape(-1)
    flat_logits = logits.float().reshape(-1)
    n_groups = int(n_nodes) * n_heads
    max_per_group = torch.full((n_groups,), -torch.inf, device=logits.device, dtype=torch.float32)
    max_per_group.scatter_reduce_(0, flat_index, flat_logits, reduce="amax", include_self=True)
    shifted = torch.clamp(flat_logits - max_per_group[flat_index], min=-60.0, max=60.0)
    exp = torch.exp(shifted)
    denom = torch.zeros((n_groups,), device=logits.device, dtype=torch.float32)
    denom.index_add_(0, flat_index, exp)
    attn = exp / denom[flat_index].clamp_min(1.0e-12)
    return attn.view_as(logits).to(dtype=logits.dtype)


class SparseEdgeGraphTransformerBlock(nn.Module):
    """Sparse edge-aware graph transformer with edge bias and value modulation.

    Directed edges are interpreted as src -> dst messages. Sequence-neighbor
    edges are generated only within identical local chain ids, so concatenated
    chain boundaries cannot create artificial covalent-neighbor edges.
    """

    edge_dim = 10

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float,
        n_heads: int,
        seq_neighbor_k: int,
        use_contact_edges: bool,
        make_contact_bidirectional: bool,
        use_seq_edges: bool,
        use_global_node: bool,
        use_chain_edge_type: bool,
        edge_hidden: int,
        score_clip: float,
        adaptive_residual_gate: bool,
    ) -> None:
        super().__init__()
        if d_model % max(1, int(n_heads)) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by sparse_graph_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = max(1, int(n_heads))
        self.head_dim = self.d_model // self.n_heads
        self.seq_neighbor_k = max(0, int(seq_neighbor_k))
        self.use_contact_edges = bool(use_contact_edges)
        self.make_contact_bidirectional = bool(make_contact_bidirectional)
        self.use_seq_edges = bool(use_seq_edges) and self.seq_neighbor_k > 0
        self.use_global_node = bool(use_global_node)
        self.use_chain_edge_type = bool(use_chain_edge_type)
        self.score_clip = float(score_clip)
        self.adaptive_residual_gate = bool(adaptive_residual_gate)

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Sequential(
            nn.Linear(self.edge_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, self.n_heads),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(self.edge_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, d_model),
        )
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.residual_gate = (
            nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            if self.adaptive_residual_gate
            else None
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.global_token = nn.Parameter(torch.zeros(1, d_model))

    def _edge_features(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        score: torch.Tensor,
        edge_type: int,
        chain_flat: torch.Tensor,
        n_res_nodes: int,
        length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = src.device
        feat = torch.zeros((src.numel(), self.edge_dim), device=device, dtype=dtype)
        if src.numel() == 0:
            return feat
        clipped = torch.clamp(score.to(device=device, dtype=dtype), min=0.0, max=max(self.score_clip, 1.0e-6))
        feat[:, 0] = clipped
        res_src = src < n_res_nodes
        res_dst = dst < n_res_nodes
        both_res = res_src & res_dst
        if bool(both_res.any()):
            src_pos = (src[both_res] % max(1, length)).to(dtype=dtype)
            dst_pos = (dst[both_res] % max(1, length)).to(dtype=dtype)
            rel = (dst_pos - src_pos) / max(1, length - 1)
            feat[both_res, 1] = torch.clamp(rel, min=-1.0, max=1.0)
            feat[both_res, 2] = torch.clamp(rel.abs(), max=1.0)
            if self.use_chain_edge_type:
                same = chain_flat[src[both_res]] == chain_flat[dst[both_res]]
                feat[both_res, 3] = same.to(dtype=dtype)
                feat[both_res, 4] = (~same).to(dtype=dtype)
        if edge_type == 0:
            feat[:, 5] = 1.0
        elif edge_type == 1:
            feat[:, 6] = 1.0
        elif edge_type == 2:
            feat[:, 7] = 1.0
        elif edge_type == 3:
            feat[:, 8] = 1.0
        elif edge_type == 4:
            feat[:, 9] = 1.0
        return feat

    def _append_edges(
        self,
        srcs: list[torch.Tensor],
        dsts: list[torch.Tensor],
        feats: list[torch.Tensor],
        src: torch.Tensor,
        dst: torch.Tensor,
        score: torch.Tensor,
        edge_type: int,
        chain_flat: torch.Tensor,
        n_res_nodes: int,
        length: int,
        dtype: torch.dtype,
    ) -> None:
        if src.numel() == 0:
            return
        srcs.append(src)
        dsts.append(dst)
        feats.append(self._edge_features(src, dst, score, edge_type, chain_flat, n_res_nodes, length, dtype))

    def _build_graph(
        self,
        mask: torch.Tensor,
        chain_ids: torch.Tensor | None,
        contact_edge_index: torch.Tensor | None,
        contact_edge_scores: torch.Tensor | None,
        aux_contact_edge_index: torch.Tensor | None,
        aux_contact_edge_scores: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        device = mask.device
        batch, length = mask.shape
        n_res_nodes = batch * length
        n_nodes = n_res_nodes + (batch if self.use_global_node else 0)
        flat_valid = mask.reshape(-1)
        if chain_ids is None:
            chain_flat = torch.zeros((n_res_nodes,), device=device, dtype=torch.long)
        else:
            chain_flat = chain_ids.to(device=device, dtype=torch.long).reshape(-1)

        srcs: list[torch.Tensor] = []
        dsts: list[torch.Tensor] = []
        feats: list[torch.Tensor] = []

        if self.use_contact_edges and contact_edge_index is not None and contact_edge_index.numel() > 0:
            edge_index = contact_edge_index.to(device=device, dtype=torch.long)
            scores = (
                torch.ones(edge_index.shape[1], device=device, dtype=dtype)
                if contact_edge_scores is None
                else contact_edge_scores.to(device=device, dtype=dtype).flatten()
            )
            valid = (
                (edge_index[0] >= 0)
                & (edge_index[1] >= 0)
                & (edge_index[0] < n_res_nodes)
                & (edge_index[1] < n_res_nodes)
                & (edge_index[0] != edge_index[1])
                & flat_valid[edge_index[0].clamp(0, max(0, n_res_nodes - 1))]
                & flat_valid[edge_index[1].clamp(0, max(0, n_res_nodes - 1))]
            )
            src = edge_index[0, valid]
            dst = edge_index[1, valid]
            score = torch.nan_to_num(scores[valid], nan=0.0, posinf=self.score_clip, neginf=0.0)
            self._append_edges(srcs, dsts, feats, src, dst, score, 0, chain_flat, n_res_nodes, length, dtype)
            if self.make_contact_bidirectional:
                self._append_edges(srcs, dsts, feats, dst, src, score, 0, chain_flat, n_res_nodes, length, dtype)

        if self.use_contact_edges and aux_contact_edge_index is not None and aux_contact_edge_index.numel() > 0:
            edge_index = aux_contact_edge_index.to(device=device, dtype=torch.long)
            scores = (
                torch.ones(edge_index.shape[1], device=device, dtype=dtype)
                if aux_contact_edge_scores is None
                else aux_contact_edge_scores.to(device=device, dtype=dtype).flatten()
            )
            valid = (
                (edge_index[0] >= 0)
                & (edge_index[1] >= 0)
                & (edge_index[0] < n_res_nodes)
                & (edge_index[1] < n_res_nodes)
                & (edge_index[0] != edge_index[1])
                & flat_valid[edge_index[0].clamp(0, max(0, n_res_nodes - 1))]
                & flat_valid[edge_index[1].clamp(0, max(0, n_res_nodes - 1))]
            )
            src = edge_index[0, valid]
            dst = edge_index[1, valid]
            score = torch.nan_to_num(scores[valid], nan=0.0, posinf=self.score_clip, neginf=0.0)
            self._append_edges(srcs, dsts, feats, src, dst, score, 4, chain_flat, n_res_nodes, length, dtype)
            if self.make_contact_bidirectional:
                self._append_edges(srcs, dsts, feats, dst, src, score, 4, chain_flat, n_res_nodes, length, dtype)

        if self.use_seq_edges:
            node_grid = torch.arange(n_res_nodes, device=device, dtype=torch.long).view(batch, length)
            chain_grid = chain_flat.view(batch, length)
            for offset in range(1, self.seq_neighbor_k + 1):
                left_valid = mask[:, :-offset]
                right_valid = mask[:, offset:]
                same_chain = chain_grid[:, :-offset] == chain_grid[:, offset:]
                keep = left_valid & right_valid & same_chain
                if bool(keep.any()):
                    src = node_grid[:, :-offset][keep]
                    dst = node_grid[:, offset:][keep]
                    score = torch.zeros(src.numel(), device=device, dtype=dtype)
                    self._append_edges(srcs, dsts, feats, src, dst, score, 1, chain_flat, n_res_nodes, length, dtype)
                    self._append_edges(srcs, dsts, feats, dst, src, score, 1, chain_flat, n_res_nodes, length, dtype)

        if self.use_global_node:
            node_grid = torch.arange(n_res_nodes, device=device, dtype=torch.long).view(batch, length)
            global_idx = torch.arange(batch, device=device, dtype=torch.long) + n_res_nodes
            for batch_idx in range(batch):
                residues = node_grid[batch_idx][mask[batch_idx]]
                if residues.numel() == 0:
                    continue
                glob = global_idx[batch_idx].expand_as(residues)
                score = torch.ones(residues.numel(), device=device, dtype=dtype)
                self._append_edges(srcs, dsts, feats, glob, residues, score, 2, chain_flat, n_res_nodes, length, dtype)
                self._append_edges(srcs, dsts, feats, residues, glob, score, 3, chain_flat, n_res_nodes, length, dtype)

        if not srcs:
            empty_edge = torch.empty((0,), device=device, dtype=torch.long)
            empty_feat = torch.empty((0, self.edge_dim), device=device, dtype=dtype)
            return empty_edge, empty_edge, empty_feat, n_nodes
        return torch.cat(srcs), torch.cat(dsts), torch.cat(feats), n_nodes

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        chain_ids: torch.Tensor | None,
        contact_edge_index: torch.Tensor | None,
        contact_edge_scores: torch.Tensor | None,
        aux_contact_edge_index: torch.Tensor | None = None,
        aux_contact_edge_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, d_model = h.shape
        n_res_nodes = batch * length
        x = self.norm(h).masked_fill(~mask.unsqueeze(-1), 0.0)
        flat = x.reshape(n_res_nodes, d_model)
        if self.use_global_node:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
            global_nodes = x.sum(dim=1) / denom + self.global_token.to(dtype=x.dtype)
            h_all = torch.cat([flat, global_nodes], dim=0)
        else:
            h_all = flat

        src, dst, edge_feat, n_nodes = self._build_graph(
            mask,
            chain_ids,
            contact_edge_index,
            contact_edge_scores,
            aux_contact_edge_index,
            aux_contact_edge_scores,
            h.dtype,
        )
        if src.numel() == 0:
            return h.masked_fill(~mask.unsqueeze(-1), 0.0)

        q = self.q_proj(h_all).view(n_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(h_all).view(n_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(h_all).view(n_nodes, self.n_heads, self.head_dim)
        bias = self.edge_bias(edge_feat).to(dtype=q.dtype)
        value_delta = self.edge_value(edge_feat).view(src.numel(), self.n_heads, self.head_dim).to(dtype=v.dtype)
        logits = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + bias
        attn = _edge_softmax(logits, dst, n_nodes)
        messages = (attn.unsqueeze(-1).to(dtype=v.dtype) * (v[src] + value_delta)).reshape(src.numel(), d_model)
        agg = torch.zeros((n_nodes, d_model), device=h.device, dtype=messages.dtype)
        agg.index_add_(0, dst, messages)
        base = h.reshape(n_res_nodes, d_model)
        update = self.out_proj(agg[:n_res_nodes].to(dtype=h.dtype))
        if self.residual_gate is not None:
            gate = self.residual_gate(torch.cat([base, update], dim=-1)).to(dtype=update.dtype)
            updated = base + gate * update
        else:
            updated = base + update
        updated = _sanitize(updated.view(batch, length, d_model)).masked_fill(~mask.unsqueeze(-1), 0.0)
        updated = _sanitize(updated + self.ffn(self.ffn_norm(updated))).masked_fill(~mask.unsqueeze(-1), 0.0)
        return updated


class MambaContextBlock(nn.Module):
    """Masked residual Mamba context block over concatenated residue order."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        bidirectional: bool,
        ff_mult: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("use_mamba_context=True requires mamba_ssm to be installed") from exc

        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if self.bidirectional else None
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        ff_hidden = max(d_hidden, int(d_model) * max(1, int(ff_mult)))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(h).masked_fill(~mask.unsqueeze(-1), 0.0)
        y = self.fwd(x)
        if self.bwd is not None:
            rev_mask = torch.flip(mask, dims=(1,))
            rev_x = torch.flip(x, dims=(1,)).masked_fill(~rev_mask.unsqueeze(-1), 0.0)
            y = 0.5 * (y + torch.flip(self.bwd(rev_x), dims=(1,)))
        h = _sanitize(h + self.out(y).to(dtype=h.dtype)).masked_fill(~mask.unsqueeze(-1), 0.0)
        h = _sanitize(h + self.ffn(self.ffn_norm(h))).masked_fill(~mask.unsqueeze(-1), 0.0)
        return h


class ESMSiteClassifier(nn.Module):
    """Residue classifier that only consumes sequence-derived tensors.

    Inputs are ESM embeddings plus sequence-order metadata.  No coordinates,
    PDB-derived DSSP/RSA, spatial neighbors, or bound-complex geometry are used.
    """

    def __init__(self, config: dict[str, Any] | ESMSiteConfig | None = None) -> None:
        super().__init__()
        if config is None:
            self.config = ESMSiteConfig()
        elif isinstance(config, ESMSiteConfig):
            self.config = config
        else:
            known = {f.name for f in ESMSiteConfig.__dataclass_fields__.values()}
            cfg = {k: v for k, v in config.items() if k in known}
            extra = {k: v for k, v in config.items() if k not in known}
            self.config = ESMSiteConfig(**cfg, extra=extra)
        c = self.config
        aux_contact_mode = str(c.aux_contact_mode or "none").lower()
        if aux_contact_mode not in {"none", "merge", "dual_branch"}:
            raise ValueError(f"Unsupported aux_contact_mode={c.aux_contact_mode!r}; use none, merge, or dual_branch")
        c.aux_contact_mode = aux_contact_mode
        fusion_mode = str(c.dual_contact_fusion_mode or "residual_mlp").lower()
        if fusion_mode not in {"residual_mlp", "gated_residual", "weighted", "attention"}:
            raise ValueError(
                f"Unsupported dual_contact_fusion_mode={c.dual_contact_fusion_mode!r}; "
                "use residual_mlp, gated_residual, weighted, or attention"
            )
        c.dual_contact_fusion_mode = fusion_mode
        c.esm_layer_fusion = str(c.esm_layer_fusion or "concat").lower()
        if c.esm_layer_fusion not in {"concat", "scalar_mix", "last"}:
            raise ValueError(f"Unsupported esm_layer_fusion={c.esm_layer_fusion!r}; use concat, scalar_mix, or last")
        c.esm_layer_count = max(1, int(c.esm_layer_count))
        c.prottrans_fusion_mode = str(c.prottrans_fusion_mode or "gated_residual").lower()
        if c.prottrans_fusion_mode not in {"gated_residual", "add", "input_branch", "projected_concat"}:
            raise ValueError(
                f"Unsupported prottrans_fusion_mode={c.prottrans_fusion_mode!r}; "
                "use gated_residual, add, input_branch, or projected_concat"
            )
        if c.prottrans_fusion_mode == "projected_concat":
            if not c.use_prottrans_embeddings:
                raise ValueError("projected_concat requires use_prottrans_embeddings=true")
            if c.d_model % 2 != 0:
                raise ValueError(f"projected_concat requires an even d_model, got {c.d_model}")
        c.prottrans_gate_input_mode = str(c.prottrans_gate_input_mode or "full").lower()
        if c.prottrans_gate_input_mode not in {"full", "simple"}:
            raise ValueError(
                f"Unsupported prottrans_gate_input_mode={c.prottrans_gate_input_mode!r}; "
                "use full or simple"
            )
        c.input_fusion_mode = str(c.input_fusion_mode or "add").lower()
        if c.input_fusion_mode not in {"add", "gated_residual", "weighted"}:
            raise ValueError(f"Unsupported input_fusion_mode={c.input_fusion_mode!r}; use add, gated_residual, or weighted")

        esm_input_dim = c.d_esm
        self.esm_scalar_mix = None
        self.esm_last_dim = None
        if c.esm_layer_fusion == "scalar_mix":
            if c.d_esm % c.esm_layer_count != 0:
                raise ValueError(f"d_esm={c.d_esm} must be divisible by esm_layer_count={c.esm_layer_count}")
            esm_input_dim = c.d_esm // c.esm_layer_count
            self.esm_scalar_mix = ESMScalarMix(c.esm_layer_count)
        elif c.esm_layer_fusion == "last":
            if c.d_esm % c.esm_layer_count != 0:
                raise ValueError(f"d_esm={c.d_esm} must be divisible by esm_layer_count={c.esm_layer_count}")
            esm_input_dim = c.d_esm // c.esm_layer_count
            self.esm_last_dim = esm_input_dim

        projected_concat = c.prottrans_fusion_mode == "projected_concat"
        esm_projected_dim = c.d_model // 2 if projected_concat else c.d_model
        prottrans_projected_dim = c.d_model // 2 if projected_concat else c.d_model

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_input_dim, esm_projected_dim),
            nn.LayerNorm(esm_projected_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.seq_proj = (
            nn.Sequential(
                nn.Linear(c.d_seq, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
                nn.Dropout(c.dropout),
            )
            if c.use_seq_features
            else None
        )
        self.prottrans_proj = (
            nn.Sequential(
                nn.Linear(c.d_prottrans, prottrans_projected_dim),
                nn.LayerNorm(prottrans_projected_dim),
                nn.GELU(),
                nn.Dropout(c.dropout),
            )
            if c.use_prottrans_embeddings
            else None
        )
        self.prottrans_gate = None
        gate_input_mult = 4 if c.prottrans_gate_input_mode == "full" else 2
        if self.prottrans_proj is not None and c.prottrans_fusion_mode == "gated_residual":
            self.prottrans_gate = nn.Sequential(
                nn.LayerNorm(c.d_model * gate_input_mult),
                nn.Linear(c.d_model * gate_input_mult, c.d_model),
                nn.Sigmoid(),
            )
            gate_linear = self.prottrans_gate[1]
            if isinstance(gate_linear, nn.Linear):
                nn.init.constant_(gate_linear.bias, float(c.prottrans_gate_bias))
        alpha = torch.tensor(float(c.prottrans_residual_alpha_init), dtype=torch.float32)
        if self.prottrans_proj is not None and c.prottrans_fusion_mode in {"gated_residual", "add"}:
            if bool(c.prottrans_residual_alpha_trainable):
                self.prottrans_residual_alpha = nn.Parameter(alpha)
            else:
                self.register_buffer("prottrans_residual_alpha", alpha)
        else:
            self.prottrans_residual_alpha = None
        self.chain_embedding = nn.Embedding(c.max_chains, c.d_model) if c.use_chain_embedding else None
        self.position_proj = (
            nn.Sequential(
                nn.Linear(2, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
                nn.Dropout(c.dropout),
            )
            if c.use_position_features
            else None
        )
        self.contact_profile_encoder = (
            ContactProfileEncoder(
                c.d_model,
                c.contact_profile_hidden,
                c.dropout,
                c.contact_profile_include_aux,
                c.contact_profile_score_clip,
            )
            if c.use_contact_profile_features
            else None
        )
        n_input_branches = (
            1
            + int(self.prottrans_proj is not None and c.prottrans_fusion_mode == "input_branch")
            + int(self.seq_proj is not None)
            + int(self.chain_embedding is not None)
            + int(self.position_proj is not None)
            + int(self.contact_profile_encoder is not None)
        )
        self.input_fusion = None
        self.input_gate = None
        self.input_weights = None
        if c.input_fusion_mode == "gated_residual":
            self.input_fusion = nn.Sequential(
                nn.LayerNorm(c.d_model * n_input_branches),
                nn.Linear(c.d_model * n_input_branches, c.d_hidden),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.d_hidden, c.d_model),
                nn.Dropout(c.dropout),
            )
            self.input_gate = nn.Sequential(
                nn.LayerNorm(c.d_model * n_input_branches),
                nn.Linear(c.d_model * n_input_branches, c.d_model),
                nn.Sigmoid(),
            )
        elif c.input_fusion_mode == "weighted":
            self.input_weights = nn.Sequential(
                nn.LayerNorm(c.d_model * n_input_branches),
                nn.Linear(c.d_model * n_input_branches, n_input_branches),
            )
        self.global_proj = (
            nn.Sequential(
                nn.Linear(c.d_model * 2, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
                nn.Dropout(c.dropout),
            )
            if c.use_global_context
            else None
        )
        self.local_blocks = nn.ModuleList(
            ResidualMLPBlock(c.d_model, c.d_hidden, c.dropout)
            for _ in range(max(1, 2 if c.n_transformer_layers <= 0 else 1))
        )
        n_tcn_layers = max(0, int(c.tcn_layers)) if c.use_tcn_context else 0
        tcn_dilations = list(c.tcn_dilations or [1])
        if not tcn_dilations:
            tcn_dilations = [1]
        self.tcn_blocks = nn.ModuleList(
            TCNContextBlock(
                c.d_model,
                c.d_hidden,
                c.dropout,
                c.tcn_kernel_size,
                tcn_dilations[layer_idx % len(tcn_dilations)],
                c.tcn_block_type,
            )
            for layer_idx in range(n_tcn_layers)
        )
        n_contact_layers = max(0, int(c.contact_graph_layers)) if c.use_contact_graph else 0
        self.contact_blocks = nn.ModuleList(
            ContactGraphBlock(c.d_model, c.d_hidden, c.dropout, c.contact_score_clip)
            for _ in range(n_contact_layers)
        )
        n_sparse_layers = max(0, int(c.sparse_graph_layers)) if c.use_sparse_graph_transformer else 0
        self.sparse_graph_blocks = nn.ModuleList(
            SparseEdgeGraphTransformerBlock(
                c.d_model,
                c.d_hidden,
                c.dropout,
                c.sparse_graph_heads,
                c.sparse_graph_seq_neighbor_k,
                c.sparse_graph_use_contact_edges,
                c.sparse_graph_make_contact_bidirectional,
                c.sparse_graph_use_seq_edges,
                c.sparse_graph_use_global_node,
                c.sparse_graph_use_chain_edge_type,
                c.sparse_graph_edge_hidden,
                c.sparse_graph_score_clip,
                c.sparse_graph_adaptive_residual_gate,
            )
            for _ in range(n_sparse_layers)
        )
        use_aux_dual_branch = bool(c.use_aux_contact_graph) and c.aux_contact_mode == "dual_branch"
        self.aux_sparse_graph_blocks = nn.ModuleList(
            SparseEdgeGraphTransformerBlock(
                c.d_model,
                c.d_hidden,
                c.dropout,
                c.sparse_graph_heads,
                c.sparse_graph_seq_neighbor_k,
                c.sparse_graph_use_contact_edges,
                c.sparse_graph_make_contact_bidirectional,
                c.sparse_graph_use_seq_edges,
                c.sparse_graph_use_global_node,
                c.sparse_graph_use_chain_edge_type,
                c.sparse_graph_edge_hidden,
                c.sparse_graph_score_clip,
                c.sparse_graph_adaptive_residual_gate,
            )
            for _ in range(n_sparse_layers if use_aux_dual_branch else 0)
        )
        self.dual_contact_fusion = None
        self.dual_contact_gate = None
        self.dual_contact_weight = None
        self.dual_contact_attn_q = None
        self.dual_contact_attn_k = None
        self.dual_contact_attn_v = None
        self.dual_contact_attn_out = None
        if use_aux_dual_branch and n_sparse_layers > 0:
            if c.dual_contact_fusion_mode in {"residual_mlp", "gated_residual"}:
                self.dual_contact_fusion = nn.Sequential(
                    nn.LayerNorm(c.d_model * 3),
                    nn.Linear(c.d_model * 3, c.d_hidden),
                    nn.GELU(),
                    nn.Dropout(c.dropout),
                    nn.Linear(c.d_hidden, c.d_model),
                    nn.Dropout(c.dropout),
                )
                if c.dual_contact_fusion_mode == "gated_residual":
                    self.dual_contact_gate = nn.Sequential(
                        nn.LayerNorm(c.d_model * 3),
                        nn.Linear(c.d_model * 3, c.d_model),
                        nn.Sigmoid(),
                    )
            elif c.dual_contact_fusion_mode == "weighted":
                self.dual_contact_weight = nn.Sequential(
                    nn.LayerNorm(c.d_model * 3),
                    nn.Linear(c.d_model * 3, 3),
                )
            elif c.dual_contact_fusion_mode == "attention":
                self.dual_contact_attn_q = nn.Linear(c.d_model, c.d_model)
                self.dual_contact_attn_k = nn.Linear(c.d_model, c.d_model)
                self.dual_contact_attn_v = nn.Linear(c.d_model, c.d_model)
                self.dual_contact_attn_out = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.Dropout(c.dropout))
        n_mamba_layers = max(0, int(c.mamba_layers)) if c.use_mamba_context else 0
        self.mamba_blocks = nn.ModuleList(
            MambaContextBlock(
                c.d_model,
                c.d_hidden,
                c.dropout,
                c.mamba_d_state,
                c.mamba_d_conv,
                c.mamba_expand,
                c.mamba_bidirectional,
                c.mamba_ff_mult,
            )
            for _ in range(n_mamba_layers)
        )
        if c.n_transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.n_heads,
                dim_feedforward=c.d_model * c.transformer_ff_mult,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=c.n_transformer_layers)
        else:
            self.context_encoder = None

        self.final_norm = nn.LayerNorm(c.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(c.d_model, c.d_hidden),
            nn.GELU(),
            nn.Dropout(c.classifier_dropout),
            nn.Linear(c.d_hidden, c.n_classes),
        )

    def forward(
        self,
        esm_embeddings: torch.Tensor,
        protein_mask: torch.Tensor,
        seq_features: torch.Tensor | None = None,
        prottrans_embeddings: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        chain_rel_pos: torch.Tensor | None = None,
        protein_rel_pos: torch.Tensor | None = None,
        contact_edge_index: torch.Tensor | None = None,
        contact_edge_scores: torch.Tensor | None = None,
        aux_contact_edge_index: torch.Tensor | None = None,
        aux_contact_edge_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.config
        mask = protein_mask.bool()
        esm_input = _sanitize(esm_embeddings)
        if self.esm_scalar_mix is not None:
            esm_input = self.esm_scalar_mix(esm_input)
        elif self.esm_last_dim is not None:
            esm_input = esm_input[..., -self.esm_last_dim :]
        esm_branch = self.esm_proj(esm_input)
        if self.prottrans_proj is not None:
            if prottrans_embeddings is not None:
                prottrans_branch = self.prottrans_proj(_sanitize(prottrans_embeddings, clip=10.0))
            else:
                prottrans_branch = torch.zeros_like(esm_branch)
            if c.prottrans_fusion_mode == "gated_residual":
                if c.prottrans_gate_input_mode == "simple":
                    gate_input = torch.cat([esm_branch, prottrans_branch], dim=-1)
                else:
                    gate_input = torch.cat(
                        [
                            esm_branch,
                            prottrans_branch,
                            esm_branch - prottrans_branch,
                            esm_branch * prottrans_branch,
                        ],
                        dim=-1,
                    )
                assert self.prottrans_gate is not None
                gate = self.prottrans_gate(gate_input).to(dtype=prottrans_branch.dtype)
                alpha = self.prottrans_residual_alpha.to(dtype=prottrans_branch.dtype, device=prottrans_branch.device)
                esm_branch = _sanitize(esm_branch + alpha * gate * prottrans_branch)
            elif c.prottrans_fusion_mode == "add":
                alpha = self.prottrans_residual_alpha.to(dtype=prottrans_branch.dtype, device=prottrans_branch.device)
                esm_branch = _sanitize(esm_branch + alpha * prottrans_branch)
            elif c.prottrans_fusion_mode == "input_branch":
                pass
            elif c.prottrans_fusion_mode == "projected_concat":
                esm_branch = _sanitize(torch.cat([esm_branch, prottrans_branch], dim=-1))
            else:
                raise RuntimeError(f"unreachable prottrans_fusion_mode={c.prottrans_fusion_mode!r}")
        branches = [esm_branch]
        if self.prottrans_proj is not None and c.prottrans_fusion_mode == "input_branch":
            if prottrans_embeddings is not None:
                branches.append(self.prottrans_proj(_sanitize(prottrans_embeddings, clip=10.0)))
            else:
                branches.append(torch.zeros_like(esm_branch))

        if self.seq_proj is not None:
            if seq_features is not None:
                branches.append(self.seq_proj(_sanitize(seq_features, clip=10.0)))
            else:
                branches.append(torch.zeros_like(esm_branch))
        if self.chain_embedding is not None:
            if chain_ids is not None:
                safe_chain_ids = torch.clamp(chain_ids.long(), min=0, max=c.max_chains - 1)
                branches.append(self.chain_embedding(safe_chain_ids))
            else:
                branches.append(torch.zeros_like(esm_branch))
        if self.position_proj is not None:
            if chain_rel_pos is not None and protein_rel_pos is not None:
                pos = torch.stack([chain_rel_pos.float(), protein_rel_pos.float()], dim=-1)
                branches.append(self.position_proj(_sanitize(pos, clip=1.0)))
            else:
                branches.append(torch.zeros_like(esm_branch))
        if self.contact_profile_encoder is not None:
            branches.append(
                self.contact_profile_encoder(
                    mask,
                    contact_edge_index,
                    contact_edge_scores,
                    aux_contact_edge_index,
                    aux_contact_edge_scores,
                ).to(dtype=esm_branch.dtype)
            )

        if c.input_fusion_mode == "add":
            h = torch.stack(branches, dim=0).sum(dim=0)
        else:
            concat_branches = torch.cat(branches, dim=-1)
            if c.input_fusion_mode == "gated_residual":
                update = self.input_fusion(concat_branches)
                gate = self.input_gate(concat_branches).to(dtype=update.dtype)
                h = esm_branch + gate * update
            elif c.input_fusion_mode == "weighted":
                weights = torch.softmax(self.input_weights(concat_branches).float(), dim=-1).to(dtype=esm_branch.dtype)
                h = sum(weights[..., idx : idx + 1] * branch for idx, branch in enumerate(branches))
            else:
                raise RuntimeError(f"unreachable input_fusion_mode={c.input_fusion_mode!r}")

        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        for block in self.local_blocks:
            h = block(h).masked_fill(~mask.unsqueeze(-1), 0.0)

        for block in self.tcn_blocks:
            h = block(h, mask)

        for block in self.contact_blocks:
            h = block(h, mask, contact_edge_index, contact_edge_scores)

        if self.aux_sparse_graph_blocks:
            h_primary = h
            for block in self.sparse_graph_blocks:
                h_primary = block(h_primary, mask, chain_ids, contact_edge_index, contact_edge_scores)
            h_aux = h
            for block in self.aux_sparse_graph_blocks:
                h_aux = block(h_aux, mask, chain_ids, aux_contact_edge_index, aux_contact_edge_scores)
            concat = torch.cat([h, h_primary, h_aux], dim=-1)
            if c.dual_contact_fusion_mode == "residual_mlp":
                h = _sanitize(h + self.dual_contact_fusion(concat)).masked_fill(~mask.unsqueeze(-1), 0.0)
            elif c.dual_contact_fusion_mode == "gated_residual":
                update = self.dual_contact_fusion(concat)
                gate = self.dual_contact_gate(concat).to(dtype=update.dtype)
                h = _sanitize(h + gate * update).masked_fill(~mask.unsqueeze(-1), 0.0)
            elif c.dual_contact_fusion_mode == "weighted":
                weights = torch.softmax(self.dual_contact_weight(concat).float(), dim=-1).to(dtype=h.dtype)
                h = (
                    weights[..., 0:1] * h
                    + weights[..., 1:2] * h_primary
                    + weights[..., 2:3] * h_aux
                )
                h = _sanitize(h).masked_fill(~mask.unsqueeze(-1), 0.0)
            elif c.dual_contact_fusion_mode == "attention":
                branch_tokens = torch.stack([h, h_primary, h_aux], dim=-2)
                q = self.dual_contact_attn_q(h).unsqueeze(-2)
                k = self.dual_contact_attn_k(branch_tokens)
                v = self.dual_contact_attn_v(branch_tokens)
                logits = (q * k).sum(dim=-1) / math.sqrt(c.d_model)
                weights = torch.softmax(logits.float(), dim=-1).to(dtype=v.dtype)
                update = (weights.unsqueeze(-1) * v).sum(dim=-2)
                h = _sanitize(h + self.dual_contact_attn_out(update).to(dtype=h.dtype)).masked_fill(
                    ~mask.unsqueeze(-1),
                    0.0,
                )
        else:
            merge_aux = bool(c.use_aux_contact_graph) and c.aux_contact_mode == "merge"
            for block in self.sparse_graph_blocks:
                h = block(
                    h,
                    mask,
                    chain_ids,
                    contact_edge_index,
                    contact_edge_scores,
                    aux_contact_edge_index if merge_aux else None,
                    aux_contact_edge_scores if merge_aux else None,
                )

        for block in self.mamba_blocks:
            h = block(h, mask)

        if self.global_proj is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
            mean_pool = h.sum(dim=1) / denom
            max_pool = h.masked_fill(~mask.unsqueeze(-1), -1.0e4).max(dim=1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            global_context = self.global_proj(torch.cat([mean_pool, max_pool], dim=-1))
            h = _sanitize(h + global_context.unsqueeze(1)).masked_fill(~mask.unsqueeze(-1), 0.0)

        if self.context_encoder is not None:
            h = self.context_encoder(h, src_key_padding_mask=~mask)
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        h = self.final_norm(_sanitize(h)).masked_fill(~mask.unsqueeze(-1), 0.0)
        logits = _sanitize(self.classifier(h))
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        return {"logits": logits, "h_residue": h}
