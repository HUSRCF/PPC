"""Sequence-only residue contact-site model using ESM embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    d_model: int = 256
    d_hidden: int = 512
    n_classes: int = 2
    dropout: float = 0.10
    classifier_dropout: float = 0.20
    max_chains: int = 128
    use_chain_embedding: bool = True
    use_seq_features: bool = True
    use_position_features: bool = True
    use_global_context: bool = True
    use_contact_graph: bool = False
    contact_graph_layers: int = 0
    contact_score_clip: float = 1.0
    n_transformer_layers: int = 0
    n_heads: int = 8
    transformer_ff_mult: int = 4
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

        self.esm_proj = nn.Sequential(
            nn.Linear(c.d_esm, c.d_model),
            nn.LayerNorm(c.d_model),
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
        n_contact_layers = max(0, int(c.contact_graph_layers)) if c.use_contact_graph else 0
        self.contact_blocks = nn.ModuleList(
            ContactGraphBlock(c.d_model, c.d_hidden, c.dropout, c.contact_score_clip)
            for _ in range(n_contact_layers)
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
        chain_ids: torch.Tensor | None = None,
        chain_rel_pos: torch.Tensor | None = None,
        protein_rel_pos: torch.Tensor | None = None,
        contact_edge_index: torch.Tensor | None = None,
        contact_edge_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.config
        mask = protein_mask.bool()
        h = self.esm_proj(_sanitize(esm_embeddings))

        if self.seq_proj is not None and seq_features is not None:
            h = h + self.seq_proj(_sanitize(seq_features, clip=10.0))
        if self.chain_embedding is not None and chain_ids is not None:
            safe_chain_ids = torch.clamp(chain_ids.long(), min=0, max=c.max_chains - 1)
            h = h + self.chain_embedding(safe_chain_ids)
        if self.position_proj is not None and chain_rel_pos is not None and protein_rel_pos is not None:
            pos = torch.stack([chain_rel_pos.float(), protein_rel_pos.float()], dim=-1)
            h = h + self.position_proj(_sanitize(pos, clip=1.0))
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        for block in self.local_blocks:
            h = block(h).masked_fill(~mask.unsqueeze(-1), 0.0)

        for block in self.contact_blocks:
            h = block(h, mask, contact_edge_index, contact_edge_scores)

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
