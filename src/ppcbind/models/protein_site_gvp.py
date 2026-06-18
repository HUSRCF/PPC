"""Protein-only residue contact-site model for PPC.

The protein-side feature flow follows the PLC Pocket-GVP-Fusion family:
physchem MLP + GVP geometry encoder + invariant vector bridge + contextual
residue transformer + per-residue classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .gvp_layers import GVPEncoder, build_protein_graph, clamp_vector_norm, sanitize_tensor


@dataclass
class ProteinSiteGVPConfig:
    d_phys: int = 62
    d_spatial_scalar: int = 89
    n_spatial_vectors: int = 8
    n_backbone_vectors: int = 4
    d_phys_encoded: int = 128
    d_gvp_scalar: int = 256
    n_gvp_vector_hidden: int = 20
    n_gvp_layers: int = 3
    d_model: int = 256
    n_heads: int = 8
    n_transformer_layers: int = 4
    transformer_ff_mult: int = 4
    dropout: float = 0.1
    classifier_dropout: float = 0.2
    n_classes: int = 2
    use_esm: bool = False
    d_esm: int = 1280
    d_esm_encoded: int = 256
    esm_dropout: float = 0.1
    gvp_edge_rbf_bins: int = 16
    gvp_edge_rbf_dmax: float = 20.0
    gvp_radius: float = 10.0
    gvp_k_neighbors: int = 20
    use_chain_edge: bool = True
    coord_noise_std: float = 0.0
    use_rmsnorm: bool = False
    rmsnorm_eps: float = 1e-6
    freeze_norms: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class PairwiseVectorBridge(nn.Module):
    """Convert vector channels [B,L,V,3] into rotation-invariant Gram features."""

    def __init__(self, n_vec_channels: int) -> None:
        super().__init__()
        self.n_vec_channels = int(n_vec_channels)
        self.register_buffer(
            "tri_i",
            torch.triu_indices(self.n_vec_channels, self.n_vec_channels)[0],
            persistent=False,
        )
        self.register_buffer(
            "tri_j",
            torch.triu_indices(self.n_vec_channels, self.n_vec_channels)[1],
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return self.n_vec_channels * (self.n_vec_channels + 1) // 2

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = clamp_vector_norm(vectors)
        gram = torch.einsum("...ic,...jc->...ij", vectors, vectors)
        return sanitize_tensor(gram[..., self.tri_i, self.tri_j], clip=100.0)


class SafeRMSNorm(nn.Module):
    """RMSNorm implemented with explicit float32 ops for backend-stable backward."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | torch.Size,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = torch.Size((normalized_shape,)) if isinstance(normalized_shape, int) else torch.Size(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        x_float = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        inv_rms = torch.rsqrt(x_float.square().mean(dim=dims, keepdim=True).clamp_min(0.0) + self.eps)
        out = (x_float * inv_rms).to(dtype=dtype)
        if self.weight is not None:
            out = out * self.weight
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class ManualSelfAttention(nn.Module):
    """Self-attention with explicit fp32 softmax and mask guards."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        logit_clip: float = 50.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.scale = self.head_dim ** -0.5
        self.logit_clip = float(logit_clip)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, length, _ = x.shape
        x = sanitize_tensor(x, clip=1.0e4)
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        scores = scores.clamp(min=-self.logit_clip, max=self.logit_clip)
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(key_mask, -1.0e4)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        if key_padding_mask is not None:
            query_mask = key_padding_mask[:, None, :, None].bool()
            attn = attn.masked_fill(query_mask, 0.0)
        attn = self.dropout(attn.to(dtype=x.dtype))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, length, self.d_model)
        out = sanitize_tensor(self.out_proj(out), clip=1.0e4)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class SafeTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer block with explicit sanitization around residuals."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = ManualSelfAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        x = sanitize_tensor(x, clip=1.0e4)

        q = sanitize_tensor(self.norm1(x), clip=1.0e4)
        attn_out = self.self_attn(q, key_padding_mask=src_key_padding_mask)
        x = sanitize_tensor(x + self.dropout1(sanitize_tensor(attn_out, clip=1.0e4)), clip=1.0e4)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        y = sanitize_tensor(self.norm2(x), clip=1.0e4)
        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        x = sanitize_tensor(x + self.dropout2(sanitize_tensor(y, clip=1.0e4)), clip=1.0e4)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return x


class SafeTransformerEncoder(nn.Module):
    """Small ModuleList wrapper matching the old context_encoder call shape."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SafeTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


def convert_layernorm_to_rmsnorm(module: nn.Module, eps: float = 1e-6) -> None:
    """Recursively replace LayerNorm with RMSNorm.

    RMSNorm avoids mean-centering and has a simpler backward path.  Use an
    explicit implementation instead of native nn.RMSNorm because ROCm kernels
    can occasionally produce non-finite affine-weight gradients here.
    LayerNorm bias is intentionally dropped because RMSNorm has only a scale.
    """
    # PyTorch TransformerEncoderLayer fastpath assumes norm1/norm2 expose
    # LayerNorm-style bias tensors.  Keep those internal norms unchanged and
    # convert the surrounding MLP/GVP/fusion norms.
    if isinstance(module, (nn.TransformerEncoder, nn.TransformerEncoderLayer)):
        return

    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            rms = SafeRMSNorm(
                child.normalized_shape,
                eps=eps,
                elementwise_affine=child.elementwise_affine,
                device=child.weight.device if child.elementwise_affine else None,
                dtype=child.weight.dtype if child.elementwise_affine else None,
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    rms.weight.copy_(child.weight.detach())
            setattr(module, name, rms)
        else:
            convert_layernorm_to_rmsnorm(child, eps=eps)


def freeze_norm_parameters(module: nn.Module) -> int:
    """Freeze affine scale parameters of normalization layers."""
    n_frozen = 0
    for child in module.modules():
        if isinstance(child, (nn.LayerNorm, SafeRMSNorm, nn.RMSNorm)):
            for param in child.parameters(recurse=False):
                if param.requires_grad:
                    param.requires_grad_(False)
                    n_frozen += param.numel()
    return n_frozen


class ProteinSiteGVP(nn.Module):
    """Residue-level protein contact-site classifier."""

    def __init__(self, config: dict[str, Any] | ProteinSiteGVPConfig | None = None) -> None:
        super().__init__()
        if config is None:
            self.config = ProteinSiteGVPConfig()
        elif isinstance(config, ProteinSiteGVPConfig):
            self.config = config
        else:
            known = {f.name for f in ProteinSiteGVPConfig.__dataclass_fields__.values()}
            cfg = {k: v for k, v in config.items() if k in known}
            extra = {k: v for k, v in config.items() if k not in known}
            self.config = ProteinSiteGVPConfig(**cfg, extra=extra)
        c = self.config

        self.physchem_encoder = nn.Sequential(
            nn.Linear(c.d_phys, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, c.d_phys_encoded),
            nn.LayerNorm(c.d_phys_encoded),
            nn.GELU(),
        )

        n_gvp_vectors = c.n_backbone_vectors + c.n_spatial_vectors
        edge_s_dim = c.gvp_edge_rbf_bins + 4 + (1 if c.use_chain_edge else 0)
        self.gvp_encoder = GVPEncoder(
            node_in_dim=(c.d_spatial_scalar, n_gvp_vectors),
            node_h_dim=(c.d_gvp_scalar, c.n_gvp_vector_hidden),
            edge_in_dim=(edge_s_dim, 2),
            n_layers=c.n_gvp_layers,
            drop_rate=c.dropout,
        )
        self.vector_bridge = PairwiseVectorBridge(c.n_gvp_vector_hidden)

        d_geo = c.d_gvp_scalar + self.vector_bridge.out_dim + c.d_phys_encoded
        self.geo_proj = nn.Sequential(
            nn.Linear(d_geo, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        if c.use_esm:
            self.esm_proj = nn.Sequential(
                nn.Linear(c.d_esm, c.d_esm_encoded),
                nn.LayerNorm(c.d_esm_encoded),
                nn.GELU(),
                nn.Dropout(c.esm_dropout),
                nn.Linear(c.d_esm_encoded, c.d_model),
                nn.LayerNorm(c.d_model),
                nn.GELU(),
            )
            self.esm_fusion_gate = nn.Sequential(
                nn.Linear(c.d_model * 2, c.d_model),
                nn.Sigmoid(),
            )
        else:
            self.esm_proj = None
            self.esm_fusion_gate = None

        self.context_encoder = SafeTransformerEncoder(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_model * c.transformer_ff_mult,
            dropout=c.dropout,
            num_layers=c.n_transformer_layers,
        )
        self.final_norm = nn.LayerNorm(c.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(c.d_model + c.d_phys_encoded, 256),
            nn.GELU(),
            nn.Dropout(c.classifier_dropout),
            nn.Linear(256, c.n_classes),
        )
        if c.use_rmsnorm:
            convert_layernorm_to_rmsnorm(self, eps=c.rmsnorm_eps)
        self.n_frozen_norm_parameters = freeze_norm_parameters(self) if c.freeze_norms else 0

    def forward(
        self,
        protein_physchem: torch.Tensor,
        protein_spatial_scalar: torch.Tensor,
        protein_spatial_vector: torch.Tensor,
        protein_backbone_vector: torch.Tensor,
        protein_coords: torch.Tensor,
        protein_mask: torch.Tensor,
        chain_ids: torch.Tensor | None = None,
        esm_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.config
        bsz, max_len, _ = protein_physchem.shape
        mask = protein_mask.bool()
        device = protein_physchem.device

        protein_physchem = sanitize_tensor(protein_physchem, clip=1.0e4)
        protein_spatial_scalar = sanitize_tensor(protein_spatial_scalar, clip=1.0e4)
        protein_spatial_vector = clamp_vector_norm(protein_spatial_vector)
        protein_backbone_vector = clamp_vector_norm(protein_backbone_vector)
        protein_coords = sanitize_tensor(protein_coords, clip=1.0e4)

        phys = sanitize_tensor(self.physchem_encoder(protein_physchem), clip=1.0e4)
        all_vectors_in = clamp_vector_norm(torch.cat([protein_backbone_vector, protein_spatial_vector], dim=2))
        n_coords = sanitize_tensor(protein_coords + protein_backbone_vector[:, :, 1, :], clip=1.0e4)
        c_coords = sanitize_tensor(protein_coords + protein_backbone_vector[:, :, 3, :], clip=1.0e4)

        h_s_gvp = protein_physchem.new_zeros((bsz, max_len, c.d_gvp_scalar))
        h_v_gvp = protein_physchem.new_zeros((bsz, max_len, c.n_gvp_vector_hidden, 3))

        n_valid = mask.sum(dim=1)
        if int(n_valid.sum().item()) > 0:
            flat_s = protein_spatial_scalar[mask]
            flat_v = all_vectors_in[mask]
            flat_ca = protein_coords[mask]
            flat_n = n_coords[mask]
            flat_c = c_coords[mask]
            flat_chain = chain_ids[mask] if chain_ids is not None else None

            if self.training and c.coord_noise_std > 0:
                noise = torch.randn_like(flat_ca) * c.coord_noise_std
                flat_ca = flat_ca + noise
                flat_n = flat_n + noise
                flat_c = flat_c + noise

            edge_indices, edge_scalars, edge_vectors = [], [], []
            offset = 0
            for batch_idx in range(bsz):
                n_i = int(n_valid[batch_idx].item())
                if n_i == 0:
                    continue
                chain_i = flat_chain[offset : offset + n_i] if flat_chain is not None and c.use_chain_edge else None
                ei, es, ev = build_protein_graph(
                    flat_ca[offset : offset + n_i],
                    flat_n[offset : offset + n_i],
                    flat_c[offset : offset + n_i],
                    k=c.gvp_k_neighbors,
                    radius=c.gvp_radius,
                    num_rbf=c.gvp_edge_rbf_bins,
                    rbf_dmax=c.gvp_edge_rbf_dmax,
                    chain_ids=chain_i,
                )
                edge_indices.append(ei + offset)
                edge_scalars.append(es)
                edge_vectors.append(ev)
                offset += n_i

            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
                edge_s = torch.cat(edge_scalars, dim=0)
                edge_v = torch.cat(edge_vectors, dim=0)
            else:
                edge_s_dim = c.gvp_edge_rbf_bins + 4 + (1 if c.use_chain_edge else 0)
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_s = protein_physchem.new_zeros((0, edge_s_dim))
                edge_v = protein_physchem.new_zeros((0, 2, 3))

            flat_h_s, flat_h_v = self.gvp_encoder(flat_s, flat_v, edge_index, edge_s, edge_v)
            flat_h_s = sanitize_tensor(flat_h_s, clip=1.0e4)
            flat_h_v = clamp_vector_norm(flat_h_v)
            h_s_gvp[mask] = flat_h_s
            h_v_gvp[mask] = flat_h_v

        v_inv = self.vector_bridge(h_v_gvp)
        h_geo = sanitize_tensor(torch.cat([h_s_gvp, v_inv, phys], dim=-1), clip=1.0e4)
        h = sanitize_tensor(self.geo_proj(h_geo), clip=1.0e4)
        h_esm = None
        if c.use_esm:
            if esm_embeddings is None:
                h_esm = torch.zeros_like(h)
            else:
                if esm_embeddings.shape[:2] != h.shape[:2]:
                    raise ValueError(
                        f"ESM residue shape {tuple(esm_embeddings.shape[:2])} "
                        f"does not match protein shape {tuple(h.shape[:2])}"
                    )
                if esm_embeddings.shape[-1] != c.d_esm:
                    raise ValueError(
                        f"ESM dim {esm_embeddings.shape[-1]} does not match configured d_esm={c.d_esm}"
                    )
                h_esm = sanitize_tensor(self.esm_proj(sanitize_tensor(esm_embeddings, clip=1.0e4)), clip=1.0e4)
            gate = self.esm_fusion_gate(torch.cat([h, h_esm], dim=-1))
            h = sanitize_tensor(gate * h + (1.0 - gate) * h_esm, clip=1.0e4)
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        h = self.context_encoder(h, src_key_padding_mask=~mask)
        h = sanitize_tensor(self.final_norm(h), clip=1.0e4).masked_fill(~mask.unsqueeze(-1), 0.0)
        logits = self.classifier(sanitize_tensor(torch.cat([h, phys], dim=-1), clip=1.0e4))
        logits = sanitize_tensor(logits, clip=1.0e4)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        out = {"logits": logits, "h_protein": h, "h_gvp_scalar": h_s_gvp, "h_gvp_vector": h_v_gvp}
        if h_esm is not None:
            out["h_esm"] = h_esm.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out
