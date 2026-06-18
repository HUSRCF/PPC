"""Minimal PyTorch GVP layers adapted from the PLC protein-side model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def sanitize_tensor(x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
    """Replace non-finite values and optionally clamp extreme activations."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        x = x.clamp(min=-clip, max=clip)
    return x


def clamp_vector_norm(v: torch.Tensor, max_norm: float = 10.0, eps: float = 1e-6) -> torch.Tensor:
    """Clamp each 3D vector channel by norm without changing valid directions."""
    v = sanitize_tensor(v)
    norm = torch.linalg.vector_norm(v.float(), dim=-1, keepdim=True).to(dtype=v.dtype)
    scale = (max_norm / norm.clamp_min(eps)).clamp_max(1.0)
    return sanitize_tensor(v * scale)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-4) -> torch.Tensor:
    """Normalize with a larger floor than F.normalize to avoid degenerate frames."""
    x = sanitize_tensor(x)
    norm = torch.linalg.vector_norm(x.float(), dim=dim, keepdim=True).to(dtype=x.dtype)
    return sanitize_tensor(x / norm.clamp_min(eps))


def build_local_frame(
    coords_ca: torch.Tensor,
    coords_n: torch.Tensor,
    coords_c: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Build residue frames with a deterministic fallback for collinear atoms."""
    fwd = safe_normalize(coords_c - coords_ca, eps=eps)
    n2ca = safe_normalize(coords_ca - coords_n, eps=eps)

    up_raw = torch.cross(fwd, n2ca, dim=-1)
    up_norm = torch.linalg.vector_norm(up_raw.float(), dim=-1, keepdim=True).to(dtype=up_raw.dtype)
    up = safe_normalize(up_raw, eps=eps)

    axis_x = torch.zeros_like(fwd)
    axis_x[..., 0] = 1.0
    axis_y = torch.zeros_like(fwd)
    axis_y[..., 1] = 1.0
    fallback_axis = torch.where(fwd[..., :1].abs() > 0.9, axis_y, axis_x)
    fallback_up = safe_normalize(torch.cross(fwd, fallback_axis, dim=-1), eps=eps)
    up = torch.where(up_norm > eps, up, fallback_up)

    right = safe_normalize(torch.cross(fwd, up, dim=-1), eps=eps)
    return torch.stack([fwd, up, right], dim=-1)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean aggregate src rows into dim-0 positions given by index."""
    src = sanitize_tensor(src)
    out = src.new_zeros((dim_size, *src.shape[1:]))
    count = src.new_zeros((dim_size, *([1] * (src.ndim - 1))))
    out.index_add_(0, index, src)
    ones = torch.ones((src.shape[0], *([1] * (src.ndim - 1))), device=src.device, dtype=src.dtype)
    count.index_add_(0, index, ones)
    return sanitize_tensor(out / count.clamp_min(1.0))


class GVP(nn.Module):
    """Geometric vector perceptron with vector gating."""

    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations: tuple = (F.relu, None),
        vector_gate: bool = True,
        eps: float = 1e-8,
        max_vector_norm: float = 10.0,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        self.vector_gate = vector_gate
        self.eps = eps
        self.max_vector_norm = max_vector_norm

        self.scalar_linear = nn.Linear(self.si + self.vi, self.so)
        if self.vo > 0 and self.vi > 0:
            self.vector_linear = nn.Linear(self.vi, self.vo, bias=False)
            if self.vector_gate:
                self.gate_linear = nn.Linear(self.si + self.vi, self.vo)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = sanitize_tensor(s, clip=1.0e4)
        v = clamp_vector_norm(v, max_norm=self.max_vector_norm)
        v_norm = torch.linalg.vector_norm(v.float(), dim=-1).to(dtype=v.dtype)
        v_norm = sanitize_tensor(v_norm.clamp_min(self.eps).clamp_max(self.max_vector_norm))
        sv = torch.cat([s, v_norm], dim=-1)
        s_out = self.scalar_linear(sv)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        s_out = sanitize_tensor(s_out, clip=1.0e4)

        if self.vo > 0 and self.vi > 0:
            v_out = self.vector_linear(v.transpose(-2, -1)).transpose(-2, -1)
            if self.vector_gate:
                v_out = v_out * torch.sigmoid(self.gate_linear(sv)).unsqueeze(-1)
            if self.vector_act is not None:
                v_out = self.vector_act(v_out)
            v_out = clamp_vector_norm(v_out, max_norm=self.max_vector_norm)
        elif self.vo > 0:
            v_out = s.new_zeros((*s.shape[:-1], self.vo, 3))
        else:
            v_out = s.new_zeros((*s.shape[:-1], 0, 3))
        return s_out, v_out


class GVPDropout(nn.Module):
    """Drop scalar channels normally and vector channels as whole 3D vectors."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p <= 0.0:
            return s, v
        s = F.dropout(s, p=self.p, training=True)
        keep = 1.0 - self.p
        mask = torch.bernoulli(v.new_full(v.shape[:-1] + (1,), keep))
        v = v * mask / keep
        return s, v


class GVPConv(nn.Module):
    """Explicit message-passing GVP convolution."""

    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        n_layers: int = 3,
        vector_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        message_in = (self.si + self.se, self.vi + self.ve)
        message_h = (self.so, self.vo)
        self.message_func = nn.ModuleList(
            GVP(message_in if i == 0 else message_h, message_h, vector_gate=vector_gate)
            for i in range(n_layers)
        )

        update_in = (self.si + self.so, self.vi + self.vo)
        self.update_func = nn.ModuleList(
            GVP(update_in if i == 0 else message_h, out_dims if i == n_layers - 1 else message_h, vector_gate=vector_gate)
            for i in range(n_layers)
        )
        self.dropout = GVPDropout(dropout)
        self.use_residual = self.si == self.so and self.vi == self.vo

    def forward(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return x_s, x_v

        x_s = sanitize_tensor(x_s, clip=1.0e4)
        x_v = clamp_vector_norm(x_v)
        edge_s = sanitize_tensor(edge_s, clip=1.0e4)
        edge_v = clamp_vector_norm(edge_v)

        src, dst = edge_index
        msg_s = torch.cat([x_s[src], edge_s], dim=-1)
        msg_v = torch.cat([x_v[src], edge_v], dim=-2)

        for layer in self.message_func:
            msg_s, msg_v = layer(msg_s, msg_v)
            msg_s, msg_v = self.dropout(msg_s, msg_v)

        aggr_s = scatter_mean(msg_s, dst, x_s.size(0))
        aggr_v = scatter_mean(msg_v, dst, x_v.size(0))

        upd_s = torch.cat([x_s, aggr_s], dim=-1)
        upd_v = torch.cat([x_v, aggr_v], dim=-2)
        for layer in self.update_func:
            upd_s, upd_v = layer(upd_s, upd_v)
            upd_s, upd_v = self.dropout(upd_s, upd_v)

        if self.use_residual:
            upd_s = upd_s + x_s
            upd_v = upd_v + x_v
        return sanitize_tensor(upd_s, clip=1.0e4), clamp_vector_norm(upd_v)


class GVPEncoder(nn.Module):
    """Stacked GVPConv encoder for residue graphs."""

    def __init__(
        self,
        node_in_dim: Tuple[int, int],
        node_h_dim: Tuple[int, int] = (256, 20),
        edge_in_dim: Tuple[int, int] = (21, 2),
        n_layers: int = 3,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.W_s_in = nn.Linear(node_in_dim[0], node_h_dim[0])
        self.W_v_in = nn.Linear(node_in_dim[1], node_h_dim[1], bias=False)
        self.layers = nn.ModuleList(
            GVPConv(node_h_dim, node_h_dim, edge_in_dim, dropout=drop_rate)
            for _ in range(n_layers)
        )
        self.norm_s = nn.LayerNorm(node_h_dim[0])

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_s = self.W_s_in(sanitize_tensor(node_s, clip=1.0e4))
        node_v = self.W_v_in(clamp_vector_norm(node_v).transpose(-2, -1)).transpose(-2, -1)
        node_v = clamp_vector_norm(node_v)
        for layer in self.layers:
            node_s, node_v = layer(node_s, node_v, edge_index, edge_s, edge_v)
        return sanitize_tensor(self.norm_s(node_s), clip=1.0e4), clamp_vector_norm(node_v)


def rbf_encode(dist: torch.Tensor, d_min: float = 0.0, d_max: float = 20.0, n_bins: int = 16) -> torch.Tensor:
    dist = sanitize_tensor(dist, clip=d_max * 10.0)
    centers = torch.linspace(d_min, d_max, n_bins, device=dist.device, dtype=dist.dtype)
    sigma = (d_max - d_min) / n_bins
    return sanitize_tensor(torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2.0 * sigma * sigma)))


def build_protein_graph(
    coords_ca: torch.Tensor,
    coords_n: torch.Tensor,
    coords_c: torch.Tensor,
    k: int = 20,
    radius: float = 10.0,
    num_rbf: int = 16,
    rbf_dmax: float = 20.0,
    chain_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a KNN plus backbone residue graph using PLC-compatible features."""
    coords_ca = sanitize_tensor(coords_ca).detach()
    coords_n = sanitize_tensor(coords_n).detach()
    coords_c = sanitize_tensor(coords_c).detach()
    if chain_ids is not None:
        chain_ids = chain_ids.detach()

    n = coords_ca.size(0)
    device = coords_ca.device
    dtype = coords_ca.dtype
    edge_s_dim = num_rbf + 4 + (1 if chain_ids is not None else 0)
    if n <= 1:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0, edge_s_dim), dtype=dtype, device=device),
            torch.zeros((0, 2, 3), dtype=dtype, device=device),
        )

    curr_k = min(k, n - 1)
    dist = torch.cdist(coords_ca.float(), coords_ca.float()).to(dtype=dtype)
    dist = sanitize_tensor(dist, clip=1.0e4)
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = dist.topk(curr_k, dim=1, largest=False)
    src_knn = torch.arange(n, device=device).unsqueeze(1).expand(-1, curr_k).reshape(-1)
    dst_knn = nn_idx.reshape(-1)

    seq = torch.arange(n, device=device)
    src_bb = torch.cat([seq[:-1], seq[1:]])
    dst_bb = torch.cat([seq[1:], seq[:-1]])

    src = torch.cat([src_knn, src_bb])
    dst = torch.cat([dst_knn, dst_bb])
    key = src * n + dst
    key = torch.unique(key, sorted=False)
    src = key // n
    dst = key % n

    edge_vec = coords_ca[dst] - coords_ca[src]
    edge_dist = torch.linalg.vector_norm(edge_vec.float(), dim=-1).to(dtype=dtype)
    edge_dist = sanitize_tensor(edge_dist, clip=1.0e4)
    seq_offset = (dst - src).abs()
    keep = (edge_dist <= radius) | (seq_offset == 1)
    src, dst = src[keep], dst[keep]
    edge_vec, edge_dist, seq_offset = edge_vec[keep], edge_dist[keep], seq_offset[keep]

    edge_s_parts = [
        rbf_encode(edge_dist, 0.0, rbf_dmax, num_rbf),
        (seq_offset == 1).float().unsqueeze(-1),
        ((seq_offset >= 2) & (seq_offset <= 5)).float().unsqueeze(-1),
        ((seq_offset >= 6) & (seq_offset <= 12)).float().unsqueeze(-1),
        (seq_offset > 12).float().unsqueeze(-1),
    ]
    if chain_ids is not None:
        edge_s_parts.append((chain_ids[src] == chain_ids[dst]).float().unsqueeze(-1))
    edge_s = torch.cat(edge_s_parts, dim=-1)

    frame = build_local_frame(coords_ca, coords_n, coords_c)

    global_dir = safe_normalize(edge_vec)
    local_dir = torch.bmm(frame[src].transpose(1, 2), global_dir.unsqueeze(-1)).squeeze(-1)
    edge_index = torch.stack([src, dst], dim=0)
    edge_v = clamp_vector_norm(torch.stack([global_dir, local_dir], dim=1), max_norm=1.0)
    edge_s = torch.nan_to_num(edge_s)
    edge_v = torch.nan_to_num(edge_v)
    return edge_index, edge_s, edge_v
