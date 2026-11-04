"""Sparse geometry modules; no external CUDA/PyG extension is required."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class EvolutionaryEdgeBias(nn.Module):
    """Bounded, zero-initialized MSA similarity prior on existing sparse edges."""

    def __init__(self, dim: int, heads: int, rank: int = 8):
        super().__init__()
        self.dim, self.heads, self.rank = dim, heads, rank
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, heads * rank, bias=False)
        self.key = nn.Linear(dim, heads * rank, bias=False)
        self.scale = nn.Parameter(torch.zeros(heads))

    def forward(self, features, mask, edge_index):
        if features is None or features.shape != (*mask.shape, self.dim):
            raise ValueError("Evolutionary edge bias requires aligned projected MSA features")
        features = features.masked_fill(~mask[..., None], 0)
        x = self.norm(features).reshape(-1, self.dim)
        q = self.query(x).reshape(-1, self.heads, self.rank)
        k = self.key(x).reshape(-1, self.heads, self.rank)
        q, k = (F.normalize(t.float(), dim=-1, eps=1.e-6) for t in (q, k))
        src, dst = edge_index.long()
        similarity = (q[dst] * k[src]).sum(-1)
        return 4. * self.scale.float().tanh() * similarity


class VectorPerceptron(nn.Module):
    """Scalar/vector channel mixing following the GVP construction (Jing et al.)."""

    def __init__(self, si: int, vi: int, so: int, vo: int):
        super().__init__()
        hidden = max(vi, vo)
        self.vector_hidden = nn.Linear(vi, hidden, bias=False)
        self.scalar = nn.Linear(si + hidden, so)
        self.vector_out = nn.Linear(hidden, vo, bias=False)
        self.gate = nn.Linear(so, vo)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.vector_hidden(v.transpose(-1, -2)).transpose(-1, -2)
        norms = torch.linalg.vector_norm(hidden.float(), dim=-1).to(s.dtype)
        scalar = F.silu(self.scalar(torch.cat([s, norms], dim=-1)))
        vector = self.vector_out(hidden.transpose(-1, -2)).transpose(-1, -2)
        vector = vector * torch.sigmoid(self.gate(scalar)).unsqueeze(-1)
        return scalar, vector


class GeometryGraphBlock(nn.Module):
    """Query-conditioned edge MLP, edge attention, or GVP-style message passing."""

    def __init__(self, dim: int, edge_dim: int, mode: str, dropout: float = .25,
                 heads: int = 8, vector_in: int = 8, vector_hidden: int = 16,
                 evolution_dim: int = 0):
        super().__init__()
        if mode not in {"edge_mlp", "edge_attention", "gvp"} or dim % heads:
            raise ValueError((mode, dim, heads))
        self.mode, self.heads, self.dim = mode, heads, dim
        self.norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.vector_drop = dropout
        if mode == "gvp":
            self.node_vector = nn.Linear(vector_in, vector_hidden, bias=False)
            self.message = VectorPerceptron(2 * dim + edge_dim, 2 * vector_hidden + 1,
                                           dim, vector_hidden)
            self.vector_readout = nn.Linear(vector_hidden, dim)
        else:
            self.message = nn.Sequential(nn.Linear(2 * dim + edge_dim, dim), nn.SiLU(),
                                         nn.Dropout(dropout), nn.Linear(dim, dim))
        if mode == "edge_attention":
            self.query, self.key = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.edge_bias = nn.Linear(edge_dim, heads)
        self.update = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                    nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.evolution_bias = None
        if evolution_dim:
            if mode != "edge_attention" or evolution_dim < 1:
                raise ValueError("MSA edge bias requires edge_attention and positive feature width")
            with torch.random.fork_rng(devices=[]):
                self.evolution_bias = EvolutionaryEdgeBias(evolution_dim, heads)

    def forward(self, h, vectors, mask, edge_index, edge_features, edge_vectors, edge_scores,
                evolutionary_features=None):
        batch, length, dim = h.shape
        n = batch * length
        if any(x is None for x in (edge_index, vectors, edge_features, edge_vectors, edge_scores)):
            raise ValueError("Enabled geometry block requires geometry inputs")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected geometry edge_index[2,E]")
        src, dst = edge_index.long()
        e = src.numel()
        if (edge_features.shape != (e, self.edge_norm.normalized_shape[0])
                or edge_vectors.shape != (e, 1, 3) or edge_scores.shape != (e,)):
            raise ValueError("Geometry edge tensor shape mismatch")
        flat_mask = mask.reshape(-1)
        if e and (bool((edge_index < 0).any()) or bool((edge_index >= n).any())
                  or not bool((src // length == dst // length).all())
                  or not bool((flat_mask[src] & flat_mask[dst]).all())):
            raise ValueError("Geometry edges cross samples, padding, or bounds")
        extra_bias = (self.evolution_bias(evolutionary_features, mask, edge_index)
                      if self.evolution_bias is not None else None)
        h = h.masked_fill(~mask.unsqueeze(-1), 0)
        flat = self.norm(h).reshape(n, dim)
        if self.mode == "gvp":
            v = self.node_vector(vectors.transpose(-1, -2)).transpose(-1, -2)
        else:
            v = vectors
        if not e:
            return h, v.masked_fill(~mask[..., None, None], 0)
        has_neighbors = torch.zeros(n, dtype=torch.bool, device=h.device)
        has_neighbors[dst] = True
        ef = self.edge_norm(edge_features.to(h.dtype))
        messages_input = torch.cat([flat[src], flat[dst], ef], dim=-1)
        weights = edge_scores.float().clamp(1.e-8, 1.)
        v_message = None
        if self.mode == "gvp":
            fv = v.reshape(n, v.shape[-2], 3)
            messages, v_message = self.message(messages_input,
                                               torch.cat([fv[src], fv[dst], edge_vectors], dim=-2))
        else:
            messages = self.message(messages_input)
        if self.mode == "edge_attention":
            dh = dim // self.heads
            query = self.query(flat).reshape(n, self.heads, dh)
            key = self.key(flat).reshape(n, self.heads, dh)
            scores = (query[dst].float() * key[src].float()).sum(-1) / math.sqrt(dh)
            scores = scores + self.edge_bias(ef).float() + weights.log()[:, None]
            if extra_bias is not None:
                scores = scores + extra_bias
            maxima = torch.full((n, self.heads), -torch.inf, device=h.device)
            maxima.scatter_reduce_(0, dst[:, None].expand_as(scores), scores, reduce="amax", include_self=True)
            numer = (scores - maxima[dst]).exp()
            denom = torch.zeros_like(maxima).index_add_(0, dst, numer)
            attention = numer / denom[dst].clamp_min(1.e-8)
            value = messages.reshape(e, self.heads, dh).float() * attention[..., None]
            aggregate = torch.zeros((n, self.heads, dh), device=h.device).index_add_(0, dst, value)
            aggregate = aggregate.reshape(n, dim)
        else:
            denom = torch.zeros(n, device=h.device).index_add_(0, dst, weights).clamp_min(1.e-8)
            aggregate = torch.zeros((n, dim), device=h.device).index_add_(
                0, dst, messages.float() * weights[:, None]) / denom[:, None]
        if v_message is not None:
            vector_aggregate = torch.zeros((n, v.shape[-2], 3), device=h.device).index_add_(
                0, dst, v_message.float() * weights[:, None, None]) / denom[:, None, None]
            # An invariant readout keeps the vector-message path useful even in a one-layer screen.
            vector_norms = torch.linalg.vector_norm(vector_aggregate, dim=-1).to(h.dtype)
            aggregate = aggregate + self.vector_readout(vector_norms).float()
            new_v = v + vector_aggregate.reshape_as(v).to(v.dtype)
            rms = new_v.float().square().sum(-1).mean(-1, keepdim=True).clamp_min(1.e-8).sqrt()
            new_v = new_v / rms[..., None].to(v.dtype)
            if self.training:
                new_v = new_v * F.dropout(torch.ones_like(new_v[..., :1]), self.vector_drop, True)
            v = torch.where(has_neighbors.reshape(batch, length, 1, 1), new_v, v)
        update = self.update(aggregate.to(h.dtype)).reshape(batch, length, dim)
        update = update.masked_fill(~has_neighbors.reshape(batch, length, 1), 0)
        result = (h + self.dropout(update)).masked_fill(~mask.unsqueeze(-1), 0)
        return result, v.masked_fill(~mask[..., None, None], 0)


class LinearAttentionContext(nn.Module):
    """Noncausal positive-kernel global attention, followed by a residual FFN."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = .25):
        super().__init__()
        if dim % heads:
            raise ValueError((dim, heads))
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        for layer in (self.output, self.ff[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h, mask):
        b, length, dim = h.shape
        q, k, v = self.qkv(self.norm(h)).reshape(b, length, 3, self.heads, dim // self.heads).unbind(2)
        # Casting inputs is insufficient: autocast can lower einsum's contractions again.
        with torch.autocast(device_type=h.device.type, enabled=False):
            q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))
            q, k = F.elu(q) + 1., F.elu(k) + 1.
            keep = mask[:, None, :, None]
            k, v = k * keep, v * keep
            kv = torch.einsum("bhld,bhle->bhde", k, v)
            denom = torch.einsum("bhld,bhd->bhl", q, k.sum(2)).clamp_min(1.e-6)
            values = torch.einsum("bhld,bhde->bhle", q, kv) / denom[..., None]
        values = values.transpose(1, 2).reshape(b, length, dim).to(h.dtype)
        out = h + self.drop(self.output(values))
        out = out + self.drop(self.ff(out))
        return out.masked_fill(~mask.unsqueeze(-1), 0)
