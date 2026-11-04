"""Opt-in post-fusion context for a single padded protein chain per sample."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PostFusionSequenceContext(nn.Module):
    def __init__(self, dim: int, mode: str, hidden: int, dropout: float = .1, scale: float = .1):
        super().__init__()
        if mode not in {"bigru", "mlp"} or hidden < 1 or not math.isfinite(scale):
            raise ValueError((mode, hidden, scale))
        if mode == "bigru" and hidden * 2 != dim:
            raise ValueError("BiGRU hidden size per direction must be half the residual dimension")
        self.mode, self.scale = mode, float(scale)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if mode == "bigru":
            self.context = nn.GRU(dim, hidden, num_layers=1, batch_first=True, bidirectional=True, dropout=0.)
        else:
            self.context = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, h, mask, chain_ids=None):
        mask = mask.bool()
        lengths = mask.sum(1)
        prefix = torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]
        if not bool((lengths > 0).all()) or not torch.equal(mask, prefix):
            raise ValueError("Post-fusion context requires nonempty prefix-padded sequences")
        if self.mode == "bigru" and chain_ids is not None:
            if bool(((chain_ids != chain_ids[:, :1]) & mask).any()):
                raise ValueError("BiGRU context supports one protein chain per sample")
        base = h.masked_fill(~mask[..., None], 0.)
        x = self.norm(base)
        if self.mode == "bigru":
            packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.context(packed)
            update, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=h.shape[1])
        else:
            update = self.context(x)
        out = base + self.scale * self.dropout(update).to(base.dtype)
        return out.masked_fill(~mask[..., None], 0.)
