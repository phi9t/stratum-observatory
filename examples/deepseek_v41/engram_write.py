from typing import Annotated
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V41Config


EngramResidual = Annotated[Tensor, "batch tokens streams hidden_dim"]
EngramHashIds = Annotated[Tensor, "batch tokens hash_columns; global table row IDs"]
EngramTokenMask = Annotated[Tensor, "batch tokens; boolean write permission"]


class EngramMemory(nn.Module):
    """Write hash-table memory into residual copies using context-dependent gates."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.streams, self.dim = config.streams, config.hidden_dim
        self.eps = config.norm_epsilon
        self.score_floor = config.engram_score_floor
        self.hash_columns = config.engram_hash_columns
        self.table = nn.Embedding(config.engram_rows, config.engram_row_dim)
        self.project = nn.Linear(
            self.hash_columns * config.engram_row_dim,
            (self.streams + 1) * self.dim, bias=False)
        self.query_scale = nn.Parameter(torch.ones(self.streams, self.dim))
        self.key_scale = nn.Parameter(torch.ones(self.streams, self.dim))

    def forward(
        self, residual: EngramResidual, hash_ids: EngramHashIds,
        token_mask: EngramTokenMask,
    ) -> EngramResidual:
        """Accept residual [B,T,S,D], global row IDs [B,T,C], and bool mask [B,T]."""
        if residual.ndim != 4 or residual.shape[-2:] != (self.streams, self.dim):
            raise ValueError("residual must have shape [batch, tokens, streams, dim]")
        if hash_ids.shape != (*residual.shape[:2], self.hash_columns):
            raise ValueError("hash IDs must match batch, tokens and hash columns")
        if token_mask.shape != residual.shape[:2] or token_mask.dtype != torch.bool:
            raise ValueError("token mask must be boolean with shape [batch, tokens]")
        with torch.autocast(device_type=residual.device.type, enabled=False):
            # Parameters may be BF16; projection and gate arithmetic accumulate in FP32.
            # Lookup appends row_dim; flatten combines hash columns with their row vectors.
            embedded = self.table(hash_ids).flatten(-2).float()
            projected = F.linear(embedded, self.project.weight.float())
            keys, value = projected.split([self.streams * self.dim, self.dim], dim=-1)
            keys = keys.unflatten(-1, (self.streams, self.dim))
            hidden = residual.float()
            # Normalize over hidden_dim separately for each residual stream.
            inverse_rms = torch.rsqrt(hidden.square().mean(-1) + self.eps)
            inverse_rms = inverse_rms * torch.rsqrt(keys.square().mean(-1) + self.eps)
            scale = self.query_scale.float() * self.key_scale.float()
            score = (hidden * scale * keys).sum(-1) * inverse_rms * self.dim ** -0.5
            signed_root = torch.copysign(score.abs().clamp_min(self.score_floor).sqrt(), score)
            gate = signed_root.sigmoid().masked_fill(~token_mask[..., None], 0)
            # Broadcast the shared value across streams, scaled by each stream gate.
            return (hidden + gate[..., None] * value[..., None, :]).to(residual.dtype)
