from typing import Annotated
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V41Config


CompressionTokens = Annotated[Tensor, "batch tokens hidden_dim"]
CompressionLatents = Annotated[Tensor, "batch complete_groups latent_dim"]
CompressionTail = Annotated[Tensor, "batch unfinished_tokens latent_dim; FP32"]
NormValues = Annotated[Tensor, "arbitrary leading axes, latent_dim"]


class CompressionNorm(nn.Module):
    """Learned RMS normalization with FP32 statistics and input-dtype output."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.head_dim))
        self.eps = config.norm_epsilon

    def forward(self, values: NormValues) -> NormValues:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(values.dtype)


class TokenCompressor(nn.Module):
    """Ratio-one path: project and normalize each token without a pooling gate."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.norm = CompressionNorm(config)
        self.project = nn.Linear(config.hidden_dim, config.head_dim, bias=False)

    def forward(self, tokens: CompressionTokens) -> CompressionLatents:
        """Map [B,T,D] to [B,T,K]; return the incoming activation dtype."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.project.in_features:
            raise ValueError("tokens must have shape [batch, tokens, input dimension]")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            projected = F.linear(tokens.float(), self.project.weight.float())
            return self.norm(projected).to(tokens.dtype)


class GroupedCompressor(nn.Module):
    """Pool complete consecutive groups; caller owns the unfinished FP32 tail."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.norm = CompressionNorm(config)
        self.ratio = config.encoder_compression_ratio
        self.latent_dim = config.head_dim
        self.project = nn.Linear(config.hidden_dim, config.head_dim, bias=False)
        self.gate = nn.Linear(config.hidden_dim, config.head_dim, bias=False)

    def empty_state(
        self, tokens: CompressionTokens,
    ) -> tuple[CompressionTail, CompressionTail]:
        """Start a new sequence; state follows the input device, never a global cache."""
        empty = tokens.new_empty(tokens.shape[0], 0, self.latent_dim, dtype=torch.float32)
        return empty, empty.clone()

    def forward(
        self, tokens: CompressionTokens, state: tuple[CompressionTail, CompressionTail],
    ) -> tuple[CompressionLatents, tuple[CompressionTail, CompressionTail]]:
        """Consume [B,T,D]; return [B,complete_groups,K] and unpooled values/scores.

        Tail tensors have shape [B,tail,K], where tail < ratio. They retain autograd
        history; detach explicitly between training segments when truncation is intended.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.project.in_features:
            raise ValueError("tokens must have shape [batch, tokens, input dimension]")
        previous_values, previous_scores = state
        expected_edges = (tokens.shape[0], self.latent_dim)
        for tail in (previous_values, previous_scores):
            if (tail.ndim != 3 or (tail.shape[0], tail.shape[2]) != expected_edges
                    or tail.shape[1] >= self.ratio):
                raise ValueError("state must be an unfinished group [batch, tail, latent]")
            if tail.device != tokens.device or tail.dtype != torch.float32:
                raise ValueError("state must use FP32 on the input device")
        if previous_values.shape != previous_scores.shape:
            raise ValueError("state values and scores must have equal shapes")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            values = torch.cat(
                [previous_values, F.linear(tokens.float(), self.project.weight.float())], dim=1)
            scores = torch.cat(
                [previous_scores, F.linear(tokens.float(), self.gate.weight.float())], dim=1)
            complete = values.shape[1] // self.ratio * self.ratio
            grouped_values = values[:, :complete].unflatten(1, (-1, self.ratio))
            grouped_scores = scores[:, :complete].unflatten(1, (-1, self.ratio))
            # Softmax competes across tokens within a group, independently per channel.
            pooled = (grouped_values * grouped_scores.softmax(dim=2)).sum(dim=2)
            # Clone the short tail so it cannot retain a large prefill storage allocation.
            next_state = (values[:, complete:].clone(), scores[:, complete:].clone())
            return self.norm(pooled).to(tokens.dtype), next_state
