# Adapted from DeepSeek V4.1 Flash, MIT; Copyright (c) 2023 DeepSeek.
import torch
from torch import Tensor
from typing import Annotated

from .config import V41Config


def dspark_projected_attention(
    queries: Annotated[Tensor, "batch draft_positions heads head_dim"],
    history_key_values: Annotated[Tensor, "batch history_positions head_dim"],
    draft_key_values: Annotated[Tensor, "batch draft_positions head_dim"],
    sink: Annotated[Tensor, "heads"],
    config: V41Config,
) -> Annotated[Tensor, "batch draft_positions heads head_dim"]:
    """Read valid history and every draft slot; return [batch, draft, heads, width].

    queries: [batch, draft, heads, width], history_key_values: [batch, history, width],
    draft_key_values: [batch, draft, width], sink: [heads].
    Inputs are projected/rotated latents. Keys and values share the same vectors.
    """
    if queries.ndim != 4 or history_key_values.ndim != 3:
        raise ValueError("queries must be rank four and history rank three")
    batch, draft_length, heads, width = queries.shape
    if (heads, width) != (config.attention_heads, config.head_dim):
        raise ValueError("query heads and width must match the V4.1 configuration")
    if draft_key_values.shape != (batch, draft_length, width):
        raise ValueError("draft key/value shape must match query batch, length and width")
    history_shape = (history_key_values.shape[0], history_key_values.shape[-1])
    if history_shape != (batch, width) or sink.shape != (heads,):
        raise ValueError("history width/batch or sink head count does not match queries")
    # Keep attention accumulation FP32 even under an outer autocast context.
    with torch.autocast(device_type=queries.device.type, enabled=False):
        key_values = torch.cat((history_key_values, draft_key_values), dim=1).float()
        # b=batch, k=draft query, h=head, d=width, n=key position.
        # Contract d: each query/head scores every retained key.
        scores = torch.einsum("bkhd,bnd->bkhn", queries.float(), key_values)
        # DSparkAttention inherits the target attention scale from the shared head width.
        scores = scores * config.head_dim ** -0.5
        # Broadcast one sink per head across batch and draft positions.
        sink_scores = sink.float().view(1, 1, heads, 1).expand(*scores.shape[:-1], 1)
        probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(dim=-1)
        # No triangular mask: the first draft slot sees the final draft slot.
        # Contract key axis n; remove the sink, which owns no value vector.
        output = torch.einsum("bkhn,bnd->bkhd", probabilities[..., :-1], key_values)
    return output.to(queries.dtype)
