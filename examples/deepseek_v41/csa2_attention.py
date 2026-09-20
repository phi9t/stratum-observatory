from dataclasses import dataclass, replace
from typing import Annotated
import torch
from torch import Tensor

from .config import V41Config


AttentionQuery = Annotated[Tensor, "batch query_tokens heads head_dim"]
LocalLatents = Annotated[Tensor, "batch local_history head_dim"]
CompressedLatents = Annotated[Tensor, "batch compressed_positions head_dim"]
AttentionSelection = Annotated[Tensor, "batch query_tokens picks; causal indices or -1"]
QueryPositions = Annotated[Tensor, "query_tokens; absolute local-history positions"]
AttentionSink = Annotated[Tensor, "heads; finite denominator-only logits"]


def sparse_attention(
    query: AttentionQuery, local_kv: LocalLatents, compressed_kv: CompressedLatents,
    selected: AttentionSelection, positions: QueryPositions, sink: AttentionSink,
    config: V41Config,
) -> AttentionQuery:
    """Attend over local and selected compressed latents using one sink-augmented softmax.

    query [B,T,H,D], caches [B,L,D]/[B,C,D], selected [B,T,K], positions [T].
    Positions address full local history; selected is already causally filtered or -1.
    A finite sink [H] has zero value. Output [B,T,H,D] retains the query dtype.
    """
    window = config.window_size
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, tokens, heads, width]")
    batch, steps, heads, width = query.shape
    if width == 0 or heads == 0:
        raise ValueError("head count and head width must be positive")
    for cache in (local_kv, compressed_kv):
        if cache.ndim != 3 or (cache.shape[0], cache.shape[-1]) != (batch, width):
            raise ValueError("cache must have shape [batch, cached positions, head width]")
    if local_kv.shape[1] == 0:
        raise ValueError("local history must include the current target position")
    if selected.ndim != 3 or selected.shape[:2] != (batch, steps):
        raise ValueError("selected indices must have shape [batch, tokens, picks]")
    if positions.shape != (steps,) or sink.shape != (heads,):
        raise ValueError("positions and sink must match token and head counts")
    offsets = torch.arange(window - 1, -1, -1, device=query.device)
    local_indices = (positions[:, None] - offsets).expand(batch, steps, window)
    valid_local = (local_indices >= 0) & (local_indices < local_kv.shape[1])
    # Shift compressed indices into the concatenated local + compressed cache.
    compressed_indices = selected + local_kv.shape[1]
    indices = torch.cat([local_indices, compressed_indices], dim=-1)
    valid_compressed = (selected >= 0) & (selected < compressed_kv.shape[1])
    valid = torch.cat([valid_local, valid_compressed], dim=-1)
    cache = torch.cat([local_kv.float(), compressed_kv.float()], dim=1)
    batch_indices = torch.arange(batch, device=query.device)[:, None, None]
    # Advanced indexing gathers each query's selected vectors from its own batch.
    gathered = cache[batch_indices, indices.masked_fill(~valid, 0)]
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.einsum('bthd,btkd->bthk', query.float(), gathered) * width ** -0.5
        logits = logits.masked_fill(~valid[:, :, None, :], -torch.inf)
        sink_logits = sink.float()[None, None, :, None].expand(batch, steps, heads, 1)
        # The sink shares normalization but contributes no value to the weighted sum.
        probability = torch.cat([logits, sink_logits], -1).softmax(-1)[..., :-1]
        return torch.einsum('bthk,btkd->bthd', probability, gathered).to(query.dtype)


@dataclass(frozen=True)
class QueryExample:
    config: V41Config
    queries: AttentionQuery
    local_kv: LocalLatents
    compressed_kv: CompressedLatents
    selected: AttentionSelection
    positions: QueryPositions
    sink: AttentionSink
    weights: Annotated[Tensor, "batch query_alternatives heads local+compressed+sink"]
    outputs: AttentionQuery


def run_query_example(config: V41Config) -> QueryExample:
    """Two counterfactual queries read exactly the same small KV dictionary on CPU.

    Only numerical sizes change from the caller's global V41Config. The token
    axis here holds alternatives at the SAME position, not successive tokens
    or layers. These illustrative vectors are not checkpoint activations.
    Real Attention layers still project fresh queries and local KV, and apply
    output projections; compressed-cache reuse does not make a query-only model.
    """
    small = replace(config, head_dim=2, attention_heads=1, window_size=2)
    local = torch.tensor([[[2., 0.], [0., 1.]]])
    compressed = torch.tensor([[[1., 2.]]])
    queries = torch.tensor([[[[2., 0.]], [[0., 2.]]]])
    positions = torch.tensor([1, 1])
    selected = torch.tensor([[[0], [0]]])
    # A fixed illustrative value for the learned per-head sink logit.
    sink = torch.tensor([.5])
    outputs = sparse_attention(queries, local, compressed, selected, positions, sink, small)

    # Expose this tiny dictionary's weights for inspection, not a second output
    # implementation. Both alternatives gather every row in this same order.
    dictionary = torch.cat([local, compressed], dim=1)
    logits = queries @ dictionary[:, None].transpose(-1, -2) / small.head_dim ** .5
    sink_logits = sink.reshape(1, 1, 1, 1).expand(1, 2, 1, 1)
    weights = torch.cat([logits, sink_logits], dim=-1).softmax(-1)
    # weights[..., -1] has no value vector; only the other lanes contribute.
    return QueryExample(small, queries, local, compressed, selected, positions,
                        sink, weights, outputs)
