from typing import Annotated
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import V41Config


IndexQuery = Annotated[Tensor, "batch query_tokens heads head_dim"]
IndexKeys = Annotated[Tensor, "batch compressed_positions head_dim"]
IndexHeadWeights = Annotated[Tensor, "batch query_tokens heads"]
IndexScores = Annotated[Tensor, "batch query_tokens compressed_positions"]
VisibleGroups = Annotated[Tensor, "batch query_tokens singleton; completed group count"]
CandidateMask = Annotated[Tensor, "batch query_tokens compressed_positions; boolean"]
SparseIndices = Annotated[Tensor, "batch query_tokens selected_positions; integer or -1"]


def compute_index_scores(
    query: IndexQuery, keys: IndexKeys, head_weights: IndexHeadWeights,
) -> IndexScores:
    """Score [B,T,H,D] queries against [B,C,D] keys with [B,T,H] learned weights."""
    if query.ndim != 4 or keys.ndim != 3 or head_weights.shape != query.shape[:-1]:
        raise ValueError("query, keys and head weights have incompatible ranks or shapes")
    if keys.shape[0] != query.shape[0] or keys.shape[-1] != query.shape[-1]:
        raise ValueError("query and keys must share batch and head width")
    heads, width = query.shape[-2:]
    if heads == 0 or width == 0:
        raise ValueError("query must have positive head count and head width")
    with torch.autocast(device_type=query.device.type, enabled=False):
        # Contract head_dim, rectify each head score, then sum weighted heads.
        products = torch.einsum('bthd,bcd->bthc', query.float(), keys.float()).relu()
        weights = head_weights.float() * (width ** -0.5 * heads ** -0.5)
        return (products * weights[..., None]).sum(dim=2)


def select_candidate_blocks(
    scores: IndexScores, visible: VisibleGroups, config: V41Config,
) -> CandidateMask:
    """Return [B,T,C] mask; visible [B,T,1] counts only completed causal groups."""
    top_blocks, block_size = config.candidate_blocks, config.candidate_block_size
    if scores.ndim != 3 or visible.shape != (*scores.shape[:2], 1):
        raise ValueError("scores and visible counts must be [B,T,C] and [B,T,1]")
    width = scores.shape[-1]
    causal = torch.arange(width, device=scores.device) < visible
    if width == 0:
        return causal
    masked = scores.float().masked_fill(~causal, -torch.inf)
    padded = F.pad(masked, (0, -width % block_size), value=-torch.inf)
    # Padding is unreachable; each block takes its best compressed-position score.
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    # Pin the newest reachable block, even when its scores are below older blocks.
    newest = (visible - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == newest, torch.inf)
    values, indices = block_scores.topk(min(top_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool)
    keep.scatter_(-1, indices, values > -torch.inf)
    return keep.repeat_interleave(block_size, -1)[..., :width] & causal


def select_sparse_indices(
    scores: IndexScores, visible: VisibleGroups, candidates: CandidateMask, config: V41Config,
) -> SparseIndices:
    """Return position-ordered compressed indices, padding unavailable picks with -1."""
    topk = config.index_topk
    if scores.ndim != 3 or visible.shape != (*scores.shape[:2], 1):
        raise ValueError("scores and visible counts must be [B,T,C] and [B,T,1]")
    if candidates.shape != scores.shape or candidates.dtype != torch.bool:
        raise ValueError("candidate mask must be boolean and match scores")
    allowed = candidates & (torch.arange(scores.shape[-1], device=scores.device) < visible)
    masked = scores.float().masked_fill(~allowed, -torch.inf)
    indices = masked.topk(min(topk, scores.shape[-1]), dim=-1).indices.sort(-1).values
    # Gather validity at selected positions; masked top-k filler becomes sentinel -1.
    return indices.masked_fill(~allowed.gather(-1, indices), -1)
