// Authored educational adaptations of DeepSeek V4.1 Flash publisher code (MIT).
export const MEMORY_EXAMPLES = [
  {
    id: 'engram-write', title: 'Engram: lookup and context-gated write', stages: ['engram'],
    sources: ['dsv41-code', 'dsv41-engram'], symbols: ['ParallelEngramEmbedding', 'Engram.forward', 'NgramHashState'], requires: [],
    summary: 'Look up the hash rows, project a key for each residual stream and a shared value, then apply the signed-square-root gate. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Hash IDs are global table row IDs already produced by the publisher hash layout; tokenization, n-gram hashing, table sharding and FP8 row dequantization are outside this excerpt. Weights may use FP32 or BF16; projection and gate arithmetic use FP32 and output keeps the residual dtype. The token mask and normalization epsilon are required caller inputs; no checkpoint scalar default is implied.',
    code: `import math
from typing import Annotated
import torch
from torch import Tensor, nn
from torch.nn import functional as F


EngramResidual = Annotated[Tensor, "batch tokens streams hidden_dim"]
EngramHashIds = Annotated[Tensor, "batch tokens hash_columns; global table row IDs"]
EngramTokenMask = Annotated[Tensor, "batch tokens; boolean write permission"]


class EngramMemory(nn.Module):
    """Write hash-table memory into residual copies using context-dependent gates."""

    def __init__(
        self, rows: int, row_dim: int, hash_columns: int,
        streams: int, dim: int, eps: float,
    ) -> None:
        super().__init__()
        dimensions = (rows, row_dim, hash_columns, streams, dim)
        if any(type(size) is not int or size <= 0 for size in dimensions):
            raise ValueError("table and residual dimensions must be positive integers")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("normalization epsilon must be finite and positive")
        self.streams, self.dim, self.eps = streams, dim, eps
        self.hash_columns = hash_columns
        self.table = nn.Embedding(rows, row_dim)
        self.project = nn.Linear(hash_columns * row_dim, (streams + 1) * dim, bias=False)
        self.query_scale = nn.Parameter(torch.ones(streams, dim))
        self.key_scale = nn.Parameter(torch.ones(streams, dim))

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
            signed_root = torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score)
            gate = signed_root.sigmoid().masked_fill(~token_mask[..., None], 0)
            # Broadcast the shared value across streams, scaled by each stream gate.
            return (hidden + gate[..., None] * value[..., None, :]).to(residual.dtype)
`
  },
  {
    id: 'csa2-compression', title: 'Compression: direct projection and completed groups', stages: ['csa2'],
    sources: ['dsv41-code'], symbols: ['Compressor', 'RMSNorm'], requires: [],
    summary: 'Keep the direct per-token path separate from learned per-channel pooling. The grouped path carries an incomplete tail explicitly between calls. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Readable reference before RoPE and quantization; FP32 projection, pooling and normalization arithmetic with input-dtype output. BF16 weights are supported, without emulating checkpoint rounding. Explicit tail tensors replace the publisher preallocated decode buffers; reset state for each new sequence. The grouped class requires a ratio greater than one. The caller supplies normalization epsilon explicitly; no checkpoint scalar default is implied.',
    code: `import math
from typing import Annotated
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CompressionTokens = Annotated[Tensor, "batch tokens hidden_dim"]
CompressionLatents = Annotated[Tensor, "batch complete_groups latent_dim"]
CompressionTail = Annotated[Tensor, "batch unfinished_tokens latent_dim; FP32"]
NormValues = Annotated[Tensor, "arbitrary leading axes, latent_dim"]


class CompressionNorm(nn.Module):
    """Learned RMS normalization with FP32 statistics and input-dtype output."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("normalization width must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("normalization epsilon must be finite and positive")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, values: NormValues) -> NormValues:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(values.dtype)


class TokenCompressor(nn.Module):
    """Ratio-one path: project and normalize each token without a pooling gate."""

    def __init__(self, dim: int, latent_dim: int, eps: float) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("input dimension must be a positive integer")
        self.norm = CompressionNorm(latent_dim, eps)
        self.project = nn.Linear(dim, latent_dim, bias=False)

    def forward(self, tokens: CompressionTokens) -> CompressionLatents:
        """Map [B,T,D] to [B,T,K]; return the incoming activation dtype."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.project.in_features:
            raise ValueError("tokens must have shape [batch, tokens, input dimension]")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            projected = F.linear(tokens.float(), self.project.weight.float())
            return self.norm(projected).to(tokens.dtype)


class GroupedCompressor(nn.Module):
    """Pool complete consecutive groups; caller owns the unfinished FP32 tail."""

    def __init__(self, dim: int, latent_dim: int, ratio: int, eps: float) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("input dimension must be a positive integer")
        if type(ratio) is not int or ratio <= 1:
            raise ValueError("group ratio must be an integer greater than one")
        self.norm = CompressionNorm(latent_dim, eps)
        self.ratio, self.latent_dim = ratio, latent_dim
        self.project = nn.Linear(dim, latent_dim, bias=False)
        self.gate = nn.Linear(dim, latent_dim, bias=False)

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
`
  },
  {
    id: 'csa2-selection', title: 'CSA2: candidate blocks and causal sparse indices', stages: ['csa2'],
    sources: ['dsv41-code'], symbols: ['Indexer.forward', 'select_candidate_blocks'], requires: [],
    summary: 'The candidate producer scores blocks by their best position and keeps the newest reachable block. Reindex layers score within that published mask; reuse layers consume the resulting indices. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Query and index keys are already projected and rotated. FP4 quantization, distributed reduction and cache publication are omitted. visible is the completed-group count floor((query_position + 1) / ratio), shaped [B,T,1]. The candidate budget must cover the requested sparse selection; unavailable picks are represented by -1.',
    code: `from typing import Annotated
import torch
from torch import Tensor
from torch.nn import functional as F


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
    scores: IndexScores, visible: VisibleGroups, top_blocks: int, block_size: int,
) -> CandidateMask:
    """Return [B,T,C] mask; visible [B,T,1] counts only completed causal groups."""
    if any(type(value) is not int or value <= 0 for value in (top_blocks, block_size)):
        raise ValueError("candidate budget and block size must be positive integers")
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
    scores: IndexScores, visible: VisibleGroups, candidates: CandidateMask, topk: int,
) -> SparseIndices:
    """Return position-ordered compressed indices, padding unavailable picks with -1."""
    if type(topk) is not int or topk <= 0:
        raise ValueError("sparse selection budget must be a positive integer")
    if scores.ndim != 3 or visible.shape != (*scores.shape[:2], 1):
        raise ValueError("scores and visible counts must be [B,T,C] and [B,T,1]")
    if candidates.shape != scores.shape or candidates.dtype != torch.bool:
        raise ValueError("candidate mask must be boolean and match scores")
    allowed = candidates & (torch.arange(scores.shape[-1], device=scores.device) < visible)
    masked = scores.float().masked_fill(~allowed, -torch.inf)
    indices = masked.topk(min(topk, scores.shape[-1]), dim=-1).indices.sort(-1).values
    # Gather validity at selected positions; masked top-k filler becomes sentinel -1.
    return indices.masked_fill(~allowed.gather(-1, indices), -1)
`
  },
  {
    id: 'csa2-attention', title: 'CSA2: local and compressed KV share a softmax', stages: ['csa2'],
    sources: ['dsv41-code', 'dsv41-kernel'], symbols: ['Attention.forward', 'sparse_attn_kernel'], requires: [],
    summary: 'Gather a causal local window and the published compressed indices, then normalize them together with the learned sink logit. The sink contributes denominator mass and no value. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Dense reference of the sparse attention core only: FP32 accumulation with query-dtype output. Queries and shared K/V latents are already projected, normalized and rotated. selected contains causal compressed indices or -1; Full/Reindex publish them and Reuse reads them unchanged. Local history uses absolute positions instead of a ring buffer. Inverse output RoPE, grouped output projections and quantization are outside this excerpt.',
    code: `from typing import Annotated
import torch
from torch import Tensor


AttentionQuery = Annotated[Tensor, "batch query_tokens heads head_dim"]
LocalLatents = Annotated[Tensor, "batch local_history head_dim"]
CompressedLatents = Annotated[Tensor, "batch compressed_positions head_dim"]
AttentionSelection = Annotated[Tensor, "batch query_tokens picks; causal indices or -1"]
QueryPositions = Annotated[Tensor, "query_tokens; absolute local-history positions"]
AttentionSink = Annotated[Tensor, "heads; finite denominator-only logits"]


def sparse_attention(
    query: AttentionQuery, local_kv: LocalLatents, compressed_kv: CompressedLatents,
    selected: AttentionSelection, positions: QueryPositions, window: int,
    sink: AttentionSink,
) -> AttentionQuery:
    """Attend over local and selected compressed latents using one sink-augmented softmax.

    query [B,T,H,D], caches [B,L,D]/[B,C,D], selected [B,T,K], positions [T].
    Positions address full local history; selected is already causally filtered or -1.
    A finite sink [H] has zero value. Output [B,T,H,D] retains the query dtype.
    """
    if type(window) is not int or window <= 0:
        raise ValueError("local window must be a positive integer")
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
`
  }
];
