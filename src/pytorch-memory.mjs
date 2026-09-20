// Authored educational adaptations of DeepSeek V4.1 Flash publisher code (MIT).
export const MEMORY_EXAMPLES = [
  {
    id: 'engram-write', title: 'Engram: lookup and context-gated write', stages: ['engram'],
    sources: ['dsv41-code', 'dsv41-engram'], symbols: ['ParallelEngramEmbedding', 'Engram.forward', 'NgramHashState'], requires: [],
    summary: 'Look up the hash rows, project a key for each residual stream and a shared value, then apply the signed-square-root gate. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Hash IDs are global table row IDs already produced by the publisher hash layout; tokenization, n-gram hashing, table sharding and FP8 row dequantization are outside this excerpt. Weights and arithmetic are FP32. The token mask and normalization epsilon are required caller inputs; no checkpoint scalar default is implied.',
    code: `import torch
from torch import nn


class ReadableEngram(nn.Module):
    def __init__(self, rows, row_dim, hash_columns, streams, dim, eps):
        super().__init__()
        self.streams, self.dim, self.eps = streams, dim, eps
        self.table = nn.Embedding(rows, row_dim)
        self.project = nn.Linear(hash_columns * row_dim, (streams + 1) * dim, bias=False)
        self.query_scale = nn.Parameter(torch.ones(streams, dim))
        self.key_scale = nn.Parameter(torch.ones(streams, dim))

    def forward(self, x, hash_ids, token_mask):
        # x: [B,T,S,D]; hash_ids: [B,T,C]; token_mask: [B,T]
        rows = self.table(hash_ids).flatten(-2)
        projected = self.project(rows)
        keys, value = projected.split([self.streams * self.dim, self.dim], -1)
        keys = keys.unflatten(-1, (self.streams, self.dim))
        # Normalize each stream over D, never across the stream axis S.
        inverse_rms = torch.rsqrt(x.square().mean(-1) + self.eps)
        inverse_rms = inverse_rms * torch.rsqrt(keys.square().mean(-1) + self.eps)
        score = (x * self.query_scale * self.key_scale * keys).sum(-1)
        score = score * inverse_rms * self.dim ** -0.5
        signed_root = torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score)
        gate = signed_root.sigmoid().masked_fill(~token_mask[..., None], 0)
        return x + gate[..., None] * value[..., None, :]
`
  },
  {
    id: 'csa2-compression', title: 'Compression: direct projection and completed groups', stages: ['csa2'],
    sources: ['dsv41-code'], symbols: ['Compressor', 'RMSNorm'], requires: [],
    summary: 'Keep the direct per-token path separate from learned per-channel pooling. The grouped path carries an incomplete tail explicitly between calls. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'FP32 teaching implementation, before RoPE and quantization. Explicit tail tensors replace the publisher preallocated decode buffers; reset state for each new sequence. The grouped class requires a ratio greater than one. The caller supplies normalization epsilon explicitly; no checkpoint scalar default is implied.',
    code: `import torch
from torch import nn


class ReadableCompressionNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class ReadableTokenCompressor(nn.Module):
    # Decoder ratio-one path: projection and norm; no pooling gate.
    def __init__(self, dim, latent_dim, eps):
        super().__init__()
        self.project = nn.Linear(dim, latent_dim, bias=False)
        self.norm = ReadableCompressionNorm(latent_dim, eps)

    def forward(self, x):
        return self.norm(self.project(x))


class ReadableGroupedCompressor(nn.Module):
    def __init__(self, dim, latent_dim, ratio, eps):
        super().__init__()
        assert ratio > 1
        self.ratio, self.latent_dim = ratio, latent_dim
        self.project = nn.Linear(dim, latent_dim, bias=False)
        self.gate = nn.Linear(dim, latent_dim, bias=False)
        self.norm = ReadableCompressionNorm(latent_dim, eps)

    def empty_state(self, x):
        empty = x.new_empty(x.shape[0], 0, self.latent_dim)
        return empty, empty.clone()

    def forward(self, x, state):
        # x: [B,T,D]; state: unpooled values/scores [B,tail,K]
        values = torch.cat([state[0], self.project(x)], dim=1)
        scores = torch.cat([state[1], self.gate(x)], dim=1)
        complete = values.shape[1] // self.ratio * self.ratio
        grouped_values = values[:, :complete].unflatten(1, (-1, self.ratio))
        grouped_scores = scores[:, :complete].unflatten(1, (-1, self.ratio))
        pooled = (grouped_values * grouped_scores.softmax(dim=2)).sum(dim=2)
        # Zero completed groups means an empty output, not a fabricated latent.
        return self.norm(pooled), (values[:, complete:], scores[:, complete:])
`
  },
  {
    id: 'csa2-selection', title: 'CSA2: candidate blocks and causal sparse indices', stages: ['csa2'],
    sources: ['dsv41-code'], symbols: ['Indexer.forward', 'select_candidate_blocks'], requires: [],
    summary: 'The candidate producer scores blocks by their best position and keeps the newest reachable block. Reindex layers score within that published mask; reuse layers consume the resulting indices. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Query and index keys are already projected and rotated. FP4 quantization, distributed reduction and cache publication are omitted. visible is the completed-group count floor((query_position + 1) / ratio), shaped [B,T,1]. The candidate budget must cover the requested sparse selection; unavailable picks are represented by -1.',
    code: `import torch
from torch.nn import functional as F


def readable_index_scores(query, keys, head_weights):
    # query [B,T,H,D], keys [B,C,D], head_weights [B,T,H].
    # Caller supplies the learned weights before publisher scale factors.
    heads, width = query.shape[-2:]
    dots = torch.einsum('bthd,bcd->bthc', query, keys).relu()
    weights = head_weights * (width ** -0.5 * heads ** -0.5)
    return (dots * weights[..., None]).sum(dim=2)


def readable_candidate_mask(scores, visible, top_blocks, block_size):
    # Publisher candidate source: every incomplete future group stays hidden.
    width = scores.shape[-1]
    causal = torch.arange(width, device=scores.device) < visible
    masked = scores.masked_fill(~causal, -torch.inf)
    padded = F.pad(masked, (0, -width % block_size), value=-torch.inf)
    blocks = padded.unflatten(-1, (-1, block_size)).amax(-1)
    newest = (visible - 1) // block_size
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    blocks = blocks.masked_fill(block_ids == newest, torch.inf)
    values, ids = blocks.topk(min(top_blocks, blocks.shape[-1]), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool)
    keep.scatter_(-1, ids, values > -torch.inf)
    return keep.repeat_interleave(block_size, -1)[..., :width] & causal


def readable_sparse_indices(scores, visible, candidates, topk):
    allowed = candidates & (torch.arange(scores.shape[-1], device=scores.device) < visible)
    masked = scores.masked_fill(~allowed, -torch.inf)
    ids = masked.topk(min(topk, scores.shape[-1]), dim=-1).indices.sort(-1).values
    # Explicit validity also handles teaching examples with undersized candidates.
    return ids.masked_fill(~allowed.gather(-1, ids), -1)
`
  },
  {
    id: 'csa2-attention', title: 'CSA2: local and compressed KV share a softmax', stages: ['csa2'],
    sources: ['dsv41-code', 'dsv41-kernel'], symbols: ['Attention.forward', 'sparse_attn_kernel'], requires: [],
    summary: 'Gather a causal local window and the published compressed indices, then normalize them together with the learned sink logit. The sink contributes denominator mass and no value. Authored adaptation of DeepSeek publisher code (MIT).',
    limitations: 'Dense FP32 reference of the sparse attention core only. Queries and shared K/V latents are already projected, normalized and rotated. selected contains causal compressed indices or -1; Full/Reindex publish them and Reuse reads them unchanged. Local history uses absolute positions instead of a ring buffer. Inverse output RoPE, grouped output projections and quantization are outside this excerpt.',
    code: `import torch


def readable_sparse_union(query, local_kv, compressed_kv, selected, positions, window, sink):
    # query [B,T,H,D], local_kv [B,L,D], compressed_kv [B,C,D]
    # selected [B,T,K] uses compressed-cache indices; positions [T] is absolute.
    batch, steps, heads, width = query.shape
    local_ids = positions[:, None] - torch.arange(window - 1, -1, -1, device=query.device)
    local_ids = local_ids.expand(batch, steps, window)
    valid_local = (local_ids >= 0) & (local_ids < local_kv.shape[1])
    compressed_ids = selected + local_kv.shape[1]
    ids = torch.cat([local_ids, compressed_ids], dim=-1)
    valid_compressed = (selected >= 0) & (selected < compressed_kv.shape[1])
    valid = torch.cat([valid_local, valid_compressed], dim=-1)
    cache = torch.cat([local_kv, compressed_kv], dim=1)
    batch_ids = torch.arange(batch, device=query.device)[:, None, None]
    gathered = cache[batch_ids, ids.masked_fill(~valid, 0)]
    logits = torch.einsum('bthd,btkd->bthk', query, gathered) * width ** -0.5
    logits = logits.masked_fill(~valid[:, :, None, :], -torch.inf)
    sink_logits = sink[None, None, :, None].expand(batch, steps, heads, 1)
    probability = torch.cat([logits, sink_logits], -1).softmax(-1)[..., :-1]
    return torch.einsum('bthk,btkd->bthd', probability, gathered)
`
  }
];
