# Adapted from DeepSeek V4.1 inference/engram.py (MIT), pinned dba1be0a40aa45a94ad051997016db3960a90277.
"""CPU full-sequence hash trace, with supplied compressed token map and small buckets.

The caller supplies tokenizer normalization's ID map. This excerpt implements the
publisher odd multipliers, distinct prime ranges, dead-token boundaries and XOR
hashing, but not tokenizer decoding or the incremental cache interface.
"""
from typing import Annotated
import math
import numpy as np
import torch


def ngram_hash_trace(
    input_ids: Annotated[torch.Tensor, "batch tokens"],
    token_mask: Annotated[torch.Tensor, "batch tokens; boolean"],
    token_map: Annotated[torch.Tensor, "raw_vocabulary; normalized IDs"],
    layer_ids: tuple[int, ...], max_ngram_size: int,
    heads: int, bucket_size: int, pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    if input_ids.ndim != 2 or token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool:
        raise ValueError('hash inputs require [batch,tokens] IDs and boolean mask')
    if min(heads, bucket_size) < 1 or max_ngram_size < 2 or not layer_ids:
        raise ValueError('hash layout requires layers, positive heads/buckets and n-grams')
    vocab_size = int(token_map.max()) + 1
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    multipliers = torch.stack([torch.tensor(np.random.default_rng(10007 * layer).integers(
        0, bound, size=max_ngram_size, dtype=np.int64) * 2 + 1) for layer in layer_ids])
    seen, primes = set(), []
    for _ in layer_ids:
        sizes = []
        for _ in range(max_ngram_size - 1):
            current = bucket_size - 1
            for _ in range(heads):
                current += 1
                while current in seen or current < 2 or any(current % d == 0 for d in range(2, math.isqrt(current) + 1)):
                    current += 1
                seen.add(current)
                sizes.append(current)
        primes.append(sizes)
    primes = torch.tensor(primes)
    offsets = torch.cat([torch.zeros(len(layer_ids), 1, dtype=torch.long), primes.cumsum(-1)[:, :-1]], -1)
    normalized = token_map[input_ids]
    compressed = normalized.masked_fill(~token_mask, -1)
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    blocked, history, masks = torch.zeros_like(token_mask), [], []
    for shift in range(max_ngram_size):
        source = compressed.gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == -1)
        history.append(torch.where(blocked, pad_id, source))
        masks.append(blocked)
    history = torch.stack(history, -1)
    products = history.unsqueeze(2) * multipliers
    rolling, hashes = products[..., 0], []
    for lookback in range(1, max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., lookback])
        hashes.append(rolling.unsqueeze(-1) % primes[:, (lookback - 1) * heads:lookback * heads])
    return dict(normalized_ids=normalized, masked_ids=compressed, history=history,
                boundary_mask=torch.stack(masks, -1), hash_multipliers=multipliers,
                hash_primes=primes, hash_offsets=offsets,
                hash_ids=torch.cat(hashes, -1) + offsets)
