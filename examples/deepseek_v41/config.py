"""The fixed V4.1 Flash path, with published dimensions and numerical settings.

Defaults follow the pinned DeepSeek V4.1 Flash config and ModelArgs. The Engram
excerpt uses the first insertion site's table. The CPU validator replaces only
sizes; it does not select a different architecture, routing rule or precision path.
No checkpoint weights are allocated by importing this file.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class V41Config:
    hidden_dim: int = 5120
    streams: int = 4
    num_layers: int = 40
    encoder_layers: int = 20
    vocab_size: int = 129280
    norm_epsilon: float = 1e-20
    mixing_epsilon: float = 1e-6
    sinkhorn_iterations: int = 20
    expert_hidden_dim: int = 2304
    num_experts: int = 384
    experts_per_token: int = 6
    router_temperature: float = 1.0
    route_scale: float = 1.5
    activation_limit: float = 10.0
    attention_heads: int = 64
    head_dim: int = 512
    window_size: int = 128
    encoder_compression_ratio: int = 2
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_blocks: int = 2048
    candidate_block_size: int = 8
    engram_score_floor: float = 1e-6
    engram_rows: int = 384006168
    engram_row_dim: int = 256
    engram_heads: int = 8
    engram_max_ngram: int = 4
    dspark_block_size: int = 5
    markov_rank: int = 256

    @property
    def engram_hash_columns(self) -> int:
        return (self.engram_max_ngram - 1) * self.engram_heads


V41_CONFIG = V41Config()
