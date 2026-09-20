"""DeepSeek-V4-Pro-DSpark draft-block math, in unquantized CPU PyTorch.

Adapted from deepseek-ai/DeepSeek-V4-Pro-DSpark inference/model.py at
7c09739fd136abfb70a49ec334157f65f45b52cd (MIT, Copyright 2023 DeepSeek).
Constructors initialize teaching weights, never checkpoint weights. The caller
supplies fused valid target history and rotary angles. This boundary excludes
ring-cache updates, quantization, target execution, final sampling and scheduling.
Pro uses current-read mHC and per-head query normalization; this is not V4.1.
"""
from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TokenStates = Annotated[Tensor, "batch draft_positions hidden_dim"]
ResidualStates = Annotated[Tensor, "batch draft_positions streams hidden_dim"]
HistoryStates = Annotated[Tensor, "batch valid_history_positions hidden_dim"]
RotaryAngles = Annotated[Tensor, "positions rotary_pairs"]
FeatureStates = Annotated[Tensor, "... feature_width"]
FeatureScale = Annotated[Tensor, "feature_width"]
RotatableStates = Annotated[Tensor, "batch positions [heads] head_dim"]
AttentionStates = Annotated[Tensor, "batch draft_positions heads head_dim"]
StreamWeights = Annotated[Tensor, "batch draft_positions streams"]
StreamMatrix = Annotated[Tensor, "batch draft_positions input_streams output_streams"]
RouteWeights = Annotated[Tensor, "batch draft_positions selected_experts"]
RouteIndices = Annotated[Tensor, "batch draft_positions selected_experts (integer expert ids)"]


@dataclass(frozen=True)
class DSparkProConfig:
    """Released Pro dimensions. Replace sizes for CPU fixtures, not mechanisms."""
    hidden_dim: int = 7168
    intermediate_dim: int = 3072
    attention_heads: int = 128
    head_dim: int = 512
    rotary_dim: int = 64
    query_rank: int = 1536
    output_groups: int = 16
    output_rank: int = 1024
    routed_experts: int = 384
    selected_experts: int = 6
    route_scale: float = 2.5
    swiglu_limit: float = 10.0
    streams: int = 4
    sinkhorn_iterations: int = 20
    window_size: int = 128
    norm_epsilon: float = 1e-6
    mixing_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        sizes = (self.hidden_dim, self.intermediate_dim, self.attention_heads,
                 self.head_dim, self.rotary_dim, self.query_rank, self.output_groups,
                 self.output_rank, self.routed_experts, self.selected_experts,
                 self.streams, self.sinkhorn_iterations, self.window_size)
        if any(n <= 0 for n in sizes):
            raise ValueError("architecture dimensions must be positive")
        if self.swiglu_limit <= 0:
            raise ValueError("this released path requires positive SwiGLU clipping")
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError("rotary width must be even and within head width")
        if self.attention_heads % self.output_groups:
            raise ValueError("output groups must divide attention heads")
        if self.selected_experts > self.routed_experts:
            raise ValueError("selected experts exceed routed experts")


def rms_norm(x: FeatureStates, weight: FeatureScale, epsilon: float) -> FeatureStates:
    """Learned RMSNorm; variance arithmetic is float32."""
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(x.dtype)


def rotate_partial(x: RotatableStates, angles: RotaryAngles, inverse: bool = False) -> RotatableStates:
    """Rotate interleaved pairs at the end of [B,S,(H),D]; leave NoPE intact."""
    width = 2 * angles.shape[-1]
    if angles.ndim != 2 or angles.shape[0] != x.shape[1] or width > x.shape[-1]:
        raise ValueError("angles must match sequence positions and fit the head")
    angle_shape = (1, x.shape[1]) + (1,) * (x.ndim - 3) + (width // 2,)
    angle = angles.reshape(angle_shape) * (-1 if inverse else 1)
    pairs = x[..., -width:].float().unflatten(-1, (width // 2, 2))
    real, imag = pairs.unbind(-1)
    rotated = torch.stack((real * angle.cos() - imag * angle.sin(),
                           real * angle.sin() + imag * angle.cos()), dim=-1).flatten(-2)
    return torch.cat((x[..., :-width], rotated.to(x.dtype)), dim=-1)


def sink_attention(
    queries: Annotated[Tensor, "batch draft_positions heads head_dim"],
    key_values: Annotated[Tensor, "batch history_plus_draft_positions head_dim"],
    sink: Annotated[Tensor, "heads"],
) -> AttentionStates:
    """Every query reads every supplied key; sink adds mass but no value."""
    scores = torch.einsum('bshd,bnd->bshn', queries.float(), key_values.float())
    scores = scores * queries.shape[-1] ** -0.5
    sink_logits = sink.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sink_logits), dim=-1).softmax(-1)
    return torch.einsum('bshn,bnd->bshd', probabilities[..., :-1], key_values.float()).to(queries.dtype)


class ProDraftAttention(nn.Module):
    """Shared main/draft K/V projection, all-slot attention, grouped output."""
    def __init__(self, config: DSparkProConfig) -> None:
        super().__init__()
        self.config = config
        d, h, w = config.hidden_dim, config.attention_heads, config.head_dim
        self.query_down = nn.Linear(d, config.query_rank, bias=False)
        self.query_norm = nn.Parameter(torch.ones(config.query_rank))
        self.query_up = nn.Linear(config.query_rank, h * w, bias=False)
        self.key_value = nn.Linear(d, w, bias=False)
        self.kv_norm = nn.Parameter(torch.ones(w))
        self.sink = nn.Parameter(torch.zeros(h))
        group_width = h * w // config.output_groups
        self.output_down = nn.Parameter(torch.empty(config.output_groups, config.output_rank, group_width))
        nn.init.normal_(self.output_down, std=group_width ** -0.5)
        self.output_up = nn.Linear(config.output_groups * config.output_rank, d, bias=False)

    def project_queries(self, x: TokenStates, angles: RotaryAngles) -> AttentionStates:
        c = self.config
        latent = rms_norm(self.query_down(x), self.query_norm, c.norm_epsilon)
        q = self.query_up(latent).unflatten(-1, (c.attention_heads, c.head_dim))
        # Pro normalizes each expanded query head; V4.1 omits this step.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + c.norm_epsilon)
        return rotate_partial(q, angles)

    def forward(self, x: TokenStates, history: HistoryStates,
                history_angles: RotaryAngles, draft_angles: RotaryAngles) -> TokenStates:
        c = self.config
        if (x.ndim != 3 or history.ndim != 3 or x.shape[0] != history.shape[0]
                or x.shape[-1] != c.hidden_dim or history.shape[-1] != c.hidden_dim
                or history.shape[1] > c.window_size):
            raise ValueError("supply matching draft features and valid window history")
        if history_angles.shape[-1] * 2 != c.rotary_dim or draft_angles.shape[-1] * 2 != c.rotary_dim:
            raise ValueError("angles must match configured rotary width")
        # Each block applies its own SAME projection to context and draft features.
        history_kv = rotate_partial(rms_norm(self.key_value(history), self.kv_norm, c.norm_epsilon), history_angles)
        draft_kv = rotate_partial(rms_norm(self.key_value(x), self.kv_norm, c.norm_epsilon), draft_angles)
        query = self.project_queries(x, draft_angles)
        values = torch.cat((history_kv, draft_kv), dim=1)
        attended = sink_attention(query, values, self.sink)
        attended = rotate_partial(attended, draft_angles, inverse=True)
        grouped = attended.reshape(*x.shape[:2], c.output_groups, -1)
        reduced = torch.einsum('bsgd,grd->bsgr', grouped, self.output_down)
        return self.output_up(reduced.flatten(-2))


class ProHyperConnection(nn.Module):
    """Predict and consume pre-weights on the current residual, not the next one."""
    def __init__(self, config: DSparkProConfig) -> None:
        super().__init__()
        self.config = config
        r = config.streams
        self.projection = nn.Linear(r * config.hidden_dim, r * (r + 2), bias=False)
        self.offset = nn.Parameter(torch.zeros(r * (r + 2)))
        self.scale = nn.Parameter(torch.ones(3))

    def read(self, residual: ResidualStates) -> tuple[TokenStates, StreamWeights, StreamMatrix]:
        c, r = self.config, self.config.streams
        if residual.ndim != 4 or residual.shape[-2:] != (r, c.hidden_dim):
            raise ValueError("residual must match configured streams and width")
        flat = residual.flatten(-2).float()
        coefficients = self.projection(flat) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + c.norm_epsilon)
        pre = torch.sigmoid(coefficients[..., :r] * self.scale[0] + self.offset[:r]) + c.mixing_epsilon
        post = 2 * torch.sigmoid(coefficients[..., r:2*r] * self.scale[1] + self.offset[r:2*r])
        logits = coefficients[..., 2*r:] * self.scale[2] + self.offset[2*r:]
        matrix = logits.unflatten(-1, (r, r)).softmax(-1) + c.mixing_epsilon
        matrix = matrix / (matrix.sum(-2, keepdim=True) + c.mixing_epsilon)
        for _ in range(c.sinkhorn_iterations - 1):
            matrix = matrix / (matrix.sum(-1, keepdim=True) + c.mixing_epsilon)
            matrix = matrix / (matrix.sum(-2, keepdim=True) + c.mixing_epsilon)
        collapsed = (residual.float() * pre.unsqueeze(-1)).sum(-2)
        return collapsed.to(residual.dtype), post, matrix

    def write(self, output: TokenStates, residual: ResidualStates,
              post: StreamWeights, matrix: StreamMatrix) -> ResidualStates:
        # Matrix axes are INPUT stream i, OUTPUT stream j.
        carried = torch.einsum('bsij,bsid->bsjd', matrix.float(), residual.float())
        expanded = post.unsqueeze(-1) * output.float().unsqueeze(-2)
        return (carried + expanded).to(output.dtype)


class ProExpert(nn.Module):
    def __init__(self, config: DSparkProConfig) -> None:
        super().__init__()
        self.limit = config.swiglu_limit
        self.gate = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: TokenStates) -> TokenStates:
        gate, up = self.gate(x), self.up(x)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.down(F.silu(gate) * up)


class ProMoE(nn.Module):
    """Score-selected draft experts plus an independent shared expert."""
    def __init__(self, config: DSparkProConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_dim, config.routed_experts, bias=False)
        self.selection_bias = nn.Parameter(torch.zeros(config.routed_experts))
        self.experts = nn.ModuleList(ProExpert(config) for _ in range(config.routed_experts))
        self.shared = ProExpert(config)

    def route(self, x: TokenStates) -> tuple[RouteWeights, RouteIndices]:
        scores = F.softplus(self.router(x).float()).sqrt()
        indices = (scores + self.selection_bias).topk(self.config.selected_experts, dim=-1).indices
        selected = scores.gather(-1, indices)  # selection bias never becomes a mixture weight
        weights = selected / selected.sum(-1, keepdim=True) * self.config.route_scale
        return weights, indices

    def forward(self, x: TokenStates) -> TokenStates:
        weights, indices = self.route(x)
        result = self.shared(x)  # no shared-expert router logit
        # Dense CPU evaluation makes the arithmetic inspectable; not an efficient dispatcher.
        for expert_id, expert in enumerate(self.experts):
            weight = (weights * (indices == expert_id)).sum(-1, keepdim=True)
            result = result + weight * expert(x)
        return result


class ProDraftBlock(nn.Module):
    """One released-Pro draft layer: current mHC → attention → current mHC → MoE."""
    def __init__(self, config: DSparkProConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_mix = ProHyperConnection(config)
        self.attention_norm = nn.Parameter(torch.ones(config.hidden_dim))
        self.attention = ProDraftAttention(config)
        self.feed_forward_mix = ProHyperConnection(config)
        self.feed_forward_norm = nn.Parameter(torch.ones(config.hidden_dim))
        self.moe = ProMoE(config)

    def forward(self, residual: ResidualStates, history: HistoryStates,
                history_angles: RotaryAngles, draft_angles: RotaryAngles) -> ResidualStates:
        x, post, matrix = self.attention_mix.read(residual)
        x = rms_norm(x, self.attention_norm, self.config.norm_epsilon)
        x = self.attention(x, history, history_angles, draft_angles)
        residual = self.attention_mix.write(x, residual, post, matrix)
        x, post, matrix = self.feed_forward_mix.read(residual)
        x = rms_norm(x, self.feed_forward_norm, self.config.norm_epsilon)
        return self.feed_forward_mix.write(self.moe(x), residual, post, matrix)
