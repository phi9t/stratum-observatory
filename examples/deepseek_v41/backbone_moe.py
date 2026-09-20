from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V41Config


ExpertTokenStates = Annotated[Tensor, "tokens hidden_dim"]
ExpertRouteWeights = Annotated[Tensor, "tokens broadcast_hidden=1"]
MixtureHiddenStates = Annotated[Tensor, "... hidden_dim"]


class SwiGLUExpert(nn.Module):
    """Clamped SwiGLU [tokens,width] -> [tokens,width], weighted before down projection."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(config.hidden_dim, config.expert_hidden_dim, bias=False)
        self.up_projection = nn.Linear(config.hidden_dim, config.expert_hidden_dim, bias=False)
        self.down_projection = nn.Linear(config.expert_hidden_dim, config.hidden_dim, bias=False)
        self.activation_limit = config.activation_limit

    def forward(
        self, inputs: ExpertTokenStates, weights: ExpertRouteWeights | float,
    ) -> ExpertTokenStates:
        gate = self.gate_projection(inputs).float().clamp(max=self.activation_limit)
        up = self.up_projection(inputs).float().clamp(-self.activation_limit, self.activation_limit)
        # Route weights broadcast across intermediate channels before down projection.
        activated = F.silu(gate) * up * weights
        return self.down_projection(activated.to(inputs.dtype))


class MixtureOfExperts(nn.Module):
    """Text-only routed mixture plus shared expert; preserves [...,width] shape and dtype."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.selection_bias = nn.Parameter(torch.zeros(config.num_experts))
        self.experts = nn.ModuleList([
            SwiGLUExpert(config) for _ in range(config.num_experts)
        ])
        self.shared_expert = SwiGLUExpert(config)
        self.topk = config.experts_per_token
        self.temperature, self.route_scale = config.router_temperature, config.route_scale

    def forward(self, inputs: MixtureHiddenStates) -> MixtureHiddenStates:
        if not inputs.is_floating_point() or inputs.shape[-1] != self.router.in_features:
            raise ValueError("mixture inputs must be floating and match the configured width")
        # Merge batch/sequence axes for dispatch; restore them after aggregation.
        tokens = inputs.reshape(-1, inputs.shape[-1])
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.router.weight.float()) / self.temperature
            scores = F.softplus(logits).sqrt()
            selected = (scores + self.selection_bias.float()).topk(self.topk, dim=-1).indices
            # Gather only selected expert scores; normalize across the selected experts.
            weights = scores.gather(-1, selected)  # bias selects but NEVER weights
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * self.route_scale
        routed = torch.zeros_like(tokens, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_indices, selection_slots = torch.where(selected == expert_id)
            contribution = expert(
                tokens[token_indices], weights[token_indices, selection_slots, None]
            ).float()
            # Add each selected expert contribution back into its original token row.
            routed = routed.index_add(0, token_indices, contribution)
        shared = self.shared_expert(tokens, 1.0).float()
        return (routed + shared).to(inputs.dtype).reshape(inputs.shape)
