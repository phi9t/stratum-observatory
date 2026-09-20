/** Source-grounded PyTorch adaptations of DeepSeek-AI's MIT-licensed reference. */
export const BACKBONE_EXAMPLES=[{
 id:'backbone-mhc',title:'Single-pass mHC and the flat encoder/decoder stack',stages:['encoder','decoder'],
 sources:['dsv41-code','dsv41-kernel'],
 symbols:['RMSNorm','Block.hc_mixes','Block.hc_pre','Block.hc_post','Block.forward','make_identity_pre_mix','Transformer.forward','hc_split_sinkhorn','hc_split_sinkhorn_kernel'],
 summary:'Track residual copies and previous-sublayer premix explicitly. The encoder boundary is a point in a flat stack, with no inserted cross-attention.',
 limitations:'Unfused PyTorch reference adapted from DeepSeek-AI (MIT). CPU float32, bfloat16 and backward behavior are tested; GPU execution and production throughput are not certified. Numerical settings are explicit inputs. No checkpoint loader, quantization, distribution, vision, Engram insertion or vocabulary head. Attention and feed-forward modules are injected: this implements residual orchestration, not a complete V4.1 model or bounded replay. Global-cache ownership belongs to the supplied attention implementation.',
 requires:[],code:`import math
from collections.abc import Sequence
from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Shapes use named axes; streams are residual copies, not attention heads.
HiddenStates = Annotated[Tensor, "... hidden_dim"]
TokenStates = Annotated[Tensor, "batch tokens hidden_dim"]
ResidualStates = Annotated[Tensor, "batch tokens streams hidden_dim"]
StreamWeights = Annotated[Tensor, "batch tokens streams"]
ResidualMixingMatrix = Annotated[Tensor, "batch tokens input_streams output_streams"]
MixingCoefficients = Annotated[Tensor, "batch tokens mixing_coefficients"]
MixingScales = Annotated[Tensor, "coefficient_groups=3"]
MixingOffsets = Annotated[Tensor, "mixing_coefficients"]


def validate_residual_dimensions(**dimensions: int) -> None:
    for name, value in dimensions.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def validate_mixing_scalars(**scalars: float) -> None:
    for name, value in scalars.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


class RMSNorm(nn.Module):
    """Normalize the last dimension, accumulating variance and scale in float32."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        validate_residual_dimensions(width=width)
        validate_mixing_scalars(epsilon=epsilon)
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, inputs: HiddenStates) -> HiddenStates:
        if not inputs.is_floating_point() or inputs.shape[-1] != self.weight.numel():
            raise ValueError("RMSNorm requires floating inputs with the configured width")
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


def split_mixing_coefficients(
    coefficients: MixingCoefficients, scales: MixingScales, offsets: MixingOffsets,
    streams: int, epsilon: float, iterations: int = 20,
) -> tuple[StreamWeights, StreamWeights, ResidualMixingMatrix]:
    """Return input/output weights [...,R] and residual matrix [...,R,R]."""
    validate_residual_dimensions(streams=streams, iterations=iterations)
    validate_mixing_scalars(epsilon=epsilon)
    coefficient_count = streams * (streams + 2)
    if (coefficients.shape[-1] != coefficient_count
            or offsets.shape != (coefficient_count,) or scales.shape != (3,)):
        raise ValueError("mixing coefficient dimensions disagree")
    coefficients, scales, offsets = coefficients.float(), scales.float(), offsets.float()
    input_weights = torch.sigmoid(
        coefficients[..., :streams] * scales[0] + offsets[:streams]
    ) + epsilon
    output_weights = 2 * torch.sigmoid(
        coefficients[..., streams:2 * streams] * scales[1] + offsets[streams:2 * streams]
    )
    matrix_logits = coefficients[..., 2 * streams:] * scales[2] + offsets[2 * streams:]
    # Split coefficient axis into input/output streams; normalize rows then columns.
    residual_matrix = matrix_logits.unflatten(-1, (streams, streams)).softmax(-1) + epsilon
    residual_matrix = residual_matrix / (residual_matrix.sum(-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        residual_matrix = residual_matrix / (residual_matrix.sum(-1, keepdim=True) + epsilon)
        residual_matrix = residual_matrix / (residual_matrix.sum(-2, keepdim=True) + epsilon)
    return input_weights, output_weights, residual_matrix


def collapse_residual_streams(
    residual: ResidualStates, input_weights: StreamWeights,
) -> TokenStates:
    """[B,T,R,D] weighted over residual copies becomes [B,T,D]."""
    if residual.ndim != 4 or input_weights.shape != residual.shape[:-1]:
        raise ValueError("input mixing weights must match [batch, tokens, streams]")
    # Broadcast each stream weight across hidden channels, then sum only streams.
    collapsed = (input_weights.float().unsqueeze(-1) * residual.float()).sum(-2)
    return collapsed.to(residual.dtype)


def combine_residual_streams(
    output: TokenStates, residual: ResidualStates,
    output_weights: StreamWeights, residual_matrix: ResidualMixingMatrix,
) -> ResidualStates:
    """Expand [B,T,D]; matrix axes are INPUT copy then OUTPUT copy."""
    if residual.ndim != 4:
        raise ValueError("residual must have shape [batch, tokens, streams, width]")
    expected_output = residual.shape[:2] + residual.shape[-1:]
    expected_matrix = residual.shape[:-1] + residual.shape[-2:-1]
    if (output.shape != expected_output or output_weights.shape != residual.shape[:-1]
            or residual_matrix.shape != expected_matrix):
        raise ValueError("sublayer output and residual mixing dimensions disagree")
    with torch.autocast(device_type=residual.device.type, enabled=False):
        # Sum input streams i into output streams j; keep batch/token/hidden axes.
        carried = torch.einsum(
            "btij,btid->btjd", residual_matrix.float(), residual.float()
        )
        # Each residual copy receives the same sublayer result with its own weight.
        expanded = output_weights.float().unsqueeze(-1) * output.float().unsqueeze(-2)
    return (expanded + carried).to(output.dtype)


class HyperConnectionMixer(nn.Module):
    """Predict next-input weights and current output/residual weights from [B,T,R,D]."""

    def __init__(
        self, width: int, streams: int, norm_epsilon: float, mixing_epsilon: float,
    ) -> None:
        super().__init__()
        validate_residual_dimensions(width=width, streams=streams)
        validate_mixing_scalars(norm_epsilon=norm_epsilon, mixing_epsilon=mixing_epsilon)
        coefficient_count = streams * (streams + 2)
        self.projection = nn.Linear(width * streams, coefficient_count, bias=False)
        self.scales = nn.Parameter(torch.ones(3))
        self.offsets = nn.Parameter(torch.zeros(coefficient_count))
        self.width, self.streams = width, streams
        self.norm_epsilon, self.mixing_epsilon = norm_epsilon, mixing_epsilon

    def forward(
        self, residual: ResidualStates,
    ) -> tuple[StreamWeights, StreamWeights, ResidualMixingMatrix]:
        if residual.ndim != 4 or residual.shape[-2:] != (self.streams, self.width):
            raise ValueError("residual must match [batch, tokens, streams, width]")
        with torch.autocast(device_type=residual.device.type, enabled=False):
            # Join streams and hidden channels while preserving each batch/token.
            flattened_streams = residual.flatten(-2).float()
            # One RMS statistic across ALL copies and width, not one per copy.
            inverse_rms = torch.rsqrt(
                flattened_streams.square().mean(-1, keepdim=True) + self.norm_epsilon
            )
            coefficients = F.linear(flattened_streams, self.projection.weight.float())
            coefficients = coefficients * inverse_rms
            return split_mixing_coefficients(
                coefficients, self.scales, self.offsets, self.streams, self.mixing_epsilon
            )


class TransformerBlock(nn.Module):
    """Apply attention/FFN with the previous sublayer's input mixing weights."""

    def __init__(
        self, width: int, streams: int, attention: nn.Module, feed_forward: nn.Module,
        norm_epsilon: float, mixing_epsilon: float,
    ) -> None:
        super().__init__()
        if not isinstance(attention, nn.Module) or not isinstance(feed_forward, nn.Module):
            raise ValueError("attention and feed_forward must be torch modules")
        self.attention, self.feed_forward = attention, feed_forward
        self.attention_norm = RMSNorm(width, norm_epsilon)
        self.feed_forward_norm = RMSNorm(width, norm_epsilon)
        self.attention_mixer = HyperConnectionMixer(width, streams, norm_epsilon, mixing_epsilon)
        self.feed_forward_mixer = HyperConnectionMixer(width, streams, norm_epsilon, mixing_epsilon)

    def forward(
        self, residual: ResidualStates, previous_input_weights: StreamWeights,
    ) -> tuple[ResidualStates, StreamWeights]:
        next_input, output_weights, residual_matrix = self.attention_mixer(residual)
        # Attention consumes the PREVIOUS block's FFN weights (initially one-hot).
        attention_input = self.attention_norm(
            collapse_residual_streams(residual, previous_input_weights)
        )
        residual = combine_residual_streams(
            self.attention(attention_input), residual, output_weights, residual_matrix
        )
        following_input, output_weights, residual_matrix = self.feed_forward_mixer(residual)
        # The attention's predicted input weights are used only now, by the FFN.
        feed_forward_input = self.feed_forward_norm(collapse_residual_streams(residual, next_input))
        residual = combine_residual_streams(
            self.feed_forward(feed_forward_input), residual, output_weights, residual_matrix
        )
        return residual, following_input


class CausalEncoderDecoder(nn.Module):
    """Flat block orchestration: [B,T,D] -> final [B,T,D], boundary [B,T,R,D]."""

    def __init__(
        self, blocks: Sequence[nn.Module], encoder_layers: int,
        width: int, streams: int, norm_epsilon: float,
    ) -> None:
        super().__init__()
        validate_residual_dimensions(width=width, streams=streams, encoder_layers=encoder_layers)
        if (encoder_layers >= len(blocks)
                or not all(isinstance(block, nn.Module) for block in blocks)):
            raise ValueError("require nonempty encoder and decoder containing torch modules")
        self.blocks = nn.ModuleList(blocks)
        self.encoder_layers, self.streams, self.width = encoder_layers, streams, width
        self.output_norm = RMSNorm(width, norm_epsilon)

    def forward(self, embedded: TokenStates) -> tuple[TokenStates, ResidualStates]:
        # Caller supplies embeddings and CAUSAL attention modules with their own cache state.
        if (embedded.ndim != 3 or embedded.shape[-1] != self.width
                or not embedded.is_floating_point()):
            raise ValueError("embeddings must be floating [batch, tokens, width]")
        # Duplicate embeddings along the new residual-stream axis.
        residual = embedded.unsqueeze(-2).repeat(1, 1, self.streams, 1)
        input_weights = embedded.new_zeros(
            *embedded.shape[:2], self.streams, dtype=torch.float32
        )
        input_weights[..., 0] = 1  # one-hot initialization, not uniform averaging
        for index, block in enumerate(self.blocks):
            residual, input_weights = block(residual, input_weights)
            if index + 1 == self.encoder_layers:
                boundary = residual.clone()  # observation; no extra cross-attention
        # The last FFN's predicted weights collapse the residual before normalization.
        return self.output_norm(collapse_residual_streams(residual, input_weights)), boundary
`
},{
 id:'backbone-moe',title:'Routed experts plus a separate shared expert',stages:['encoder','decoder'],
 sources:['dsv41-code'],symbols:['Gate.forward','Expert.forward','MoE.forward'],
 summary:'Correction bias chooses routed experts; unbiased scores weight their outputs. The shared expert runs independently of routed top-k.',
 limitations:'Unfused PyTorch reference adapted from DeepSeek-AI (MIT). Text-only sqrt-softplus routing, normalized routed top-k greater than one, and one shared expert. Dimensions, temperature, routing scale and positive SwiGLU clamp are explicit inputs. CPU float32, bfloat16 and backward behavior are tested. No image routing bias, quantization, expert parallelism, checkpoint loading, GPU certification or throughput claim.',
 requires:[],code:`import math
from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ExpertTokenStates = Annotated[Tensor, "tokens hidden_dim"]
ExpertRouteWeights = Annotated[Tensor, "tokens broadcast_hidden=1"]
MixtureHiddenStates = Annotated[Tensor, "... hidden_dim"]


class SwiGLUExpert(nn.Module):
    """Clamped SwiGLU [tokens,width] -> [tokens,width], weighted before down projection."""

    def __init__(self, width: int, hidden_width: int, activation_limit: float) -> None:
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (width, hidden_width)):
            raise ValueError("expert dimensions must be positive integers")
        if not math.isfinite(activation_limit) or activation_limit <= 0:
            raise ValueError("activation_limit must be finite and positive")
        self.gate_projection = nn.Linear(width, hidden_width, bias=False)
        self.up_projection = nn.Linear(width, hidden_width, bias=False)
        self.down_projection = nn.Linear(hidden_width, width, bias=False)
        self.activation_limit = activation_limit

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

    def __init__(
        self, width: int, hidden_width: int, experts: int, topk: int,
        temperature: float, route_scale: float, activation_limit: float,
    ) -> None:
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (width, hidden_width, experts, topk)):
            raise ValueError("mixture dimensions must be positive integers")
        if not 1 < topk <= experts:
            raise ValueError("routed top-k must be greater than one and no larger than experts")
        if any(not math.isfinite(value) or value <= 0 for value in (temperature, route_scale)):
            raise ValueError("temperature and route_scale must be finite and positive")
        self.router = nn.Linear(width, experts, bias=False)
        self.selection_bias = nn.Parameter(torch.zeros(experts))
        self.experts = nn.ModuleList([
            SwiGLUExpert(width, hidden_width, activation_limit) for _ in range(experts)
        ])
        self.shared_expert = SwiGLUExpert(width, hidden_width, activation_limit)
        self.topk, self.temperature, self.route_scale = topk, temperature, route_scale

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
`
}];
