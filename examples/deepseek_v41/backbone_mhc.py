from collections.abc import Iterator, Sequence
from typing import Annotated

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V41Config


# Shapes use named axes; streams are residual copies, not attention heads.
HiddenStates = Annotated[Tensor, "... hidden_dim"]
TokenStates = Annotated[Tensor, "batch tokens hidden_dim"]
ResidualStates = Annotated[Tensor, "batch tokens streams hidden_dim"]
StreamWeights = Annotated[Tensor, "batch tokens streams"]
ResidualMixingMatrix = Annotated[Tensor, "batch tokens input_streams output_streams"]
MixingCoefficients = Annotated[Tensor, "batch tokens mixing_coefficients"]
MixingScales = Annotated[Tensor, "coefficient_groups=3"]
MixingOffsets = Annotated[Tensor, "mixing_coefficients"]


class RMSNorm(nn.Module):
    """Normalize the last dimension, accumulating variance and scale in float32."""

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_dim))
        self.epsilon = config.norm_epsilon

    def forward(self, inputs: HiddenStates) -> HiddenStates:
        if not inputs.is_floating_point() or inputs.shape[-1] != self.weight.numel():
            raise ValueError("RMSNorm requires floating inputs with the configured width")
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(inputs.dtype)


def split_mixing_coefficients(
    config: V41Config, coefficients: MixingCoefficients,
    scales: MixingScales, offsets: MixingOffsets,
) -> tuple[StreamWeights, StreamWeights, ResidualMixingMatrix]:
    """Return input/output weights [...,R] and residual matrix [...,R,R]."""
    streams, epsilon = config.streams, config.mixing_epsilon
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
    for residual_matrix in sinkhorn_steps(matrix_logits.unflatten(-1, (streams, streams)), config):
        pass  # Keep only the last state; tracing is an opt-in observation.

    return input_weights, output_weights, residual_matrix


def sinkhorn_steps(
    matrix_logits: ResidualMixingMatrix, config: V41Config,
) -> Iterator[ResidualMixingMatrix]:
    """Yield source-ordered normalizations without retaining all matrices."""
    epsilon = config.mixing_epsilon
    matrix = matrix_logits.float().softmax(-1) + epsilon
    yield matrix
    matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
    yield matrix
    for _ in range(config.sinkhorn_iterations - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + epsilon)
        yield matrix
        matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
        yield matrix


def sinkhorn_trace(
    matrix_logits: ResidualMixingMatrix, config: V41Config,
) -> Annotated[Tensor, "balancing_step batch tokens input_streams output_streams"]:
    """Stack [balancing_step, batch, tokens, input_streams, output_streams]."""
    return torch.stack(list(sinkhorn_steps(matrix_logits, config)))


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

    def __init__(self, config: V41Config) -> None:
        super().__init__()
        self.config = config
        coefficient_count = config.streams * (config.streams + 2)
        self.projection = nn.Linear(
            config.hidden_dim * config.streams, coefficient_count, bias=False
        )
        self.scales = nn.Parameter(torch.ones(3))
        self.offsets = nn.Parameter(torch.zeros(coefficient_count))
        self.width, self.streams = config.hidden_dim, config.streams
        self.norm_epsilon = config.norm_epsilon

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
                self.config, coefficients, self.scales, self.offsets
            )


class TransformerBlock(nn.Module):
    """Apply attention/FFN with the previous sublayer's input mixing weights."""

    def __init__(
        self, config: V41Config, attention: nn.Module, feed_forward: nn.Module,
    ) -> None:
        super().__init__()
        self.attention, self.feed_forward = attention, feed_forward
        self.attention_norm = RMSNorm(config)
        self.feed_forward_norm = RMSNorm(config)
        self.attention_mixer = HyperConnectionMixer(config)
        self.feed_forward_mixer = HyperConnectionMixer(config)

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

    def __init__(self, config: V41Config, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(blocks) != config.num_layers or not 0 < config.encoder_layers < len(blocks):
            raise ValueError("blocks must match the configured encoder/decoder stack")
        self.blocks = nn.ModuleList(blocks)
        self.encoder_layers = config.encoder_layers
        self.streams, self.width = config.streams, config.hidden_dim
        self.output_norm = RMSNorm(config)

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
