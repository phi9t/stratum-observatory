"""CPU mHC teaching fixture with injected linear attention and FFN stand-ins.

Residual mechanics execute backbone_mhc.py, whose publisher reference is the
pinned DeepSeek V4.1 inference/model.py (dba1be0a40aa45a94ad051997016db3960a90277).
Anchors: Block.hc_mixes, Block.hc_pre, Block.hc_post and Block.forward.
These illustrative weights do not establish checkpoint or GPU parity.
"""
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbone_mhc import (
    HyperConnectionMixer, RMSNorm, ResidualStates, StreamWeights, TransformerBlock,
    collapse_residual_streams, combine_residual_streams, sinkhorn_trace,
)
from .config import V41_CONFIG, V41Config


@dataclass(frozen=True)
class ResidualFixture:
    config: V41Config
    block: TransformerBlock
    residual: ResidualStates
    previous_p: StreamWeights


def residual_fixture() -> ResidualFixture:
    """Keep the checkpoint's stream count and normalization algorithm; shrink width."""
    config = replace(V41_CONFIG, hidden_dim=2)
    # Isolate constructor randomness from callers; every used weight is overwritten.
    with torch.random.fork_rng(devices=[]):
        block = TransformerBlock(config, nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        block.attention.weight.copy_(torch.tensor([[.5, -.25], [.75, .5]]))
        block.feed_forward.weight.copy_(torch.tensor([[-.25, .75], [.5, .25]]))
        for index, mixer in enumerate((block.attention_mixer, block.feed_forward_mixer)):
            # Nonseparable logits produce an asymmetric matrix after balancing.
            projection = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
            mixer.projection.weight.copy_(((projection * (7 + 2 * index) + 3) % 23 - 11) / 32)
            mixer.scales.copy_(torch.tensor([.5, .75, 1.]))
            mixer.offsets.copy_((torch.arange(24, dtype=torch.float32) % 7 - 3) / 8)
    return ResidualFixture(config, block,
                           torch.tensor([[[[1., -1.], [2., .5], [-.5, 1.5], [.25, -2.]]]]),
                           torch.tensor([[[.7, .1, .15, .05]]]))


def residual_example() -> dict[str, Any]:
    """Return the shared trace schema, retaining batch/token/stream axes explicitly."""
    # Local import permits worked_examples to register this trace without a cycle.
    from .worked_examples import tensor_row

    fixture = residual_fixture()
    config, block = fixture.config, fixture.block
    stream_axes = ['batch', 'tokens', 'streams']
    residual_axes = stream_axes + ['hidden_dim']
    token_axes = ['batch', 'tokens', 'hidden_dim']
    rows: list[dict[str, Any]] = []

    def record(identifier, title, operation, value, axes):
        rows.append(tensor_row(identifier, title, operation, value, axes))

    def stage(prefix: str, residual: ResidualStates, previous_p: StreamWeights,
              mixer: HyperConnectionMixer, norm: RMSNorm,
              sublayer: nn.Linear) -> tuple[ResidualStates, StreamWeights]:
        flat = residual.flatten(-2).float()
        inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_epsilon)
        projected = F.linear(flat, mixer.projection.weight.float())
        coefficients = projected * inverse_rms
        next_p, q, matrix = mixer(residual)
        matrix_logits = (coefficients[..., 2 * config.streams:] * mixer.scales[2]
                         + mixer.offsets[2 * config.streams:]).unflatten(-1, (config.streams, config.streams))
        iterations = sinkhorn_trace(matrix_logits, config)
        read = collapse_residual_streams(residual, previous_p)
        normalized = norm(read)
        output = sublayer(normalized)
        carried = torch.einsum('btij,btid->btjd', matrix.float(), residual.float())
        written = q.float().unsqueeze(-1) * output.float().unsqueeze(-2)
        following = combine_residual_streams(output, residual, q, matrix)
        label = 'Attention' if prefix == 'attention' else 'FFN'
        items = [
            ('current_p', 'Current read weights inherited from the preceding sublayer', 'p_previous', previous_p, stream_axes),
            ('matrix_logits', 'Unbalanced carry logits', 'α_M c_M + β_M', matrix_logits, ['batch', 'tokens', 'input_streams', 'output_streams']),
            ('sinkhorn_matrices', 'Every executed Sinkhorn balancing state', 'softmax plus epsilon; column; then row/column pairs', iterations, ['balancing_step', 'batch', 'tokens', 'input_streams', 'output_streams']),
            ('sinkhorn_row_sums', 'Row sums at every balancing step', 'sum over output streams', iterations.sum(-1), ['balancing_step', 'batch', 'tokens', 'input_streams']),
            ('sinkhorn_column_sums', 'Column sums at every balancing step', 'sum over input streams', iterations.sum(-2), ['balancing_step', 'batch', 'tokens', 'output_streams']),
            ('sinkhorn_row_error', 'Maximum row-sum deviation from one', 'max(abs(row_sum - 1))', (iterations.sum(-1) - 1).abs().amax(-1), ['balancing_step', 'batch', 'tokens']),
            ('sinkhorn_column_error', 'Maximum column-sum deviation from one', 'max(abs(column_sum - 1))', (iterations.sum(-2) - 1).abs().amax(-1), ['balancing_step', 'batch', 'tokens']),
            ('flattened', 'Flatten residual streams', 'flatten(X)', flat, ['batch', 'tokens', 'streams_times_hidden_dim']),
            ('inverse_rms', 'Inverse RMS across all streams and channels', 'rsqrt(mean(flatten(X)²) + ε_norm)', inverse_rms, ['batch', 'tokens', 'singleton']),
            ('projection', 'Coefficient projection before RMS scaling', 'flatten(X) Wᵀ', projected, ['batch', 'tokens', 'coefficients']),
            ('coefficients', 'RMS-scaled coefficients', '(flatten(X) Wᵀ) × inverse_rms', coefficients, ['batch', 'tokens', 'coefficients']),
            ('next_p', 'Input weights for the following sublayer', 'sigmoid(α_p c_p + β_p) + ε_mix', next_p, stream_axes),
            ('q', 'Current output write weights', '2 sigmoid(α_q c_q + β_q)', q, stream_axes),
            ('matrix', 'Current carry matrix: input rows, output columns', 'Sinkhorn(softmax_rows(α_M c_M + β_M) + ε_mix)', matrix, ['batch', 'tokens', 'input_streams', 'output_streams']),
            ('row_sums', 'Approximate row sums', 'sum over output streams', matrix.sum(-1), ['batch', 'tokens', 'input_streams']),
            ('column_sums', 'Approximate column sums', 'sum over input streams', matrix.sum(-2), ['batch', 'tokens', 'output_streams']),
            ('read', 'Read using previous input weights', 'Σᵢ p_previous[i] X[i,:]', read, token_axes),
            ('normalized', 'Normalize collapsed read', 'RMSNorm(read)', normalized, token_axes),
            ('output', 'Injected linear stand-in output', 'linear stand-in(normalized); illustrative, not checkpoint attention or MoE', output, token_axes),
            ('carried', 'Carry input streams into output streams', 'carried[j,d] = Σᵢ M[i,j] X[i,d]', carried, residual_axes),
            ('written', 'Write sublayer output into every stream', 'written[j,d] = q[j] y[d]', written, residual_axes),
            ('residual', 'Residual after carry and write', 'X_next = Mᵀ X + q y', following, residual_axes),
        ]
        for name, title, operation, value, axes in items:
            record(prefix + '_' + name, label + ': ' + title, operation, value, axes)
        return following, next_p

    with torch.no_grad():
        attention_residual, attention_p = stage('attention', fixture.residual, fixture.previous_p,
                                               block.attention_mixer, block.attention_norm, block.attention)
        final_residual, final_p = stage('ffn', attention_residual, attention_p,
                                       block.feed_forward_mixer, block.feed_forward_norm, block.feed_forward)
        final_collapse = collapse_residual_streams(final_residual, final_p)
        record('final_collapse', 'Illustrative stack-exit collapse', 'Σᵢ p_FFN[i] X_final[i,:]', final_collapse, token_axes)
        record('final_normalized', 'Illustrative stack-exit normalization', 'RMSNorm(final collapse)', RMSNorm(config)(final_collapse), token_axes)
    return dict(
        title='Residual streams: read, carry, write, then hand off input weights',
        scope='CPU illustrative fixture with injected linear attention and FFN stand-ins. Residual mechanics use the executable mHC modules; this is not checkpoint or GPU parity. Final collapse illustrates a stack exit after this block.',
        settings=dict(batch=1, tokens=1, streams=config.streams, hidden_dim=config.hidden_dim,
                      norm_epsilon=config.norm_epsilon, mixing_epsilon=config.mixing_epsilon,
                      sinkhorn_iterations=config.sinkhorn_iterations),
        inputs=[tensor_row('residual', 'Incoming nonidentical residual streams', 'X', fixture.residual, residual_axes),
                tensor_row('previous_p', 'Input weights inherited from the preceding sublayer', 'p_previous', fixture.previous_p, stream_axes),
                tensor_row('attention_linear', 'Injected attention stand-in weights', 'W_attention', block.attention.weight, ['output_hidden_dim', 'input_hidden_dim']),
                tensor_row('ffn_linear', 'Injected FFN stand-in weights', 'W_FFN', block.feed_forward.weight, ['output_hidden_dim', 'input_hidden_dim'])],
        steps=rows)
