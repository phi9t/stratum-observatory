"""Run reduced-shape CPU module checks: python -m examples.deepseek_v41.validate.

These checks use authored reference modules without checkpoint weights. They do not
establish full-model parity, upstream implementation parity, or GPU certification.
DeepSpec checks concern its separate offline evaluator, not V4.1 serving.
"""
from dataclasses import replace

import torch
from torch.nn import functional as F

from .config import V41_CONFIG, V41Config
from .backbone_mhc import (
    TransformerBlock, CausalEncoderDecoder, split_mixing_coefficients,
    combine_residual_streams,
)
from .backbone_moe import MixtureOfExperts, SwiGLUExpert
from .csa2_compression import GroupedCompressor, TokenCompressor
from .csa2_selection import (
    compute_index_scores, select_candidate_blocks, select_sparse_indices,
)
from .csa2_attention import sparse_attention
from .engram_write import EngramMemory
from .dspark_attention import dspark_projected_attention
from .dspark_heads import DSparkOutputHeads
from .deepspec_verifier import (
    deepspec_residual_probabilities, deepspec_verify_probabilities,
    deepspec_confident_prefix, deepspec_reuse_state,
)


# One fixture changes dimensions only; all numerical rules retain their fixed defaults.
SMALL_CONFIG = replace(
    V41_CONFIG, hidden_dim=4, streams=2, num_layers=2, encoder_layers=1,
    vocab_size=8, expert_hidden_dim=6, num_experts=3, experts_per_token=2,
    attention_heads=2, head_dim=4, window_size=2, index_heads=2,
    index_head_dim=4, index_topk=3, candidate_blocks=1, candidate_block_size=2,
    engram_rows=16, engram_row_dim=2, engram_heads=1, engram_max_ngram=3,
    dspark_block_size=3, markov_rank=2,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def backward(loss: torch.Tensor, *inputs: torch.Tensor) -> None:
    loss.backward()
    for value in inputs:
        require(value.grad is not None, "missing input gradient")
        require(bool(torch.isfinite(value.grad).all()), "nonfinite input gradient")
        require(bool(value.grad.abs().sum() > 0), "zero input gradient")


def normalize(values: torch.Tensor, config: V41Config) -> torch.Tensor:
    return values / (values.square().mean(-1, keepdim=True) + config.norm_epsilon).sqrt()


def check_backbone_mhc(config: V41Config) -> None:
    count = config.streams * (config.streams + 2)
    pre, post, matrix = split_mixing_coefficients(
        config, torch.zeros(1, 1, count), torch.ones(3), torch.zeros(count))
    torch.testing.assert_close(pre, torch.full_like(pre, .5 + config.mixing_epsilon))
    torch.testing.assert_close(post, torch.ones_like(post))
    torch.testing.assert_close(matrix.sum(-2), torch.ones(1, 1, config.streams),
                               atol=3 * config.mixing_epsilon, rtol=0)
    residual = torch.tensor([10., 20.]).reshape(1, 1, 2, 1).expand(1, 1, 2, config.hidden_dim)
    mixed = combine_residual_streams(
        torch.full((1, 1, config.hidden_dim), 2.), residual,
        torch.tensor([[[1., 2.]]]), torch.tensor([[[[.8, .2], [.3, .7]]]]))
    expected = torch.tensor([16., 20.]).reshape(1, 1, 2, 1).expand_as(mixed)
    torch.testing.assert_close(mixed, expected)
    blocks = [TransformerBlock(config, torch.nn.Identity(), torch.nn.Identity())
              for _ in range(config.num_layers)]
    with torch.no_grad():
        for block in blocks:
            block.attention_mixer.projection.weight.zero_()
            block.feed_forward_mixer.projection.weight.zero_()
    stack = CausalEncoderDecoder(config, blocks)
    inputs = torch.arange(1., 1. + config.hidden_dim).reshape(1, 1, -1).requires_grad_()
    output, boundary = stack(inputs)
    torch.testing.assert_close(output, normalize(inputs, config))
    require(boundary.shape == (1, 1, config.streams, config.hidden_dim), "boundary shape")
    backward(output.sum() + boundary.sum(), inputs)


def check_backbone_moe(config: V41Config) -> None:
    mixture = MixtureOfExperts(config)
    with torch.no_grad():
        mixture.router.weight.zero_()
        mixture.router.weight[:, 0] = torch.tensor([3., 0., 1.])
        mixture.selection_bias.copy_(torch.tensor([-100., 100., 200.]))
        for index, expert in enumerate([*mixture.experts, mixture.shared_expert]):
            expert.gate_projection.weight.fill_(.1 * (index + 1))
            expert.up_projection.weight.fill_(.2)
            expert.down_projection.weight.fill_(.3)
    inputs = torch.ones(1, 2, config.hidden_dim, requires_grad=True)
    output = mixture(inputs)
    scores = F.softplus(torch.tensor([0., 1.]) / config.router_temperature).sqrt()
    weights = scores / scores.sum() * config.route_scale
    expert_values = F.silu(torch.tensor([.2, .3, .4]) * config.hidden_dim)
    expert_values *= config.hidden_dim * .2 * config.expert_hidden_dim * .3
    expected = (weights * expert_values[:2]).sum() + expert_values[-1]
    torch.testing.assert_close(output, torch.full_like(output, expected))
    backward(output.sum(), inputs)
    require(mixture.selection_bias.grad is None, "selection bias must not weight outputs")
    expert = SwiGLUExpert(config)
    with torch.no_grad():
        for projection in (expert.gate_projection, expert.up_projection, expert.down_projection):
            projection.weight.zero_()
            projection.weight[0, 0] = 1
    extreme = torch.zeros(2, config.hidden_dim)
    extreme[:, 0] = torch.tensor([20., -20.])
    expected = torch.zeros_like(extreme)
    expected[:, 0] = F.silu(torch.tensor([config.activation_limit, -20.])) * torch.tensor(
        [config.activation_limit, -config.activation_limit])
    torch.testing.assert_close(expert(extreme, 1.), expected)


def check_csa2_compression(config: V41Config) -> None:
    grouped, token = GroupedCompressor(config), TokenCompressor(config)
    inputs = torch.randn(1, 5, config.hidden_dim, requires_grad=True)
    full, full_tail = grouped(inputs, grouped.empty_state(inputs))
    values = F.linear(inputs, grouped.project.weight)
    scores = F.linear(inputs, grouped.gate.weight)
    expected_groups = []
    for start in range(0, 4, config.encoder_compression_ratio):
        stop = start + config.encoder_compression_ratio
        weights = scores[:, start:stop].softmax(1)
        expected_groups.append((values[:, start:stop] * weights).sum(1))
    expected = normalize(torch.stack(expected_groups, 1), config)
    torch.testing.assert_close(full, expected)
    state, pieces = grouped.empty_state(inputs), []
    for chunk in (inputs[:, :1], inputs[:, 1:3], inputs[:, 3:]):
        emitted, state = grouped(chunk, state)
        pieces.append(emitted)
    torch.testing.assert_close(torch.cat(pieces, 1), full)
    for actual, reference in zip(state, full_tail):
        torch.testing.assert_close(actual, reference)
        require(actual.shape[1] == 1 and actual.dtype == torch.float32, "unfinished tail contract")
    unpooled = token(inputs)
    torch.testing.assert_close(unpooled, normalize(F.linear(inputs, token.project.weight), config))
    backward(full.sum() + unpooled.sum(), inputs)


def check_csa2_selection(config: V41Config) -> None:
    query = torch.ones(1, 1, config.index_heads, config.index_head_dim, requires_grad=True)
    keys = torch.tensor([8., 7., 1., 2., 100.]).reshape(1, 5, 1).expand(1, 5, config.index_head_dim)
    head_weights = torch.ones(1, 1, config.index_heads)
    scores = compute_index_scores(query, keys, head_weights)
    scale = (config.index_heads * config.index_head_dim) ** .5
    torch.testing.assert_close(scores, torch.tensor([[[8., 7., 1., 2., 100.]]]) * scale)
    visible = torch.tensor([[[4]]])
    candidates = select_candidate_blocks(scores, visible, config)
    # The newest reachable block wins despite lower scores; future index 4 is excluded.
    torch.testing.assert_close(candidates, torch.tensor([[[False, False, True, True, False]]]))
    selected = select_sparse_indices(scores, visible, candidates, config)
    require(sorted(selected[selected >= 0].tolist()) == [2, 3], "causal newest-block selection")
    require(int((selected == -1).sum()) == 1, "unavailable pick sentinel")
    backward(scores.sum(), query)


def check_csa2_attention(config: V41Config) -> None:
    query = torch.zeros(1, 1, config.attention_heads, config.head_dim)
    local = torch.tensor([1., 3., 99.]).reshape(1, 3, 1).expand(1, 3, config.head_dim).clone().requires_grad_()
    compressed = torch.full((1, 1, config.head_dim), 5., requires_grad=True)
    output = sparse_attention(query, local, compressed, torch.tensor([[[0, -1]]]),
                              torch.tensor([1]), torch.zeros(config.attention_heads), config)
    # Zero scores: local 1 + local 3 + compressed 5 + zero-valued sink, divided by four.
    torch.testing.assert_close(output, torch.full_like(output, 9 / 4))
    backward(output.sum(), local, compressed)
    torch.testing.assert_close(local.grad[:, 2], torch.zeros_like(local.grad[:, 2]))


def check_engram_write(config: V41Config) -> None:
    memory = EngramMemory(config)
    with torch.no_grad():
        memory.table.weight.fill_(1)
        memory.project.weight.fill_(1 / (config.engram_hash_columns * config.engram_row_dim))
    residual = torch.ones(1, 2, config.streams, config.hidden_dim, requires_grad=True)
    ids = torch.zeros(1, 2, config.engram_hash_columns, dtype=torch.long)
    output = memory(residual, ids, torch.tensor([[True, False]]))
    gate = torch.sigmoid(torch.tensor(config.hidden_dim ** .25))
    torch.testing.assert_close(output[:, 0], torch.full_like(output[:, 0], 1 + gate))
    torch.testing.assert_close(output[:, 1], residual[:, 1])
    backward(output.sum(), residual)
    require(memory.table.weight.grad is not None, "Engram lookup gradient")


def check_dspark_attention(config: V41Config) -> None:
    query = torch.zeros(1, config.dspark_block_size, config.attention_heads, config.head_dim)
    history = torch.ones(1, 1, config.head_dim, requires_grad=True)
    draft = torch.zeros(1, config.dspark_block_size, config.head_dim)
    draft[:, -1] = 7
    draft.requires_grad_()
    output = dspark_projected_attention(query, history, draft,
                                       torch.zeros(config.attention_heads), config)
    expected = 8 / (1 + config.dspark_block_size + 1)
    torch.testing.assert_close(output, torch.full_like(output, expected))
    backward(output[:, 0].sum(), history, draft)
    require(bool(draft.grad[:, -1].abs().sum() > 0), "first draft slot must see final draft slot")


def check_dspark_heads(config: V41Config) -> None:
    heads = DSparkOutputHeads(config)
    with torch.no_grad():
        heads.markov_embed.weight.copy_(torch.arange(config.vocab_size * config.markov_rank).reshape(
            config.vocab_size, config.markov_rank).float() / 10)
        heads.markov_head.weight.copy_(torch.arange(config.vocab_size * config.markov_rank).reshape(
            config.vocab_size, config.markov_rank).float() / 20)
        heads.confidence.weight.fill_(1)
    hidden = torch.ones(1, config.dspark_block_size, config.hidden_dim, requires_grad=True)
    base = torch.zeros(1, config.dspark_block_size, config.vocab_size, requires_grad=True)
    seed = torch.tensor([1])
    tokens, logits, confidence = heads(hidden, base, seed, 1.)
    torch.testing.assert_close(tokens[:, 0], seed)
    for position in range(config.dspark_block_size):
        embedding = heads.markov_embed(tokens[:, position])
        expected = embedding @ heads.markov_head.weight.T
        torch.testing.assert_close(logits[:, position], expected)
        torch.testing.assert_close(confidence[:, position], hidden[:, position].sum(-1) + embedding.sum(-1))
    require(bool((confidence > 1).all()), "confidence must remain raw logits")
    backward(logits.sum() + confidence.sum(), hidden, base)


def check_deepspec_verifier(config: V41Config) -> None:
    proposed = torch.tensor([0, 1, 2])
    draft = F.one_hot(proposed, config.vocab_size).float()
    target = F.one_hot(torch.tensor([0, 3, 2, 7]), config.vocab_size).float()
    uniforms = torch.full((3,), .5)
    committed, accepted = deepspec_verify_probabilities(proposed, draft, target, uniforms)
    require(accepted == 1, "acceptance must stop at first rejection")
    torch.testing.assert_close(committed, torch.tensor([0, 3]))
    target[:3] = draft
    committed, accepted = deepspec_verify_probabilities(proposed, draft, target, uniforms)
    require(accepted == 3, "identical distributions must accept every proposal")
    torch.testing.assert_close(committed, torch.tensor([0, 1, 2, 7]))
    torch.testing.assert_close(deepspec_residual_probabilities(draft[0], draft[0]), draft[0])
    require(deepspec_confident_prefix(torch.tensor([3., -3., 3.]), .5) == 1, "confidence prefix")
    cache = torch.arange(12.).reshape(6, 2)
    features = torch.arange(8.).reshape(4, 2)
    target_cache, draft_cache, retained = deepspec_reuse_state(cache, cache, features, 2, 1)
    torch.testing.assert_close(target_cache, cache[:4])
    torch.testing.assert_close(draft_cache, cache[:2])
    torch.testing.assert_close(retained, features[:2])
    cache.zero_()
    require(bool(target_cache.sum() > 0), "cropped state must release speculative backing storage")


def main() -> None:
    torch.manual_seed(41)
    checks = (
        check_backbone_mhc, check_backbone_moe, check_csa2_compression,
        check_csa2_selection, check_csa2_attention, check_engram_write,
        check_dspark_attention, check_dspark_heads, check_deepspec_verifier,
    )
    print("Reduced-shape CPU module checks; no checkpoint weights or full-model parity.")
    for check in checks:
        check(SMALL_CONFIG)
        print(f"PASS {check.__name__.removeprefix('check_')}")
    print(f"PASS {len(checks)} module checks on CPU float32; no GPU certification.")
    print("DeepSpec is checked as a separate offline evaluator, not V4.1 serving.")


if __name__ == "__main__":
    main()
