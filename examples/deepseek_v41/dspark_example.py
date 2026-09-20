"""dSpark boundary trace: projected attention and output heads, with teaching weights.

Executes the V4.1 helpers, not a draft backbone or released checkpoint. Attention
inputs are supplied after projection/rotation; output hidden and base logits are
independent supplied boundaries. No connection between them is simulated.
"""
from dataclasses import replace
from typing import Any

import torch
from torch.nn import functional as F

from .config import V41_CONFIG
from .dspark_attention import dspark_projected_attention
from .dspark_heads import DSparkOutputHeads


def dspark_fixture() -> dict[str, Any]:
    """Shrink dimensions while preserving all draft positions and the head mechanism."""
    config = replace(V41_CONFIG, hidden_dim=2, attention_heads=2, head_dim=2,
                     vocab_size=4, markov_rank=2)
    with torch.random.fork_rng(devices=[]):
        heads = DSparkOutputHeads(config)
    with torch.no_grad():
        heads.markov_embed.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., .5], [.5, -1.]]))
        heads.markov_head.weight.copy_(torch.tensor([[.8, -.3], [-.4, .7], [.2, .5], [-.6, -.2]]))
        heads.confidence.weight.copy_(torch.tensor([[.3, -.2, .9, -.7]]))
    return dict(config=config, heads=heads,
                queries=torch.tensor([[[[1., 0.], [0., 1.]], [[.5, 1.], [-1., .5]],
                                       [[1., 1.], [.5, -.5]], [[-.5, 1.], [1., .5]],
                                       [[.25, -.5], [-.5, -.25]]]]),
                history_kv=torch.tensor([[[1., -.5], [-.25, 1.]]]),
                draft_kv=torch.tensor([[[.5, .25], [-1., .5], [.25, -1.], [.75, .5], [-.5, -.75]]]),
                sink=torch.tensor([.25, -.5]),
                hidden=torch.tensor([[[.5, -1.], [1., .25], [-.5, .75], [.25, .5], [-.75, -.25]]]),
                base_logits=torch.tensor([[[.2, -.1, .4, 0.], [-.2, .5, .1, -.3],
                                           [.3, .1, -.4, .2], [0., -.2, .4, .1], [.1, .2, -.1, .3]]]),
                seed=torch.tensor([0], dtype=torch.long))


def dspark_example() -> dict[str, Any]:
    """Trace two explicit boundaries without pretending to execute the missing stack."""
    from .worked_examples import tensor_row

    f = dspark_fixture()
    config, heads = f['config'], f['heads']
    inputs, steps = [], []

    def record(destination, identifier, title, operation, value, axes):
        destination.append(tensor_row(identifier, title, operation, value, axes))

    for identifier, title, axes in [
        ('queries', 'Supplied projected and rotated queries', ['batch', 'draft_positions', 'heads', 'head_dim']),
        ('history_kv', 'Supplied valid history KV only', ['batch', 'history_positions', 'head_dim']),
        ('draft_kv', 'Supplied projected and rotated draft KV', ['batch', 'draft_positions', 'head_dim']),
        ('sink', 'Attention sink logits, with no value', ['heads']),
        ('hidden', 'Supplied pre-final-norm collapsed hidden', ['batch', 'draft_positions', 'hidden_dim']),
        ('base_logits', 'Supplied logits from shared vocabulary head', ['batch', 'draft_positions', 'vocabulary']),
        ('seed', 'Preceding token included in output', ['batch']),
    ]:
        record(inputs, identifier, title, 'supplied boundary input', f[identifier], axes)
    for identifier, title, value, axes in [
        ('markov_embedding_weights', 'Teaching Markov embedding weights', heads.markov_embed.weight, ['vocabulary', 'markov_rank']),
        ('markov_head_weights', 'Teaching Markov vocabulary weights', heads.markov_head.weight, ['vocabulary', 'markov_rank']),
        ('confidence_weights', 'Teaching raw confidence weights', heads.confidence.weight, ['output', 'hidden_plus_markov']),
    ]:
        record(inputs, identifier, title, 'explicit illustrative weights', value, axes)

    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        kv = torch.cat((f['history_kv'], f['draft_kv']), dim=1)
        scores = torch.einsum('bkhd,bnd->bkhn', f['queries'], kv) * config.head_dim ** -.5
        scores = torch.cat((scores, f['sink'].view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)), -1)
        attention_probabilities = scores.softmax(-1)
        attention_output = dspark_projected_attention(f['queries'], f['history_kv'], f['draft_kv'], f['sink'], config)
        ids, logits, raw = heads(f['hidden'], f['base_logits'], f['seed'], temperature=1.)
        predecessor = heads.markov_embed(ids[:, :-1])
        bias = F.linear(predecessor, heads.markov_head.weight)
        conditional = raw.sigmoid()
        quantities = [
            ('attention_scores', 'History, every draft slot, then sink', 'Q Kᵀ / sqrt(head_dim); append per-head sink', scores, ['batch', 'draft_positions', 'heads', 'history_plus_draft_plus_sink']),
            ('attention_probabilities', 'Bidirectional draft attention probabilities', 'softmax over history + all draft slots + sink', attention_probabilities, ['batch', 'draft_positions', 'heads', 'history_plus_draft_plus_sink']),
            ('attention_output', 'Canonical projected attention output', 'sum over real KV; sink owns no value', attention_output, ['batch', 'draft_positions', 'heads', 'head_dim']),
            ('sampled_ids', 'Seed followed by sampled draft tokens', 'DSparkOutputHeads; sequential Markov-conditioned sampling', ids, ['batch', 'seed_plus_draft_positions']),
            ('predecessor_embeddings', 'Embedding of each preceding token', 'MarkovEmbedding(output_ids[:, :-1])', predecessor, ['batch', 'draft_positions', 'markov_rank']),
            ('markov_bias', 'Predecessor-conditioned vocabulary bias', 'MarkovHead(predecessor_embedding)', bias, ['batch', 'draft_positions', 'vocabulary']),
            ('corrected_logits', 'Canonical corrected logits', 'base_logits + Markov bias', logits, ['batch', 'draft_positions', 'vocabulary']),
            ('draft_probabilities', 'Draft sampling distribution', 'softmax(corrected_logits / temperature)', logits.softmax(-1), ['batch', 'draft_positions', 'vocabulary']),
            ('raw_confidence', 'Canonical raw confidence score', 'linear(concat(pre-final-norm hidden, predecessor embedding))', raw, ['batch', 'draft_positions']),
            ('conditional_estimates', 'Illustrative conditional acceptance estimates', 'sigmoid(raw confidence); teaching weights are uncalibrated', conditional, ['batch', 'draft_positions']),
            ('cumulative_survival', 'Illustrative probability that the prefix survives', 'cumulative product of conditional estimates', conditional.cumprod(-1), ['batch', 'draft_positions']),
        ]
        for identifier, title, operation, value, axes in quantities:
            record(steps, identifier, title, operation, value, axes)
    return dict(
        title='dSpark: bidirectional attention, Markov sampling and confidence',
        scope='CPU teaching fixture using the V4.1 projected-attention and output-head helpers. Projected attention inputs and output-head inputs are separately supplied boundaries; no draft backbone, projection, rotation, cache update or target verifier executes here. Explicit teaching weights are not checkpoint weights. Sigmoid and cumulative survival illustrate confidence interpretation, not measured acceptance or GPU performance.',
        settings=dict(batch=1, history_positions=2, draft_positions=config.dspark_block_size,
                      heads=config.attention_heads, head_dim=config.head_dim,
                      hidden_dim=config.hidden_dim, vocabulary=config.vocab_size,
                      markov_rank=config.markov_rank, temperature=1., random_seed=23),
        inputs=inputs, steps=steps)
