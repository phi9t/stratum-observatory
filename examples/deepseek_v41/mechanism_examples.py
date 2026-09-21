"""Reduced deterministic teaching fixtures; no trained weights or serving claims."""
from dataclasses import replace
import torch
from .config import V41_CONFIG
from .csa2_compression import GroupedCompressor, TokenCompressor
from .csa2_selection import compute_index_scores, candidate_block_trace, select_sparse_indices
from .csa2_attention import sparse_attention
from .engram_write import EngramMemory
from .deepspec_verifier import deepspec_verify_probabilities, deepspec_residual_probabilities, deepspec_reuse_state


def csa2_example():
    from .worked_examples import tensor_row as row
    config = replace(V41_CONFIG, hidden_dim=2, head_dim=2, attention_heads=1,
                     index_heads=1, index_head_dim=2, window_size=2,
                     encoder_compression_ratio=2, candidate_block_size=1,
                     candidate_blocks=2, index_topk=2)
    tokens = torch.tensor([[[1., 0.], [0., 2.], [2., 1.], [-1., 1.], [.5, -.5]]])
    with torch.random.fork_rng(devices=[]):
        compressor, local_project = GroupedCompressor(config), TokenCompressor(config)
    with torch.no_grad():
        compressor.project.weight.copy_(torch.eye(2))
        compressor.gate.weight.copy_(torch.tensor([[.5, -.25], [-.5, .75]]))
        local_project.project.weight.copy_(torch.eye(2))
        compressed, tail = compressor(tokens, compressor.empty_state(tokens))
        local = local_project(tokens)
    positions = torch.tensor([0, 1, 3, 3])
    queries = torch.tensor([[[[1., 0.]], [[1., 0.]], [[1., 0.]], [[0., 1.]]]])
    visible = ((positions + 1) // config.encoder_compression_ratio)[None, :, None]
    scores = compute_index_scores(queries, compressed, torch.ones(1, len(positions), 1))
    candidates = candidate_block_trace(scores, visible, config)
    selected = select_sparse_indices(scores, visible, candidates['candidate_mask'], config)
    local_indices = positions[:, None] - torch.arange(config.window_size - 1, -1, -1)
    valid_local = local_indices >= 0
    gathered = torch.cat([local[0, local_indices.clamp_min(0)], compressed[0, selected[0].clamp_min(0)]], 1)
    valid = torch.cat([valid_local, selected[0] >= 0], -1)
    sink = torch.tensor([.5])
    logits = torch.einsum('td,tkd->tk', queries[0, :, 0], gathered) * config.head_dim ** -.5
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = torch.cat([logits, sink.expand(len(positions), 1)], -1)
    weights = logits.softmax(-1)
    output = sparse_attention(queries, local, compressed, selected, positions, sink, config)
    membership = (torch.arange(tokens.shape[1])[None] <= positions[:, None]) & (torch.arange(tokens.shape[1])[None] > positions[:, None] - config.window_size)
    steps = []
    def add(name, title, op, tensor, axes):
        steps.append(row(name, title, op, tensor, axes))
    projected = tokens @ compressor.project.weight.T
    complete = tokens.shape[1] // config.encoder_compression_ratio * config.encoder_compression_ratio
    pooling = (tokens @ compressor.gate.weight.T)[:, :complete].unflatten(1, (-1, config.encoder_compression_ratio)).softmax(2)
    add('pooling_weights', 'Per-channel group pooling weights', 'softmax of compression gates over group tokens', pooling[0], ['groups', 'group_tokens', 'head_dim'])
    add('projected_values', 'Projected compression values', 'tokens W_compressᵀ', projected[0], ['positions', 'head_dim'])
    add('compressed_kv', 'Completed normalized compressed entries', 'GroupedCompressor', compressed[0], ['compressed_position', 'head_dim'])
    add('unfinished_tail', 'Unfinished projected tail; unavailable to attention', 'GroupedCompressor next_state values', tail[0][0], ['unfinished_positions', 'head_dim'])
    add('local_kv', 'Normalized local KV', 'TokenCompressor', local[0], ['positions', 'head_dim'])
    add('visible_groups', 'Completed causal group count', 'floor((query_position + 1) / compression_ratio)', visible[0], ['queries', 'singleton'])
    add('compressed_available', 'Compressed-entry causal availability', 'entry_index < completed_group_count', (torch.arange(compressed.shape[1]) < visible)[0], ['queries', 'compressed_position'])
    add('local_membership', 'Causal sliding-window membership', 'position - window < local_position <= position', membership, ['queries', 'positions'])
    add('index_scores', 'Index scores over compressed keys', 'compute_index_scores', scores[0], ['queries', 'compressed_position'])
    add('candidate_mask', 'Retained causal candidate groups', 'candidate_block_trace', candidates['candidate_mask'][0], ['queries', 'compressed_position'])
    add('selected_indices', 'Selected compressed entries', 'select_sparse_indices', selected[0], ['queries', 'selected_positions'])
    add('gathered_values', 'Local slots followed by compressed slots', 'gather local window and selected compressed entries; masked slots receive zero mass', gathered, ['queries', 'dictionary_slots', 'head_dim'])
    add('logits', 'Joint logits with denominator-only sink', '[local logits; compressed logits; sink]', logits, ['queries', 'dictionary_plus_sink'])
    add('weights', 'One normalization over the union and sink', 'softmax(logits)', weights, ['queries', 'dictionary_plus_sink'])
    add('outputs', 'Canonical attention outputs', 'sparse_attention', output[0, :, 0], ['queries', 'head_dim'])
    return dict(title='CSA2: compression, causal availability and a joint read', scope='Reduced CPU teaching fixture. Index keys reuse the illustrative compressed vectors. Final query alternatives occupy the same position and use unchanged KV. No learned checkpoint projections or positional rotations are claimed.', settings=dict(compression_ratio=config.encoder_compression_ratio, window=config.window_size, head_dim=config.head_dim), inputs=[row('tokens', 'Illustrative token vectors', 'supplied tokens', tokens[0], ['positions', 'hidden_dim']), row('positions', 'Query positions; last pair are alternatives', 'absolute positions', positions, ['queries']), row('queries', 'Projected illustrative query alternatives', 'supplied Q', queries[0, :, 0], ['queries', 'head_dim'])], steps=steps)


def engram_example():
    from .worked_examples import tensor_row as row
    from .engram_hash import ngram_hash_trace
    config = replace(V41_CONFIG, hidden_dim=2, engram_row_dim=2)
    # Same normalized spelling may have different raw embedding IDs; this tiny
    # supplied map isolates hashing from the tokenizer's Unicode normalizer.
    tokens = torch.tensor([[0, 1, 2, 1]])
    token_map = torch.tensor([0, 1, 1])
    mask = torch.tensor([[False, True, False, True]])
    hashes = ngram_hash_trace(tokens, mask, token_map, config.engram_layer_ids,
                             config.engram_max_ngram, config.engram_heads, bucket_size=3)
    residual = torch.tensor([[[[1., -.5], [.25, 1.], [-1., .5], [.5, .75]]]]).expand(1, 4, 4, 2).clone()
    steps = []
    for name, axes in [('normalized_ids', ['batch', 'tokens']), ('masked_ids', ['batch', 'tokens']),
                       ('history', ['batch', 'tokens', 'lookback']), ('boundary_mask', ['batch', 'tokens', 'lookback']),
                       ('hash_multipliers', ['sites', 'lookback']), ('hash_primes', ['sites', 'hash_columns']),
                       ('hash_offsets', ['sites', 'hash_columns'])]:
        record = row(name, name.replace('_', ' ').capitalize(), 'ngram_hash_trace; publisher hash algorithm with teaching bucket sizes', hashes[name], axes)
        if name == 'hash_multipliers':
            # Exact int64 decimal text survives JavaScript's narrower integer range.
            record['values'] = [[str(value) for value in site] for site in hashes[name].tolist()]
        steps.append(record)
    for index, site in enumerate(['site_a', 'site_b']):
        ids = hashes['hash_ids'][:, :, index]
        table_rows = int(hashes['hash_primes'][index].sum())
        with torch.random.fork_rng(devices=[]):
            memory = EngramMemory(replace(config, engram_rows=table_rows))
        with torch.no_grad():
            memory.table.weight.copy_(((torch.arange(table_rows * 2).reshape(table_rows, 2) * (index + 2)) % 11 - 5) / 8)
            width = config.engram_hash_columns * config.engram_row_dim
            memory.project.weight.copy_(((torch.arange(10 * width).reshape(10, width) * (index + 3)) % 13 - 6) / 8)
            trace = memory.trace(residual, ids, mask)
        steps.append(row(site + '_hash_ids', 'Global row IDs at layer site', 'XOR products modulo layer-specific prime ranges plus offsets', ids, ['batch', 'tokens', 'hash_columns']))
        for name, axes in [('lookup', ['batch', 'tokens', 'hash_columns', 'row_dim']), ('keys', ['batch', 'tokens', 'streams', 'hidden_dim']), ('value', ['batch', 'tokens', 'hidden_dim']), ('scores', ['batch', 'tokens', 'streams']), ('gates', ['batch', 'tokens', 'streams']), ('write', ['batch', 'tokens', 'streams', 'hidden_dim']), ('output', ['batch', 'tokens', 'streams', 'hidden_dim'])]:
            steps.append(row(site + '_' + name, site.replace('_', ' ').title() + ': ' + name, 'EngramMemory.trace: layer-specific hashes, table and projection; shared token history', trace[name], axes))
    return dict(title='Engram: the same token history at different layer sites', scope='CPU publisher hash algorithm with small teaching buckets and illustrative independent layer tables. Compressed token map is supplied; full tokenizer normalization and incremental cache are outside this fixture. Dead positions stop lookback and suppress writes. Hidden vectors stay fixed to isolate layer differences. Large int64 multipliers are encoded as exact decimal strings.', settings=dict(streams=config.streams, hidden_dim=config.hidden_dim, hash_columns=config.engram_hash_columns, layer_ids=list(config.engram_layer_ids), sites=len(config.engram_layer_ids), teaching_bucket_size=3), inputs=[row('tokens', 'Shared raw token history', 'illustrative token IDs', tokens, ['batch', 'tokens']), row('token_map', 'Supplied normalized ID map', 'raw token ID to compressed ID; distinct raw IDs can share a normalized ID', token_map, ['raw_vocabulary']), row('write_mask', 'Dead positions suppress writes and stop lookback', 'supplied image/dead-token mask', mask, ['batch', 'tokens']), row('residual', 'Same residual input at both sites', 'X', residual, ['batch', 'tokens', 'streams', 'hidden_dim'])], steps=steps)


def verification_example():
    from .worked_examples import tensor_row as row
    proposed = torch.tensor([0, 1, 0])
    draft = torch.tensor([[.5, .25, .25], [.2, .6, .2], [.5, .25, .25]])
    target = torch.tensor([[.75, .125, .125], [.2, .1, .7], [.75, .125, .125], [.2, .3, .5]])
    uniforms = torch.tensor([.5, .5, .1])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        committed, accepted = deepspec_verify_probabilities(proposed, draft, target, uniforms)
    rows = torch.arange(len(proposed))
    ratios = (target[rows, proposed] / draft[rows, proposed]).clamp_max(1)
    prefix = (uniforms < ratios).long().cumprod(0)
    correction = deepspec_residual_probabilities(target[accepted], draft[accepted])
    start = 2
    target_cache = torch.arange(start + len(proposed) + 1)[:, None]
    draft_cache = torch.arange(start + len(proposed))[:, None]
    features = torch.arange(len(proposed) + 1)[:, None]
    retained = deepspec_reuse_state(target_cache, draft_cache, features, start, accepted)
    inputs = [row('proposed', 'Proposed draft tokens', 'supplied proposal', proposed, ['proposal_positions']), row('draft_probabilities', 'Draft distributions', 'supplied normalized probabilities', draft, ['proposal_positions', 'vocabulary']), row('target_probabilities', 'Target verification distributions and bonus', 'supplied normalized probabilities', target, ['proposal_plus_bonus_positions', 'vocabulary']), row('uniforms', 'Fixed acceptance draws', 'supplied uniforms', uniforms, ['proposal_positions'])]
    steps = [row('acceptance_probabilities', 'Per-token acceptance probabilities', 'min(1, p_target(token) / p_draft(token))', ratios, ['proposal_positions']), row('prefix_acceptance', 'Only the contiguous accepted prefix survives', 'cumprod(uniform < acceptance_probability)', prefix, ['proposal_positions']), row('accepted_count', 'Accepted proposal count', 'deepspec_verify_probabilities', torch.tensor(accepted), []), row('correction_distribution', 'Distribution after first rejection', 'deepspec_residual_probabilities: normalize(max(target - draft, 0))', correction, ['vocabulary']), row('committed_tokens', 'Accepted prefix followed by correction', 'deepspec_verify_probabilities', committed, ['committed_positions']), row('retained_counts', 'Retained target cache, draft cache and verified features', 'lengths from deepspec_reuse_state; correction not yet cached', torch.tensor([len(x) for x in retained]), ['state_kind'])]
    for name, tensor in zip(['target_cache', 'draft_cache', 'verified_features'], retained):
        steps.append(row('retained_' + name, 'Retained ' + name.replace('_', ' '), 'deepspec_reuse_state', tensor, ['retained_positions', 'feature']))
    return dict(title='Verification: acceptance, correction and retained cache state', scope='Deterministic CPU example of the checked-in DeepSpec evaluator helpers. This is an offline Qwen3/Gemma4 evaluator boundary, not a V4.1 production verifier, scheduler, or measured acceptance result. Distributions are supplied independently of the dSpark teaching head.', settings=dict(start=start, random_seed=31, correction_cached=False, state_order=['target cache', 'draft cache', 'verified features']), inputs=inputs, steps=steps)
