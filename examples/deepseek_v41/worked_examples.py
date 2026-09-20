"""Deterministic CPU teaching traces; illustrative vectors, no checkpoint weights.

Only fixture sizes replace V41Config settings. The replay experiment deliberately
uses a simple residual update, not a complete V4.1 layer or serving implementation.
Its compressed dictionary is fixed independently of the local hidden-state path.
"""
from dataclasses import replace
import json
import math
from typing import Any

import torch
from torch import Tensor

from .config import V41_CONFIG
from . import csa2_attention, csa2_selection


def json_values(value: Any) -> Any:
    """Encode nonfinite mask/pin values explicitly, without invalid JSON numbers."""
    if isinstance(value, torch.Tensor):
        return json_values(value.detach().cpu().tolist())
    if isinstance(value, list):
        return [json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'inf' if value > 0 else '-inf'
    return value


def tensor_row(identifier: str, title: str, operation: str, tensor: Tensor, axes: list[str]) -> dict[str, Any]:
    return dict(id=identifier, title=title, operation=operation,
                shape=list(tensor.shape), axes=axes, dtype=str(tensor.dtype), values=json_values(tensor))


def query_example() -> dict[str, Any]:
    example = csa2_attention.run_query_example(V41_CONFIG)
    dictionary = torch.cat([example.local_kv, example.compressed_kv], dim=1)[0]
    queries = example.queries[0, :, 0]
    logits = queries @ dictionary.T / example.config.head_dim ** .5
    logits = torch.cat([logits, example.sink.expand(queries.shape[0], 1)], dim=-1)
    weights = example.weights[0, :, 0]
    contributions = weights[:, :-1, None] * dictionary[None]
    return dict(
        title='Changing a query over fixed KV',
        scope='CPU illustrative vectors. Query alternatives share one position and unchanged KV; these are not successive decoder layers.',
        settings=dict(batch=example.queries.shape[0], local_positions=example.local_kv.shape[1],
                      compressed_positions=example.compressed_kv.shape[1],
                      head_dim=example.config.head_dim, heads=example.config.attention_heads,
                      window=example.config.window_size, position=example.positions[0].item()),
        inputs=[tensor_row('queries', 'Query alternatives', 'Q', queries, ['alternatives', 'head_dim']),
                tensor_row('dictionary', 'Local rows then compressed row', '[local KV; compressed KV]', dictionary, ['dictionary', 'head_dim']),
                tensor_row('sink', 'Fixed sink logit', 's', example.sink, ['heads'])],
        steps=[tensor_row('logits', 'Scaled logits, including sink', '[QKᵀ / √D; s]', logits, ['alternatives', 'dictionary_plus_sink']),
               tensor_row('weights', 'Joint softmax weights', 'softmax(logits)', weights, ['alternatives', 'dictionary_plus_sink']),
               tensor_row('contributions', 'Weighted value contributions', 'pᵢ Vᵢ; sink value is zero', contributions, ['alternatives', 'dictionary', 'head_dim']),
               tensor_row('outputs', 'Attention outputs', 'csa2_attention.run_query_example → sparse_attention: Σ pᵢ Vᵢ', example.outputs[0, :, 0], ['alternatives', 'head_dim'])])


def selection_example() -> dict[str, Any]:
    config = replace(V41_CONFIG, index_heads=1, index_head_dim=2,
                     candidate_block_size=2, candidate_blocks=2, index_topk=2)
    keys = torch.tensor([[[9., 1.], [8., 2.], [4., 9.], [3., 8.], [1., 7.], [0., 10.]]])
    source = torch.tensor([[[[1., 0.]]]])
    selector = torch.tensor([[[[0., 1.]]]])
    weights = torch.ones(1, 1, 1)
    visible = torch.tensor([[[5]]])
    source_scores = csa2_selection.compute_index_scores(source, keys, weights)
    source_selected = csa2_selection.select_sparse_indices(
        source_scores, visible, torch.ones_like(source_scores, dtype=torch.bool), config)
    trace = csa2_selection.candidate_block_trace(source_scores, visible, config)
    # The source layer supplies candidates; a later selector computes fresh scores.
    selector_scores = csa2_selection.compute_index_scores(selector, keys, weights)
    selected = csa2_selection.select_sparse_indices(selector_scores, visible, trace['candidate_mask'], config)
    def row(name: str, title: str, operation: str, tensor: Tensor, axes: list[str]) -> dict[str, Any]:
        return tensor_row(name, title, operation, tensor[0, 0], axes)
    return dict(
        title='Candidate blocks, then fresh sparse selection',
        scope='Reduced CPU fixture. Candidate-source and later-selector queries differ; the later selector cannot recover keys excluded by the shared candidate mask.',
        settings=dict(heads=config.index_heads, head_dim=config.index_head_dim,
                      block_size=config.candidate_block_size, candidate_blocks=config.candidate_blocks,
                      topk=config.index_topk, visible_groups=visible.item()),
        inputs=[tensor_row('keys', 'Compressed index keys', 'K_index', keys[0], ['compressed_position', 'head_dim']),
                row('source_queries', 'Candidate-source query heads', 'Q_source', source, ['heads', 'head_dim']),
                row('selector_queries', 'Later-selector query heads', 'Q_selector', selector, ['heads', 'head_dim']),
                row('head_weights', 'Head weights', 'w', weights, ['heads'])],
        steps=[row('source_scores', 'Candidate-source scores', 'csa2_selection.compute_index_scores: Σ wₕ ReLU(qₕ · k) / √(D H)', source_scores, ['compressed_position']),
               row('source_selected_indices', 'Source-layer direct sparse selection', 'csa2_selection.select_sparse_indices: top-k across all completed causal groups', source_selected, ['selected_position']),
               row('causal_scores', 'Completed causal groups only', 'score if j < visible else −∞', trace['causal_scores'], ['compressed_position']),
               row('block_maxima', 'Maximum score per candidate block', 'csa2_selection.candidate_block_trace: max within each block', trace['block_maxima'], ['block']),
               row('newest_block', 'Newest reachable block', 'floor((visible − 1) / block_size)', trace['newest_block'], ['singleton']),
               row('pinned_block_scores', 'Pin the newest reachable block', 'replace newest block score with +∞', trace['pinned_block_scores'], ['block']),
               row('candidate_mask', 'Expand retained blocks and reapply causality', 'top candidate blocks ∩ causal positions', trace['candidate_mask'], ['compressed_position']),
               row('selector_scores', 'Fresh later-selector scores', 'csa2_selection.compute_index_scores: Σ wₕ ReLU(qₕ · k) / √(D H)', selector_scores, ['compressed_position']),
               row('selected_indices', 'Final position-ordered selection', 'csa2_selection.select_sparse_indices: top-k within candidates, then sort positions', selected, ['selected_position'])])


def replay_example() -> dict[str, Any]:
    config = replace(V41_CONFIG, attention_heads=1, head_dim=2, window_size=2)
    initial = torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [2., 0.]]], dtype=torch.float32)
    global_kv = torch.tensor([[[.5, 1.5]]], dtype=torch.float32)
    sink = torch.tensor([.5])
    depth_count = 2
    steps: list[dict[str, Any]] = []
    runs: dict[str, list[dict[str, Any]]] = {}
    for name, start in [('full', 0), ('truncated', 2)]:
        hidden = initial[:, start:].clone()
        positions = torch.arange(hidden.shape[1])
        selected = torch.zeros(1, hidden.shape[1], 1, dtype=torch.long)
        depth_rows: list[dict[str, Any]] = []
        def record(depth: int) -> None:
            depth_rows.append(dict(depth=depth, positions=list(range(initial.shape[1])),
                                   shape=[initial.shape[1], config.head_dim], dtype=str(hidden.dtype),
                                   axes=['positions', 'head_dim'],
                                   missing_positions=list(range(start)),
                                   values=[None] * start + json_values(hidden[0])))
        record(0)
        for depth in range(1, depth_count + 1):
            attended = csa2_attention.sparse_attention(
                hidden[:, :, None], hidden, global_kv, selected, positions, sink, config)[:, :, 0]
            steps.append(tensor_row(f'{name}_attention_{depth}', f'{name.title()} attention at depth {depth}',
                                    'csa2_attention.sparse_attention(local hidden, fixed global KV, fixed sink)', attended[0], ['positions', 'head_dim']))
            hidden = hidden + attended
            steps.append(tensor_row(f'{name}_hidden_{depth}', f'{name.title()} residual at depth {depth}',
                                    'h_next = h + attention(h)', hidden[0], ['positions', 'head_dim']))
            record(depth)
        runs[name] = depth_rows
    full_final = torch.tensor(runs['full'][-1]['values'][-1])
    truncated_final = torch.tensor(runs['truncated'][-1]['values'][-1])
    steps.append(tensor_row('final_difference', 'Final target difference', 'full target − truncated target', full_final - truncated_final, ['head_dim']))
    return dict(
        title='A local replay window has dependencies through depth',
        scope='Tiny illustrative local dependency experiment with a simple residual update. This is not a full V4.1 decoder. Global KV and sink stay fixed in both runs.',
        settings=dict(layers=depth_count, window=config.window_size, head_dim=config.head_dim,
                      replay_start=2, target_position=initial.shape[1]-1),
        inputs=[tensor_row('initial_hidden', 'Initial local hidden vectors', 'h⁰', initial[0], ['positions', 'head_dim']),
                tensor_row('global_kv', 'Separate fixed global dictionary', 'compressed KV', global_kv[0], ['compressed_position', 'head_dim']),
                tensor_row('sink', 'Fixed sink logit', 's', sink, ['heads'])],
        steps=steps, **runs,
        missing_dependency=dict(position=1, via_position=2, target_position=3,
                                from_depth=0, via_depth=1, target_depth=2))


def run_worked_examples() -> dict[str, dict[str, Any]]:
    """Return strict JSON-compatible traces calculated by the checked-in CPU code."""
    with torch.no_grad():
        return dict(replay=replay_example(), selection=selection_example(), query=query_example())


if __name__ == '__main__':
    print(json.dumps(run_worked_examples(), allow_nan=False, sort_keys=True))
