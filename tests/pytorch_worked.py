"""Scalar oracles for the reader-visible CPU traces."""
import json
import math
import unittest
from typing import Any
from unittest.mock import patch

import torch
from examples.deepseek_v41 import worked_examples, csa2_attention
from examples.deepseek_v41.config import V41_CONFIG


def rows(example: dict[str, Any]) -> dict[str, Any]:
    return {row['id']: row['values'] for row in example['inputs'] + example['steps']}


def attention_scalar(query: list[float], dictionary: list[list[float]], sink: float) -> list[float]:
    masses = [math.exp(sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query)))
              for key in dictionary]
    denominator = sum(masses) + math.exp(sink)
    return [sum(m * key[d] for m, key in zip(masses, dictionary)) / denominator
            for d in range(len(query))]


class WorkedExamplesTest(unittest.TestCase):
    def test_trace_is_strict_json_and_query_uses_existing_example(self) -> None:
        with patch.object(csa2_attention, 'run_query_example', wraps=csa2_attention.run_query_example) as run:
            data = worked_examples.run_worked_examples()
        run.assert_called_once_with(V41_CONFIG)
        self.assertEqual(set(json.loads(json.dumps(data, allow_nan=False))), {'replay', 'selection', 'query'})
        values = rows(data['query'])
        dictionary = values['dictionary']
        for index, (query, output) in enumerate(zip(values['queries'], values['outputs'])):
            logits = [sum(q*k for q,k in zip(query, key)) / math.sqrt(len(query)) for key in dictionary] + values['sink']
            masses = [math.exp(value) for value in logits]
            weights = [mass / sum(masses) for mass in masses]
            contributions = [[weights[j] * x for x in key] for j, key in enumerate(dictionary)]
            torch.testing.assert_close(torch.tensor(values['logits'][index]), torch.tensor(logits))
            torch.testing.assert_close(torch.tensor(values['weights'][index]), torch.tensor(weights))
            torch.testing.assert_close(torch.tensor(values['contributions'][index]), torch.tensor(contributions))
            torch.testing.assert_close(torch.tensor(output), torch.tensor(attention_scalar(query, dictionary, values['sink'][0])))

    def test_rows_name_axes_and_preserve_dtype(self) -> None:
        examples = worked_examples.run_worked_examples()
        for example in examples.values():
            for row in example['inputs'] + example['steps']:
                self.assertEqual(len(row['axes']), len(row['shape']), row['id'])
                self.assertTrue(all(axis and not axis.startswith('axis') for axis in row['axes']))
                self.assertIn(row['dtype'], {'torch.float32', 'torch.int64', 'torch.bool'})
        selection = {row['id']: row for row in examples['selection']['inputs'] + examples['selection']['steps']}
        self.assertEqual(selection['source_queries']['axes'], ['heads', 'head_dim'])
        self.assertEqual(selection['source_scores']['axes'], ['compressed_position'])
        self.assertEqual(selection['block_maxima']['axes'], ['block'])
        self.assertEqual(selection['candidate_mask']['dtype'], 'torch.bool')
        self.assertEqual(selection['selected_indices']['dtype'], 'torch.int64')
        query = {row['id']: row for row in examples['query']['steps']}
        self.assertEqual(query['logits']['axes'], ['alternatives', 'dictionary_plus_sink'])
        for name in ('full', 'truncated'):
            for row in examples['replay'][name]:
                self.assertEqual(row['axes'], ['positions', 'head_dim'])
                self.assertEqual(len(row['axes']), len(row['shape']))

    def test_hierarchy_uses_source_candidates_and_later_scores(self) -> None:
        data = worked_examples.run_worked_examples()['selection']
        values = rows(data)
        keys = values['keys']
        def score(query: list[list[float]]) -> list[float]:
            return [sum(max(0, sum(q * k for q, k in zip(head, key))) * weight
                        for head, weight in zip(query, values['head_weights'])) /
                    math.sqrt(len(query) * len(keys[0])) for key in keys]
        source = score(values['source_queries'])
        later = score(values['selector_queries'])
        torch.testing.assert_close(torch.tensor(values['source_scores']), torch.tensor(source))
        torch.testing.assert_close(torch.tensor(values['selector_scores']), torch.tensor(later))
        visible = data['settings']['visible_groups']
        source_expected = sorted(sorted(range(visible), key=lambda i: source[i], reverse=True)[:data['settings']['topk']])
        self.assertEqual(values['source_selected_indices'], source_expected)
        self.assertNotEqual(values['source_selected_indices'], values['selected_indices'])
        size = data['settings']['block_size']
        causal = source[:visible] + [float('-inf')] * (len(source) - visible)
        maxima = [max(causal[i:i+size]) for i in range(0, len(causal), size)]
        torch.testing.assert_close(torch.tensor(values['block_maxima']), torch.tensor(maxima))
        self.assertEqual(values['causal_scores'][-1], '-inf')
        self.assertEqual(values['pinned_block_scores'][-1], 'inf')
        self.assertEqual(values['candidate_mask'], [True, True, False, False, True, False])
        allowed = [i for i, keep in enumerate(values['candidate_mask']) if keep]
        expected = sorted(sorted(allowed, key=lambda i: later[i], reverse=True)[:data['settings']['topk']])
        self.assertEqual(values['selected_indices'], expected)
        self.assertNotEqual(expected, sorted(sorted(range(visible), key=lambda i: later[i], reverse=True)[:2]))

    def test_replay_missing_dependency_matches_independent_scalar_layers(self) -> None:
        data = worked_examples.run_worked_examples()['replay']
        values = rows(data)
        dictionary = values['global_kv']
        initial = values['initial_hidden']
        results = []
        for start, name in [(0, 'full'), (2, 'truncated')]:
            hidden = initial[start:]
            for depth in range(1, 3):
                following, attention_rows = [], []
                for i, query in enumerate(hidden):
                    local = hidden[max(0, i-1):i+1]
                    attended = attention_scalar(query, local + dictionary, values['sink'][0])
                    attention_rows.append(attended)
                    following.append([x+y for x,y in zip(query, attended)])
                torch.testing.assert_close(torch.tensor(values[f'{name}_attention_{depth}']), torch.tensor(attention_rows))
                torch.testing.assert_close(torch.tensor(values[f'{name}_hidden_{depth}']), torch.tensor(following))
                hidden = following
                actual = data[name][depth]['values'][start:]
                torch.testing.assert_close(torch.tensor(actual), torch.tensor(hidden))
            results.append(hidden[-1])
        self.assertNotEqual(results[0], results[1])
        torch.testing.assert_close(torch.tensor(values['final_difference']), torch.tensor([a-b for a,b in zip(*results)]))
        # Last token agrees at the first layer; its neighbour already diverges.
        torch.testing.assert_close(torch.tensor(data['full'][1]['values'][-1]), torch.tensor(data['truncated'][1]['values'][-1]))
        self.assertNotEqual(data['full'][1]['values'][-2], data['truncated'][1]['values'][-2])
        self.assertEqual(data['truncated'][0]['values'][:2], [None, None])
        self.assertEqual(data['truncated'][0]['missing_positions'], [0, 1])
        self.assertEqual(data['truncated'][0]['shape'], [len(initial), len(initial[0])])
        self.assertEqual(data['truncated'][0]['dtype'], 'torch.float32')
        self.assertEqual(data['missing_dependency']['position'], 1)

    def test_replay_does_not_mutate_caches(self) -> None:
        original = csa2_attention.sparse_attention
        calls = []
        def checked(*args: Any) -> torch.Tensor:
            snapshots = [x.clone() for x in args[:-1]]
            result = original(*args)
            for value, snapshot in zip(args[:-1], snapshots):
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
            calls.append(result)
            return result
        with patch.object(csa2_attention, 'sparse_attention', side_effect=checked):
            worked_examples.run_worked_examples()
        self.assertEqual(len(calls), 5)


if __name__ == '__main__':
    unittest.main()
