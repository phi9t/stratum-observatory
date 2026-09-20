"""Independent scalar oracle for the fixed-dictionary query experiment."""
import math
import unittest
from unittest.mock import patch

import torch

from examples.deepseek_v41 import csa2_attention
from examples.deepseek_v41.config import V41_CONFIG


class QueryExampleTest(unittest.TestCase):
    def test_two_queries_use_actual_core_without_mutating_shared_inputs(self):
        observed = []
        original = csa2_attention.sparse_attention

        def checked(*args):
            snapshots = [value.clone() for value in args[:-1]]
            output = original(*args)
            for value, before in zip(args[:-1], snapshots):
                torch.testing.assert_close(value, before, rtol=0, atol=0)
            observed.append(output)
            return output

        with patch.object(csa2_attention, 'sparse_attention', side_effect=checked):
            result = csa2_attention.run_query_example(V41_CONFIG)
        self.assertEqual(len(observed), 1)
        torch.testing.assert_close(result.outputs, observed[0])
        self.assertFalse(torch.allclose(result.outputs[:, 0], result.outputs[:, 1]))
        self.assertFalse(torch.allclose(result.weights[:, 0], result.weights[:, 1]))
        self.assertEqual(result.positions.tolist(), [1, 1])
        self.assertEqual(result.selected.tolist(), [[[0], [0]]])

    def test_manual_joint_softmax_includes_zero_value_sink(self):
        result = csa2_attention.run_query_example(V41_CONFIG)
        dictionary = [[2., 0.], [0., 1.], [1., 2.]]
        for index, query in enumerate([[2., 0.], [0., 2.]]):
            numerators = [math.exp(sum(q * k for q, k in zip(query, row)) / math.sqrt(2))
                          for row in dictionary]
            sink_mass = math.exp(.5)
            weights = [n / (sum(numerators) + sink_mass) for n in numerators + [sink_mass]]
            expected = [sum(weights[j] * row[d] for j, row in enumerate(dictionary))
                        for d in range(2)]
            torch.testing.assert_close(result.weights[0, index, 0], torch.tensor(weights))
            torch.testing.assert_close(result.outputs[0, index, 0], torch.tensor(expected))
            self.assertGreater(weights[-1], 0)
            self.assertLess(sum(weights[:-1]), 1)
            # This tempting alternative normalizes each source separately then adds.
            wrong = [sum(numerators[j] * dictionary[j][d] for j in range(2)) /
                     (sum(numerators[:2]) + sink_mass) +
                     numerators[2] * dictionary[2][d] / (numerators[2] + sink_mass)
                     for d in range(2)]
            self.assertFalse(torch.allclose(result.outputs[0, index, 0], torch.tensor(wrong)))

    def test_raising_sink_reduces_output_without_changing_dictionary(self):
        result = csa2_attention.run_query_example(V41_CONFIG)
        config = result.config
        larger_sink = result.sink + 3
        actual = csa2_attention.sparse_attention(
            result.queries, result.local_kv, result.compressed_kv, result.selected,
            result.positions, larger_sink, config)
        self.assertTrue(bool((actual < result.outputs).all()))


if __name__ == '__main__':
    unittest.main()
