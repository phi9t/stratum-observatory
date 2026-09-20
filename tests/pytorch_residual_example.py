"""Independent scalar checks for the illustrative mHC teaching trace."""
import json
import math
import unittest

import torch

from examples.deepseek_v41.residual_example import residual_example, residual_fixture


class ResidualExampleTest(unittest.TestCase):
    def setUp(self):
        self.trace = residual_example()
        self.rows = {row['id']: row for row in self.trace['inputs'] + self.trace['steps']}

    def value(self, identifier):
        return torch.tensor(self.rows[identifier]['values'])

    def test_schema_is_finite_deterministic_and_keeps_stream_axis(self):
        self.assertEqual(json.dumps(self.trace, allow_nan=False),
                         json.dumps(residual_example(), allow_nan=False))
        self.assertEqual(self.rows['residual']['shape'], [1, 1, 4, 2])
        for row in self.rows.values():
            self.assertEqual(len(row['axes']), len(row['shape']))
            self.assertEqual(list(self.value(row['id']).shape), row['shape'])

    def test_scalar_projection_split_sinkhorn_and_two_sublayers(self):
        fixture = residual_fixture()
        residual = self.value('residual')[0, 0].tolist()
        previous = self.value('previous_p')[0, 0].tolist()
        for prefix, mixer, sublayer in [
            ('attention', fixture.block.attention_mixer, fixture.block.attention),
            ('ffn', fixture.block.feed_forward_mixer, fixture.block.feed_forward),
        ]:
            flat = [v for stream in residual for v in stream]
            inverse = 1 / math.sqrt(sum(v*v for v in flat) / len(flat) + fixture.config.norm_epsilon)
            coefficients = [sum(w*x for w, x in zip(row, flat)) * inverse
                            for row in mixer.projection.weight.tolist()]
            torch.testing.assert_close(self.value(prefix + '_coefficients')[0, 0], torch.tensor(coefficients))
            scales, offsets = mixer.scales.tolist(), mixer.offsets.tolist()
            sigmoid = lambda x: 1 / (1 + math.exp(-x))
            next_p = [sigmoid(coefficients[i]*scales[0] + offsets[i]) + fixture.config.mixing_epsilon for i in range(4)]
            q = [2*sigmoid(coefficients[4+i]*scales[1] + offsets[4+i]) for i in range(4)]
            matrix = []
            for i in range(4):
                logits = [coefficients[8+4*i+j]*scales[2] + offsets[8+4*i+j] for j in range(4)]
                exp = [math.exp(v-max(logits)) for v in logits]
                matrix.append([v/sum(exp)+fixture.config.mixing_epsilon for v in exp])
            def columns():
                sums = [sum(matrix[i][j] for i in range(4)) + fixture.config.mixing_epsilon for j in range(4)]
                for i in range(4):
                    for j in range(4):
                        matrix[i][j] /= sums[j]
            columns()
            for _ in range(fixture.config.sinkhorn_iterations - 1):
                matrix = [[v/(sum(row)+fixture.config.mixing_epsilon) for v in row] for row in matrix]
                columns()
            torch.testing.assert_close(self.value(prefix+'_next_p')[0, 0], torch.tensor(next_p))
            torch.testing.assert_close(self.value(prefix+'_q')[0, 0], torch.tensor(q))
            torch.testing.assert_close(self.value(prefix+'_matrix')[0, 0], torch.tensor(matrix))
            for suffix in ('row_sums', 'column_sums'):
                torch.testing.assert_close(self.value(prefix+'_'+suffix), torch.ones(1, 1, 4), atol=3e-6, rtol=0)
            read = [sum(previous[i]*residual[i][d] for i in range(4)) for d in range(2)]
            norm = [v / math.sqrt(sum(x*x for x in read)/2 + fixture.config.norm_epsilon) for v in read]
            output = [sum(w*v for w, v in zip(row, norm)) for row in sublayer.weight.tolist()]
            carried = [[sum(matrix[i][j]*residual[i][d] for i in range(4)) for d in range(2)] for j in range(4)]
            written = [[q[j]*output[d] for d in range(2)] for j in range(4)]
            for name, expected in [('read', read), ('normalized', norm), ('output', output), ('carried', carried), ('written', written)]:
                torch.testing.assert_close(self.value(prefix+'_'+name)[0, 0], torch.tensor(expected))
            residual = [[carried[j][d]+written[j][d] for d in range(2)] for j in range(4)]
            torch.testing.assert_close(self.value(prefix+'_residual')[0, 0], torch.tensor(residual))
            previous = next_p
        final = [sum(previous[i]*residual[i][d] for i in range(4)) for d in range(2)]
        torch.testing.assert_close(self.value('final_collapse')[0, 0], torch.tensor(final))
        normalized = [v / math.sqrt(sum(x*x for x in final)/2 + fixture.config.norm_epsilon) for v in final]
        torch.testing.assert_close(self.value('final_normalized')[0, 0], torch.tensor(normalized))

    def test_full_block_exactly_matches_staged_trace(self):
        fixture = residual_fixture()
        with torch.no_grad():
            residual, weights = fixture.block(fixture.residual, fixture.previous_p)
        torch.testing.assert_close(residual, self.value('ffn_residual'), rtol=0, atol=0)
        torch.testing.assert_close(weights, self.value('ffn_next_p'), rtol=0, atol=0)

    def test_wrong_current_premix_and_transpose_are_detectable(self):
        for prefix, incoming, previous in [('attention', 'residual', 'previous_p'),
                                           ('ffn', 'attention_residual', 'attention_next_p')]:
            x = self.value(incoming)
            wrong_read = (x*self.value(prefix+'_next_p').unsqueeze(-1)).sum(-2)
            correct_read = self.value(prefix+'_read')
            self.assertGreater((wrong_read-correct_read).abs().max().item(), .05)
            self.assertGreater(((wrong_read / wrong_read.square().mean(-1, keepdim=True).sqrt())
                               - self.value(prefix+'_normalized')).abs().max().item(), .01)
            wrong_carry = self.value(prefix+'_matrix') @ x
            self.assertGreater((wrong_carry-self.value(prefix+'_carried')).abs().max().item(), .05)


if __name__ == '__main__':
    unittest.main()
