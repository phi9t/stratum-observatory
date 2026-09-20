"""Scalar oracles and counterexamples for the dSpark boundary teaching trace."""
import json
import math
import unittest

import torch

from examples.deepseek_v41.dspark_example import dspark_example, dspark_fixture
from examples.deepseek_v41.dspark_attention import dspark_projected_attention


class DSparkExampleTest(unittest.TestCase):
    def setUp(self):
        self.trace = dspark_example()
        self.rows = {r['id']: r for r in self.trace['inputs'] + self.trace['steps']}

    def value(self, name):
        return torch.tensor(self.rows[name]['values'])

    def test_determinism_schema_and_rng_isolation(self):
        state = torch.random.get_rng_state().clone()
        self.assertEqual(json.dumps(self.trace, allow_nan=False), json.dumps(dspark_example(), allow_nan=False))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for row in self.rows.values():
            self.assertEqual(len(row['axes']), len(row['shape']))
            self.assertEqual(list(self.value(row['id']).shape), row['shape'])

    def test_attention_scalar_oracle_includes_sink_without_value(self):
        q = self.value('queries')[0].tolist()
        kv = self.value('history_kv')[0].tolist() + self.value('draft_kv')[0].tolist()
        sink = self.value('sink').tolist()
        expected = []
        for position in q:
            heads = []
            for h, query in enumerate(position):
                scores = [sum(x*y for x, y in zip(query, key))/math.sqrt(len(query)) for key in kv] + [sink[h]]
                weights = [math.exp(x-max(scores)) for x in scores]
                weights = [x/sum(weights) for x in weights]
                heads.append([sum(weights[i]*kv[i][d] for i in range(len(kv))) for d in range(len(query))])
            expected.append(heads)
        torch.testing.assert_close(self.value('attention_output')[0], torch.tensor(expected))
        self.assertTrue((self.value('attention_probabilities')[..., -1] > 0).all())

    def test_markov_and_confidence_scalar_oracles_use_predecessor(self):
        ids = self.value('sampled_ids')[0].long().tolist()
        embeddings = self.value('markov_embedding_weights').tolist()
        head = self.value('markov_head_weights').tolist()
        confidence = self.value('confidence_weights')[0].tolist()
        hidden = self.value('hidden')[0].tolist()
        base = self.value('base_logits')[0].tolist()
        survival = 1.
        for i, predecessor in enumerate(ids[:-1]):
            e = embeddings[predecessor]
            bias = [sum(w*x for w, x in zip(row, e)) for row in head]
            logits = [x+y for x, y in zip(base[i], bias)]
            exp = [math.exp(x-max(logits)) for x in logits]
            probabilities = [v/sum(exp) for v in exp]
            raw = sum(w*x for w, x in zip(confidence, hidden[i]+e))
            conditional = 1/(1+math.exp(-raw))
            survival *= conditional
            for name, expected in [('predecessor_embeddings', e), ('markov_bias', bias), ('corrected_logits', logits), ('draft_probabilities', probabilities)]:
                torch.testing.assert_close(self.value(name)[0, i], torch.tensor(expected))
            self.assertAlmostEqual(self.value('raw_confidence')[0, i].item(), raw, places=6)
            self.assertAlmostEqual(self.value('conditional_estimates')[0, i].item(), conditional, places=6)
            self.assertAlmostEqual(self.value('cumulative_survival')[0, i].item(), survival, places=6)
        current = self.value('markov_embedding_weights')[self.value('sampled_ids')[:, 1:].long()]
        wrong = torch.cat((self.value('hidden'), current), -1) @ self.value('confidence_weights').T
        self.assertGreater((wrong.squeeze(-1)-self.value('raw_confidence')).abs().max().item(), .05)

    def test_future_draft_and_predecessor_are_observable(self):
        f = dspark_fixture()
        changed = f['draft_kv'].clone()
        changed[:, -1] += torch.tensor([2., -1.])
        output = dspark_projected_attention(f['queries'], f['history_kv'], changed, f['sink'], f['config'])
        self.assertGreater((output[:, 0]-self.value('attention_output')[:, 0]).abs().max().item(), .05)
        alternate = (self.value('base_logits')[:, 0] + self.value('markov_head_weights') @ self.value('markov_embedding_weights')[2]).softmax(-1)
        self.assertGreater((alternate-self.value('draft_probabilities')[:, 0]).abs().max().item(), .01)


if __name__ == '__main__':
    unittest.main()
