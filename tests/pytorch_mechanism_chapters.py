import unittest
import torch
from examples.deepseek_v41.worked_examples import run_worked_examples


def rows(example):
    return {r['id']: torch.tensor([[int(x) for x in site] for site in r['values']]) if r['id'] == 'hash_multipliers' else torch.tensor(r['values']) for r in example['inputs'] + example['steps'] if '-inf' not in str(r['values'])}


class MechanismChapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = run_worked_examples()

    def test_compressed_groups_and_local_membership_are_causal(self):
        r = rows(self.examples['csa2'])
        self.assertEqual(r['visible_groups'].flatten().tolist(), [0, 1, 2, 2])
        self.assertEqual(r['local_membership'][0].tolist(), [True, False, False, False, False])
        self.assertTrue(torch.allclose(r['weights'].sum(-1), torch.ones(4)))
        self.assertFalse(torch.allclose(r['outputs'][-1], r['outputs'][-2]))
        expected = torch.einsum('tk,tkd->td', r['weights'][:, :-1], r['gathered_values'])
        torch.testing.assert_close(r['outputs'], expected)

    def test_sinkhorn_steps_and_previous_gate_handoff(self):
        r = rows(self.examples['residual'])
        for prefix in ['attention', 'ffn']:
            trace = r[prefix + '_sinkhorn_matrices']
            torch.testing.assert_close(trace[-1], r[prefix + '_matrix'])
            torch.testing.assert_close(trace.sum(-1), r[prefix + '_sinkhorn_row_sums'])
            torch.testing.assert_close(trace.sum(-2), r[prefix + '_sinkhorn_column_sums'])
        torch.testing.assert_close(r['attention_current_p'], r['previous_p'])
        torch.testing.assert_close(r['ffn_current_p'], r['attention_next_p'])
        self.assertFalse(torch.allclose(r['attention_current_p'], r['attention_next_p']))

    def test_engram_sites_preserve_mask_and_have_different_gates(self):
        r = rows(self.examples['engram'])
        self.assertFalse(torch.equal(r['site_a_hash_ids'], r['site_b_hash_ids']))
        self.assertTrue((r['hash_multipliers'] % 2 == 1).all())
        self.assertEqual(r['history'][0, 3, 1:].tolist(), [0, 0, 0])
        self.assertFalse(torch.allclose(r['site_a_lookup'], r['site_b_lookup']))
        self.assertFalse(torch.allclose(r['site_a_gates'], r['site_b_gates']))
        for site in ['site_a', 'site_b']:
            torch.testing.assert_close(r[site + '_output'][:, 0], r['residual'][:, 0])
            self.assertEqual(r[site + '_gates'][:, 0].count_nonzero().item(), 0)

    def test_hashes_match_scalar_xor_and_disjoint_ranges(self):
        r = rows(self.examples['engram'])
        for site_index, site in enumerate(['site_a', 'site_b']):
            multipliers = r['hash_multipliers'][site_index].tolist()
            primes = r['hash_primes'][site_index].tolist()
            offsets = r['hash_offsets'][site_index].tolist()
            heads = len(primes) // (len(multipliers) - 1)
            for position, history in enumerate(r['history'][0].tolist()):
                running = history[0] * multipliers[0]
                expected = []
                for order in range(1, len(multipliers)):
                    running ^= history[order] * multipliers[order]
                    for head in range(heads):
                        column = (order - 1) * heads + head
                        expected.append(running % primes[column] + offsets[column])
                self.assertEqual(r[site + '_hash_ids'][0, position].tolist(), expected)
            self.assertEqual(offsets, [sum(primes[:i]) for i in range(len(primes))])
        self.assertEqual(len(set(r['hash_primes'].flatten().tolist())), r['hash_primes'].numel())

    def test_sinkhorn_snapshots_match_scalar_normalization_order(self):
        r = rows(self.examples['residual'])
        eps = self.examples['residual']['settings']['mixing_epsilon']
        for prefix in ['attention', 'ffn']:
            snapshots = r[prefix + '_sinkhorn_matrices'][:, 0, 0]
            for step in range(1, len(snapshots)):
                previous = snapshots[step - 1].tolist()
                expected = []
                for i, values in enumerate(previous):
                    expected.append([value / (sum(previous[k][j] for k in range(len(previous))) + eps)
                                     if step % 2 else value / (sum(values) + eps)
                                     for j, value in enumerate(values)])
                torch.testing.assert_close(snapshots[step], torch.tensor(expected))

    def test_verification_commits_prefix_then_correction_and_crops_cache(self):
        r = rows(self.examples['verification'])
        self.assertEqual(r['accepted_count'].item(), 1)
        self.assertEqual(r['committed_tokens'].tolist(), [0, 2])
        self.assertEqual(r['retained_counts'].tolist(), [4, 2, 2])
        self.assertEqual(r['prefix_acceptance'].tolist(), [1, 0, 0])
        self.assertEqual(r['correction_distribution'].tolist(), [0., 0., 1.])

if __name__ == '__main__':
    unittest.main()
