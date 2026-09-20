"""Reduced CPU tests for the displayed dSpark educational modules."""
import unittest
import torch
from pytorch_test_support import load_examples


class DSparkExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_examples()

    def test_attention_sees_future_draft_slot_and_sink(self):
        attention = self.ns['dspark_projected_attention']
        q = torch.zeros(1, 2, 1, 1)
        history = torch.tensor([[[2.0]]])
        draft = torch.tensor([[[4.0], [10.0]]])
        out = attention(q, history, draft, torch.zeros(1), 1.0)
        torch.testing.assert_close(out, torch.full_like(out, 4.0))
        # First slot sees the final draft slot; an autoregressive mask is wrong.
        changed = draft.clone()
        changed[:, 1] = 18
        self.assertGreater(attention(q, history, changed, torch.zeros(1), 1.0)[0, 0, 0, 0], out[0, 0, 0, 0])
        # Sink contributes denominator mass, never a value vector.
        no_sink = attention(q, history, draft, torch.full((1,), -100.0), 1.0)
        torch.testing.assert_close(no_sink, torch.full_like(out, 16 / 3))

    def test_markov_conditions_on_preceding_sample_and_confidence_is_raw(self):
        module = self.ns['DSparkOutputHeads'](2, 2, 2)
        with torch.no_grad():
            module.markov_embed.weight.copy_(torch.eye(2))
            module.markov_head.weight.copy_(torch.tensor([[-100.0, 100.0], [100.0, -100.0]]))
            module.confidence.weight.fill_(0)
            module.confidence.weight[0, 2:] = torch.tensor([3.0, -3.0])
        hidden = torch.zeros(1, 2, 2)
        ids, logits, confidence = module(hidden, torch.zeros(1, 2, 2), torch.tensor([0]), 1.0)
        self.assertEqual(ids.tolist(), [[0, 1, 0]])
        self.assertEqual(confidence.tolist(), [[3.0, -3.0]])
        self.assertGreater(logits[0, 1, 0], logits[0, 1, 1])

    def test_verification_commits_only_prefix_and_residual(self):
        verify = self.ns['deepspec_verify_probabilities']
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        draft = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        tokens, accepted = verify(torch.tensor([0, 0]), draft, target, torch.tensor([0.5, 0.0]))
        self.assertEqual(accepted, 0)
        self.assertEqual(tokens.tolist(), [1])

    def test_all_accepted_uses_bonus_distribution(self):
        verify = self.ns['deepspec_verify_probabilities']
        draft = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.cat([draft, torch.tensor([[1.0, 0.0]])])
        tokens, accepted = verify(torch.tensor([0, 1]), draft, target, torch.tensor([0.3, 0.4]))
        self.assertEqual((tokens.tolist(), accepted), ([0, 1, 0], 2))

    def test_empty_proposal_and_zero_residual_mass(self):
        verify = self.ns['deepspec_verify_probabilities']
        tokens, accepted = verify(torch.empty(0, dtype=torch.long), torch.empty(0, 2), torch.tensor([[0.0, 1.0]]), torch.empty(0))
        self.assertEqual((tokens.tolist(), accepted), ([1], 0))
        residual = self.ns['deepspec_residual_probabilities']
        p = torch.tensor([0.25, 0.75])
        torch.testing.assert_close(residual(p, p), p)

    def test_confidence_prefix_and_state_ownership(self):
        prefix = self.ns['deepspec_confident_prefix']
        self.assertEqual(prefix(torch.tensor([3.0, -3.0, 3.0]), 0.5), 1)
        crop = self.ns['deepspec_reuse_state']
        target, draft, features = crop(torch.arange(9), torch.arange(8), torch.arange(4), 5, 2)
        self.assertEqual(target.tolist(), list(range(8)))
        self.assertEqual(draft.tolist(), list(range(5)))
        self.assertEqual(features.tolist(), [0, 1, 2])
        # Returned state owns storage; stale speculative slots cannot be read via it.
        self.assertEqual(target.untyped_storage().nbytes(), target.numel() * target.element_size())


if __name__ == '__main__':
    unittest.main()
