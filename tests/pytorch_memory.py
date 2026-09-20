import unittest
import torch
from pytorch_test_support import load_examples


class MemoryExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_examples()

    def test_engram_mask_and_signed_stream_gates(self):
        m = self.ns['ReadableEngram'](1, 1, 1, 2, 1, eps=1e-6)
        with torch.no_grad():
            m.table.weight.fill_(1)
            m.project.weight.copy_(torch.tensor([[1.], [-1.], [2.]]))
        x = torch.ones(1, 2, 2, 1)
        y = m(x, torch.zeros(1, 2, 1, dtype=torch.long), torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(y[:, 1], x[:, 1]))
        self.assertGreater(y[0, 0, 0, 0].item(), 2)
        self.assertLess(y[0, 0, 1, 0].item(), 2)
        self.assertAlmostEqual((y[0, 0, :, 0] - 1).sum().item(), 2, places=5)

    def test_group_compressor_preserves_partial_tail(self):
        m = self.ns['ReadableGroupedCompressor'](2, 2, 2, eps=1e-6)
        with torch.no_grad():
            m.project.weight.copy_(torch.eye(2))
            m.gate.weight.zero_()
        x = torch.tensor([[[1., 3.], [3., 1.], [2., 4.], [4., 2.]]])
        whole, tail = m(x, m.empty_state(x))
        first, tail = m(x[:, :3], m.empty_state(x))
        self.assertEqual(tail[0].shape[1], 1)
        last, tail = m(x[:, 3:], tail)
        torch.testing.assert_close(torch.cat([first, last], 1), whole)
        self.assertEqual(tail[0].shape[1], 0)
        torch.testing.assert_close(whole, torch.ones_like(whole), atol=1e-6, rtol=1e-6)

    def test_pooling_is_per_channel_and_waits_for_complete_group(self):
        m = self.ns['ReadableGroupedCompressor'](2, 2, 2, eps=1e-6)
        with torch.no_grad():
            m.project.weight.copy_(torch.eye(2))
            m.gate.weight.copy_(torch.eye(2))
        m.norm = torch.nn.Identity()
        first, tail = m(torch.tensor([[[1., 3.]]]), m.empty_state(torch.ones(1, 1, 2)))
        self.assertEqual(first.shape, (1, 0, 2))
        result, tail = m(torch.tensor([[[3., 1.]]]), tail)
        expected = (1 + 3 * torch.exp(torch.tensor(2.))) / (1 + torch.exp(torch.tensor(2.)))
        torch.testing.assert_close(result, torch.full((1, 1, 2), expected))

    def test_index_relu_precedes_weighted_head_sum(self):
        q = torch.tensor([[[[1.], [-1.]]]])
        k = torch.tensor([[[2.]]])
        weights = torch.tensor([[[1., 10.]]])
        result = self.ns['readable_index_scores'](q, k, weights)
        torch.testing.assert_close(result, torch.tensor([[[2 ** .5]]]))

    def test_projection_compressor_has_no_pooling_gate(self):
        m = self.ns['ReadableTokenCompressor'](2, 2, eps=1e-6)
        self.assertFalse(hasattr(m, 'gate'))
        self.assertEqual(m(torch.ones(1, 3, 2)).shape, (1, 3, 2))

    def test_candidates_pin_latest_and_exclude_future(self):
        scores = torch.tensor([[[100., 90., 80., 1., 1000., 2000.]]])
        visible = torch.tensor([[[4]]])
        mask = self.ns['readable_candidate_mask'](scores, visible, 1, 2)
        self.assertEqual(mask.tolist(), [[[False, False, True, True, False, False]]])
        idx = self.ns['readable_sparse_indices'](scores, visible, mask, 2)
        self.assertEqual(idx.tolist(), [[[2, 3]]])
        none = self.ns['readable_sparse_indices'](scores, torch.zeros_like(visible), mask, 2)
        self.assertEqual(none.tolist(), [[[-1, -1]]])

    def test_sparse_union_sink_and_causal_window(self):
        q = torch.zeros(1, 2, 1, 1)
        local = torch.tensor([[[2.], [4.]]])
        compressed = torch.tensor([[[10.]]])
        selected = torch.tensor([[[-1], [0]]])
        y = self.ns['readable_sparse_union'](q, local, compressed, selected, torch.arange(2), 1, torch.zeros(1))
        torch.testing.assert_close(y.flatten(), torch.tensor([1., 14. / 3]))
        changed = local.clone()
        changed[:, 1] = 400
        z = self.ns['readable_sparse_union'](q, changed, compressed, selected, torch.arange(2), 1, torch.zeros(1))
        torch.testing.assert_close(y[:, 0], z[:, 0])
