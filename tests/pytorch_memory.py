import unittest
from dataclasses import replace
import torch
from examples.deepseek_v41.config import V41_CONFIG
from examples.deepseek_v41.engram_write import EngramMemory
from examples.deepseek_v41.csa2_compression import CompressionNorm, TokenCompressor, GroupedCompressor
from examples.deepseek_v41.csa2_selection import compute_index_scores, select_candidate_blocks, select_sparse_indices
from examples.deepseek_v41.csa2_attention import sparse_attention


# Change only fixture sizes; published numerical settings stay in the global config.
SMALL = replace(V41_CONFIG, hidden_dim=2, streams=2, head_dim=2, attention_heads=2,
                engram_rows=4, engram_row_dim=2, engram_heads=1, engram_max_ngram=2,
                candidate_blocks=1, candidate_block_size=2, index_topk=2, window_size=2)


class MemoryExamplesTest(unittest.TestCase):
    def test_config_constructor_allocates_only_reduced_fixture(self):
        config = replace(V41_CONFIG, hidden_dim=2, streams=2, engram_rows=4,
                         engram_row_dim=2, engram_heads=1, engram_max_ngram=2)
        memory = EngramMemory(config)
        self.assertEqual(memory.table.weight.shape, (4, 2))
        self.assertEqual(memory.project.weight.shape, (6, 2))

    def test_engram_mask_and_signed_stream_gates(self):
        m = EngramMemory(replace(SMALL, hidden_dim=1, engram_rows=1, engram_row_dim=1))
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
        m = GroupedCompressor(SMALL)
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
        m = GroupedCompressor(SMALL)
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
        result = compute_index_scores(q, k, weights)
        torch.testing.assert_close(result, torch.tensor([[[2 ** .5]]]))

    def test_projection_compressor_has_no_pooling_gate(self):
        m = TokenCompressor(SMALL)
        self.assertFalse(hasattr(m, 'gate'))
        self.assertEqual(m(torch.ones(1, 3, 2)).shape, (1, 3, 2))

    def test_candidates_pin_latest_and_exclude_future(self):
        scores = torch.tensor([[[100., 90., 80., 1., 1000., 2000.]]])
        visible = torch.tensor([[[4]]])
        mask = select_candidate_blocks(scores, visible, SMALL)
        self.assertEqual(mask.tolist(), [[[False, False, True, True, False, False]]])
        idx = select_sparse_indices(scores, visible, mask, SMALL)
        self.assertEqual(idx.tolist(), [[[2, 3]]])
        none = select_sparse_indices(scores, torch.zeros_like(visible), mask, SMALL)
        self.assertEqual(none.tolist(), [[[-1, -1]]])

    def test_sparse_union_sink_and_causal_window(self):
        q = torch.zeros(1, 2, 1, 1)
        local = torch.tensor([[[2.], [4.]]])
        compressed = torch.tensor([[[10.]]])
        selected = torch.tensor([[[-1], [0]]])
        y = sparse_attention(q, local, compressed, selected, torch.arange(2), torch.zeros(1), replace(SMALL, window_size=1))
        torch.testing.assert_close(y.flatten(), torch.tensor([1., 14. / 3]))
        changed = local.clone()
        changed[:, 1] = 400
        z = sparse_attention(q, changed, compressed, selected, torch.arange(2), torch.zeros(1), replace(SMALL, window_size=1))
        torch.testing.assert_close(y[:, 0], z[:, 0])

    def test_engram_rejects_incompatible_write_mask(self):
        memory = EngramMemory(SMALL)
        with self.assertRaisesRegex(ValueError, "token mask"):
            memory(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 1, dtype=torch.long),
                   torch.ones(1, 1))

    def test_norm_uses_configured_latent_axis(self):
        norm = CompressionNorm(SMALL)
        values = torch.tensor([[[3., 4.]]])
        expected = values / (12.5 + SMALL.norm_epsilon) ** .5
        torch.testing.assert_close(norm(values), expected)

    def test_bfloat16_modules_preserve_dtype_and_finite_gradients(self):
        torch.manual_seed(7)
        memory = EngramMemory(SMALL).to(torch.bfloat16)
        residual = torch.randn(1, 3, 2, 2, dtype=torch.bfloat16, requires_grad=True)
        output = memory(residual, torch.ones(1, 3, 1, dtype=torch.long),
                        torch.ones(1, 3, dtype=torch.bool))
        self.assertEqual(output.dtype, residual.dtype)
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        for parameter in memory.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for constructor in (TokenCompressor, GroupedCompressor):
            module = constructor(SMALL).to(torch.bfloat16)
            inputs = torch.randn(1, 3, 2, dtype=torch.bfloat16, requires_grad=True)
            if isinstance(module, GroupedCompressor):
                output, state = module(inputs, module.empty_state(inputs))
                self.assertEqual(state[0].dtype, torch.float32)
            else:
                output = module(inputs)
            self.assertEqual(output.dtype, inputs.dtype)
            output.float().square().mean().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
            for parameter in module.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_partial_state_rejects_wrong_batch_and_completed_group(self):
        module = GroupedCompressor(SMALL)
        inputs = torch.ones(1, 1, 2)
        for shape in ((2, 1, 2), (1, 2, 2), (1, 1, 3)):
            with self.assertRaises(ValueError):
                module(inputs, (torch.zeros(shape), torch.zeros(shape)))

    def test_empty_candidates_and_invalid_visibility_shape(self):
        empty = torch.empty(1, 2, 0)
        visible = torch.zeros(1, 2, 1, dtype=torch.long)
        self.assertEqual(select_candidate_blocks(empty, visible, SMALL).shape,
                         empty.shape)
        with self.assertRaises(ValueError):
            select_candidate_blocks(empty, visible.squeeze(-1), SMALL)

    def test_attention_bfloat16_float32_sink_and_backward(self):
        query = torch.ones(1, 1, 1, 2, dtype=torch.bfloat16, requires_grad=True)
        local = torch.ones(1, 1, 2, dtype=torch.bfloat16, requires_grad=True)
        output = sparse_attention(
            query, local, local[:, :0], torch.empty(1, 1, 0, dtype=torch.long),
            torch.tensor([0]), torch.zeros(1, requires_grad=True), replace(SMALL, window_size=1))
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(local.grad).all())

    def test_outer_cpu_autocast_preserves_explicit_fp32_math(self):
        torch.manual_seed(19)
        memory = EngramMemory(SMALL)
        residual = torch.randn(1, 3, 2, 2, requires_grad=True)
        hashes = torch.ones(1, 3, 1, dtype=torch.long)
        mask = torch.ones(1, 3, dtype=torch.bool)
        tokens = torch.randn(1, 3, 2, requires_grad=True)
        grouped = GroupedCompressor(SMALL)
        direct = TokenCompressor(SMALL)
        query = torch.randn(1, 3, 2, 2, requires_grad=True)
        keys = torch.randn(1, 3, 2, requires_grad=True)
        weights = torch.randn(1, 3, 2)
        selected = torch.tensor([[[0], [0], [0]]])
        sink = torch.randn(2, requires_grad=True)

        def evaluate():
            compressed, state = grouped(tokens, grouped.empty_state(tokens))
            return [memory(residual, hashes, mask), direct(tokens), compressed, *state,
                    compute_index_scores(query, keys, weights),
                    sparse_attention(
                        query, keys, keys[:, :1], selected, torch.arange(3), sink, SMALL)]

        expected = evaluate()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            actual = evaluate()
        for reference, observed in zip(expected, actual):
            self.assertEqual(observed.dtype, torch.float32)
            torch.testing.assert_close(reference, observed, rtol=0, atol=0)
        sum(value.square().mean() for value in actual).backward()
        for inputs in (residual, tokens, query, keys, sink):
            self.assertTrue(torch.isfinite(inputs.grad).all())
