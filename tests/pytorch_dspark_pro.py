"""Independent arithmetic checks for the Pro DSpark draft block, without weights."""
from dataclasses import replace
import math
import unittest
import torch
from examples.deepseek_v41.dspark_pro import (
    DSparkProConfig, ProDraftAttention, ProHyperConnection, ProMoE, ProDraftBlock,
    rotate_partial, sink_attention,
)


def small_config():
    return replace(DSparkProConfig(), hidden_dim=4, intermediate_dim=3,
                   attention_heads=2, head_dim=4, rotary_dim=2, query_rank=3,
                   output_groups=2, output_rank=2, routed_experts=8)


class ProDraftTests(unittest.TestCase):
    def test_rejects_unreleased_unclipped_mode(self):
        with self.assertRaises(ValueError):
            replace(small_config(), swiglu_limit=0)

    def test_sink_and_all_draft_slots_against_scalar_average(self):
        q = torch.zeros(1, 2, 2, 4)
        kv = torch.tensor([[[2., 0., 0., 0.], [4., 0., 0., 0.], [10., 0., 0., 0.]]])
        out = sink_attention(q, kv, torch.zeros(2))
        self.assertEqual(out[0, 0, 0, 0].item(), 4.)  # three values + zero-valued sink
        kv[0, -1, 0] = 18
        self.assertEqual(sink_attention(q, kv, torch.zeros(2))[0, 0, 0, 0].item(), 6.)
        torch.testing.assert_close(sink_attention(q, kv, torch.full((2,), math.log(2.))),
                                   kv.sum(1)[:, None, None, :].expand_as(q) / 5)

    def test_partial_rotation_and_inverse_against_scalar_trigonometry(self):
        x = torch.tensor([[[1., 2., 3., 4.]]])
        angle = torch.tensor([[math.pi / 2]])
        result = rotate_partial(x, angle)
        torch.testing.assert_close(result, torch.tensor([[[1., 2., -4., 3.]]]))
        torch.testing.assert_close(rotate_partial(result, angle, inverse=True), x)

    def test_query_has_pro_specific_unit_rms_after_low_rank_projection(self):
        torch.manual_seed(11)
        config = small_config()
        attn = ProDraftAttention(config)
        q = attn.project_queries(torch.tensor([[[1., 2., 3., 4.]]]), torch.zeros(1, 1))
        torch.testing.assert_close(q.square().mean(-1), torch.ones(1, 1, 2), atol=2e-5, rtol=0)
        # Multiplying wq_b must cancel through Pro's per-head RMS normalization.
        with torch.no_grad(): attn.query_up.weight.mul_(3)
        changed = attn.project_queries(torch.tensor([[[1., 2., 3., 4.]]]), torch.zeros(1, 1))
        torch.testing.assert_close(q, changed, atol=2e-5, rtol=0)

    @torch.no_grad()
    def test_grouped_attention_output_against_scalar_reference(self):
        torch.manual_seed(7)
        config = small_config()
        attn = ProDraftAttention(config)
        x, history = torch.randn(1, 2, 4), torch.randn(1, 1, 4)
        draft_angles = torch.tensor([[0.3], [0.7]])
        history_angles = torch.tensor([[0.1]])
        # Scalar projection/normalization/rotation oracle does not call module helpers.
        def project(row, weight):
            return [sum(float(a) * float(b) for a, b in zip(row, w)) for w in weight]
        def norm(row, weight):
            denominator = math.sqrt(sum(v*v for v in row) / len(row) + config.norm_epsilon)
            return [v * float(w) / denominator for v, w in zip(row, weight)]
        def rotate(row, angle):
            a, b = row[-2:]
            return row[:-2] + [a*math.cos(angle)-b*math.sin(angle), a*math.sin(angle)+b*math.cos(angle)]
        keys = []
        for row, angle in zip(torch.cat((history, x), 1)[0], [0.1, 0.3, 0.7]):
            keys.append(rotate(norm(project(row, attn.key_value.weight), attn.kv_norm), angle))
        expected = []
        for row, angle in zip(x[0], [0.3, 0.7]):
            latent = norm(project(row, attn.query_down.weight), attn.query_norm)
            q = project(latent, attn.query_up.weight)
            group_outputs = []
            for head in range(2):
                query = rotate(norm(q[4*head:4*head+4], [1.]*4), angle)
                scores = [math.exp(sum(a*b for a,b in zip(query, key))/2) for key in keys]
                denominator = sum(scores) + math.exp(float(attn.sink[head]))
                output = [sum(score*key[d] for score,key in zip(scores, keys))/denominator for d in range(4)]
                output = rotate(output, -angle)
                group_outputs.extend(project(output, attn.output_down[head]))
            expected.append(project(group_outputs, attn.output_up.weight))
        actual = attn(x, history, history_angles, draft_angles)
        torch.testing.assert_close(actual, torch.tensor([expected]), atol=2e-6, rtol=2e-5)

    def test_current_read_mhc_and_residual_axis(self):
        config = small_config()
        mix = ProHyperConnection(config)
        with torch.no_grad():
            mix.projection.weight.zero_()
            mix.offset.zero_()
        residual = torch.arange(16.).reshape(1, 1, 4, 4)
        collapsed, post, matrix = mix.read(residual)
        torch.testing.assert_close(collapsed, residual.sum(2) * (0.5 + config.mixing_epsilon))
        torch.testing.assert_close(post, torch.ones_like(post))
        # Independent scalar sum across input streams; asymmetric matrix detects transpose.
        matrix = torch.arange(16.).reshape(1, 1, 4, 4) / 16
        output = torch.tensor([[[2., 3., 5., 7.]]])
        actual = mix.write(output, residual, post, matrix)
        expected = torch.empty_like(actual)
        for j in range(4):
            for d in range(4):
                expected[0, 0, j, d] = output[0, 0, d] + sum(
                    matrix[0, 0, i, j] * residual[0, 0, i, d] for i in range(4))
        torch.testing.assert_close(actual, expected)
        # Pro reads weights predicted from THIS residual, rather than delayed V4.1 weights.
        with torch.no_grad(): mix.projection.weight.fill_(0.01)
        self.assertFalse(torch.equal(mix.read(residual)[0], collapsed))

    def test_bias_selects_experts_but_does_not_weight_shared_or_routed_scores(self):
        config = small_config()
        moe = ProMoE(config)
        x = torch.ones(1, 1, 4)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.selection_bias.copy_(torch.arange(8.))
        weights, indices = moe.route(x)
        self.assertEqual(set(indices.flatten().tolist()), {2, 3, 4, 5, 6, 7})
        torch.testing.assert_close(weights, torch.full_like(weights, config.route_scale / 6))
        # Identical expert functions make the exact oracle (route_scale + 1)*expert(x).
        with torch.no_grad():
            for expert in [*moe.experts, moe.shared]:
                expert.gate.weight.fill_(0.1)
                expert.up.weight.fill_(0.2)
                expert.down.weight.fill_(0.3)
        gate, up = 0.4, 0.8
        expected = (config.route_scale + 1) * 3 * 0.3 * (gate / (1 + math.exp(-gate))) * up
        torch.testing.assert_close(moe(x), torch.full_like(x, expected))

    def test_full_block_history_dependency_and_gradients(self):
        torch.manual_seed(42)
        config = small_config()
        block = ProDraftBlock(config)
        residual = torch.randn(1, 2, 4, 4, requires_grad=True)
        history = torch.randn(1, 3, 4, requires_grad=True)
        main_angles, draft_angles = torch.zeros(3, 1), torch.zeros(2, 1)
        result = block(residual, history, main_angles, draft_angles)
        changed = block(residual, -history, main_angles, draft_angles)
        self.assertFalse(torch.allclose(result, changed))
        result.square().sum().backward()
        for parameter in (residual, history, block.attention.query_down.weight,
                          block.moe.router.weight, block.attention_mix.projection.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_invalid_architecture_and_history_rejected(self):
        with self.assertRaises(ValueError): replace(small_config(), selected_experts=9)
        with self.assertRaises(ValueError): replace(small_config(), rotary_dim=3)
        with self.assertRaises(ValueError): replace(small_config(), output_groups=3)
        config = small_config()
        attn = ProDraftAttention(config)
        with self.assertRaises(ValueError):
            attn(torch.zeros(1, 2, 4), torch.zeros(1, 129, 4),
                 torch.zeros(129, 1), torch.zeros(2, 1))


if __name__ == '__main__':
    unittest.main()
