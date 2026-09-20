import unittest
from dataclasses import replace
from examples.deepseek_v41.config import V41_CONFIG
from examples.deepseek_v41.backbone_mhc import (
    RMSNorm, HyperConnectionMixer, TransformerBlock, CausalEncoderDecoder,
    split_mixing_coefficients, combine_residual_streams,
)
from examples.deepseek_v41.backbone_moe import SwiGLUExpert, MixtureOfExperts
import torch


class BackboneConfigTest(unittest.TestCase):
    def test_shared_config_drives_all_backbone_modules(self):
        config = replace(V41_CONFIG, hidden_dim=4, streams=2, num_layers=2,
                         encoder_layers=1, expert_hidden_dim=7, num_experts=3,
                         experts_per_token=2)
        moe = MixtureOfExperts(config)
        block = TransformerBlock(config, torch.nn.Identity(), moe)
        model = CausalEncoderDecoder(config, [block, TransformerBlock(
            config, torch.nn.Identity(), MixtureOfExperts(config))])
        self.assertEqual(moe.router.weight.shape, (3, 4))
        self.assertEqual(moe.shared_expert.up_projection.weight.shape, (7, 4))
        self.assertEqual(block.attention_mixer.projection.weight.shape, (8, 8))
        output, boundary = model(torch.randn(1, 3, 4))
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertEqual(boundary.shape, (1, 3, 2, 4))

    def test_stack_rejects_missing_configured_blocks(self):
        config = replace(V41_CONFIG, hidden_dim=4, streams=2, num_layers=2,
                         encoder_layers=1)
        with self.assertRaisesRegex(ValueError, "configured encoder/decoder stack"):
            CausalEncoderDecoder(config, [torch.nn.Identity()])



class BackboneExamplesTest(unittest.TestCase):
    def test_rmsnorm_uses_width_not_stream_axis(self):
        norm = RMSNorm(replace(V41_CONFIG, hidden_dim=2))
        x = torch.tensor([[[[3., 4.], [0., 2.]]]])
        expected = torch.tensor([[[[3 / (12.5 + V41_CONFIG.norm_epsilon)**0.5, 4 / (12.5 + V41_CONFIG.norm_epsilon)**0.5],
                                   [0., 2 / (2 + V41_CONFIG.norm_epsilon)**0.5]]]])
        torch.testing.assert_close(norm(x), expected)

    def test_sinkhorn_and_residual_matrix_orientation(self):
        split = split_mixing_coefficients
        pre, post, comb = split(replace(V41_CONFIG, streams=2), torch.zeros(1, 1, 8), torch.ones(3), torch.zeros(8))
        torch.testing.assert_close(pre, torch.full_like(pre, .500001))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-2), torch.ones(1, 1, 2), atol=3e-6, rtol=0)
        residual = torch.tensor([[[[10.], [20.]]]])
        # First matrix axis is INPUT copy; second is OUTPUT copy.
        matrix = torch.tensor([[[[.8, .2], [.3, .7]]]])
        result = combine_residual_streams(torch.tensor([[[2.]]]), residual,
                                  torch.tensor([[[1., 2.]]]), matrix)
        torch.testing.assert_close(result, torch.tensor([[[[16.], [20.]]]]))
        with self.assertRaises(ValueError):
            split(replace(V41_CONFIG, streams=2), torch.zeros(1, 1, 7), torch.ones(3), torch.zeros(8))

    def test_previous_premix_not_current_attention_premix(self):
        class Capture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = None
            def forward(self, x):
                self.seen = x.clone()
                return torch.zeros_like(x)
        class FixedMix(torch.nn.Module):
            def __init__(self, pre):
                super().__init__()
                self.pre = torch.tensor(pre)
            def forward(self, x):
                b, t, r, _ = x.shape
                return (self.pre.expand(b, t, r), torch.zeros(b, t, r),
                        torch.eye(r).expand(b, t, r, r))
        attention, ffn = Capture(), Capture()
        block = TransformerBlock(replace(V41_CONFIG, hidden_dim=2, streams=2), attention, ffn)
        block.attention_mixer = FixedMix([0., 1.])
        block.feed_forward_mixer = FixedMix([.25, .75])
        x = torch.tensor([[[[3., 4.], [4., 3.]]]])
        output, next_pre = block(x, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(attention.seen, block.attention_norm(x[:, :, 0]))
        torch.testing.assert_close(ffn.seen, block.feed_forward_norm(x[:, :, 1]))
        torch.testing.assert_close(output, x)
        torch.testing.assert_close(next_pre, torch.tensor([[[.25, .75]]]))

    def test_flat_stack_boundary_and_final_collapse(self):
        class Step(torch.nn.Module):
            def __init__(self, inc, pre):
                super().__init__()
                self.inc, self.pre, self.seen = inc, torch.tensor(pre), None
            def forward(self, x, pre):
                self.seen = pre.clone()
                return x + torch.tensor(self.inc).reshape(1, 1, 2, 1), self.pre.expand(*x.shape[:2], 2)
        first, second = Step([1., 2.], [0., 1.]), Step([3., 4.], [1., 0.])
        stack = CausalEncoderDecoder(replace(V41_CONFIG, num_layers=2, encoder_layers=1, hidden_dim=2, streams=2), [first, second])
        stack.output_norm = torch.nn.Identity()
        output, boundary = stack(torch.tensor([[[1., 2.]]]))
        torch.testing.assert_close(first.seen, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(second.seen, torch.tensor([[[0., 1.]]]))
        torch.testing.assert_close(boundary, torch.tensor([[[[2., 3.], [3., 4.]]]]))
        torch.testing.assert_close(output, torch.tensor([[[5., 6.]]]))
        with self.assertRaises(ValueError):
            CausalEncoderDecoder(replace(V41_CONFIG, num_layers=2, encoder_layers=0, hidden_dim=2, streams=2), [first, second])

    def test_moe_bias_selects_but_does_not_weight_and_shared_always_runs(self):
        class ConstantExpert(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, x, weights):
                return torch.full_like(x, self.value) * weights
        moe = MixtureOfExperts(replace(V41_CONFIG, hidden_dim=2, expert_hidden_dim=3, num_experts=3, experts_per_token=2))
        moe.experts = torch.nn.ModuleList([ConstantExpert(1000), ConstantExpert(5), ConstantExpert(7)])
        moe.shared_expert = ConstantExpert(3)
        with torch.no_grad():
            # Raw logits [3, 0, 1]; bias excludes the highest raw-scoring expert.
            moe.router.weight.copy_(torch.tensor([[3., 0.], [0., 0.], [1., 0.]]))
            moe.selection_bias.copy_(torch.tensor([-100., 100., 200.]))
        scores = torch.nn.functional.softplus(torch.tensor([0., 1.])).sqrt()
        expected = 3 + 1.5 * (scores * torch.tensor([5., 7.])).sum() / scores.sum()
        torch.testing.assert_close(moe(torch.ones(1, 2, 2)), torch.full((1, 2, 2), expected))
        self.assertEqual(moe.router.out_features, 3)  # shared has no router logit

    def test_expert_always_uses_asymmetric_checkpoint_clamps(self):
        expert = SwiGLUExpert(replace(V41_CONFIG, hidden_dim=1, expert_hidden_dim=1))
        with torch.no_grad():
            expert.gate_projection.weight.fill_(1)
            expert.up_projection.weight.fill_(1)
            expert.down_projection.weight.fill_(1)
        x = torch.tensor([[20.], [-20.]])
        # Positive gate clamps at 10, negative gate remains -20;
        # up branch clamps on both sides, including -10.
        expected = torch.nn.functional.silu(torch.tensor([[10.], [-20.]])) * torch.tensor([[10.], [-10.]])
        torch.testing.assert_close(expert(x, 1.0), expected)

    def test_bfloat16_stack_and_moe_have_finite_gradients(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                config = replace(V41_CONFIG, hidden_dim=4, streams=2, num_layers=2,
                                 encoder_layers=1, expert_hidden_dim=8, num_experts=3,
                                 experts_per_token=2)
                blocks = [TransformerBlock(
                    config, torch.nn.Linear(4, 4, bias=False), MixtureOfExperts(config)
                ) for _ in range(config.num_layers)]
                model = CausalEncoderDecoder(config, blocks).to(dtype)
                inputs = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
                output, boundary = model(inputs)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(boundary.dtype, dtype)
                (output.float().square().mean() + boundary.float().square().mean()).backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0)
                # Router selection biases are nondifferentiable by design.
                for name, parameter in model.named_parameters():
                    if name.endswith('selection_bias'):
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                coefficients = blocks[0].attention_mixer(boundary.detach())
                self.assertTrue(all(value.dtype == torch.float32 for value in coefficients))

    def test_autocast_preserves_float32_mixing_and_normalization_math(self):
        mixer = HyperConnectionMixer(replace(V41_CONFIG, hidden_dim=4, streams=2))
        residual = torch.randn(1, 3, 2, 4)
        expected = mixer(residual)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = mixer(residual)
        for result, reference in zip(actual, expected):
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, reference)
        norm = RMSNorm(replace(V41_CONFIG, hidden_dim=4)).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1., 2., 3., 4.]))
        inputs = torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16)
        reference = (inputs.float() / (inputs.float().square().mean(-1, keepdim=True) + V41_CONFIG.norm_epsilon).sqrt())
        reference = (reference * norm.weight.float()).to(inputs.dtype)
        torch.testing.assert_close(norm(inputs), reference, rtol=0, atol=0)

    def test_tensor_shape_guards_preserve_residual_ownership(self):
        config = replace(V41_CONFIG, hidden_dim=4, streams=2, num_experts=3,
                         experts_per_token=2, expert_hidden_dim=8)
        with self.assertRaises(ValueError):
            RMSNorm(config)(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            HyperConnectionMixer(config)(torch.ones(1, 2, 4, 2))
        with self.assertRaises(ValueError):
            MixtureOfExperts(config)(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            combine_residual_streams(torch.ones(1, 2, 4), torch.ones(1, 2, 2, 4),
                                     torch.ones(1, 2, 2), torch.ones(1, 2, 2, 3))


if __name__ == '__main__':
    unittest.main()
