import unittest
import torch
from pytorch_test_support import load_examples


class BackboneExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_examples()

    def test_rmsnorm_uses_width_not_stream_axis(self):
        norm = self.ns['RMSNorm'](2, 1e-6)
        x = torch.tensor([[[[3., 4.], [0., 2.]]]])
        expected = torch.tensor([[[[3 / (12.5 + 1e-6)**0.5, 4 / (12.5 + 1e-6)**0.5],
                                   [0., 2 / (2 + 1e-6)**0.5]]]])
        torch.testing.assert_close(norm(x), expected)

    def test_sinkhorn_and_residual_matrix_orientation(self):
        split = self.ns['split_mixing_coefficients']
        pre, post, comb = split(torch.zeros(1, 1, 8), torch.ones(3), torch.zeros(8), 2, 1e-6)
        torch.testing.assert_close(pre, torch.full_like(pre, .500001))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-2), torch.ones(1, 1, 2), atol=3e-6, rtol=0)
        residual = torch.tensor([[[[10.], [20.]]]])
        # First matrix axis is INPUT copy; second is OUTPUT copy.
        matrix = torch.tensor([[[[.8, .2], [.3, .7]]]])
        result = self.ns['combine_residual_streams'](torch.tensor([[[2.]]]), residual,
                                  torch.tensor([[[1., 2.]]]), matrix)
        torch.testing.assert_close(result, torch.tensor([[[[16.], [20.]]]]))
        with self.assertRaises(ValueError):
            split(torch.zeros(1, 1, 7), torch.ones(3), torch.zeros(8), 2, 1e-6)

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
        block = self.ns['TransformerBlock'](2, 2, attention, ffn, 1e-6, 1e-6)
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
        stack = self.ns['CausalEncoderDecoder']([first, second], 1, 2, 2, 1e-6)
        stack.output_norm = torch.nn.Identity()
        output, boundary = stack(torch.tensor([[[1., 2.]]]))
        torch.testing.assert_close(first.seen, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(second.seen, torch.tensor([[[0., 1.]]]))
        torch.testing.assert_close(boundary, torch.tensor([[[[2., 3.], [3., 4.]]]]))
        torch.testing.assert_close(output, torch.tensor([[[5., 6.]]]))
        with self.assertRaises(ValueError):
            self.ns['CausalEncoderDecoder']([first, second], 0, 2, 2, 1e-6)

    def test_moe_bias_selects_but_does_not_weight_and_shared_always_runs(self):
        class ConstantExpert(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, x, weights):
                return torch.full_like(x, self.value) * weights
        moe = self.ns['MixtureOfExperts'](2, 3, 3, 2, 1.0, 1.5, 10.0)
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
        with self.assertRaises(ValueError):
            self.ns['MixtureOfExperts'](2, 3, 3, 1, 1.0, 1.5, 10.0)

    def test_expert_always_uses_asymmetric_checkpoint_clamps(self):
        expert = self.ns['SwiGLUExpert'](1, 1, 10.0)
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
                blocks = [self.ns['TransformerBlock'](
                    4, 2, torch.nn.Linear(4, 4, bias=False),
                    self.ns['MixtureOfExperts'](4, 8, 3, 2, 1.0, 1.5, 10.0),
                    1e-6, 1e-6) for _ in range(2)]
                model = self.ns['CausalEncoderDecoder'](blocks, 1, 4, 2, 1e-6).to(dtype)
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
        mixer = self.ns['HyperConnectionMixer'](4, 2, 1e-6, 1e-6)
        residual = torch.randn(1, 3, 2, 4)
        expected = mixer(residual)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = mixer(residual)
        for result, reference in zip(actual, expected):
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, reference)
        norm = self.ns['RMSNorm'](4, 1e-6).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1., 2., 3., 4.]))
        inputs = torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16)
        reference = (inputs.float() / (inputs.float().square().mean(-1, keepdim=True) + 1e-6).sqrt())
        reference = (reference * norm.weight.float()).to(inputs.dtype)
        torch.testing.assert_close(norm(inputs), reference, rtol=0, atol=0)

    def test_invalid_dimensions_and_nonfinite_constructor_scalars(self):
        constructors = [
            lambda: self.ns['RMSNorm'](0, 1e-6),
            lambda: self.ns['RMSNorm'](4, float('nan')),
            lambda: self.ns['HyperConnectionMixer'](4, 0, 1e-6, 1e-6),
            lambda: self.ns['HyperConnectionMixer'](4, 2, 1e-6, float('inf')),
            lambda: self.ns['SwiGLUExpert'](4, -1, 10.),
            lambda: self.ns['SwiGLUExpert'](4, 8, float('nan')),
            lambda: self.ns['MixtureOfExperts'](4, 8, 3, 2, float('nan'), 1.5, 10.),
            lambda: self.ns['MixtureOfExperts'](4, 8, 3, 2, 1., float('inf'), 10.),
            lambda: self.ns['MixtureOfExperts'](4, 8, 3, 2.5, 1., 1.5, 10.),
            lambda: self.ns['CausalEncoderDecoder']([], 1, 4, 2, 1e-6),
        ]
        for construct in constructors:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()


if __name__ == '__main__':
    unittest.main()
