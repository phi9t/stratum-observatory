import unittest
import torch
from pytorch_test_support import load_examples


class BackboneExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_examples()

    def test_rmsnorm_uses_width_not_stream_axis(self):
        norm = self.ns['BB_RMSNorm'](2, 1e-6)
        x = torch.tensor([[[[3., 4.], [0., 2.]]]])
        expected = torch.tensor([[[[3 / (12.5 + 1e-6)**0.5, 4 / (12.5 + 1e-6)**0.5],
                                   [0., 2 / (2 + 1e-6)**0.5]]]])
        torch.testing.assert_close(norm(x), expected)

    def test_sinkhorn_and_residual_matrix_orientation(self):
        split = self.ns['bb_split_sinkhorn']
        pre, post, comb = split(torch.zeros(1, 1, 8), torch.ones(3), torch.zeros(8), 2, 1e-6)
        torch.testing.assert_close(pre, torch.full_like(pre, .500001))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-2), torch.ones(1, 1, 2), atol=3e-6, rtol=0)
        residual = torch.tensor([[[[10.], [20.]]]])
        # First matrix axis is INPUT copy; second is OUTPUT copy.
        matrix = torch.tensor([[[[.8, .2], [.3, .7]]]])
        result = self.ns['bb_post'](torch.tensor([[[2.]]]), residual,
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
        block = self.ns['BB_Block'](2, 2, attention, ffn, 1e-6, 1e-6)
        block.attn_mix = FixedMix([0., 1.])
        block.ffn_mix = FixedMix([.25, .75])
        x = torch.tensor([[[[3., 4.], [4., 3.]]]])
        output, next_pre = block(x, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(attention.seen, block.attn_norm(x[:, :, 0]))
        torch.testing.assert_close(ffn.seen, block.ffn_norm(x[:, :, 1]))
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
        stack = self.ns['BB_FlatCED']([first, second], 1, 2, 2, 1e-6)
        stack.norm = torch.nn.Identity()
        output, boundary = stack(torch.tensor([[[1., 2.]]]))
        torch.testing.assert_close(first.seen, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(second.seen, torch.tensor([[[0., 1.]]]))
        torch.testing.assert_close(boundary, torch.tensor([[[[2., 3.], [3., 4.]]]]))
        torch.testing.assert_close(output, torch.tensor([[[5., 6.]]]))
        with self.assertRaises(ValueError):
            self.ns['BB_FlatCED']([first, second], 0, 2, 2, 1e-6)

    def test_moe_bias_selects_but_does_not_weight_and_shared_always_runs(self):
        class ConstantExpert(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, x, weights):
                return torch.full_like(x, self.value) * weights
        moe = self.ns['BB_MoE'](2, 3, 3, 2, 1.0, 1.5, 10.0)
        moe.experts = torch.nn.ModuleList([ConstantExpert(1000), ConstantExpert(5), ConstantExpert(7)])
        moe.shared = ConstantExpert(3)
        with torch.no_grad():
            # Raw logits [3, 0, 1]; bias excludes the highest raw-scoring expert.
            moe.router.weight.copy_(torch.tensor([[3., 0.], [0., 0.], [1., 0.]]))
            moe.bias.copy_(torch.tensor([-100., 100., 200.]))
        scores = torch.nn.functional.softplus(torch.tensor([0., 1.])).sqrt()
        expected = 3 + 1.5 * (scores * torch.tensor([5., 7.])).sum() / scores.sum()
        torch.testing.assert_close(moe(torch.ones(1, 2, 2)), torch.full((1, 2, 2), expected))
        self.assertEqual(moe.router.out_features, 3)  # shared has no router logit
        with self.assertRaises(ValueError):
            self.ns['BB_MoE'](2, 3, 3, 1, 1.0, 1.5, 10.0)

    def test_expert_always_uses_asymmetric_checkpoint_clamps(self):
        expert = self.ns['BB_Expert'](1, 1, 10.0)
        with torch.no_grad():
            expert.gate.weight.fill_(1)
            expert.up.weight.fill_(1)
            expert.down.weight.fill_(1)
        x = torch.tensor([[20.], [-20.]])
        # Positive gate clamps at 10, negative gate remains -20;
        # up branch clamps on both sides, including -10.
        expected = torch.nn.functional.silu(torch.tensor([[10.], [-20.]])) * torch.tensor([[10.], [-10.]])
        torch.testing.assert_close(expert(x, 1.0), expected)


if __name__ == '__main__':
    unittest.main()
