"""Independent scalar oracles for the image teaching path (no upstream execution)."""
from dataclasses import replace
import math
import unittest

import torch
from examples.deepseek_v41.config import V41_CONFIG
from examples.deepseek_v41.vision_inputs import (VisionPatchEmbed, VisionAttention,
    VisionBlock, VisionAligner, VisionInputs, vision_positions)
from examples.deepseek_v41.vision_example import inputs_example


def small_config():
    return replace(V41_CONFIG, hidden_dim=3, vision_dim=4, vision_heads=1,
                   vision_hidden_dim=3, vision_layers=1, vision_patch_size=2)


def linear(x, layer):
    return [[sum(a * b for a, b in zip(row, weight)) + (layer.bias[j].item() if layer.bias is not None else 0)
             for j, weight in enumerate(layer.weight.tolist())] for row in x]


def norm(x, layer):
    return [[v / math.sqrt(sum(t*t for t in row)/len(row)+layer.epsilon) * w
             for v, w in zip(row, layer.weight.tolist())] for row in x]


def attention_oracle(x, module, height, width):
    cfg = module.config
    dim = cfg.vision_dim // cfg.vision_heads
    qkv = linear(x, module.qkv)
    output = []
    q, k, v = [], [], []
    for part, destination in enumerate((q, k, v)):
        for position, row in enumerate(qkv):
            heads = []
            for head in range(cfg.vision_heads):
                vector = row[part*cfg.vision_dim+head*dim:part*cfg.vision_dim+(head+1)*dim]
                if part != 2:
                    half = dim//2
                    angles = [coord / cfg.vision_rope_theta**(2*i/half)
                              for coord in (position//width, position%width) for i in range(half//2)]
                    vector = ([vector[i]*math.cos(a)-vector[i+half]*math.sin(a) for i,a in enumerate(angles)] +
                              [vector[i+half]*math.cos(a)+vector[i]*math.sin(a) for i,a in enumerate(angles)])
                heads.append(vector)
            destination.append(heads)
    for query in q:
        merged = []
        for head in range(cfg.vision_heads):
            scores = [sum(a*b for a,b in zip(query[head], key[head]))/math.sqrt(dim) for key in k]
            weights = [math.exp(s-max(scores)) for s in scores]
            merged.extend(sum(w*value[head][i] for w,value in zip(weights,v))/sum(weights) for i in range(dim))
        output.append(merged)
    return linear(output, module.output)


class VisionInputTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.cfg = small_config()

    def test_patch_flatten_is_channel_then_row_then_column(self):
        patch = VisionPatchEmbed(self.cfg)
        pixels = torch.arange(48, dtype=torch.float32).reshape(4,3,2,2)/10
        expected = linear(pixels.reshape(4,-1).tolist(), patch.projection)
        torch.testing.assert_close(patch(pixels), torch.tensor(expected))

    def test_attention_is_bidirectional_with_split_half_two_axis_rope(self):
        module = VisionAttention(replace(self.cfg, vision_dim=16, vision_heads=2))
        x = torch.arange(96, dtype=torch.float32).reshape(6,16)/30
        cos, sin = vision_positions(module.config, 2,3, x.device)
        expected = torch.tensor(attention_oracle(x.tolist(), module, 2,3))
        torch.testing.assert_close(module(x,cos,sin), expected, atol=2e-6, rtol=2e-5)
        altered = x.clone(); altered[-1] += 2
        self.assertFalse(torch.allclose(module(x,cos,sin)[0], module(altered,cos,sin)[0]))

    def test_block_pre_norm_residual_and_swiglu_order(self):
        block = VisionBlock(self.cfg)
        x = torch.tensor([[1.,2.,-1.,3.],[-2.,1.,3.,.5]])
        cos,sin = vision_positions(self.cfg, 1,2,x.device)
        attention = attention_oracle(norm(x.tolist(),block.norm1), block.attention,1,2)
        residual = [[a+b for a,b in zip(row,out)] for row,out in zip(x.tolist(),attention)]
        projected = linear(norm(residual,block.norm2),block.mlp.gate_up)
        activated = [[g/(1+math.exp(-g))*u for g,u in zip(row[:3],row[3:])] for row in projected]
        ffn = linear(activated,block.mlp.down)
        expected = [[a+b for a,b in zip(row,out)] for row,out in zip(residual,ffn)]
        torch.testing.assert_close(block(x,cos,sin),torch.tensor(expected))

    def test_aligner_pads_and_concatenates_channel_first_then_exact_gelu(self):
        module = VisionAligner(self.cfg)
        height,width = 4,5
        x = torch.arange(height*width*4,dtype=torch.float32).reshape(-1,4)/100
        grouped = [[x[(top+h)*width+left+w,c].item() if top+h<height and left+w<width else 0
                    for c in range(4) for h in range(3) for w in range(3)]
                   for top in range(0,height,3) for left in range(0,width,3)]
        torch.testing.assert_close(module.group_patches(x,height,width),torch.tensor(grouped))
        hidden = linear(grouped,module.input)
        activated = [[v*.5*(1+math.erf(v/math.sqrt(2))) for v in row] for row in hidden]
        torch.testing.assert_close(module(x,height,width),torch.tensor(linear(activated,module.output)))

    def test_end_to_end_gradient_and_shape(self):
        module = VisionInputs(self.cfg)
        patches = torch.arange(48,dtype=torch.float32).reshape(4,3,2,2).div(20).requires_grad_()
        output = module(patches,2,2)
        self.assertEqual(list(output.shape),[1,3])
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(patches.grad).all())
        self.assertGreater(patches.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
        epsilon = 1e-3
        plus,minus = patches.detach().clone(),patches.detach().clone()
        plus[0,0,0,0] += epsilon; minus[0,0,0,0] -= epsilon
        finite_difference = (module(plus,2,2).square().sum()-module(minus,2,2).square().sum())/(2*epsilon)
        torch.testing.assert_close(patches.grad[0,0,0,0],finite_difference,atol=3e-5,rtol=3e-3)
        with self.assertRaises(ValueError): module(patches,1,3)

    def test_trace_is_deterministic_and_ends_at_actual_module_output(self):
        first,second = inputs_example(),inputs_example()
        self.assertEqual(first,second)
        self.assertEqual(first['steps'][-1]['id'],'visual_tokens')
        self.assertEqual(first['steps'][-1]['shape'],[1,3])


if __name__ == '__main__':
    unittest.main()
