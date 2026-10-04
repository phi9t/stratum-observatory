"""Run the pinned publisher's complete CPU eager model, never downloaded weights.
Import cached source directly; no upstream source is copied into this deliverable.
SCHEMATICS_PYTHON defaults to /data03/yph/venv/bin/python in the run instructions.
"""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '.sources-cache/deepseek-v3'
package = types.ModuleType('pinned_deepseek')
package.__path__ = []
sys.modules[package.__name__] = package

def load(name, path):
    spec = importlib.util.spec_from_file_location(f'pinned_deepseek.{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

# Transformers 5 removed this tracing-only availability helper. No arithmetic
# is changed; torch.fx is installed in this CPU environment.
import transformers.utils.import_utils as import_utils
if not hasattr(import_utils, 'is_torch_fx_available'):
    import_utils.is_torch_fx_available = lambda: True

configuration = load('configuration_deepseek', CACHE / 'configuration/configuration_deepseek.py')
upstream = load('modeling_deepseek', CACHE / 'reference/modeling_deepseek.py')
TOY = dict(vocab_size=24, hidden_size=12, num_hidden_layers=4, num_attention_heads=2,
           intermediate_size=16, moe_intermediate_size=8, n_routed_experts=8,
           n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=2,
           moe_layer_freq=1, n_group=4, topk_group=2, q_lora_rank=6,
           kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
           max_position_embeddings=32, rms_norm_eps=1e-6, attention_bias=False,
           attention_dropout=0.0, hidden_act='silu', scoring_func='sigmoid',
           topk_method='noaux_tc', norm_topk_prob=True, routed_scaling_factor=2.5,
           rope_theta=10000, rope_scaling=dict(type='yarn', factor=40,
           original_max_position_embeddings=4096, beta_fast=32, beta_slow=1,
           mscale=1, mscale_all_dim=1), tie_word_embeddings=False, ep_size=1)
config = configuration.DeepseekV3Config(**TOY)
config._attn_implementation = 'eager'
torch.manual_seed(17)
model = upstream.DeepseekV3ForCausalLM(config).float().eval()
# Replace every parameter, including uninitialized correction bias. Nonidentity
# gains and nonzero biases prevent accidental omissions from passing unnoticed.
with torch.no_grad():
    for name, parameter in model.named_parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.13 + (0.8 if 'layernorm' in name or '.norm.' in name else 0))

def dump(t):
    return dict(shape=list(t.shape), data=[float(f'{x:.8g}') for x in t.flatten().tolist()])
weights = {name: dump(p) for name, p in model.named_parameters()}
with torch.no_grad():
    for name, p in model.named_parameters():
        p.copy_(torch.tensor(weights[name]['data']).view_as(p))
trace = {}
hooks = []
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.self_attn.kv_a_layernorm.register_forward_hook(lambda m, a, y, i=i: trace.update({f'h{i}.kn': dump(y)})))
    hooks.append(layer.self_attn.kv_b_proj.register_forward_hook(lambda m, a, y, i=i: trace.update({f'h{i}.kb': dump(y)})))
    hooks.append(layer.self_attn.register_forward_hook(lambda m, a, y, i=i: trace.update({f'h{i}.softmax': dump(y[1])})))
    if i >= TOY['first_k_dense_replace']:
        def capture(m, a, y, i=i):
            ids, weights = y
            # PyTorch sorted=False order is unspecified; canonicalize id/weight pairs.
            order = ids.argsort(dim=-1)
            trace[f'h{i}.topk'] = dump(ids.gather(-1, order))
            trace[f'h{i}.routeweights'] = dump(weights.gather(-1, order))
            scores = torch.nn.functional.linear(a[0].reshape(-1, TOY['hidden_size']), m.weight).sigmoid()
            corrected = scores + m.e_score_correction_bias
            gs = corrected.view(-1, TOY['n_group'], TOY['n_routed_experts']//TOY['n_group']).topk(2, dim=-1).values.sum(-1)
            trace[f'h{i}.groups'] = dump(gs.topk(TOY['topk_group'], dim=-1).indices.sort(-1).values)
        hooks.append(layer.mlp.gate.register_forward_hook(capture))
ids = [[1, 7, 3, 9, 2, 5]]
with torch.no_grad():
    result = model(torch.tensor(ids), use_cache=False, output_attentions=True)
fixture = dict(upstream='pinned modeling_deepseek.DeepseekV3ForCausalLM',
    revision=json.loads((ROOT/'models/deepseek-v3/sources.json').read_text())['checkpoint']['revision'],
    source_sha256=hashlib.sha256((CACHE/'reference/modeling_deepseek.py').read_bytes()).hexdigest(),
    torch_version=torch.__version__, transformers_version=transformers.__version__,
    config=TOY, input_ids=ids, weights=weights, logits=dump(result.logits), trace=trace)
path = Path(__file__).with_name('fixture.json')
path.write_text(json.dumps(fixture, separators=(',', ':'))+'\n')
print(f'Wrote {path}: {path.stat().st_size} bytes, {len(weights)} parameters')
