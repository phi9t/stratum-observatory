"""Execute pinned upstream source, never pretrained weights. Writes a CPU teaching fixture."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
import transformers
root = Path(__file__).resolve().parents[3]
cache = root / '.sources-cache/glm-5.3'
base = 'transformers.models.glm_moe_dsa'
for name, source in [('configuration_glm_moe_dsa','configuration'),('modeling_glm_moe_dsa','reference')]:
    spec = importlib.util.spec_from_file_location(base+'.'+name,cache/source/'src/transformers/models/glm_moe_dsa'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
Config = sys.modules[base+'.configuration_glm_moe_dsa'].GlmMoeDsaConfig
Model = sys.modules[base+'.modeling_glm_moe_dsa'].GlmMoeDsaForCausalLM
config = Config(vocab_size=24,hidden_size=16,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=2,n_shared_experts=1,n_routed_experts=4,num_experts_per_tok=2,q_lora_rank=8,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,index_n_heads=8,index_head_dim=8,index_topk=3,mlp_layer_types=['dense','dense','sparse','sparse','sparse','sparse'],indexer_types=['full','full','full','shared','full','shared'],rope_parameters={'rope_type':'default','rope_theta':8000000},pad_token_id=None)
config._attn_implementation='eager'
config._experts_implementation='eager'
torch.manual_seed(20)
model=Model(config).eval()
with torch.no_grad():
    for p in model.parameters(): p.add_(0.05*torch.randn_like(p))
    for layer in model.model.layers:
        if hasattr(layer.mlp,'gate'): layer.mlp.gate.e_score_correction_bias.copy_(0.2*torch.randn(4))
def dump(t): return dict(shape=list(t.shape),data=t.detach().flatten().tolist())
trace={}
for i,layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(lambda m,a,o,i=i: trace.update({f'l{i}.indices':dump(o[2])}))
    layer.self_attn.q_a_layernorm.register_forward_hook(lambda m,a,o,i=i: trace.update({f'l{i}.qnorm':dump(o)}))
    if hasattr(layer.mlp,'gate'):
        layer.mlp.gate.register_forward_hook(lambda m,a,o,i=i: trace.update({f'l{i}.router_logits':dump(o[0]),f'l{i}.route_weights':dump(o[1])}))
ids=torch.tensor([[3,7,2,11,5,9]])
with torch.no_grad(): output=model(ids,use_cache=False,output_attentions=True)
for i,t in enumerate(output.attentions): trace[f'l{i}.attention']=dump(t)
weights={n:dump(p) for n,p in model.named_parameters()}
buffers={n:dump(t) for n,t in model.named_buffers() if n.endswith('e_score_correction_bias')}
fixture=dict(upstream='pinned GlmMoeDsaForCausalLM',sources=json.loads((root/'models/glm-5.3/sources.json').read_text()),transformers_version=transformers.__version__,input_ids=ids.tolist(),weights=weights,buffers=buffers,logits=dump(output.logits),trace=trace)
path=Path(__file__).with_name('fixture.json'); path.write_text(json.dumps(fixture,separators=(',',':'))+'\n'); print(path,path.stat().st_size)
