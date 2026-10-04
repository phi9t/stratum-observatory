"""Execute unchanged pinned publisher module definitions on CPU, never trained weights.
AST selection removes legacy transformers import/decorator glue, not module arithmetic.
Embedding/final norm/head are assembled as the publisher Model/CausalLM constructors do.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '.sources-cache/kimi-k2/reference/modeling_deepseek.py'
CONFIG = ROOT / '.sources-cache/kimi-k2/config/config.json'
TOY = dict(vocab=24, hidden=12, layers=4, heads=2, qRank=6, kvRank=4,
           nope=4, rope=4, v=4, dense=16, expert=6, experts=6, shared=1,
           topK=2, firstDense=2, eps=1e-6, theta=50000., factor=32.,
           original=4096, betaFast=1., betaSlow=1., mscale=1., allDim=1., scale=2.827)
KEEP = {'DeepseekV3RMSNorm','DeepseekV3RotaryEmbedding','DeepseekV3LinearScalingRotaryEmbedding',
        'DeepseekV3DynamicNTKScalingRotaryEmbedding','DeepseekV3YarnRotaryEmbedding',
        'DeepseekV3MLP','MoEGate','DeepseekV3MoE','DeepseekV3Attention','DeepseekV3DecoderLayer',
        'yarn_find_correction_dim','yarn_find_correction_range','yarn_get_mscale',
        'yarn_linear_ramp_mask','rotate_half','apply_rotary_pos_emb'}
text = SOURCE.read_text()
tree = ast.parse(text)
selected = [node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in KEEP]
# Evaluated type hints and framework symbols are provided without changing definitions.
namespace = dict(globals(), ACT2FN={'silu': F.silu}, DeepseekV3Config=SimpleNamespace,
                 Cache=object, dist=torch.distributed)
# ATTENTION_CLASSES is needed only when constructing the decoder, after definition loading.
exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),namespace)
namespace['ATTENTION_CLASSES']={'eager':namespace['DeepseekV3Attention']}
base=json.loads(CONFIG.read_text());base.pop('quantization_config',None)
base.update(vocab_size=TOY['vocab'], hidden_size=TOY['hidden'],num_hidden_layers=TOY['layers'],
            num_attention_heads=TOY['heads'],q_lora_rank=TOY['qRank'],kv_lora_rank=TOY['kvRank'],
            qk_nope_head_dim=TOY['nope'],qk_rope_head_dim=TOY['rope'],v_head_dim=TOY['v'],
            intermediate_size=TOY['dense'],moe_intermediate_size=TOY['expert'],n_routed_experts=TOY['experts'],
            num_experts_per_tok=TOY['topK'],first_k_dense_replace=TOY['firstDense'],ep_size=1,
            max_position_embeddings=16,_attn_implementation='eager')
cfg=SimpleNamespace(**base)
torch.manual_seed(17)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(cfg.vocab_size,cfg.hidden_size)
        self.model.layers=nn.ModuleList([namespace['DeepseekV3DecoderLayer'](cfg,i) for i in range(cfg.num_hidden_layers)])
        self.model.norm=namespace['DeepseekV3RMSNorm'](cfg.hidden_size,eps=cfg.rms_norm_eps)
        self.lm_head=nn.Linear(cfg.hidden_size,cfg.vocab_size,bias=False)
model=Decoder().eval()
with torch.no_grad():
    for name,p in model.named_parameters():
        # Explicit finite initialization also replaces upstream empty correction biases.
        p.copy_(torch.randn_like(p)*0.15+(1.0 if 'norm.weight' in name else 0.0))
ids=torch.tensor([[3,7,2,11,5],[8,1,6,4,9]])
seq=ids.shape[1]
mask=torch.zeros(seq,seq).masked_fill(torch.triu(torch.ones(seq,seq,dtype=torch.bool),diagonal=1),float('-inf'))
mask=mask[None,None].expand(ids.shape[0],1,seq,seq)
positions=torch.arange(seq)[None]
trace={}
def hook(key):
    def save(module,inp,out):
        trace[key]=out.detach() if torch.is_tensor(out) else out
    return save
model.model.layers[0].self_attn.kv_a_layernorm.register_forward_hook(hook('h0.attn.latent'))
model.model.layers[2].mlp.gate.register_forward_hook(hook('router2'))
with torch.no_grad():
    x=model.model.embed_tokens(ids)
    for i,layer in enumerate(model.model.layers):
        outputs=layer(x,attention_mask=mask,position_ids=positions,output_attentions=True)
        x=outputs[0]
        trace[f'h{i}.attn.weights']=outputs[1]
        trace[f'h{i}.out']=x
    logits=model.lm_head(model.model.norm(x)).float()
def dump(t):
    return dict(shape=list(t.shape),data=[round(v,8) for v in t.flatten().tolist()])
ridx,rweight=trace.pop('router2')
fixture=dict(upstream='pinned publisher eager modules (unchanged AST definitions)',
             source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),torch_version=torch.__version__,
             config=TOY,input_ids=ids.tolist(),weights={n:dump(p) for n,p in model.named_parameters()},
             logits=dump(logits),trace={k:dump(v) for k,v in trace.items()},router2=dict(indices=dump(ridx),weights=dump(rweight)))
path=Path(__file__).with_name('fixture.json');path.write_text(json.dumps(fixture,separators=(',',':'))+'\n')
print(f'Wrote {path}: {path.stat().st_size} bytes, {len(fixture["weights"])} parameters')
