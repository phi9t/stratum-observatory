"""CPU COMPONENT parity, never a substitute whole-model fixture.
Execute exact AST definitions from pinned upstream files without their GPU import glue.
No function body is changed. The full publisher model still cannot run here.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '.sources-cache' / 'kimi-k3'
TARGET = Path(__file__).with_name('component-fixture.json')

def definitions(path, names, namespace):
    text = path.read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    # Future annotations prevent imports needed only for type hints.
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return hashlib.sha256(text.encode()).hexdigest()

def pack(x):
    return {'shape': list(x.shape), 'data': x.detach().float().flatten().tolist()}

def perturb(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * .15 + (.9 if 'norm' in name else .02))

def weights(model, prefix):
    return {prefix + '.' + name: pack(p) for name, p in model.named_parameters()}

def main():
    torch.manual_seed(73)
    ns = {'torch':torch,'nn':nn,'F':F,'math':math,'ACT2FN':{},'ALL_ATTENTION_FUNCTIONS':{},'KimiLinearConfig':SimpleNamespace}
    sha = definitions(CACHE/'reference/modeling_kimi_linear.py', ['SituAndMul','_get_situ_activation_params','KimiRMSNorm','KimiMLP','KimiBlockSparseMLP','KimiMoEGate','KimiSparseMoeBlock','repeat_kv','eager_attention_forward','KimiMLAAttention','_apply_attn_res'], ns)
    config = dict(hidden_size=12, intermediate_size=16, moe_intermediate_size=8, num_experts=6, num_experts_per_token=2, num_shared_experts=2, routed_expert_hidden_size=6, latent_moe_use_norm=True, hidden_act='situ', activation_situ_beta=4., activation_situ_linear_beta=25., rms_norm_eps=1e-5, routed_scaling_factor=1., moe_router_activation_func='sigmoid', num_expert_group=1,topk_group=1,moe_renormalize=True,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=6,kv_lora_rank=4,v_head_dim=4,qk_nope_head_dim=4,qk_rope_head_dim=2,mla_use_nope=True,mla_use_output_gate=True,_attn_implementation='eager')
    c=SimpleNamespace(**config)
    x=torch.randn(1,3,12)*.7
    moe=ns['KimiSparseMoeBlock'](c).eval();perturb(moe)
    mla=ns['KimiMLAAttention'](c,0).eval();perturb(mla)
    mask=torch.where(torch.tril(torch.ones(3,3,dtype=torch.bool)),0.,float('-inf'))[None,None]
    ids,w=moe.gate(x)
    gain=ns['KimiRMSNorm'](12,eps=c.rms_norm_eps);proj=nn.Linear(12,1,bias=False);perturb(gain);perturb(proj)
    bank=torch.randn(3,2,12);prefix=x.reshape(3,12)
    result={'scope':'CPU components only; full decoder logits and fused GPU parity unverified','source_sha256':sha,'config':config,'x':pack(x),'moe':{'weights':weights(moe,'moe'),'output':pack(moe(x)),'ids':pack(ids),'mixture_weights':pack(w)},'mla':{'weights':weights(mla,'mla'),'output':pack(mla(x,attention_mask=mask))},'depth':{'bank':pack(bank),'gain':pack(gain.weight),'proj':pack(proj.weight),'output':pack(ns['_apply_attn_res'](prefix,bank,proj,gain))}}
    kns={'torch':torch};result['kda_source_sha256']=definitions(CACHE/'kda-math/fla/ops/kda/naive.py',['naive_recurrent_kda'],kns)
    gns={'torch':torch,'F':F};result['gate_source_sha256']=definitions(CACHE/'kda-gate/fla/ops/kda/gate.py',['naive_kda_lowerbound_gate'],gns)
    q=torch.randn(1,3,2,4);k=torch.randn_like(q);v=torch.randn_like(q);g=torch.randn_like(q);beta=torch.rand(1,3,2);a=torch.randn(2);dt=torch.randn(8)
    gate=gns['naive_kda_lowerbound_gate'](g,a,dt,lower_bound=-5.)
    o,state=kns['naive_recurrent_kda'](q,k,v,gate,beta,output_final_state=True)
    result['kda']={key:pack(val) for key,val in dict(q=q,k=k,v=v,g=g,beta=beta,a=a,dt=dt,gate=gate,output=o,state=state).items()}
    TARGET.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(f'{TARGET}: {TARGET.stat().st_size} bytes; upstream CPU components only')

if __name__=='__main__':main()
