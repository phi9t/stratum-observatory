"""Reconcile scalar counts using tensor NAMES only; never fetch weights or headers.
Shapes derive from pinned constructors, config and FP8 block dimensions. Unknown
names fail closed. Run from schematics; output retained under model directory.
"""
import json,math,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
cache=root/'.sources-cache/glm-5.3'
c=json.loads((cache/'config/config.json').read_text())
index_file=json.loads((cache/'tensor-index/model.safetensors.index.json').read_text());index=index_file['weight_map']
h=c['hidden_size'];v=c['vocab_size'];heads=c['num_attention_heads'];q=c['q_lora_rank'];kv=c['kv_lora_rank'];rope=c['qk_rope_head_dim'];nope=c['qk_nope_head_dim'];vd=c['v_head_dim'];ix=c['index_head_dim'];ih=c['index_n_heads']
att={'q_a_proj':(q,h),'q_b_proj':(heads*(nope+rope),q),'kv_a_proj_with_mqa':(kv+rope,h),'kv_b_proj':(heads*(nope+vd),kv),'o_proj':(h,heads*vd),'q_a_layernorm':(q,),'kv_a_layernorm':(kv,),'indexer.wq_b':(ih*ix,q),'indexer.wk':(ix,h),'indexer.weights_proj':(ih,h),'indexer.k_norm':(ix,)}
def shape(name):
 if name in ('model.embed_tokens.weight','lm_head.weight'):return(v,h)
 if name=='model.norm.weight':return(h,)
 parts=name.split('.');layer=int(parts[2]);sub='.'.join(parts[3:-1]);suffix=parts[-1]
 if sub in ('input_layernorm','post_attention_layernorm','enorm','hnorm','shared_head.norm'):return(h,)
 if sub=='eh_proj':return(h,2*h)
 if sub.startswith('self_attn.'):return att[sub[len('self_attn.'):]]
 if name.endswith('e_score_correction_bias'):return(c['n_routed_experts'],)
 if sub=='mlp.gate':return(c['n_routed_experts'],h)
 if sub.startswith('mlp.experts.') or sub.startswith('mlp.shared_experts.'):
  width=c['moe_intermediate_size']*(c['n_shared_experts'] if 'shared_experts' in sub else 1)
 elif sub.startswith('mlp.'):
  width=c['intermediate_size']
 else:raise ValueError(name)
 proj=sub.split('.')[-1]
 if proj in ('gate_proj','up_proj'):return(width,h)
 if proj=='down_proj':return(h,width)
 raise ValueError(name)
totals=dict(decoder_trainable=0,mtp_trainable=0,correction_buffers=0,fp8_scales=0)
for name in index:
 dims=shape(name)
 if name.endswith('weight_scale_inv'):
  if len(dims)!=2:raise ValueError(name)
  block=c['quantization_config']['weight_block_size'];count=math.ceil(dims[0]/block[0])*math.ceil(dims[1]/block[1]);key='fp8_scales'
 else:
  count=math.prod(dims)
  key='correction_buffers' if name.endswith('e_score_correction_bias') else 'mtp_trainable' if name.startswith('model.layers.78.') else 'decoder_trainable'
 totals[key]+=count
hub=json.loads((cache/'model-info.json').read_text())['safetensors']['total']
result=dict(method='Constructor-shape inference over every pinned tensor-index name; no weights downloaded',tensor_names=len(index),totals=totals,total=sum(totals.values()),hub_total=hub,match=totals['decoder_trainable']+totals['mtp_trainable']+totals['correction_buffers']==hub)
hub_info=json.loads((cache/'model-info.json').read_text())['safetensors']['parameters']
result['storage_bytes']=hub_info['BF16']*2+hub_info['F8_E4M3']+hub_info['F32']*4+totals['fp8_scales']*4
result['index_storage_bytes']=index_file['metadata']['total_size']
result['storage_match']=result['storage_bytes']==result['index_storage_bytes']
result['hub_accounted']=totals['decoder_trainable']+totals['mtp_trainable']+totals['correction_buffers']
print(json.dumps(result,indent=2));assert result['match'] and result['storage_match']
result['bindings']={str(p.relative_to(cache)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cache/'config/config.json',cache/'tensor-index/model.safetensors.index.json',cache/'model-info.json',cache/'reference/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py',cache/'serving-mtp/vllm/models/deepseek_v32/nvidia/mtp.py']}
path=Path(__file__).with_name('storage-audit.json')
if '--check' in sys.argv: assert json.loads(path.read_text())==result
else: path.write_text(json.dumps(result,indent=2)+'\n')
