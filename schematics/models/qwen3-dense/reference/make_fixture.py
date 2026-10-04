"""Generate CPU float32 upstream parity, without checkpoint weights.
Imports the pinned modeling file directly; framework dependencies are installed transformers.
"""
import importlib.util
import json
import hashlib
from pathlib import Path
import torch
import transformers
from transformers import Qwen3Config
ROOT = Path(__file__).resolve().parents[3]
source = ROOT / '.sources-cache/qwen3-dense/reference/src/transformers/models/qwen3/modeling_qwen3.py'
spec = importlib.util.spec_from_file_location('transformers.models.qwen3.pinned_qwen3', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TOY = dict(vocab_size=40, hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=6, max_position_embeddings=16, rope_theta=1000000, rms_norm_eps=1e-6, tie_word_embeddings=False, use_sliding_window=False, sliding_window=None, attention_bias=False, attention_dropout=0.0)
config = Qwen3Config(**TOY)
config._attn_implementation = 'eager'
torch.manual_seed(0)
model = module.Qwen3ForCausalLM(config).float().eval()
with torch.no_grad():
    for p in model.parameters(): p.add_(0.05 * torch.randn_like(p))
ids = torch.tensor([[3,17,5,29,11,0,38,8]])
with torch.no_grad(): output = model(ids, use_cache=False, output_attentions=True)
def dump(t): return dict(shape=list(t.shape), data=t.flatten().tolist())
result = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_revision=json.loads((ROOT / 'models/qwen3-dense/sources.json').read_text())['sources'][1]['revision'], upstream='pinned transformers Qwen3ForCausalLM', transformers_version=transformers.__version__, config=TOY, input_ids=ids.tolist(), weights={k:dump(v) for k,v in model.state_dict().items()}, logits=dump(output.logits), attention_layer0=dump(output.attentions[0]))
Path(__file__).with_name('fixture.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print('wrote upstream fixture')
