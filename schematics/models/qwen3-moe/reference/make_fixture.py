"""CPU fixture from pinned transformers v4.51.3; no checkpoint weights loaded.
Run with PYTHONPATH=.sources-cache/qwen3-moe/python and SCHEMATICS_PYTHON.
Dependencies are isolated in the ignored source cache, never installed globally.
"""
import hashlib
import inspect
import json
from pathlib import Path
import torch
import transformers
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe import modeling_qwen3_moe, configuration_qwen3_moe
from transformers import modeling_rope_utils

ROOT = Path(__file__).resolve().parents[3]
SOURCES = json.loads((Path(__file__).parents[1] / "sources.json").read_text())
assert transformers.__version__ == "4.51.3"
bindings = {}
for sid, module in [("reference", modeling_qwen3_moe), ("configuration", configuration_qwen3_moe), ("rope-utils", modeling_rope_utils)]:
    entry = next(s for s in SOURCES["sources"] if s["id"] == sid)
    pinned = ROOT / ".sources-cache/qwen3-moe" / sid / entry["path"]
    actual = Path(inspect.getfile(module)).read_bytes()
    assert actual == pinned.read_bytes(), f"{sid} implementation differs from pinned source"
    bindings[sid] = dict(revision=entry["revision"], sha256=hashlib.sha256(actual).hexdigest())

TOY = dict(vocab_size=32, hidden_size=12, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=2, head_dim=4,
           intermediate_size=24, moe_intermediate_size=8, num_experts=4,
           num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
           mlp_only_layers=[], rope_theta=1000000, rms_norm_eps=1e-6,
           attention_bias=False, attention_dropout=0, use_sliding_window=False,
           tie_word_embeddings=False)
torch.set_num_threads(1)
torch.manual_seed(17)
config = Qwen3MoeConfig(**TOY)
config._attn_implementation = "eager"
model = Qwen3MoeForCausalLM(config).float().eval()
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.12 * torch.randn_like(p))
ids = torch.tensor([[3, 7, 11, 2, 19, 5], [8, 1, 13, 9, 4, 20]])
with torch.no_grad():
    output = model(ids, use_cache=False, output_attentions=True, output_router_logits=True)

def dump(t):
    return dict(shape=list(t.shape), data=t.flatten().tolist())
routers = []
for logits in output.router_logits:
    scores = torch.softmax(logits, dim=-1)
    weights, selected = torch.topk(scores, config.num_experts_per_tok, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    routers.append(dict(logits=dump(logits), scores=dump(scores), selected=dump(selected), weights=dump(weights)))
fixture = dict(upstream="transformers.Qwen3MoeForCausalLM", version=transformers.__version__, bindings=bindings,
               config=TOY, input_ids=ids.tolist(), weights={n:dump(p) for n,p in model.state_dict().items()},
               logits=dump(output.logits), attention=[dump(t) for t in output.attentions], routers=routers)
path = Path(__file__).with_name("fixture.json")
path.write_text(json.dumps(fixture, separators=(",", ":"))+"\n")
print(f"{path}: {path.stat().st_size} bytes; {len(fixture['weights'])} tensors")
