"""Writes fixture.json: a toy-width GPT-2 from the upstream implementation, with its
weights, one input and the logits and attention weights it produces.

The TypeScript reference in model.ts must reproduce these numbers
(tests/reference.test.ts). Run with any Python that has torch and transformers:

    python models/gpt2/reference/make_fixture.py
"""

import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

TOY = dict(vocab_size=40, n_positions=16, n_embd=24, n_layer=2, n_head=3)

torch.manual_seed(0)
config = GPT2Config(**TOY, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
config._attn_implementation = "eager"
model = GPT2LMHeadModel(config).eval()
# The default init leaves biases at zero and norms at identity; perturb everything so
# a wrong bias or gain cannot hide.
with torch.no_grad():
    for parameter in model.parameters():
        parameter.add_(0.05 * torch.randn_like(parameter))

input_ids = torch.tensor([[3, 17, 5, 29, 11, 0, 38, 8]])
with torch.no_grad():
    output = model(input_ids, output_attentions=True)


def dump(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "data": [round(v, 7) for v in tensor.flatten().tolist()]}


weights = {
    name: dump(parameter)
    for name, parameter in model.state_dict().items()
    if not name.endswith(".attn.bias") and not name.endswith("masked_bias") and name != "lm_head.weight"
}
fixture = {
    "upstream": "transformers.GPT2LMHeadModel",
    "transformers_version": __import__("transformers").__version__,
    "config": TOY,
    "input_ids": input_ids.tolist(),
    "weights": weights,
    "logits": dump(output.logits),
    "attention_layer0": dump(output.attentions[0]),
}
path = Path(__file__).with_name("fixture.json")
path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(f"wrote {path} ({path.stat().st_size} bytes, {len(weights)} tensors)")
