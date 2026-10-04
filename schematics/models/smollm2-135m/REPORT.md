# SmolLM2-135M schematic report

Built for **HuggingFaceTB/SmolLM2-135M**, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Source pins and reading notes are
in sources.json and NOTES.md. No shared kit, tool, root application or verifier edits.

## Delivered

Self-contained `schematics/dist/smollm2-135m.html`: ten sections covering whole
circuit, tensors, every layer, residual block, operation-level attention and MLP,
computed toy experiments, parameter ledger, serving and provenance. Shared embedding
ownership is drawn explicitly. Layer, op, query, rotary position, toy head/layer/seed,
batch and context controls use URL state. Figures scroll inside their panels on mobile.

The exact derived ledger is **134515008**, equal to the Hub metadata and the clean
reference's real-width shape-mode count. CPU float32 parity against the pinned
Transformers modeling file passes with perturbed random toy weights: max absolute
logits difference **1.7881393432617188e-7**, attention difference
**2.9802322387695312e-8**. The toy uses two layers, width 24, six query heads,
two KV heads; its three-to-one sharing ratio matches the checkpoint. Unique parameter
names agree, including the absence of a separate tied head.

Fixture generation ran with matching Transformers 4.51.3 in an isolated uv environment
using the existing CPU torch installation. Reproduce with a Python having that version:
`SCHEMATICS_PYTHON=/path/to/python`; run `$SCHEMATICS_PYTHON models/smollm2-135m/reference/make_fixture.py`
from schematics. The default shared Python had incompatible Transformers 5.17.0;
no upstream source was patched to work around it.

## Scope and limits

Numerical execution covers unpadded prefill to all-position logits. Diagrams additionally
explain logical cached decoding; decode parity, padded/static cache fixtures, kernel
rounding, GPU execution, throughput and trained-checkpoint behavior are unverified.
No trained weights were downloaded or bundled. Fixture tensors are newly initialized
toy parameters, not weights from the requested checkpoint.

Sampling, tokenizer behavior, training losses, alternative task heads, quantized kernels,
LoRA and other checkpoint variants are excluded. These are not branches in the drawn
base-checkpoint forward pass. Reference class exclusions are explicit in coverage.json.
Transformers v4.51.3 is the pinned page reference, not a claim of historical training
version. Publisher README names Nanotron; current pinned Nanotron supplies a comparison,
not a historical training receipt or numerical parity guarantee. Its optional muP scaling
and current noninterleaved inference branch are not used as the parity oracle (NOTES.md).
No unavailable gate is labelled PASS; no gate was skipped. Root engineering/science/runtime
commands are outside this self-contained schematic subtree and were NOT_RUN.

## Browser and artifact identity

Browser: Google Chrome **145.0.7632.75**. Final layout gate checked **2033 text elements**
across desktop light/dark and mobile light. Play found **90 controls**: 38 buttons,
38 operations, six ranges, eight radios; **12/12 sampled controls** changed the page,
and reload restored their URL state. This sampling does not claim exhaustive testing of
all controls. Screenshots: **33 PNGs**, hero and all ten sections in those three views,
generated with `npm run shoot -- smollm2-135m` and visually inspected against the skill
examples. Wide figures/tables deliberately scroll inside panels; no whole-page overflow.

Artifact: **513436 bytes**; SHA-256 `c99f8e5b58e285cb20d974466bebb28c0030fa9e87433f6acf7a11e161000150`.

## Final check table

Command: `npm run check -- smollm2-135m`

| Gate | Result | Evidence |
|---|---|---|
| pins | PASS | No findings. |
| typecheck | PASS | No findings. |
| literals | PASS | No findings. |
| format | PASS | No findings. |
| tests | PASS | Tests  12 passed (12); runs the clean reference's tests |
| facts | PASS | No findings. |
| coverage | PASS | No findings. |
| parameters | PASS | parameterTotal 134515008 equals safetensors.total |
| build | PASS | dist/smollm2-135m.html 513436 bytes sha256 c99f8e5b58e285cb20d974466bebb28c0030fa9e87433f6acf7a11e161000150 |
| layout | PASS | desktop-light, desktop-dark, mobile-light: 2033 text elements checked |
| play | PASS | 90 controls (38 button, 38 op, 6 range, 8 radio); 12 of 12 sampled changed the page; reload kept #q=12&runLayer=1&ropePos=4096&runHead=2&runQ=4 |

**ok: 11 pass, 0 fail, 0 skipped.** Twelve tests passed.

## Self-review against the skill rubric

| Rubric | Result | Evidence |
|---|---|---|
| 1 exact identity | PASS | Base checkpoint and full commit shown in Sources. |
| 2 pins/facts/coverage/parameters | PASS | Final table above, no parameter exception. |
| 3 five source spot checks | PASS | q_proj/view (LlamaAttention.forward), apply_rotary_pos_emb, repeat_kv/eager_attention_forward, softmax/eager_attention_forward, gate multiply/LlamaMLP.forward; order and axes checked against pinned code. |
| 4 layer kinds | PASS | Uniform LlamaDecoderLayer construction; all live layer tiles selectable. |
| 5 serving comparison | PASS | Packed QKV/gate-up, deferred residual sums, backend cache, tied head loading and distributed KV ownership. |
| 6 numeric facts | PASS | literals/facts gates; model values imported or derived, geometry in LAYOUT. |
| 7a clean reference | PASS | Upstream fixture logits, attentions, unique parameter names and fixture source binding tested. |
| 7b real shape count | PASS | Exact Hub/ledger/reference agreement. |
| 7 detail levels | PASS | All ten required sections; inspectors supply op detail. |
| 8 attention/FFN operations | PASS | Projections/reshape, rotary angles/trig/rotation, cache, repeat, dot, scale, mask, softmax, mix/merge/output; gate/up/SiLU/multiply/down. |
| 9 inspectors | PASS | Purpose, formulas, shapes, parameter shapes where allocated, pinned implementation references. |
| 10 observations beyond config | PASS | Cache-before-repeat ordering, no Q/K head norms, shared rotary tables and serving's delayed addition. |
| 11 layer selection | PASS | Changes overview highlight, block title, input/output ownership sentences. |
| 12 mechanism widgets | PASS | Causal grid, rotary dials, executed shared-KV probability/vector comparison and rotary score/pair experiment. |
| 13 controls/state | PASS | Play gate sampled meaningful changes and reload restoration. |
| 14 screenshots | PASS | Light/dark/mobile inspected; no label collision or unreadable palette. |
| 15 lane grammar | PASS | Ink stream, blue attention, orange MLP, violet positions/state/tie, teal residual wires. |
| 16 ledes/captions | PASS | Explain each section's purpose and what each figure demonstrates. |
| 17 coherent design | PASS | Shared monograph kit, consistent headings, spacing, figures and tables. |
| 18 output-width gauges | PASS | Every graph data-path node supplies a width; score gauge prefix assumption explicit. |
| 19 study aids | PASS | Every section ends with Takeaways; mechanisms have reasoning checks. |
| 20 mechanism benefit | PASS | Three-to-one cache savings, tied-head parameter saving and MLP width shown from facts. |
| 21 glossary | PASS | Defines stream, norm, heads, mask, cache, GQA, dot products, RoPE, SiLU, tying and execution terms. |
| 22 honesty | PASS | Scope, exclusions, unverified items and no skipped gates recorded above. |
