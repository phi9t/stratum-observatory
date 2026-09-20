# STRATUM Observatory — design contract

## Purpose
A comparative scientific instrument: inspect the same tensor-to-system causal chain across DeepSeek V2/V3, Qwen3 dense/MoE, Qwen3.8 dense/Flash-Next, and Kimi K3. The deliverable is a portable offline HTML application and reusable source. Nine exact checkpoint presets, not invented family-wide architectures.

## First principles
1. Shape and dependency are the invariant; presentation is a projection. Core model definitions feed the diagram, accounting and comparisons.
2. A view has one primary question. Architecture asks what is computed; training asks what persists and synchronizes; serving asks what state survives each token; hardware asks where bytes and compute live; fabric asks how ownership crosses machines; evidence asks which claims are supported.
3. Empty space is routing capacity. Operator typography is measured/checked in the browser; orthogonal paths use reserved lanes. Core labels cannot ellipsize. The canvas scrolls before text becomes tiny.
4. Visual hierarchy follows causal hierarchy. Whole model → selected layer → attention and FFN → optional state/branches. Delayed recurrence is distinguished from same-step dependencies.
5. Position is memory. Sibling operations share semantic rows across presets. Comparison shares context, precision and units.
6. Color is semantic: cyan = selected/computation, amber = retained state/assumption, violet = communication, subdued neutral = structural context. Paper and graphite use the same grammar.
7. Facts, derivations, assumptions and observations are different objects. An inspected URL is not an accepted byte snapshot; engineering success is not publication or runtime certification.

## Modules
`models.mjs`: exact preset identities, field values, provenance links and hardware identities.
`core.mjs`: validated scenario state, shape/parameter/cache/training/communication calculations. No DOM.
`scene.mjs`: visual IR, stable nodes/ports, geometric assertions.
`diagram-model.mjs`, `diagram-systems.mjs`, `diagram-meta.mjs`: domain motifs and scenes; no duplicated aggregate accounting. `diagrams.mjs` dispatches them.
`app.mjs`: navigation, layer selection, inspector, trace, comparison, evidence, URL state, SVG/JSON export.
`styles.css`: typography, tokens, responsive composition, focus and print.
`build.mjs`: deterministic local-source bundler, no external runtime assets.
`verification/`: existing publication/runtime trust policy retained; new target is explicitly a pending evidence migration.

## Scientific boundary
DeepSeek base decoder totals exclude MTP; Qwen3 variants distinguish dense vs MoE; Qwen3.8-27B is hybrid dense, Flash-Next adds QSA/PLE/hyperconnections; Kimi K3 uses vector-decay KDA, gated NoPE MLA, latent MoE and depth attention residuals. Estimated storage uses declared uniform precision, not a validated quantization recipe. Parameter and cache subtotals declare excluded vision/MTP/auxiliary states. No chosen hardware implies an actual publisher deployment.

## Interaction
Global Models, GPU Hardware and Networking navigation; four model tabs (architecture, training, serving, evidence), exact preset selector, layer/motif selection, shared context/batch/precision/hardware controls, component inspector, dependency trace, focus mode, zoom, keyboard, URL restore, portable SVG/JSON exports. No decorative fake control.

## Release
Run numeric and semantic tests, existing verifier self-tests, deterministic build twice and browser tests against the resulting HTML. Check representative images manually. Export fresh engineering results and separate blocked scientific/runtime reports. No certificates or source hashes are invented.


## Dedicated DeepSeek V4.1 delivery

`dist/deepseek-v4.1.html` is a dedicated entry point built from the shared application shell.
It fixes the checkpoint identity, opens the whole-network CED view, and puts that view first
in its architecture navigation. The landing directory opens this page; its Home link
returns to the directory. Existing model hash scenarios and the STRATUM programmatic seam remain
compatible. Serving exposes prefill/decode phase inspection with distinct participating depths.

The build emits every registered portable HTML file. Engineering binds each file digest and compares them
across repeated builds. Browser and presentation tests visit the dedicated page; captures bind
its own artifact identity. `npm start` serves the current built pages so links have the same
meaning in development, offline files, and Pages.

See [Site sections](SITE_SECTIONS.md) for canonical URLs, workload ownership and legacy bookmarks.


The landing page is a static learning directory. Modeling Foundations groups DeepSeek V2/V3
and Qwen3, while Qwen3.8, Kimi K3 and DeepSeek V4.1 Flash keep dedicated entries. GPU
Hardware and Networking are shared systems pages. Data Representation is a compact general
reference for tokenization, packing, attention versus loss masking, and storage formats;
its documentation links are unaccepted references pending source anchoring and review.
`PAGES` enumerates interactive artifacts and `GUIDES` static documents; delivery identity
covers both. Legacy scenario fragments redirect to their canonical interactive page, while
ordinary in-page anchors remain on the directory.

## Detailed causal encoder/decoder chapters

Encoder and Decoder use the shared attention/MoE block renderer. The Encoder selector
contains absolute layers 1 through `encoderLayers`; Decoder contains `encoderLayers + 1`
through `layers`. The model identity states the two counts separately. Encoder ends in
a residual handoff, while Decoder begins at that handoff and retains the final norm/head.
The shared ownership indices remain absolute across the complete backbone.

The redundant V4.1 Decoder block tab is removed. Legacy `stage=block` links resolve to
the chapter owning the requested absolute layer, preserving that layer. Explicit chapter
links clamp out-of-range layers into their own range. Both tile clicks and keyboard
selection keep the reader in the chapter. Worked explanations, mathematics, source
citations and checked-in Python remain beneath the detailed schematic.

The portable build gzip-compresses only the embedded Python-source JSON using Node's
built-in zlib. A modern browser's native DecompressionStream restores the exact UTF-8
source before the application modules initialize. This preserves readable displayed
code and byte-exact downloads within the existing single-file size budget, without
external assets or an added runtime package. Browser tests check restored code against
the checked-in files; engineering checks repeat-build byte identity.

## Worked-example reading order

The computed examples lead with the mechanism and a short visual trace. Replay follows
one dependency from an omitted token through depth to the changed target output;
selection groups source scores, candidate blocks and selected positions; the query
example places the fixed dictionary beside its different readouts. These are views of
the existing CPU artifact, with no new browser-side model arithmetic.

Complete tensors, named axes/dtypes and fixture settings are available in a closed
"Inspect CPU tensors and fixture settings" disclosure. Execution notes and source contracts follow the examples
in a separate disclosure. Print expands both and restores the reader's prior state.
Primary architecture schematics remain the first view on each chapter.

Tensor dimensions use ungrouped integers inside brackets so every comma separates
axes. Encoder and decoder chapter selectors preserve absolute layer IDs and arrange
them in the same producer/reuse bundles as the CED overview. Bundle boundaries come
from `layerKind`; they are not a second authored schedule. Every tile remains an
independent keyboard-accessible selection of the detailed attention-plus-MoE layer.

All reader-facing numbers omit thousands separators, including controls and prose.
Numeric quantities and tensor shapes use code glyphs at the surrounding type size;
existing mathematical contracts retain their code formatting. This is typography,
not a change to model arithmetic, tensor axes or computation.
