# STRATUM Observatory — design contract

## Purpose
A comparative scientific instrument: inspect the same tensor-to-system causal chain across DeepSeek V2/V3, Qwen3 dense/MoE, Qwen3.8 dense/Flash-Next, and Kimi K3. The deliverable is a portable offline HTML application and reusable source. Eight exact checkpoint presets, not invented family-wide architectures.

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
Seven lenses (architecture, compare, training, serving, hardware, network, evidence), exact preset selector, layer/motif selection, shared context/batch/precision/hardware controls, component inspector, dependency trace, focus mode, zoom, keyboard, URL restore, portable SVG/JSON exports. No decorative fake control.

## Release
Run numeric and semantic tests, existing verifier self-tests, deterministic build twice and browser tests against the resulting HTML. Check representative images manually. Export fresh engineering results and separate blocked scientific/runtime reports. No certificates or source hashes are invented.
