# STRATUM — Kimi K3 systems atlas

## Scientific-verifier extension (v2.1)

The atlas now includes executable scientific-verification gates and domain-specific coding-agent specifications. Start with **[the verifier index](docs/verifiers/README.md)** and **[the agent runbook](docs/verifiers/AGENT_RUNBOOK.md)**. The original model, hardware and networking diagrams are unchanged. This adds validation profiles for H100/H200, B200, B300/GB300; it does not add new GPU selections to the visual interface.

**The initial scientific publication audit is BLOCKED.** Source URLs/commits have been cataloged, but raw upstream snapshots, many fine-grained code anchors, signed independent reviews, several scenario records and real execution receipts are still missing. Existing source-backed wording in the original atlas is not a certification under the new policy. See `verification/reports/publication.md` and `VERIFIER_STATUS.md`.

```sh
npm run verify:engineering  # Local verifier + atlas tests, deterministic builds, Chromium
npm run verify:science      # Source/semantic publication gate; initially exits 2
npm run verify:runtime      # Also requires declared GPU/fabric/framework executions
npm run verify              # Engineering AND publication, intentionally no bypass
```

Runtime remains dependency-free. Verifier prerequisites: Python 3.11+, Node.js 20+, OpenSSL with Ed25519 support; browser checks additionally need Python Playwright and Chromium. The CI workflow runs engineering and publication separately and makes both prerequisites for Pages deployment. No remote repository settings or reviewer identities have been configured by creating this archive.


A self-contained, source-backed web atlas of Kimi K3: **model architecture, pretraining / agentic RL, serving, accelerator hardware and networking**. Graphite and warm-paper themes; orthogonal SVG schematics; clickable equations and provenance; URL-persisted configurations; SVG / JSON exports. No telemetry or external runtime assets.

## Open

Open `dist/index.html` in a full browser. Everything is in this one file: no server, installation, API key or CDN. A restricted chat attachment preview may disable JavaScript; open the file in a browser instead.

To work on source: `npm start` serves the ES-module version at `http://localhost:8000`. Requires Node 20+ for tests/build and Python 3 for the optional server.

## Explore

**Model:** select any of the 93 decoder layers with the dropdown or miniature layer matrix. Layers 4, 8, …, 92 and 93 show gated MLA; the remaining 69 show KDA. Layer 1 alone has a dense SiTU FFN. Other layers show Stable LatentMoE: 3584-wide routed path, 896 experts/top 16, two full-width shared experts. Depth-bank reads and local partial sums implement Block AttnRes.

**Training:** switch between the report’s pretraining mechanisms and agentic-RL state lifecycles. Shows PP/virtual stages, EP/MoonEP, KCP, ZeRO-1 and Pipeline ZeRO-2, Per-Head Muon, tensor-level activation policies, external prefix retention and AgentENV. Columns group mechanisms; arrows indicate selected dependencies, not a full execution schedule.

**Serving:** vary context up to 1,048,576 tokens, batch, cache representation, precision, weight-budget scenario and replica size. Compare fixed KDA matrices and convolution buffers with growing MLA history. The independent prefix example requires a checkpoint shared by all KDA groups, then returns to prefill for the uncached prompt suffix. Decode begins only after the whole prompt is processed.

**Hardware:** documented DGX H100/H200 logical connectivity: eight GPUs, four NVSwitches, host / PCIe, eight 400Gb/s compute NICs. The dense-BF16 roofline is an ideal bound. This is an illustrative Hopper substrate, not Moonshot’s actual physical cluster.

**Network:** three distinct contracts: latent expert dispatch, KCP transition/zero-state all-gather, and one full request cache handoff. Link efficiency and oversubscription are assumptions. The output is a bandwidth-only component, not measured latency or throughput.

Click an operation for equations, source links and scope notes. `1`–`5` switch lenses, `/` or `Ctrl/Cmd+K` searches, `Escape` closes dialogs, `0` resets zoom. On narrow screens the diagram scrolls at legible intrinsic width; labels are not squashed to fit.

## Scientific boundaries

- **Reported versus counted:** 2.78T total / 104.2B active parameters are publisher-rounded figures. The exact routed matrix subtotal is `92 × 896 × 3 × 3584 × 3072`. This is not an exact full-checkpoint manifest count.
- **NoPE still retains 64 shared key channels.** The released attention code disables rotation, but retains the legacy-named 64-dimensional shared branch. Optimized compressed cache is `512 + 64 = 576` elements per global layer/token. Reference expanded cache is `96 × (192 + 128)`.
- **KDA state is not all serving memory.** The recurrent matrix is fixed per request. MLA history grows with tokens; convolution history, retained snapshots, replication and engine metadata are separate. The calculator counts running matrices, a reference-shaped BF16 convolution buffer, and logical MLA history. State dtype is an explicit assumption.
- **Mixed MXFP4, not uniform four-bit:** the scenario counts routed expert weights at `0.5 + 1/32` bytes/value, including one uint8 scale per 32 weights. The rounded remainder is budgeted at BF16. Padding, other metadata and backend workspace are not included. The approximate 1.56TB result is not a measured file size or a fit guarantee.
- **AttnRes is depth attention:** softmax is over depth representations at the same token. Local sums are updated within a block; source banks are incrementally extended. Eight decoder groups plus the embedding are not nine repeated transformer blocks.
- **Report versus base release:** the report describes one MTP layer; the released text configuration sets `num_nextn_predict_layers=0`. No MTP/draft weights or cache are silently included in the base graph/budget.
- **Training is not an inherited FSDP/Adam example.** No uniform bytes-per-parameter training ledger, exact parallel degrees or cluster size is invented.
- **Cache hit is not permission to decode early.** Restore a common prefix, finish prefill for remaining prompt tokens, then begin generation.
- **Traffic scope matters:** MoE is one layer/source rank/direction under uniform destinations. KCP is one layer, all 96 heads local, with both matrices exchanged. Cache handoff is one whole request, not per-rank TP traffic.
- GB/TB are decimal; KiB/MiB/GiB are binary. NVLink900GB/s is per-GPU bidirectional aggregate; a 400Gb/s NIC is50GB/s in one direction.

Primary sources are in `src/core.mjs` and the Method & sources dialog. The report is linked to arXiv v1; release-config and code links point to upstream `main` and can change. `evidence/source-ledger.json` is a manually transcribed field ledger, **not an original byte-for-byte source snapshot**. Verify upstream changes before extending the model.

## Verify

```sh
npm test
npm run build
python3 tests/browser.test.py
# or: npm run verify
```

Browser tests need Python `playwright` and Chromium. Set `CHROMIUM_PATH` to the browser executable; default `/usr/bin/chromium`. Tests run against the built portable HTML, without a web server.

The delivered edition passes **25 Node tests**, **302 rendered configuration checks**, and **40 responsive combinations** in Chromium. Representative exports and real interactions are checked. This is not Safari/Firefox certification, an exhaustive Cartesian product of settings, or a proof of every semantic property. See `VERIFICATION.md`.

```sh
python3 tests/capture.py  # optional high-resolution previews
```

## Source map

| Module | Responsibility |
|---|---|
| `src/core.mjs` | Configuration, evidence, pure accounting, validation |
| `src/scene.mjs` | Nodes, groups, edges, labels, sum primitives |
| `src/diagrams.mjs` | K3-specific model, training, serving, hardware and network templates |
| `src/app.mjs` | State, controls, SVG renderer, inspector, exports, browser geometry audit |
| `src/styles.css` | Visual grammar, themes, responsive viewport |
| `build.mjs` | Dependency-free portable HTML build |
| `tests/` | Numeric, semantic, structural, browser and interaction checks |

`STRATUM.getState()`, `getScene()`, `setState({...})` and `geometryReport()` are available in the browser console. The geometry audit uses actual SVG text bounding boxes. It checks text/node overlap, overflow, wire/text and unrelated wire/node intersections. It does not certify every edge crossing or hidden semantic ambiguity.

This is a **template-based atlas**, not an arbitrary-model importer or automatic graph-layout solver. Coordinates are authored and measured after rendering; a new motif must pass the browser verifier. The engineering contract is in `AGENTS.md` and `docs/AGENT_DESIGN_SPEC.md`.

Visual inspiration: Edward Z. Yang’s infra-oriented DeepSeek-V3 schematic. Application code and layouts are original; no paper figures, upstream source code, model weights or font files are bundled. The upstream models and papers retain their own terms. No GitHub publication or public deployment is implied by this source archive.
