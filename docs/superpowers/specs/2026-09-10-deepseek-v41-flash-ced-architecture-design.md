> September 12 amendment: the earlier absence-of-framework observations below are historical.
> Exact V4.1 SGLang and Miles development branches are now identified in
> [V41-SERVING-RESEARCH](../../design/V41-SERVING-RESEARCH.md). The dedicated-page landing
> incorporates SGLang cache packing and qualifies reported phase depth versus optional replay.
> Original source observations are retained below; they do not describe current availability.

# DeepSeek-V4.1-Flash: Causal Encoder-Decoder architecture — design

**Date:** 2026-09-10
**Status:** approved for planning
**Target:** add `dsv41` as the ninth model, with a dedicated whole-network page for its
Causal Encoder-Decoder architecture.

## 1. Why this is not a variant

Every model currently in the Observatory is a uniform stack: `layers` blocks, one attention
kind per layer, one `active` parameter count, one KV cache formula. DeepSeek-V4.1-Flash breaks
four of those assumptions at once:

1. **Two phases with different depth participation.** Prefill computes only layers 0–19;
   decode computes all 40. This is the 8B/16B active-parameter split, not a marketing figure.
2. **Two KV caches** with different precision, different layer ownership, and different growth:
   global KV (FP4, four layers, linear in context) and SWA KV (FP8, all layers, bounded at 128).
3. **Four attention modes** in one stack, where the repo models one kind per layer.
4. **A 4-stream residual** (Single-Pass mHC) distinct from the existing `hyper` and `attnres`.

Adding it as a row in `MODELS` would misrepresent all four. Hence a dedicated page.

## 2. Verified facts and their provenance

All bytes pinned at revision `dba1be0a40aa45a94ad051997016db3960a90277`.

| Fact | Source | Locator |
| --- | --- | --- |
| 40 layers, `d`=5120, 20-layer encoder + 20-layer decoder | config + report | `num_hidden_layers`; §4.2.1 |
| **All 40 layers causal** | report | line 314: "40 causal Transformer layers" |
| Encoder: SWA×2, then `[Full, Reuse×5]`×3 | config + report | `compress_ratios`, `kv_source_layer_ids`; §4.2.1 |
| Decoder: `[Full, Reuse×3]`, then `[Reindex, Reuse×3]`×4 | config + report | `index_source_layer_ids`; §4.2.1 |
| Decoder global KV projected from `H₂₀` | report | §2.2 eq. (1) |
| SWA computed per-layer from own `H_l`, all layers | report + code | §2.2; `Attention._window_kv` |
| Main KV = FP4 E2M1, one E4M3 scale / 16 channels | report | §2.4 |
| **SWA KV retained at FP8** | report | §2.4 "due to its sensitivity to quantization" |
| Global KV = 890 B/token | report + card | §1; abstract |
| 64 query heads, `head_dim`=512, **`kvHeads`=1** | config | `num_key_value_heads: 1` |
| Output projection LoRA, 8 groups × rank 1024 | config + code | `o_groups`, `o_lora_rank`; `Attention.forward` einsum |
| 384 routed + 1 shared expert, top-6, `expertFF`=2304 | config | `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `moe_intermediate_size` |
| Routing = `sqrtsoftplus` = `softplus(s).sqrt()` | code | `Gate.forward` |
| mHC: 4 streams, 20 Sinkhorn iterations | config + code | `hc_mult`, `hc_sinkhorn_iters`; `Block.hc_mixes` |
| mHC is "single-pass": each sublayer's coefficients feed the **next** | code | `Block.forward` docstring |
| Engram = 196,613,849,600 params, FP8-resident, layers 1 and 14 | config + code | `(384006168 + 384016682) × 256`; `ParallelEngramEmbedding` |
| Engram fetches 24 hash columns = `(4−1)×8` | code | `Engram.__init__` |
| Hierarchical Sparse Indexer: 2048 blocks × 8 = 16,384 candidates | config + report | `candidate_topk_blocks`; §2.3.2 |
| DSpark = 3 draft layers, block size 5, own 128-expert MoE top-3 | config | `num_nextn_predict_layers`, `dspark_*` |
| **DSpark is non-causal within its 5-token block** | code | `get_dspark_topk_idxs` expands one index row to all queries |
| 552B backbone; 8B prefill / 16B decode | report | §4.2.1 |
| Context 1M via YaRN factor 16 over 65,536 | config | `rope_scaling` |

### 2.1 The 890 B/token figure

Main KV is fully derivable from stated facts — FP4 at 0.5 B/channel plus a 1-byte E4M3 scale per
16 channels = 0.5625 B/channel, stored only by the four Full-mode layers:

```
layers 2, 8, 14  (m=2):  512/2 × 0.5625 = 144 B each  →  432 B
layer  20        (m=1):  512   × 0.5625 =              →  288 B
                                        main KV total =    720 B/token
```

The 170 B residual to 890 is the indexer K for those same four layers
(`index_head_dim` 128 → 320 channels/token). Its storage precision is stated nowhere in the
config or the report. **720 B is therefore the derived figure and 890 B the reported one; the
residual is labelled, not reverse-engineered.** A precision that closes the gap exactly can be
guessed at, and guessing it would be fabricating evidence to make a number agree.

## 3. Non-goals

- No `framework_code` source. `deepseek_v41` does not exist in transformers, SGLang, or vLLM
  (all three verified 404). `deepseek_v4` exists in two of them and is a **different model**;
  citing it would be inventing a branch to fit a template.
- No tuning of `cacheLedger` to reproduce 890 B/token.
- No DSpark parameter count. Neither config nor report states one.
- The tech report PDF is not a byte-verified source: `sources.py` decodes strictly as UTF-8.
  It appears in `SOURCES` (display-only citations) and not in `sources.json`.

## 4. Design

### 4.1 Sources and the policy root

`REPOS` in `verification/evidence.py:25` is a strict allowlist; an unlisted repo yields
`SOURCE_AUTHORITY` **FAIL**. `deepseek-ai/DeepSeek-V4.1-Flash` must be added to
`publisher_config` and `publisher_code`. This is an evidenced addition — the HF model API
confirms the publisher ships modeling code — and `tools/audit-publisher-code.py` re-records
the audit that `CONFIG_ONLY_PUBLISHERS` is tested against.

Six records in `sources.json`, all `kind: git`, all pinned at `dba1be0a…`:

| id | path | role |
| --- | --- | --- |
| `dsv41-config` | `config.json` | `publisher_config` |
| `dsv41-code` | `inference/model.py` | `publisher_code` |
| `dsv41-engram` | `inference/engram.py` | `publisher_code` |
| `dsv41-kernel` | `inference/kernel.py` | `publisher_code` |
| `dsv41-vision` | `inference/vision.py` | `publisher_code` |
| `dsv41-card` | `README.md` | `paper` |

### 4.2 Model schema

Five fields join `common`, defaulting to the absent case so the existing eight stay valid:
`encoderLayers: 0`, `activeDecode: 0`, `window: 0`, `oRank: 0`, `oGroups: 0`. These carry the
established "common defaults for absent mechanisms are not claims that the corresponding
checkpoint has those operators" statement.

Everything else is `dsv41`-only, following the `attnResBlock` / `mixRank` / `globalEvery`
precedent: `swaLayers`, `kvSourceLayers`, `indexSourceLayers`, `compressEncoder`,
`compressDecoder`, `candidateBlocks`, `candidateBlockSize`, `candidateSource`, `indexerHeads`,
`indexerBudget`, `engramLayers`, `engramNgram`, `engramHeads`, `engramHeadDim`, `sinkhornIters`,
`routeScale`, `swigluLimit`, `dsparkBlock`, `dsparkExperts`, `dsparkTopK`, `dsparkTargets`,
`visionLayers`, `visionD`, `visionHeads`, `visionFF`, `visionPatch`, `kvBytesPerToken`,
`kvFormat`, `swaKvFormat`.

New enum values: `architecture: 'ced-csa2'`, `residual: 'mhc'`, `scoring: 'sqrtsoftplus'`.

`mtp: 3` follows `ds3`'s layer-count convention. `parameterLedger`'s existing
`m.id==='q38flash'` special case therefore correctly excludes it from resident parameters;
that pre-existing ambiguity is left alone rather than widened.

### 4.3 `layerKind` and `layerPartition`

```js
if (m.architecture === 'ced-csa2') {
  const l = layer - 1;                              // report indexes from 0
  if (l < m.swaLayers) return 'swa';
  if (m.kvSourceLayers.includes(l)) return 'csa2-full';      // must precede reindex:
  if (m.indexSourceLayers.includes(l)) return 'csa2-reindex'; // kvSource ⊂ indexSource
  return 'csa2-reuse';
}
```

`layerPartition` gains four keys, summing to `m.layers`:

```
{ swa: 2, 'csa2-full': 4, 'csa2-reindex': 4, 'csa2-reuse': 30 }   // = 40
```

DSpark's three draft layers are **not** partition members: `model.py` builds `self.layers`
(40 blocks) separately from `self.mtp` (3 `DSparkBlock`s), and including them would make the
partition sum to 43 ≠ `m.layers`. They are carried by `mtp: 3` and drawn on the CED page.

The schematics matrix goes 504 → **544** layers.

### 4.4 Ledgers

**`cacheLedger(id, settings)`** gains `{contextLinear, bounded}` for all nine models.

Two corrections to the earlier framing, found while reading the current implementation:

1. **`global` is already taken.** `cacheLedger` returns `global = p.mla + p.gqa + p.qsa`, a *layer
   count*, consumed by `app.mjs:69` and `diagram-systems.mjs:14,19`. Reusing the name for a byte
   total would silently change what those consumers mean. The new fields are therefore
   `contextLinear` and `bounded`, which say what the property is rather than where it lives.
2. **The existing eight do not report zero.** `recurrent` and `conv` carry no `S` term, so
   `kimi3`, `q3827` and `q38flash` already hold bounded state. The split is a *reclassification
   of existing terms*, not a field bolted on for the ninth model:

```
contextLinear = mla + gqa + indexer      // every term carrying S
bounded       = recurrent + conv         // every term that does not
total         = contextLinear + bounded  // value unchanged for all nine models
```

For `dsv41`, `contextLinear` is the FP4 compressed main KV over the four Full-mode layers and
`bounded` is the FP8 SWA window. Because `total` is unchanged, no existing `total` binding moves;
only the two new fields need claims.

**`servingLedger(id, {phase})`** gains a phase dimension: `prefill` sums layers
`0 … encoderLayers-1`, `decode` sums all `layers`. This makes 8B a derived consequence of
`encoderLayers: 20` rather than a transcribed constant.

**`parameterLedger`** needs the new attention shape: `wq_a` (d × qRank) + `wq_b`
(qRank × heads·headDim) + `wkv` (d × headDim, a **single** latent for all 64 heads) +
`wo_a` (heads·headDim/oGroups × oGroups·oRank) + `wo_b` (oGroups·oRank × d), plus the per-head
`attn_sink`. Engram's 196,613,849,600 parameters join `resident` as `ngram`, matching
`q38flash`. The exact value is required, not the card's rounded "196B" — see §5.6.

### 4.5 Diagrams

**Block view** (`diagram-model.mjs`) — mHC 4-stream residual, CSA2 dual-path attention
(sliding window ⊕ compressed, concatenated into one sparse-attention call), per-head attention
sink, clamped SwiGLU at threshold 10, and the 384/top-6/1-shared MoE with `sqrtsoftplus`
routing and a selection-only correction bias.

**`buildResidual`** gains an `mhc` branch beside `hyper` and `attnres`, stating that mHC's
coefficients are consumed by the following sublayer.

State wiring the page needs, none of which is optional:
`sanitizeState`'s `choices.stage` must gain `'ced'`; `app.mjs:19` must offer the stage when
`m.encoderLayers` is set; and `setState`'s model-switch guard at `app.mjs:11` must reset
`stage` to `'block'` when moving to a model without `encoderLayers`, exactly as it already does
for `ple`, `residual` and `inputs`. Omitting the guard leaves a stale stage rendering the wrong
model.

`layerPartition` keys are `layerKind` return values, so they are hyphenated and need bracket
access in `cacheLedger`. `comparisonTable`'s `k.toUpperCase()` will render them as
`CSA2-FULL` etc., which is correct and needs no special case.

**New page: `stage: 'ced'`** (`diagrams.mjs` dispatch, gated on `m.encoderLayers`, exactly as
`ple` is gated on `q38flash`). A whole-network page, the Figure 3 analogue:

- Text and vision embedding → Single-Pass mHC → 4-stream embedding
- Causal Encoder box: SWA×2, then three `[Full, Reuse×5]` groups
- `Enc. Hidden States` (4 streams) → **CED** edge → decoder Full-mode layer 20
- Decoder box: `[Full, Reuse×3]`, then four `[Reindex, Reuse×3]` groups
- Engram box feeding layers 1 and 14
- Candidate Pool sourced from layer 20, feeding the Reindex layers
- DSpark head reading the attention inputs of layers 37–39

### 4.6 Verification

Approximately 455 new claims: ~50 field, ~360 scene, ~45 for the CED page. Anchor the
config-derived ones with `tools/anchor-configs.py` against the acquired bytes; hand-anchor code
claims to `python_symbol`s in `model.py`; record every genuinely unanchorable citation in
`docs/verifiers/ANCHOR_RESIDUE.md` with a decision, per existing practice.

The architecture-specific verifiers of §5 land in `verification/ced.py`, called from
`schematics.py`, each with a negative test that constructs a deliberately broken architecture
and asserts the check fires.

The composition baseline grows because the sample set grows: record with
`npm run verify:aesthetics -- --record` and promote deliberately. Previews recapture via
`npm run capture`. `VERIFICATION.md` renumbers from reports generated on the current bytes.

### 4.7 Innovation coverage

The CED page carries a written exposition of what is new in this architecture: prose and tables,
not only a diagram. Eight innovations get an entry — CED, CSA2's three modes, the Hierarchical
Sparse Indexer, Single-Pass mHC, Engram, DSpark, FP4 main KV, and SWA Bounded Replay.

Prose that asserts an architectural fact needs the same evidence chain as a number in a diagram.
Two rules make that enforceable rather than aspirational:

1. **Every number is derived, never retyped.** `CLAUDE.md` is explicit that "every displayed
   total is derived from it, never re-typed in a template". Each innovation record therefore
   carries `effect` as a **function** of the model, not a string. A literal cannot drift from the
   data because a literal cannot be written in the first place. This follows the existing
   `comparisonTable()` pattern at `app.mjs:64`, which already derives every cell from
   `cacheLedger` and `parameterLedger`.
2. **Every factual statement cites a source.** Each record carries `sources: [...]` of ids that
   must resolve in `SOURCES`. Clickable source citations are the established provenance feature
   and `presentations.py` explicitly permits them.

Records live in a new `src/innovations.mjs` (data and row construction, no DOM, per the module
boundaries in `CLAUDE.md`), inserted into `build.mjs`'s hardcoded `files` array between
`'diagrams'` and `'app'` — it needs `core.mjs` for the derivations. Rendering is an
`innovationCoverage()` function in `app.mjs` mirroring `comparisonTable()`, reusing the existing
`.table-scroll` / `.comparison-table` styling so no new type sizes or colour tokens enter the
frozen scales.

Record shape:

```js
{ id:'ced', title:'Causal Encoder-Decoder',
  replaces:'Every layer deriving its own global KV',
  mechanism:'Decoder global KV is projected from the final encoder hidden state H20 …',
  effect:m=>`prefill computes ${m.encoderLayers} of ${m.layers} layers`,
  sources:['dsv41-code','dsv41-card'] }
```

Accessibility and legibility are not optional here: this is reading-size **chrome** prose, which
the aesthetics gate holds to **AAA (7:1)**, and `presentations.py` checks heading order and
accessible names. The table needs real `<thead>`/`<th>` semantics, as `comparisonTable()` has.

Two checks keep it honest:

| Code | Severity | Property |
| --- | --- | --- |
| `COVERAGE_DERIVED` | FAIL | Every innovation record's `effect` is a function, and its rendered output equals an independent Python computation for each derivable quantity |
| `COVERAGE_UNCITED` | FAIL | Every record carries at least one source id, and every id resolves in `SOURCES` |

`COVERAGE_DERIVED` is the check that matters. It is what stops the exposition becoming a second,
unverified copy of the architecture that drifts from the ledgers.

## 5. Architecture-specific verifiers

A verifier that restates the config proves nothing. Every check below is either a **cross-check
between two independent derivations** or a **structural invariant that a mis-transcription
breaks**. They live in a new `verification/ced.py`, called from `schematics.py` — not a fifth
software gate, because adding a verification domain would force both a contract and a runtime
obligation, and the Pages deploy contract is written against four software gates.

### 5.1 The strongest check: publisher agreement

`MODELS.dsv41` is a hand transcription of `config.json`. Transcription error is the dominant
failure mode when adding an architecture, and nothing in the repo currently catches it.

`CED_PUBLISHER_AGREEMENT` re-derives **every** `dsv41` field directly from the acquired
`config.json` bytes and asserts equality with the transcribed entry. Python reads the
publisher's bytes; JavaScript reads our transcription; they must agree field for field.

`.verification-cache/` is git-ignored by design, so this check uses the established
`require_cache()` pattern: it runs locally and `skipTest`s in CI with an explicit reason. It is
deliberately not weakened into a comparison against committed values, which would be
self-consistency rather than verification.

### 5.2 CSA2 mode assignment

The config states mode assignment as three arrays; the report states it as a group structure
(§4.2.1). Deriving it both ways and comparing is a genuine independent cross-check.

| Code | Severity | Property |
| --- | --- | --- |
| `CED_MODE_DISAGREEMENT` | FAIL | Modes derived from the config arrays must equal modes generated from the report's group structure, layer for layer |
| `CSA2_INDEX_SUBSET` | FAIL | `kvSourceLayers ⊆ indexSourceLayers` — a Full layer computes its own indexer K |
| `CED_REUSE_UNREACHABLE` | FAIL | Every Reuse/Reindex layer has a preceding same-ratio Full layer to read from |
| `CED_BAND_RATIO` | FAIL | `compress_ratio` is constant within each band (2 encoder, 1 decoder) |
| `CED_MTP_RATIO` | FAIL | Every draft layer has `compress_ratio == 0`, as `DSparkAttention` asserts |
| `CSA2_CANDIDATE_SOURCE` | FAIL | `candidateSource` is the first decoder Full layer, and no Reindex layer precedes it — Reindex layers search a pool that must already exist |

`CED_REUSE_UNREACHABLE` is the one that matters most: it is the invariant that makes
layer-dimension cache sharing well-founded, and it is invisible to every existing check.

### 5.3 CED KV provenance and the two phases

| Code | Severity | Property |
| --- | --- | --- |
| `CED_KV_PROVENANCE` | FAIL | Exactly one `kvSource` in the decoder band, and it is the first decoder layer. This is the structural form of eq. (1): no decoder layer derives global KV from its own hidden state |
| `CED_PARTITION_SUM` | FAIL | `layerPartition` sums to `m.layers` |
| `CED_PHASE_DEPTH` | FAIL | `servingLedger(id,{phase:'prefill'})` covers exactly `encoderLayers` layers; `'decode'` covers `layers` |
| `CED_ACTIVE_RATIO` | BLOCKED | `active/activeDecode` agrees with `encoderLayers/layers` within a policy tolerance. BLOCKED rather than FAIL because the published actives are rounded to whole billions and embedding/Engram terms are depth-independent |

### 5.4 KV cache budget

Two independent implementations of the same arithmetic — the JS ledger and the Python verifier —
must agree, and the reported total must remain physically explainable.

| Code | Severity | Property |
| --- | --- | --- |
| `KV_BUDGET_DISAGREEMENT` | FAIL | Python-derived main-KV bytes/token equals `cacheLedger().contextLinear` per token |
| `KV_RESIDUAL_IMPLAUSIBLE` | FAIL | `(kvBytesPerToken − derived) / indexerChannelsPerToken` lies in `[0.5, 1.0]` B/channel — between the FP4 floor and FP8 |
| `KV_BOUNDED_INVARIANT` | FAIL | `cacheLedger().bounded` is identical at two different context lengths, **for all nine models** — a bounded term that moves with `S` is misclassified |

`KV_BOUNDED_INVARIANT` generalises the SWA property into an invariant every model must satisfy,
which is what makes the reclassification verifiable rather than cosmetic.

`KV_RESIDUAL_IMPLAUSIBLE` is how the 890 B/token figure is held accountable without inventing
the indexer precision. Today the residual is `170 / 320 = 0.53` B/channel, comfortably inside
the bound. If either the reported figure or the indexer geometry changes such that the residual
implies an impossible precision, that is a real contradiction and the gate says so.

### 5.5 Single-Pass mHC

| Code | Severity | Property |
| --- | --- | --- |
| `MHC_COEFF_COUNT` | FAIL | Coefficient count equals `(2 + streams) × streams` = 24, derived from `streams` alone and cross-checked against the code's expression |
| `MHC_STREAM_BALANCE` | FAIL | Every mHC block carries exactly `streams` inbound and `streams` outbound residual edges; `hc_pre` nodes are `streams`→1 and `hc_post` nodes are `1+streams`→`streams` |

`MHC_STREAM_BALANCE` is a scene-graph invariant. It catches a residual drawn as an ordinary sum,
which is precisely the misrepresentation the project forbids.

### 5.6 Engram

| Code | Severity | Property |
| --- | --- | --- |
| `ENGRAM_TOTAL_DERIVATION` | FAIL | `sum(engram_num_embeddings) × engramHeadDim == ngram` |
| `ENGRAM_HASH_COLUMNS` | FAIL | `n_hash_cols == (engramNgram − 1) × engramHeads` = 24, cross-checked against `Engram.__init__` |
| `ENGRAM_LAYER_BOUNDS` | FAIL | One embedding table per Engram layer, and every Engram layer id `< layers` |

`ENGRAM_TOTAL_DERIVATION` forces the exact figure. `768022850 × 256 = 196,613,849,600`, so
`ngram` must be recorded as `196613849600` — not the card's rounded "196B". A rounded value
fails this check, which is the intended behaviour: the derived number is available, so the
rounded one is not good enough.

### 5.7 DSpark

| Code | Severity | Property |
| --- | --- | --- |
| `DSPARK_TARGET_COUNT` | FAIL | `len(dsparkTargets) == mtp` — one backbone target per draft layer |
| `DSPARK_BLOCK_TRIVIAL` | FAIL | `dsparkBlock > 1`; a block size of 1 would be ordinary autoregression, not semi-autoregressive drafting |
| `DSPARK_MOE_DISTINCT` | FAIL | Draft-layer expert counts differ from the backbone's, per `get_moe_config` |

### 5.8 CED page topology

The new page asserts specific structure, so the structure is checked rather than trusted.

| Code | Severity | Property |
| --- | --- | --- |
| `CED_SCENE_TOPOLOGY` | FAIL | Encoder band has exactly `encoderLayers` tiles and the decoder band `layers − encoderLayers`; exactly one CED edge from encoder output to the decoder Full layer; the Candidate Pool fans out to exactly the Reindex layers; Engram fans out to exactly `len(engramLayers)` layers; **no edge runs from any decoder node back into the encoder** |

The last clause is the one-way-handoff property that distinguishes CED from a shared-KV
Transformer, and it is a graph property rather than a caption.

### 5.9 Innovation coverage

`COVERAGE_DERIVED` and `COVERAGE_UNCITED` from §4.7 are implemented in the same module. The
derived-output comparison reuses the §5.4 KV arithmetic and the §5.6 Engram derivation, so the
exposition is checked against the same independent Python computation the ledgers are.

### 5.10 Test protocol

Every verifier above ships with a negative test that constructs a deliberately broken
architecture and asserts the check fires. A verifier that passes on its first run against real
data has not been shown to check anything. Node-side counterparts in `tests/core.test.mjs`
cover the JS derivations so the two-implementation cross-checks in §5.1 and §5.4 are real
rather than one implementation compared against itself.

## 6. Expected end state

| Gate | Expected | Why |
| --- | --- | --- |
| engineering | PASS | 544 layers, 0 geometry defects, deterministic build |
| schematics | PASS | scene-vs-model agreement across the new matrix, plus every §5 check |
| aesthetics | PASS | after baseline promotion |
| presentations | PASS | CED page and innovation coverage in previews, the export matrix, heading order and AAA contrast |
| science | **exit 2 BLOCKED** | no new FAIL codes |
| runtime | exit 2 BLOCKED | unchanged |

`npm run test:verifiers` gains `verification/tests/test_ced.py`; `npm test` gains the
Node-side derivation tests that make the §5.1 and §5.4 cross-checks two-implementation rather
than self-referential.

### 6.1 The honest cost, stated up front

`required_roles` demands `('publisher_code','framework_code')` for model-domain claims, and
`dual_code_roles` demands independent readings from two different organizations. No framework
implements this model. Therefore **every non-presentation model-domain `dsv41` claim earns two
BLOCKED findings** — `CLAIM_ROLE` and `MODEL_DUAL_CODE` — roughly 100 in total.

These cannot be cleared honestly today. They must not be cleared by:

- classifying claims `assumed` (which makes `required_roles` return `()`), or
- citing `deepseek_v4` as a framework reading of `deepseek_v41`, or
- adding a second DeepSeek repository as the "independent" organization.

The model is reported as uncorroborated until a framework ships an implementation. That is the
fail-closed design behaving correctly, and it is the price of adding a model this new.

## 7. Risks

- **The `cacheLedger` contract change re-binds every existing model's cache claims.** Expected
  and deliberate; it is the one change here that touches all nine models. `CONSUMER_VALUE`
  findings will fire until the bindings are updated.
- **The five `common` additions add ~40 claims of churn** across the existing eight models.
- **`sqrtsoftplus`, `'mhc'` and `'ced-csa2'` widen closed value sets** that `schematics.py`
  enforces; the enums must be extended in step or the gate fails honestly.
- **New motifs may surface aesthetics defects** in states never previously measured. Fix the
  tokens, never the policy.
- **Scene-vs-model consistency may surface label/ledger disagreements.** That is the gate
  working; keep diagram fixes separate from verifier changes.

## 8. Known and unclosed

- No independent framework corroboration exists (§6.1).
- The 170 B indexer-K residual in the 890 B/token figure is labelled, not derived (§2.1).
- No DSpark parameter count is published.
- The tech report PDF cannot be byte-pinned.
- `CED_PUBLISHER_AGREEMENT` (§5.1) cannot run in CI, because `.verification-cache/` is
  git-ignored by design and CI therefore holds no publisher bytes. It skips with an explicit
  reason rather than passing vacuously.
- `parameterLedger`'s `m.id==='q38flash'` special case for `mtp` semantics remains; this design
  follows the `ds3` convention rather than widening the debt.
