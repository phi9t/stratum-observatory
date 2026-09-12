# DeepSeek-V4.1-Flash CED Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DeepSeek-V4.1-Flash as the ninth model (`dsv41`) with a dedicated whole-network CED page, a written innovation exposition, and architecture-specific verifiers that cross-check our transcription against publisher bytes.

**Architecture:** The model breaks four schema assumptions (two prefill/decode phases, two KV caches, four attention modes, a 4-stream residual), so it gets new `layerKind` values, a reclassified `cacheLedger` contract, a phase-aware `servingLedger`, a `stage:'ced'` page, and a new `verification/ced.py` called from `schematics.py`. Verifiers are cross-checks between two independent derivations, never restatements of the config.

**Tech Stack:** Zero-dependency ESM (Node 20+), uv-managed CPython 3.13.12, Playwright 1.55.0, `unittest`, `node --test`.

**Spec:** `docs/superpowers/specs/2026-09-10-deepseek-v41-flash-ced-architecture-design.md`

## Global Constraints

- **24 claims declare they reconcile every supported model, and adding a ninth changes their substance, not only their digest.** The 15 `function.*` and 9 `file.*` claims listed in Task 2 Step 7 each carry `dependencies` naming `MODELS.<id>.d` for all eight current models, and each states "code review must reconcile all supported models". Every one must gain `MODELS.dsv41.d`. These are all `classification: derived` and carry `REVIEW_REQUIRED` BLOCKED, so adding the dependency declares what a review must cover — it does not assert that one happened. Task 7's `CLAIM_MODEL_COVERAGE` enforces this so the manifest cannot silently omit a model, and the 2,520 per-scene claims are unaffected because each depends on its own single model.
- **Every changed consumer needs its exact binding in the same task that changes it.** `CLAUDE.md` states this as a non-negotiable, and the audit enforces it: `file.src/*.mjs`, `function.*`, `MODELS.*` fields and scene signatures are all claimed consumers, and changing one without rebinding its claim raises `CONSUMER_VALUE` or `COVERAGE_MISSING` — both **FAIL**, which turns the audit FAIL instead of BLOCKED and breaks `test_cold_cache_is_blocked_not_certified`. Task 13 is not a deferral point for claims your task invalidates; it covers anchoring, the residue record, and scene claims no earlier task owns.
- **Gate order for any task touching `src/` or `tests/` (controller ruling T456a, revised twice).** The governing rule is simpler than the sequence: **`VERIFICATION.md` and `verification/reports/publication.json` must agree with each other at the moment `verify:engineering` runs**, because engineering's first step is the verifier suite and `test_reported_numbers` cross-checks the two. Hence **digest early, figures late**: paste the current `toolchain_digest` *before* engineering (it is not cross-checked against the publication report, and pasting converges because `VERIFICATION.md` is not a verifier input), but do **not** paste any gate *figure* until after the final `verify:science`. Restoring "the committed report" is only correct while `VERIFICATION.md` still holds its committed figures — once you have edited them, the committed report no longer agrees, and engineering fails on a one-digit mismatch. This has now bitten five times on this branch. `verify:engineering`'s first step is the verifier suite, which reads `verification/reports/publication.json` **from disk** — and a publication report generated against a stale receipt writes staleness FAILs that then make engineering fail. The convergent order is: rebind claims → `verify:science` **diagnostic-only, to confirm `CONSUMER_VALUE` and `COVERAGE_MISSING` are 0, then restore the committed report** → `npm run capture` → `verify:presentations` → `verify:engineering` → `verify:science` for real → paste figures → `verify:schematics` / `verify:aesthetics` / `verify:runtime` → `test_reported_numbers`. **Do not leave the first science result on disk.** Before committing, list **all six** report timestamps and confirm each postdates your last `src/` change — and confirm the receipt's `verifier_sha256` equals a freshly derived `toolchain_digest`, which is the authoritative check rather than the timestamps. `--record` on the aesthetics baseline drops `$comment`; restore it.
- **Any task that edits `verification/*`, `tools/*`, workflows, schemas, trust, `package.json` or `tests/` moves `toolchain_digest`; any task that edits `verification/manifests/*` moves the scientific lock. Both are quoted in `VERIFICATION.md`, and `verification/tests/test_reported_numbers.py` fails when they drift.** As that task's last step before committing, run `uv run --frozen python -m unittest verification.tests.test_reported_numbers` and paste the digests it names into `VERIFICATION.md`'s "Artifact identity" section. This converges: `VERIFICATION.md` is not a verifier input, so writing the digest does not move it. Edit the verifier last, then paste. Every task leaves the suite green.

- Publisher revision is `dba1be0a40aa45a94ad051997016db3960a90277`. Every source record pins it. Never a branch name.
- Exit codes: `0` PASS, `1` contradiction/malformed input, `2` BLOCKED or NOT_RUN. **BLOCKED is not failing.**
- `ngram` must be exactly `196613849600`. The card's rounded "196B" fails `ENGRAM_TOTAL_DERIVATION` by design.
- No `framework_code` source exists for this model. `CLAIM_ROLE` and `MODEL_DUAL_CODE` BLOCKED findings are the correct output and must **not** be silenced by classifying claims `assumed`, by citing `deepseek_v4` as a reading of `deepseek_v41`, or by adding a second DeepSeek repository as the "independent" organization.
- Do not tune `cacheLedger` to reproduce 890 B/token. 720 B is derived; the 170 B residual is labelled.
- "Every displayed total is derived from it, never re-typed in a template." No numeric literal in a template or an innovation record.
- No branch invented to fit a template. Preserve exact variant identity, shape contracts, precision/rate units.
- Do not turn missing numeric evidence into zero. No DSpark parameter count exists; do not invent one.
- Never hand-edit `lock_sha256`, `probe_schema_sha256`, `contract_sha256`, `hardware_sha256`, or a recorded source `sha256`. Re-acquire or re-probe.
- No `continue-on-error`, fake raw hashes, invented reviewer keys, or copied signatures.
- `build.mjs` supports **only** single-line `import {…} from './x.mjs';` and `export const|let|function|class` / `export {…};`. Anything else throws `Unsupported module syntax`.
- Write a failing behavioural/negative test first, implement, then run `npm run verify:engineering`. A verifier that passes on its first run against real data has not been shown to check anything.
- Previews come from `npm run capture` on the current build, never hand-edited.
- `VERIFICATION.md` figures must come from a report generated on the current bytes.

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `verification/evidence.py` | Evidence policy root (`REPOS` allowlist) | Modify: authorize the new publisher repo |
| `verification/manifests/sources.json` | Byte-verified source catalog | Modify: six `dsv41-*` records |
| `src/models.mjs` | Checkpoint facts and source identities | Modify: `SOURCES`, `CONFIG_PINS`, `common`, `MODELS.dsv41` |
| `src/core.mjs` | Pure ledgers and layer maths | Modify: `layerKind`, `layerPartition`, `cacheLedger`, `parameterLedger`, `servingLedger`, `sanitizeState`, `DEFAULT_STATE` |
| `src/diagram-model.mjs` | Per-block model motifs | Modify: mHC residual, CSA2 attention, clamped SwiGLU |
| `src/diagram-meta.mjs` | Comparison / evidence / focal views | Modify: `buildResidual` mHC branch; add `buildCED` |
| `src/diagrams.mjs` | Template dispatch | Modify: route `stage:'ced'` |
| `src/innovations.mjs` | **New.** Innovation records and row construction. No DOM. | Create |
| `src/app.mjs` | UI, state, export | Modify: stage offer, model-switch guard, `innovationCoverage()` |
| `build.mjs` | Deterministic bundler | Modify: add `'innovations'` to the hardcoded `files` array |
| `verification/ced.py` | **New.** Architecture-specific invariants and cross-checks | Create |
| `verification/schematics.py` | Diagram semantics gate | Modify: call `verification/ced.py` |
| `verification/tests/test_ced.py` | **New.** Negative tests per check | Create |
| `tests/core.test.mjs` | Node ledger/layer tests | Modify: JS-side derivations |
| `tests/scene.test.mjs` | Scene contract tests | Modify: CED page and 544-layer matrix |
| `verification/manifests/observatory.json` | Scientific target | Modify: ~455 new claims |

---

### Task 1: Authorize the publisher and lock its bytes

**Files:**
- Modify: `verification/evidence.py:25-32` (`REPOS`)
- Modify: `verification/manifests/sources.json`
- Test: `verification/tests/test_ced.py` (create)

**Task 1 does not touch `src/models.mjs`.** `file.src/models.mjs` is a claimed consumer, so editing it without rebinding its claim raises `CONSUMER_VALUE` (FAIL) and turns the audit FAIL instead of BLOCKED. The `SOURCES` and `CONFIG_PINS` additions therefore belong to Task 2, which owns that file — see the controller ruling there.

**Interfaces:**
- Consumes: nothing.
- Produces: accepted `sha256` values in `sources.json` for six `dsv41-*` records; `deepseek-ai/DeepSeek-V4.1-Flash` authorized in `REPOS`. Task 2 consumes these ids from `src/models.mjs`.

**Context you need:** `REPOS` in `verification/evidence.py` is a **strict allowlist**. `check_source` raises `SOURCE_AUTHORITY` **FAIL** (not BLOCKED) for any repo not listed under its role. So no `dsv41` source can pass until the repo is authorized. This is a trust-root edit and the user approved it explicitly.

- [ ] **Step 1: Write the failing test**

Create `verification/tests/test_ced.py`:

```python
import unittest
from verification.evidence import REPOS

DSV41 = 'deepseek-ai/DeepSeek-V4.1-Flash'

class PublisherAuthorizationTests(unittest.TestCase):
    def test_the_publisher_is_authorized_for_the_roles_its_bytes_are_cited_under(self):
        """Six dsv41 sources are cited: one config and one card, four modeling files.
        An unlisted repo is SOURCE_AUTHORITY FAIL, so both roles must be authorized."""
        self.assertIn(DSV41, REPOS['publisher_config'])
        self.assertIn(DSV41, REPOS['publisher_code'])

    def test_no_framework_is_authorized_to_stand_in_for_this_model(self):
        """No framework implements deepseek_v41. deepseek_v4 is a different model and
        must never be cited as an independent reading of this one."""
        self.assertNotIn(DSV41, REPOS['framework_code'])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: FAIL on the first test with `'deepseek-ai/DeepSeek-V4.1-Flash' not found in {...}`. The second test passes already; that is fine, it is a guard against a later shortcut.

- [ ] **Step 3: Authorize the repository**

In `verification/evidence.py`, add `'deepseek-ai/DeepSeek-V4.1-Flash'` to the `publisher_config` set and to the `publisher_code` set. Do **not** add it to `CONFIG_ONLY_PUBLISHERS` — this publisher ships modeling code in `inference/`.

- [ ] **Step 4: Re-record the publisher-code audit**

Run: `uv run --frozen python tools/audit-publisher-code.py`

This regenerates `evidence/publisher-code/audit.json`, which a test compares against `CONFIG_ONLY_PUBLISHERS`. Read the diff before accepting it. If the tool reports that this publisher ships no code, stop — that contradicts the file listing and means the evidence, not the code, needs revisiting.

- [ ] **Step 5: Run the test again**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: PASS, 2 tests.

- [ ] **Step 6: Add the six source records**

In `verification/manifests/sources.json`, append to `sources`. Provide only the identity fields; `acquire` fills `sha256`, `cache`, `retrieved_at` and `etag`. Repeat this shape for each of the six:

```json
{
  "id": "dsv41-config",
  "role": "publisher_config",
  "kind": "git",
  "provider": "huggingface",
  "repo": "deepseek-ai/DeepSeek-V4.1-Flash",
  "path": "config.json",
  "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
  "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/config.json",
  "license": "Acquire for local verification; preserve upstream license and notices; no vendored source in this release."
}
```

**`sources.json` uses `ROLES` from `verification/evidence.py`, which is a different vocabulary
from `src/models.mjs`'s `SOURCES`.** `ROLES` is `{publisher_config, publisher_code,
framework_code, vendor_document, library_code, paper, tokenizer_code, checkpoint_metadata}` —
there is no `report`. The two already diverge by design: `k3-paper` is `role:'report'` in
`models.mjs` and `role:'paper'` in `sources.json`. So the model card takes `paper` here, which is
the vocabulary's only prose role, and keeps `report` in `models.mjs`. `paper` is absent from
`REPOS`, so no repo-authority restriction applies; unlike the four arxiv papers, which are
`kind:'document'` and need a `version`, ours is `kind:'git'` at a 40-hex revision, which is a
stronger pin.

The remaining five, all same `repo`/`revision`/`license`/`provider`/`kind`:

| id | path | role |
| --- | --- | --- |
| `dsv41-code` | `inference/model.py` | `publisher_code` |
| `dsv41-engram` | `inference/engram.py` | `publisher_code` |
| `dsv41-kernel` | `inference/kernel.py` | `publisher_code` |
| `dsv41-vision` | `inference/vision.py` | `publisher_code` |
| `dsv41-card` | `README.md` | `paper` |

- [ ] **Step 7: Acquire the bytes**

`npm run verify:sources` is incomplete on its own — `acquire` requires `--candidate`, refuses to overwrite the accepted lock, and refuses a candidate path that already exists. Run:

```bash
uv run --frozen python -m verification.sources acquire \
  --candidate .verification-cache/candidates/dsv41-sources.json \
  --only dsv41-config --only dsv41-code --only dsv41-engram \
  --only dsv41-kernel --only dsv41-vision --only dsv41-card
```

Expected: six results, all `"status": "PASS"`, exit 0. If a corporate TLS proxy interferes, set `UV_NATIVE_TLS=1`.

- [ ] **Step 8: Verify the acquired hashes against the expected bytes**

Compare the candidate's `sha256` values against these, which were computed from the pinned revision. **A mismatch means the upstream bytes moved and the revision pin is wrong — stop and re-pin. Never edit a recorded hash to match.**

| id | sha256 |
| --- | --- |
| `dsv41-config` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `dsv41-card` | `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251` |
| `dsv41-code` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `dsv41-engram` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `dsv41-kernel` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `dsv41-vision` | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |

```bash
python3 -c "
import json
b=json.load(open('.verification-cache/candidates/dsv41-sources.json'))
for s in b['catalog']['sources']:
    if s['id'].startswith('dsv41-'): print(s['id'], s.get('sha256'))
"
```

- [ ] **Step 9: Promote the four identity fields into the accepted lock**

Copy `sha256`, `cache`, `retrieved_at` and `etag` for the six records from the candidate into `verification/manifests/sources.json`. Promote **only** these six; leave every pre-existing record untouched.

- [ ] **Step 10: Confirm nothing regressed**

Run: `uv run --frozen python -m unittest discover -s verification/tests -v`
Expected: PASS. `SOURCE_AUTHORITY` must not appear for any `dsv41` source.

- [ ] **Step 11: Update VERIFICATION.md and commit**

Run `uv run --frozen python -m unittest verification.tests.test_reported_numbers` and paste the digests it names into the "Artifact identity" section, per the global constraint. Then:

```bash
git add verification/evidence.py verification/manifests/sources.json VERIFICATION.md \
        verification/tests/test_ced.py verification/tests/test_evidence.py evidence/publisher-code/audit.json
git commit -m "Authorize the DeepSeek-V4.1-Flash publisher and lock its bytes

REPOS is a strict allowlist and an unlisted repo is SOURCE_AUTHORITY FAIL, so
the publisher must be authorized before any of its bytes can be cited. The
audit is re-recorded rather than asserted.

No framework_code source: deepseek_v41 is absent from transformers, SGLang and
vLLM. deepseek_v4 is a different model and is not cited as a reading of this
one. A test guards that."
```

---

### Task 2: Transcribe the checkpoint and classify its layers

> **Controller ruling (pre-flight):** Tasks 2 and 3 are dispatched together as one unit.
> Task 2 alone leaves the suite red — `layerKind` falls through to `return m.architecture`,
> so `layerPartition` increments `r['ced-csa2']` from `undefined` to `NaN`, which propagates
> into `cacheLedger` and makes existing scene assertions fail on non-finite geometry. A task
> that cannot leave the suite green is not right-sized. Do all of Task 2 and all of Task 3,
> run the full suite once at the end, and commit as two commits.

**Files:**
- Modify: `src/models.mjs:52` (`common`), `src/models.mjs:53-60` (`MODELS`)
- Test: `tests/core.test.mjs`

**Interfaces:**
- Consumes: source ids from Task 1.
- Produces: `MODELS.dsv41` with `architecture:'ced-csa2'`, `residual:'mhc'`, `scoring:'sqrtsoftplus'`, `encoderLayers:20`, `activeDecode:16e9`, `window:128`, `oRank:1024`, `oGroups:8`. Later tasks read all of these.

**Context you need:** `common` supplies defaults so every model has every field; five new fields go there so the existing eight stay valid. Everything model-specific stays on `dsv41` alone, following the existing `attnResBlock` (kimi3) and `mixRank` (q38flash) precedent. `nopeDim` is `headDim - ropeDim = 512 - 64 = 448`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core.test.mjs`:

```js
test('dsv41 transcribes the published shapes and derives nothing by rounding', () => {
  const m = MODELS.dsv41;
  assert.equal(m.layers, 40);
  assert.equal(m.encoderLayers, 20);
  assert.equal(m.d, 5120);
  assert.equal(m.heads, 64);
  assert.equal(m.kvHeads, 1, 'one latent KV serves all 64 query heads');
  assert.equal(m.headDim, 512);
  assert.equal(m.nopeDim, m.headDim - m.ropeDim);
  assert.equal(m.experts, 384);
  assert.equal(m.topK, 6);
  assert.equal(m.shared, 1);
  assert.equal(m.denseLayers, 0, 'every Transformer block is MoE');
  assert.equal(m.architecture, 'ced-csa2');
  assert.equal(m.residual, 'mhc');
  assert.equal(m.scoring, 'sqrtsoftplus');
  assert.equal(m.active, 8e9);
  assert.equal(m.activeDecode, 16e9);
  // the exact figure is derivable, so the card's rounded "196B" is not good enough
  assert.equal(m.ngram, 196613849600);
  assert.equal(m.engramTables.reduce((a, b) => a + b, 0) * m.engramHeadDim, m.ngram);
});

test('the eight existing models keep the common defaults for absent CED mechanisms', () => {
  for (const id of ['ds2', 'ds3', 'q38', 'q330', 'q3235', 'q3827', 'q38flash', 'kimi3']) {
    const m = MODELS[id];
    assert.equal(m.encoderLayers, 0, `${id} has no encoder/decoder split`);
    assert.equal(m.activeDecode, 0, `${id} publishes one active figure`);
    assert.equal(m.window, 0, `${id} has no sliding-window branch`);
    assert.equal(m.oRank, 0);
    assert.equal(m.oGroups, 0);
  }
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'dsv41 transcribes' tests/core.test.mjs`
Expected: FAIL — `MODELS.dsv41` is undefined.

- [ ] **Step 3: Add the app-level citations** (moved here from Task 1 by controller ruling T1c)

In `src/models.mjs`, add to `SOURCES`. Note `dsv41-report` is display-only and deliberately absent from `sources.json`, because `sources.py` decodes strictly as UTF-8 and this is a PDF:

```js
 'dsv41-config':{title:'DeepSeek-V4.1-Flash • released configuration',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json',role:'publisher_config',anchor:'text_config: compress_ratios, kv_source_layer_ids, index_source_layer_ids, hc_mult, engram_num_embeddings'},
 'dsv41-code':{title:'DeepSeek-V4.1-Flash • publisher reference implementation',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py',role:'publisher_code',anchor:'Attention; Compressor; Indexer; Block.hc_mixes; DSparkAttention'},
 'dsv41-engram':{title:'DeepSeek-V4.1-Flash • Engram n-gram hashing',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py',role:'publisher_code',anchor:'EngramLayout; NgramHashState'},
 'dsv41-kernel':{title:'DeepSeek-V4.1-Flash • sparse attention and quantization kernels',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py',role:'publisher_code',anchor:'sparse_attn; act_quant; hc_split_sinkhorn'},
 'dsv41-vision':{title:'DeepSeek-V4.1-Flash • vision encoder',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/vision.py',role:'publisher_code',anchor:'ViT; Aligner'},
 'dsv41-card':{title:'DeepSeek-V4.1-Flash • model card',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash',role:'report',anchor:'552B backbone; 8B prefill / 16B decode; 890 bytes per token'},
 'dsv41-report':{title:'DeepSeek-V4.1-Flash technical report',url:HF+'deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf',role:'report',anchor:'§2.2 Causal Encoder-Decoder; §2.3 CSA2; §4.2.1 model setups'},
```

Then add `"dsv41":"dba1be0a40aa45a94ad051997016db3960a90277"` to `CONFIG_PINS`. The loop below `CONFIG_PINS` rewrites `/blob/main/` to the pinned revision for `-config` keys and for the ids listed in its `['ds2','ds3','k3']` array; add `'dsv41'` to that array so `dsv41-code` is pinned too.

- [ ] **Step 4: Extend `common`**

Add exactly these five to the `common` object on `src/models.mjs:52`, keeping it one line as the file's style requires: `encoderLayers:0,activeDecode:0,window:0,oRank:0,oGroups:0`

- [ ] **Step 5: Add the model**

Append to `MODELS`, one line per model as the file's style requires (wrapped here for reading only):

```js
 dsv41:{...common,id:'dsv41',name:'DeepSeek V4.1 Flash',family:'DeepSeek',repo:'deepseek-ai/DeepSeek-V4.1-Flash',generation:'V4.1-Flash',architecture:'ced-csa2',layers:40,encoderLayers:20,d:5120,ff:2304,heads:64,kvHeads:1,headDim:512,vDim:512,ropeDim:64,nopeDim:448,qRank:1280,oRank:1024,oGroups:8,experts:384,shared:1,topK:6,expertFF:2304,denseLayers:0,scoring:'sqrtsoftplus',routeScale:1.5,swigluLimit:10,window:128,swaLayers:2,kvSourceLayers:[2,8,14,20],indexSourceLayers:[2,8,14,20,24,28,32,36],compressEncoder:2,compressDecoder:1,indexer:128,indexerHeads:32,indexerBudget:512,candidateBlocks:2048,candidateBlockSize:8,candidateSource:20,residual:'mhc',streams:4,sinkhornIters:20,ngram:196613849600,engramLayers:[1,14],engramTables:[384006168,384016682],engramNgram:4,engramHeads:8,engramHeadDim:256,mtp:3,dsparkBlock:5,dsparkExperts:128,dsparkTopK:3,dsparkTargets:[37,38,39],vision:true,visionLayers:32,visionD:1024,visionHeads:16,visionFF:2816,visionPatch:14,vocab:129280,total:552e9,active:8e9,activeDecode:16e9,maxContext:1048576,configMax:1048576,kvBytesPerToken:890,kvFormat:'FP4 E2M1',swaKvFormat:'FP8',qkNorm:false,summary:'Causal encoder-decoder. Sparse reads shared across layers.',sources:['dsv41-config','dsv41-code','dsv41-engram','dsv41-kernel','dsv41-vision','dsv41-card','dsv41-report']}
```

`qkNorm:false` matches the MLA family: this model normalizes the query LoRA rank and the KV latent, not per-head Q/K as Qwen3 does.

- [ ] **Step 6: Run both tests**

Run: `node --test tests/core.test.mjs`
Expected: PASS. Other tests may now fail because `layerKind` does not know `'ced-csa2'` — that is Task 3. If any failure is **not** a `RangeError` or unknown-architecture error from `layerKind`/`layerPartition`, stop and report it.

- [ ] **Step 7: Rebind the claims this file's bytes invalidate**

`file.src/models.mjs` is a claimed consumer with a locked `expected` digest, and `MODELS.dsv41`'s
~70 fields are new consumers with no claim at all. Until both are handled, `check_coverage` raises
`CONSUMER_VALUE` and `COVERAGE_MISSING` — both FAIL — which turns the audit FAIL rather than
BLOCKED and breaks `test_cold_cache_is_blocked_not_certified`.

`verification/tests/test_observatory.py` also hardcodes the inventory shape and must move with it:
`len(inv['models'])` `8` → `9`, and `samples['decoder_layers']` `504` → `544`. That test asserts
`check_coverage(inv, claims) == []`, so it is the gate proving this step is complete.

**Also add `MODELS.dsv41.d` to the dependency list of all 24 all-models claims**, which is what
makes their "reconcile all supported models" statement true of nine rather than eight:

```
function.cacheLedger      function.evidenceSummary  function.ffnKind
function.formatBytes      function.formatNum        function.formatParams
function.formatTime       function.getModel         function.layerKind
function.layerPartition   function.networkLedger    function.parameterLedger
function.roofline         function.servingLedger    function.trainingLedger
file.build.mjs            file.src/app.mjs          file.src/core.mjs
file.src/diagram-meta.mjs file.src/diagram-model.mjs file.src/diagram-systems.mjs
file.src/diagrams.mjs     file.src/models.mjs       file.src/scene.mjs
```

`file.src/styles.css` and `file.index.html` are `presentation` with empty dependencies and are
deliberately not in this list.

Author a claim per new `MODELS.dsv41.*` field and rebind `file.src/models.mjs`. Classification
rules are in Task 13's brief and apply identically here: `id`/`name`/`family`/`sources` are
`presentation`; everything else in the `model` domain earns `CLAIM_ROLE` and `MODEL_DUAL_CODE`
BLOCKED because no framework implements this model, **which is the expected output and must not be
silenced by classifying a shape claim `assumed`**.

- [ ] **Step 8: Commit**

```bash
git add src/models.mjs src/core.mjs tests/core.test.mjs \
        verification/manifests/observatory.json verification/tests/test_observatory.py
git commit -m "Transcribe DeepSeek-V4.1-Flash into the model inventory

Five fields join common so the existing eight stay valid; everything
model-specific stays on dsv41, as attnResBlock and mixRank already do.

ngram is recorded as the derived 196613849600 rather than the card's rounded
196B, and a test ties it to engramTables x engramHeadDim so the two cannot
drift."
```

---

### Task 3: Teach the layer matrix the four CSA2 modes
>
> **Dispatched together with Task 2 — see the ruling there.**

**Files:**
- Modify: `src/core.mjs:17` (`layerKind`), `src/core.mjs:18` (`layerPartition`)
- Test: `tests/core.test.mjs`

**Interfaces:**
- Consumes: `MODELS.dsv41` from Task 2.
- Produces: `layerKind(id, layer)` returning `'swa' | 'csa2-full' | 'csa2-reindex' | 'csa2-reuse'` for `dsv41`; `layerPartition(id)` with those four keys added.

**Context you need:** `layerKind` takes a **one-indexed** layer; the config and report index from zero, so subtract one before comparing against `kvSourceLayers` / `indexSourceLayers`. `kvSourceLayers` is a **subset** of `indexSourceLayers` (layers 2, 8, 14, 20 appear in both), so the Full test must come first or every Full layer is misreported as Reindex. `layerPartition` keys are exactly `layerKind`'s return values, so they are hyphenated and need bracket access from other code.

DSpark's three draft layers are deliberately **not** partition members: `model.py` builds `self.layers` (40 blocks) separately from `self.mtp` (3 `DSparkBlock`s), and including them would make the partition sum to 43 rather than `m.layers`.

- [ ] **Step 1: Write the failing test**

```js
test('dsv41 layer kinds follow the published group structure', () => {
  const m = MODELS.dsv41;
  // encoder: SWA x2, then three groups of six led by a Full layer
  assert.equal(layerKind(m, 1), 'swa');
  assert.equal(layerKind(m, 2), 'swa');
  for (const l of [3, 9, 15]) assert.equal(layerKind(m, l), 'csa2-full', `layer ${l} owns its KV`);
  for (const l of [4, 8, 10, 14, 16, 20]) assert.equal(layerKind(m, l), 'csa2-reuse');
  // decoder: [Full, Reuse x3] then [Reindex, Reuse x3] x4
  assert.equal(layerKind(m, 21), 'csa2-full', 'the decoder projects its KV from H20 here');
  for (const l of [25, 29, 33, 37]) assert.equal(layerKind(m, l), 'csa2-reindex');
  for (const l of [22, 23, 24, 38, 39, 40]) assert.equal(layerKind(m, l), 'csa2-reuse');
});

test('dsv41 partition sums to its layer count and excludes the draft layers', () => {
  const p = layerPartition('dsv41');
  assert.deepEqual(p.swa, 2);
  assert.deepEqual(p['csa2-full'], 4);
  assert.deepEqual(p['csa2-reindex'], 4);
  assert.deepEqual(p['csa2-reuse'], 30);
  const sum = Object.values(p).reduce((a, b) => a + b, 0);
  assert.equal(sum, MODELS.dsv41.layers, 'DSpark draft layers are mtp, not backbone layers');
});

test('every model partitions to exactly its layer count', () => {
  for (const m of Object.values(MODELS)) {
    const sum = Object.values(layerPartition(m)).reduce((a, b) => a + b, 0);
    assert.equal(sum, m.layers, `${m.id} partition must sum to ${m.layers}`);
  }
});

test('layerKind still rejects layers outside the backbone', () => {
  assert.throws(() => layerKind('dsv41', 41), RangeError, 'draft layers are not addressable here');
  assert.throws(() => layerKind('dsv41', 0), RangeError);
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'dsv41 layer kinds' tests/core.test.mjs`
Expected: FAIL — `layerKind` falls through to `return m.architecture` and yields `'ced-csa2'`.

- [ ] **Step 3: Add the dispatch branch**

In `layerKind`, after the existing `RangeError` guard and **before** the `hybrid-kda` branch:

```js
if(m.architecture==='ced-csa2'){const l=layer-1;if(l<m.swaLayers)return'swa';if(m.kvSourceLayers.includes(l))return'csa2-full';if(m.indexSourceLayers.includes(l))return'csa2-reindex';return'csa2-reuse';}
```

- [ ] **Step 4: Add the partition keys**

In `layerPartition`, extend the initializer:

```js
const r={mla:0,gqa:0,kda:0,delta:0,qsa:0,swa:0,'csa2-full':0,'csa2-reindex':0,'csa2-reuse':0};
```

- [ ] **Step 5: Run the tests**

Run: `node --test tests/core.test.mjs`
Expected: PASS, including `every model partitions to exactly its layer count` across all nine.

- [ ] **Step 6: Commit**

```bash
git add src/core.mjs tests/core.test.mjs
git commit -m "Derive the four CSA2 modes per layer

Mode assignment falls out of compress_ratios crossed with kv_source and
index_source layers. kvSourceLayers is a subset of indexSourceLayers, so Full
is tested first; otherwise every Full layer reports as Reindex.

The partition sums to 40, not 43: DSpark's draft layers are mtp, matching how
model.py builds self.mtp separately from self.layers."
```

---

### Task 4: Reclassify the cache into context-linear and bounded

**Files:**
- Modify: `src/core.mjs:39-52` (`cacheLedger`)
- Test: `tests/core.test.mjs`

**Interfaces:**
- Consumes: `layerPartition` keys from Task 3.
- Produces: `cacheLedger(id, settings)` returning the existing fields plus `contextLinear`, `bounded`, `mainKV`, `windowKV`. `total` is **unchanged in value for all nine models**.

**Context you need:** `cacheLedger` already returns a field called `global`, and it is a **layer count** (`p.mla+p.gqa+p.qsa`), consumed by `src/app.mjs:69` and `src/diagram-systems.mjs:14,19`. Do not reuse that name for a byte total. The new byte fields are `contextLinear` and `bounded`.

The reclassification is of terms that already exist: `mla`, `gqa` and `indexer` all carry an `S` factor; `recurrent` and `conv` do not. So `kimi3`, `q3827` and `q38flash` already hold bounded state and must not report `bounded: 0`.

`dsv41`'s KV precision is trained in — §2.4 introduces QAT specifically to enable FP4 main KV — so it is an architectural fact, not a deployment choice, and the user-selectable `s.kvBytes` deliberately does not apply to it. FP4 E2M1 with one E4M3 scale per 16 channels is `0.5 + 1/16 = 0.5625` bytes per channel; SWA KV is retained at FP8, `1` byte per channel.

- [ ] **Step 1: Write the failing test**

```js
test('every model splits its cache into context-linear and bounded halves', () => {
  for (const m of Object.values(MODELS)) {
    const c = cacheLedger(m, { context: 8192 });
    assert.equal(c.contextLinear + c.bounded, c.total, `${m.id} halves must sum to total`);
    assert.ok(c.contextLinear >= 0 && c.bounded >= 0, `${m.id} no negative cache`);
  }
});

test('bounded state does not move with context, for any model', () => {
  for (const m of Object.values(MODELS)) {
    const near = cacheLedger(m, { context: 4096 }).bounded;
    const far = cacheLedger(m, { context: 262144 }).bounded;
    assert.equal(near, far, `${m.id} bounded term must not scale with S`);
  }
});

test('models with recurrent state report it as bounded, not zero', () => {
  for (const id of ['q3827', 'q38flash', 'kimi3']) {
    assert.ok(cacheLedger(id, { context: 8192 }).bounded > 0, `${id} holds recurrent state`);
  }
});

test('dsv41 global KV is 720 bytes per token from the four Full-mode layers', () => {
  const one = cacheLedger('dsv41', { context: 1, batch: 1 });
  // 3 encoder Full layers at m=2 and 1 decoder Full layer at m=1, FP4 at 0.5625 B/channel
  assert.equal(one.mainKV, 720);
  assert.equal(one.contextLinear, 720);
  const ten = cacheLedger('dsv41', { context: 10, batch: 1 });
  assert.equal(ten.mainKV, 7200, 'main KV is linear in context');
});

test('dsv41 SWA KV is bounded by the window across all forty layers', () => {
  const c = cacheLedger('dsv41', { context: 1048576, batch: 1 });
  assert.equal(c.windowKV, 128 * 40 * 512, 'window x layers x headDim at FP8');
  assert.equal(c.bounded, c.windowKV);
});

test('dsv41 reports its global-history layers as the thirty-eight CSA2 layers', () => {
  assert.equal(cacheLedger('dsv41', {}).global, 38, 'the two SWA-only layers hold no global history');
  assert.equal(cacheLedger('dsv41', {}).linear, 0, 'no recurrent attention in this model');
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'context-linear' tests/core.test.mjs`
Expected: FAIL — `c.contextLinear` is `undefined`, so the sum is `NaN`.

- [ ] **Step 3: Implement the split**

Rewrite `cacheLedger`'s body between the `const linear=` line and the `return`:

```js
 const csa2=p['csa2-full']+p['csa2-reindex']+p['csa2-reuse'];
 const linear=p.kda+p.delta,global=p.mla+p.gqa+p.qsa+csa2;
 const mla=p.mla*B*S*(m.kvRank+m.ropeDim)*s.kvBytes;
 const gqa=(p.gqa+p.qsa)*B*S*2*m.kvHeads*m.headDim*s.kvBytes;
 const recurrent=linear*B*m.linearHeads*m.linearHeadDim*m.linearHeadDim*s.stateBytes;
 const conv=linear*B*(2*m.linearKeyHeads+m.linearHeads)*m.linearHeadDim*m.convKernel*2;
 const indexer=p.qsa ? p.qsa*B*Math.ceil(S/m.indexerCompression)*m.indexer*s.kvBytes : 0;
 // FP4 E2M1 plus one E4M3 scale per 16 channels. The precision is trained in (QAT), so it is
 // an architectural fact and s.kvBytes deliberately does not apply. SWA KV is retained at FP8.
 let mainKV=0,windowKV=0;
 if(m.architecture==='ced-csa2'){
  for(const l of m.kvSourceLayers){const ratio=l<m.encoderLayers?m.compressEncoder:m.compressDecoder;mainKV+=B*S*(m.headDim/ratio)*(0.5+1/16);}
  windowKV=B*m.window*m.layers*m.headDim*1;
 }
 const history=mla+gqa+mainKV;
 const contextLinear=history+indexer;
 const bounded=recurrent+conv+windowKV;
 const total=contextLinear+bounded;
```

Add `mainKV,windowKV,contextLinear,bounded` to the returned object. Extend `formula` with a `ced-csa2` case ahead of the existing ternary:

```js
   formula:m.architecture==='ced-csa2'?`${m.kvSourceLayers.length} Full-mode layers × B × S × (${m.headDim} / m) × b_FP4  +  B × ${m.window} × ${m.layers} × ${m.headDim} × b_FP8`:p.mla?`…`:`…`
```

Add one exclusion when `m.architecture==='ced-csa2'`:

```js
 if(m.architecture==='ced-csa2')exclusions.push('Indexer K storage, candidate-pool bookkeeping and SWA Bounded Replay recomputation');
```

- [ ] **Step 4: Run the tests**

Run: `node --test tests/core.test.mjs`
Expected: PASS. Every model's `total` must be numerically identical to before this task — if any pre-existing total-based assertion fails, the split is wrong, not the old test.

- [ ] **- [ ] **Step 5: Commit: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.src/core.mjs`, `function.cacheLedger`

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 6: Commit**

```bash
git add src/core.mjs tests/core.test.mjs
git commit -m "Split every cache into its context-linear and bounded halves

The terms already existed: mla, gqa and indexer carry S; recurrent and conv do
not. This names the distinction rather than adding a field, so total is
unchanged in value for all nine models and no total binding moves.

The new byte fields are contextLinear and bounded, because cacheLedger's
existing 'global' is a layer count that app.mjs and diagram-systems.mjs read.

dsv41 derives 720 B/token of FP4 main KV over its four Full-mode layers and a
bounded FP8 SWA window. Its KV precision is trained in via QAT, so kvBytes
does not apply and the ledger says so."
```

---

### Task 5: Derive the CED parameter count

**Files:**
- Modify: `src/core.mjs:20-38` (`parameterLedger`)
- Test: `tests/core.test.mjs`

**Interfaces:**
- Consumes: `MODELS.dsv41`.
- Produces: `parameterLedger('dsv41')` with `approximate:false` and an exactly derived `main`.

**Context you need:** `main` starts as `m.total` with `approximate=true`; only the `gqa` and `mla` branches replace it with a calculation. Adding a third calculated branch means `dsv41` displays a derived figure, as `ds2`/`ds3` already do.

Shapes come from `Attention.__init__` and `Gate.__init__` in `inference/model.py`. `wo_a` is block-diagonal over `oGroups`, so its parameter count reduces to `oRank * heads * headDim`. The MoE gate carries a second bias (`bias_vl`) because vision is enabled. mHC contributes `2 × mixHC × (streams × d)` plus two bias vectors and two 3-element scales per block, where `mixHC = (2 + streams) × streams`.

Engram's embedding tables are **not** part of this: they stay in `ngram`, exactly as `q38flash` does. Engram's `wkv` projection is a real backbone parameter and is counted.

**Do not invent a DSpark parameter count** — none is published. Exclude it and say so in `scope`.

- [ ] **Step 1: Write the failing test**

```js
test('dsv41 parameters are derived, not rounded', () => {
  const r = parameterLedger('dsv41');
  assert.equal(r.approximate, false, 'this branch calculates rather than quoting');
  assert.equal(r.ngram, 196613849600);
  assert.equal(r.resident, r.main + r.ngram);
});

test('the derived dsv41 backbone reconciles with the published 552B', () => {
  const r = parameterLedger('dsv41');
  const published = 552e9;
  const drift = Math.abs(r.main - published) / published;
  assert.ok(drift < 0.01, `derived ${r.main} drifts ${(drift * 100).toFixed(3)}% from 552B`);
  assert.ok(r.main < published * 1.01, 'the text backbone must not exceed the published figure');
});

test('dsv41 does not fabricate a DSpark parameter count', () => {
  const r = parameterLedger('dsv41');
  assert.equal(r.mtp, 0, 'no DSpark parameter count is published; mtp:3 is a layer count');
  assert.equal(r.hasMtp, true, 'the mechanism exists even though its size is unpublished');
  assert.match(r.scope, /DSpark/, 'scope must name what it excludes');
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'dsv41 parameters are derived' tests/core.test.mjs`
Expected: FAIL — `approximate` is `true` because no branch matches `'ced-csa2'`.

- [ ] **Step 3: Add the branch**

Insert after the `mla` branch, before `const ngram=`:

```js
 }else if(m.architecture==='ced-csa2'){
   const a=m.d*m.qRank+m.qRank+m.qRank*m.heads*m.headDim+m.d*m.headDim+m.headDim+m.oRank*m.heads*m.headDim+m.oGroups*m.oRank*m.d+m.heads;
   const mixHC=(2+m.streams)*m.streams;
   const mhc=m.layers*(2*mixHC*m.streams*m.d+2*mixHC+6);
   const gate=m.layers*(m.d*m.experts+m.experts*(m.vision?2:1));
   const engram=m.engramLayers.length*((m.engramNgram-1)*m.engramHeads*m.engramHeadDim*m.d*(m.streams+1)+2*m.streams*m.d);
   attention=m.layers*a;
   main=embedding+attention+routed+shared+gate+mhc+engram+(2*m.layers+1)*m.d;approximate=false;
 }
```

- [ ] **Step 4: Extend the scope string**

`scope` is currently a two-way ternary on `m.id==='q38flash'`. Add a `ced-csa2` case ahead of it:

```js
m.architecture==='ced-csa2'?'Calculated text backbone: attention, MoE, gating, mHC and Engram projections, untied vocabulary and norms. Engram tables are carried separately as n-gram residency. Excludes the vision encoder, the DSpark draft layers — whose parameter count is unpublished — and quantization metadata.':
```

- [ ] **Step 5: Run the tests and read the reconciliation**

Run: `node --test --test-name-pattern 'dsv41' tests/core.test.mjs`
Expected: PASS. The derived backbone should land near `551.8e9` against a published `552e9`, roughly 0.03% drift.

**If drift exceeds 1%, do not widen the tolerance.** Re-derive the shapes against `Attention.__init__` and `Gate.__init__`; a large gap means a term is wrong or double-counted, and the test is doing its job.

- [ ] **- [ ] **Step 6: Commit: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.src/core.mjs`, `function.parameterLedger`

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 7: Commit**

```bash
git add src/core.mjs tests/core.test.mjs
git commit -m "Derive the DeepSeek-V4.1-Flash parameter count

Third calculated branch beside gqa and mla. Shapes come from the publisher's
Attention.__init__ and Gate.__init__: a single 512-wide latent KV for all 64
query heads, a block-diagonal output projection over eight groups, a second
routing bias because vision is enabled, and the per-block mHC coefficients.

A test reconciles the derived backbone against the published 552B rather than
displaying the published figure, so a wrong shape shows up as drift.

No DSpark parameter count is invented, because none is published."
```

---

### Task 6: Make the ledgers phase-aware

**Files:**
- Modify: `src/core.mjs:6` (`DEFAULT_STATE`), `src/core.mjs:7` (`sanitizeState` choices), `src/core.mjs:53` (`servingLedger`)
- Test: `tests/core.test.mjs`

**Interfaces:**
- Consumes: `cacheLedger` from Task 4, `parameterLedger` from Task 5, `MODELS.dsv41.encoderLayers`.
- Produces: `servingLedger(id, {phase})` returning `phase`, `phaseLayers`, `phaseActive` alongside its existing fields. `phase` is `'prefill' | 'decode'`, default `'decode'`.

**Context you need:** `sanitizeState` clamps every field against a whitelist in its `choices` object; a value absent from `choices` is silently replaced by the `DEFAULT_STATE` value. So `phase` must be added to **both** or it will never take effect.

For the eight models without a phase split, `encoderLayers` is `0` and `activeDecode` is `0`, so both phases must return `m.layers` and `m.active` — the phase dimension is a no-op for them, and a test pins that.

- [ ] **Step 1: Write the failing test**

```js
test('dsv41 prefill computes only the encoder half', () => {
  const pre = servingLedger('dsv41', { phase: 'prefill' });
  const dec = servingLedger('dsv41', { phase: 'decode' });
  assert.equal(pre.phaseLayers, 20);
  assert.equal(dec.phaseLayers, 40);
  assert.equal(pre.phaseActive, 8e9);
  assert.equal(dec.phaseActive, 16e9);
});

test('the prefill saving is the depth ratio the architecture claims', () => {
  const m = MODELS.dsv41;
  const depthRatio = m.encoderLayers / m.layers;
  const activeRatio = servingLedger(m, { phase: 'prefill' }).phaseActive /
                      servingLedger(m, { phase: 'decode' }).phaseActive;
  // published actives are rounded to whole billions, so this is a tolerance not an identity
  assert.ok(Math.abs(activeRatio - depthRatio) < 0.05,
    `active ratio ${activeRatio} should track depth ratio ${depthRatio}`);
});

test('phase is a no-op for models without an encoder/decoder split', () => {
  for (const id of ['ds2', 'ds3', 'q38', 'q330', 'q3235', 'q3827', 'q38flash', 'kimi3']) {
    const pre = servingLedger(id, { phase: 'prefill' });
    const dec = servingLedger(id, { phase: 'decode' });
    assert.equal(pre.phaseLayers, MODELS[id].layers, `${id} computes every layer at prefill`);
    assert.equal(pre.phaseActive, dec.phaseActive, `${id} publishes one active figure`);
  }
});

test('an unknown phase falls back to the default rather than propagating', () => {
  assert.equal(servingLedger('dsv41', { phase: 'wishful' }).phase, 'decode');
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'prefill computes only' tests/core.test.mjs`
Expected: FAIL — `pre.phaseLayers` is `undefined`.

- [ ] **Step 3: Add the state field**

In `DEFAULT_STATE`, add `phase:'decode'`. In `sanitizeState`'s `choices`, add `phase:['prefill','decode']`.

- [ ] **Step 4: Extend `servingLedger`**

```js
export function servingLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),p=parameterLedger(m),cache=cacheLedger(m,s);const weights=p.resident*s.weightBytes;
 const phaseLayers=s.phase==='prefill'&&m.encoderLayers?m.encoderLayers:m.layers;
 const phaseActive=s.phase==='prefill'?m.active:(m.activeDecode||m.active);
 return{...cache,weights,perRank:(weights+cache.total)/s.ranks,replicas:s.placement==='split'?2:1,fleetWeights:weights*(s.placement==='split'?2:1),phase:s.phase,phaseLayers,phaseActive,scope:p.scope};}
```

- [ ] **Step 5: Run the tests**

Run: `node --test tests/core.test.mjs`
Expected: PASS, all four.

- [ ] **- [ ] **Step 6: Run the full engineering gate: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.src/core.mjs`, `function.sanitizeState`, `function.servingLedger`, plus the new
`DEFAULT_STATE.phase` field, which is `presentation` because its consumer key starts with
`DEFAULT_STATE.`

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 7: Run the full engineering gate**

Run: `npm run verify:engineering`
Expected: PASS. This is the first point where the whole matrix, both builds and the browser suite run against the new model. Report the state count and geometry-defect count.

- [ ] **Step 8: Commit**

```bash
git add src/core.mjs tests/core.test.mjs
git commit -m "Make serving phase-aware so prefill depth is derived

CED computes only the bottom half during prefill, which is what the 8B/16B
split is. phaseLayers now falls out of encoderLayers instead of 8B being a
transcribed constant, and a tolerance test ties the active ratio to the depth
ratio.

phase is a no-op for the eight models without a split, and a test pins that so
the dimension cannot quietly change their numbers."
```

---

### Task 7: Structural invariants for CSA2, CED, Engram and DSpark

**Files:**
- Create: `verification/ced.py`
- Modify: `verification/schematics.py:200-226` (`run`)
- Test: `verification/tests/test_ced.py`

**Interfaces:**
- Consumes: `MODELS.dsv41` via `tools/export-verification.mjs`.
- Produces: `judge_architecture(models, samples=None) -> list[Finding]`, `judge_claim_model_coverage(claims, models) -> list[Finding]`, `modes_from_report_structure(...) -> list[str]`, `inventory(root) -> tuple[dict, dict]`. Task 8 adds `judge_kv_budget` and `judge_mhc` to the same module; Task 9 adds `judge_publisher_agreement`.

**Context you need:** `schematics.py` builds findings with `_ok(code,subject,message)` for PASS and `_bad(code,subject,message,remedy)` for FAIL, both wrapping `Finding(code,domain,status,subject,message,remedy)` from `.model`. Emit a PASS finding for every check that runs, so a reader can tell the check happened — `aesthetics.py` was once caught reporting nothing when a check silently did not run.

Read the model inventory the way `aesthetics.py:370-376` does, from `tools/export-verification.mjs`, so the verifier reads what the application actually exposes rather than a literal.

The point of `CED_MODE_DISAGREEMENT` is that the two derivations are **independent**: one reads the config's `kvSourceLayers`/`indexSourceLayers` arrays, the other generates the assignment from the report's §4.2.1 prose ("three identically configured groups of six layers"; "five groups of four layers"). Agreement is evidence. If you implement the second derivation by reading the same arrays, the check becomes a tautology and is worthless.

- [ ] **Step 1: Write the failing tests**

Append to `verification/tests/test_ced.py`:

```python
from verification.ced import (judge_architecture, modes_from_report_structure,
                              judge_mhc_streams, judge_claim_model_coverage)
# samples_fixture arrives with Task 8; judge_architecture takes samples optionally, so the
# invariant tests below call it with the model map alone.

def dsv41_fixture(**overrides):
    """The transcribed shape, so a test can break one field at a time."""
    m = {'id':'dsv41','architecture':'ced-csa2','layers':40,'encoderLayers':20,'swaLayers':2,
         'kvSourceLayers':[2,8,14,20],'indexSourceLayers':[2,8,14,20,24,28,32,36],
         'compressEncoder':2,'compressDecoder':1,'candidateSource':20,
         'engramLayers':[1,14],'engramTables':[384006168,384016682],'engramHeadDim':256,
         'engramNgram':4,'engramHeads':8,'ngram':196613849600,
         'mtp':3,'dsparkTargets':[37,38,39],'dsparkBlock':5,'dsparkExperts':128,
         'dsparkTopK':3,'experts':384,'topK':6,'streams':4,'headDim':512,
         'indexer':128,'kvBytesPerToken':890,'active':8e9,'activeDecode':16e9,
         'residual':'mhc'}
    m.update(overrides)
    return {'dsv41': m}

def codes(findings, status=None):
    return {f.code for f in findings if status is None or f.status == status}

class ArchitectureInvariantTests(unittest.TestCase):
    def test_the_transcribed_architecture_satisfies_every_invariant(self):
        f = judge_architecture(dsv41_fixture())
        self.assertEqual(codes(f, 'FAIL'), set(), 'the real shape must be clean')
        self.assertIn('CED_MODE_DISAGREEMENT', codes(f, 'PASS'), 'the check must report that it ran')

    def test_the_two_mode_derivations_are_compared_and_can_disagree(self):
        """Move one Full layer in the config arrays only. The report-derived structure is
        unchanged, so the two must diverge."""
        f = judge_architecture(dsv41_fixture(kvSourceLayers=[2,9,14,20]))
        self.assertIn('CED_MODE_DISAGREEMENT', codes(f, 'FAIL'))

    def test_a_full_layer_outside_the_index_sources_is_caught(self):
        f = judge_architecture(dsv41_fixture(indexSourceLayers=[8,14,20,24,28,32,36]))
        self.assertIn('CSA2_INDEX_SUBSET', codes(f, 'FAIL'))

    def test_a_reuse_layer_with_nothing_to_reuse_is_caught(self):
        """Drop the encoder's first Full layer: layers 3-7 then read a cache no layer wrote."""
        f = judge_architecture(dsv41_fixture(kvSourceLayers=[8,14,20]))
        self.assertIn('CED_REUSE_UNREACHABLE', codes(f, 'FAIL'))

    def test_a_decoder_deriving_its_own_kv_is_caught(self):
        """Two decoder KV sources means the decoder no longer projects solely from H20."""
        f = judge_architecture(dsv41_fixture(kvSourceLayers=[2,8,14,20,30]))
        self.assertIn('CED_KV_PROVENANCE', codes(f, 'FAIL'))

    def test_a_candidate_pool_built_after_its_consumers_is_caught(self):
        f = judge_architecture(dsv41_fixture(candidateSource=28))
        self.assertIn('CSA2_CANDIDATE_SOURCE', codes(f, 'FAIL'))

    def test_a_rounded_engram_total_is_caught(self):
        f = judge_architecture(dsv41_fixture(ngram=196_000_000_000))
        self.assertIn('ENGRAM_TOTAL_DERIVATION', codes(f, 'FAIL'))

    def test_mismatched_engram_tables_and_layers_are_caught(self):
        f = judge_architecture(dsv41_fixture(engramTables=[384006168]))
        self.assertIn('ENGRAM_LAYER_BOUNDS', codes(f, 'FAIL'))

    def test_a_wrong_hash_column_count_is_caught(self):
        f = judge_architecture(dsv41_fixture(engramNgram=1))
        self.assertIn('ENGRAM_HASH_COLUMNS', codes(f, 'FAIL'))

    def test_a_draft_head_without_one_target_per_layer_is_caught(self):
        f = judge_architecture(dsv41_fixture(dsparkTargets=[38,39]))
        self.assertIn('DSPARK_TARGET_COUNT', codes(f, 'FAIL'))

    def test_a_trivial_draft_block_is_caught(self):
        f = judge_architecture(dsv41_fixture(dsparkBlock=1))
        self.assertIn('DSPARK_BLOCK_TRIVIAL', codes(f, 'FAIL'))

    def test_a_draft_moe_identical_to_the_backbone_is_caught(self):
        f = judge_architecture(dsv41_fixture(dsparkExperts=384, dsparkTopK=6))
        self.assertIn('DSPARK_MOE_DISTINCT', codes(f, 'FAIL'))

    def test_a_manifest_claim_omitting_a_supported_model_is_caught(self):
        """The 24 all-models claims say they reconcile every supported model. One that names
        eight of nine is asserting completeness it does not have."""
        claims = [{'id': 'function.cacheLedger',
                   'dependencies': ['MODELS.ds2.d', 'MODELS.ds3.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'FAIL'))

    def test_a_single_model_scene_claim_is_not_required_to_cover_all_models(self):
        """2,520 per-scene claims each depend on their own one model; that is correct."""
        claims = [{'id': 'scene.ds2.model.block.norm1.abc',
                   'dependencies': ['MODELS.ds2.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertEqual(codes(f, 'FAIL'), set())

    def test_a_complete_all_models_claim_passes(self):
        claims = [{'id': 'file.src/core.mjs',
                   'dependencies': ['MODELS.ds2.d', 'MODELS.ds3.d', 'MODELS.dsv41.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'PASS'))

    def test_models_without_the_architecture_are_left_alone(self):
        f = judge_architecture({'ds3': {'id':'ds3','architecture':'mla','layers':61}})
        self.assertEqual(f, [], 'these invariants apply only to ced-csa2')

class ReportStructureTests(unittest.TestCase):
    def test_the_report_group_structure_reproduces_the_published_assignment(self):
        modes = modes_from_report_structure(swa=2, encoder=20, total=40,
                                            encoder_group=6, decoder_group=4)
        self.assertEqual(len(modes), 40)
        self.assertEqual([i for i, m in enumerate(modes) if m == 'csa2-full'], [2, 8, 14, 20])
        self.assertEqual([i for i, m in enumerate(modes) if m == 'csa2-reindex'], [24, 28, 32, 36])
        self.assertEqual(modes[:2], ['swa', 'swa'])
        self.assertEqual(modes.count('csa2-reuse'), 30)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.ced'`.

- [ ] **Step 3: Create the module**

`verification/ced.py`. The report-structure derivation must not consult the config arrays:

```python
"""Architecture-specific invariants for the Causal Encoder-Decoder family.

Every check here is either a cross-check between two independent derivations or a
structural property that a mis-transcription breaks. A check that restates config.json
proves nothing and does not belong in this module.
"""
from __future__ import annotations
import json,subprocess
from pathlib import Path
from .model import Finding

ARCH='ced-csa2'

def _ok(code,subject,message):return Finding(code,'schematics','PASS',subject,message)
def _bad(code,subject,message,remedy):return Finding(code,'schematics','FAIL',subject,message,remedy)

def modes_from_report_structure(*,swa:int,encoder:int,total:int,encoder_group:int,decoder_group:int)->list[str]:
    """Generate the mode assignment from the report's prose, §4.2.1, and nothing else.

    "For the first two layers, we use pure sliding window attention. The remaining 18
    encoder layers ... are divided into three identically configured groups of six layers.
    In each group, the first layer operates in Full Mode, and the remaining five ... in
    Reuse Mode. The 20 decoder layers ... are divided into five groups of four layers. In
    the first group, the first layer operates in Full Mode ... The remaining four groups
    ... the first layer operates in Reindex Mode."

    Deliberately takes no layer-id arrays: reading them would make the comparison in
    judge_architecture a tautology.
    """
    modes=['swa']*swa
    for _ in range(swa,encoder,encoder_group):
        modes.append('csa2-full');modes+=['csa2-reuse']*(encoder_group-1)
    for i,_ in enumerate(range(encoder,total,decoder_group)):
        modes.append('csa2-full' if i==0 else 'csa2-reindex');modes+=['csa2-reuse']*(decoder_group-1)
    return modes

def modes_from_config(m:dict)->list[str]:
    """The same assignment as the application derives it: from the pinned layer-id arrays."""
    out=[]
    for l in range(m['layers']):
        if l<m['swaLayers']:out.append('swa')
        elif l in m['kvSourceLayers']:out.append('csa2-full')
        elif l in m['indexSourceLayers']:out.append('csa2-reindex')
        else:out.append('csa2-reuse')
    return out
```

Then `judge_architecture(models)`, iterating only models whose `architecture` is `ARCH`, emitting one PASS or FAIL per check. Implement, in order:

1. `CED_MODE_DISAGREEMENT` — `modes_from_config(m)` equals `modes_from_report_structure(swa=m['swaLayers'], encoder=m['encoderLayers'], total=m['layers'], encoder_group=6, decoder_group=4)`. Remedy: "Re-derive the layer-id arrays from the publisher config; do not adjust the report structure to match a transcription."
2. `CSA2_INDEX_SUBSET` — `set(m['kvSourceLayers']) <= set(m['indexSourceLayers'])`.
3. `CED_REUSE_UNREACHABLE` — walking layers in order, every layer whose mode is `csa2-reuse` or `csa2-reindex` has some earlier layer in the same band (`l < encoderLayers` versus `l >= encoderLayers`) present in `kvSourceLayers`.
4. `CED_BAND_RATIO` — `compressEncoder` and `compressDecoder` are both `>= 1`, and they differ; identical ratios would make the two bands indistinguishable.
5. `CED_KV_PROVENANCE` — exactly one element of `kvSourceLayers` is `>= encoderLayers`, and it equals `encoderLayers`.
6. `CSA2_CANDIDATE_SOURCE` — `candidateSource == encoderLayers`, and no element of `indexSourceLayers` that is `>= encoderLayers` is `< candidateSource`.
7. `ENGRAM_TOTAL_DERIVATION` — `sum(m['engramTables']) * m['engramHeadDim'] == m['ngram']`.
8. `ENGRAM_HASH_COLUMNS` — `(m['engramNgram']-1) * m['engramHeads'] == 24`.
9. `ENGRAM_LAYER_BOUNDS` — `len(engramTables) == len(engramLayers)` and every Engram layer id is `< layers`.
10. `DSPARK_TARGET_COUNT` — `len(m['dsparkTargets']) == m['mtp']` and every target is `< layers`.
11. `DSPARK_BLOCK_TRIVIAL` — `m['dsparkBlock'] > 1`.
12. `DSPARK_MOE_DISTINCT` — `(m['dsparkExperts'], m['dsparkTopK']) != (m['experts'], m['topK'])`.
13. `CED_MTP_RATIO` — every draft layer index (`m['layers']` through `m['layers']+m['mtp']-1`) is absent from `kvSourceLayers` and `indexSourceLayers`. `DSparkAttention.forward` opens with `assert self.compress_ratio == 0`, so a draft layer owning a compressed cache contradicts the publisher's own assertion.
14. `CED_PARTITION_SUM` — `sum(samples['partitions'][id].values()) == m['layers']`. Takes the partition from the inventory, so it checks what the application computes rather than re-deriving it here. `samples` defaults to `None`; when it is absent, emit this one check as **BLOCKED** naming the missing inventory rather than skipping it silently, so a caller that forgets to pass samples is visible in the report.
15. `CED_PHASE_DEPTH` — `m['encoderLayers'] > 0` and `m['encoderLayers'] < m['layers']`; a phase split that covers every layer or none is not a split.
16. `CED_ACTIVE_RATIO` — **BLOCKED, not FAIL**: `abs(m['active']/m['activeDecode'] - m['encoderLayers']/m['layers'])` is within `0.05`. BLOCKED because the published actives are rounded to whole billions and the embedding and Engram terms are depth-independent, so an exact identity would be overfitting. The message must print both ratios.

17. `CLAIM_MODEL_COVERAGE` — **a check on the manifest, not on a model.** Read
    `verification/manifests/observatory.json` and, for every claim whose `dependencies` name two
    or more distinct `MODELS.<id>.d` entries, require that set to equal the full model set from
    the inventory. Today 24 claims declare they reconcile every supported model; a manifest that
    omits one is asserting completeness it does not have. This is a checklist replaced by a check:
    it makes the next person to add a model unable to forget these 24, without anyone maintaining
    a list. Findings name the claim id and the missing model ids. Severity FAIL.

`judge_architecture` therefore takes the inventory `samples` as an optional second argument —
`judge_architecture(models, samples=None)` — rather than shelling out to the exporter again. Only
check 14 reads it, and it reports BLOCKED when samples is absent, mirroring how Task 9's
`judge_publisher_agreement` blocks on an absent cache.

Add one negative test for that path:

```python
    def test_an_absent_inventory_blocks_the_partition_check(self):
        f = judge_architecture(dsv41_fixture())
        self.assertIn('CED_PARTITION_SUM', codes(f, 'BLOCKED'))
        self.assertNotIn('CED_PARTITION_SUM', codes(f, 'PASS'))
```

Add an inventory reader mirroring `aesthetics.py:370-376`:

```python
def inventory(root:Path)->tuple[dict,dict]:
    """Read what the application actually exposes, never a literal copy of it.
    Returns (models, samples) from one exporter run."""
    out=subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,
                       capture_output=True,check=True)
    doc=json.loads(out.stdout)
    return doc['models'],doc['samples']
```

- [ ] **Step 4: Run the tests**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: PASS. Every negative test must fail for **its own** code — if breaking `kvSourceLayers` trips four checks at once, the checks are entangled; assert the specific code is present, which the tests already do.

- [ ] **Step 5: Wire it into the gate**

In `verification/schematics.py`, import and call once per run, after the per-scene loop:

```python
from .ced import judge_architecture,inventory
...
    findings.extend(judge_row_alignment(data,tolerance))
    inv_models,inv_samples=inventory(root)   # one subprocess, both halves
    findings.extend(judge_architecture(inv_models,inv_samples))
    target=loads((root/'verification/manifests/observatory.json').read_text())
    findings.extend(judge_claim_model_coverage(target['claims'],inv_models))
```

- [ ] **Step 6: Run the gate**

Run: `npm run verify:schematics`
Expected: PASS. The report's "Checks that passed" table must now list the new codes. If any is absent, the check did not run and the PASS finding is missing.

- [ ] **Step 7: Commit**

```bash
git add verification/ced.py verification/schematics.py verification/tests/test_ced.py
git commit -m "Add architecture invariants for CSA2, CED, Engram and DSpark

CED_MODE_DISAGREEMENT derives the mode assignment twice and compares: once from
the pinned layer-id arrays, once generated from the report's group structure.
The second derivation deliberately takes no layer ids, because reading them
would make the comparison a tautology.

CED_REUSE_UNREACHABLE is the invariant that makes layer-dimension cache sharing
well-founded -- every Reuse layer must have an earlier same-band Full layer to
read -- and nothing checked it before.

ENGRAM_TOTAL_DERIVATION rejects the card's rounded 196B, because the exact
figure is derivable from the two table sizes.

Each check emits a PASS finding when it runs, so a silently skipped check is
visible in the report rather than indistinguishable from a clean one."
```

---

### Task 8: Cross-check the KV budget and the mHC coefficients

**Files:**
- Modify: `verification/ced.py`, `verification/schematics.py`
- Modify: `tools/export-verification.mjs`
- Test: `verification/tests/test_ced.py`

**Interfaces:**
- Consumes: `cacheLedger` fields from Task 4; `inventory` from Task 7.
- Produces: `judge_kv_budget(models, samples) -> list[Finding]`, `judge_mhc(models) -> list[Finding]`.

**Context you need:** `tools/export-verification.mjs` already emits `samples.logical_state`, the full `cacheLedger` output per model at the default state (`B=1`, `S=32768`). `samples` and `models` are **not** claim-bearing — `check_coverage` reads only `fields`, `functions` and `scenes` — so adding to `samples` costs no claims.

To check the bounded invariant from Python you need the ledger at two contexts, so add a second sample. Bytes per token is `mainKV / (B * S)`.

The residual bound is the honest way to hold the reported 890 B/token accountable. Derived main KV is 720 B; indexer K over the same four Full-mode layers is `3 × (128/2) + 1 × (128/1) = 320` channels per token; the residual of 170 B implies `0.53` B/channel, which sits between the FP4 floor (`0.5`) and FP8 (`1.0`). **Do not narrow this bound to make it tighter, and do not pick a precision that makes 890 come out exactly — the report states none.**

- [ ] **Step 1: Write the failing tests**

```python
from verification.ced import judge_kv_budget, judge_mhc

def samples_fixture(main_per_token=720, bounded=128*40*512, S=32768):
    return {'logical_state': {'dsv41': {'mainKV': main_per_token*S, 'bounded': bounded,
                                        'contextLinear': main_per_token*S, 'B': 1, 'S': S}},
            'logical_state_far': {'dsv41': {'bounded': bounded, 'B': 1, 'S': S*4}}}

class KvBudgetTests(unittest.TestCase):
    def test_the_real_budget_agrees_between_python_and_the_ledger(self):
        f = judge_kv_budget(dsv41_fixture(kvBytesPerToken=890), samples_fixture())
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('KV_BUDGET_DISAGREEMENT', codes(f, 'PASS'))
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'PASS'))

    def test_a_ledger_that_drifts_from_the_python_derivation_is_caught(self):
        f = judge_kv_budget(dsv41_fixture(kvBytesPerToken=890), samples_fixture(main_per_token=640))
        self.assertIn('KV_BUDGET_DISAGREEMENT', codes(f, 'FAIL'))

    def test_a_reported_total_below_the_derived_main_kv_is_caught(self):
        """890 -> 600 would make the residual negative, i.e. main KV alone exceeds the total."""
        f = judge_kv_budget(dsv41_fixture(kvBytesPerToken=600), samples_fixture())
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'FAIL'))

    def test_a_residual_implying_an_impossible_precision_is_caught(self):
        """1200 B leaves 480 B over 320 channels = 1.5 B/channel, wider than FP8."""
        f = judge_kv_budget(dsv41_fixture(kvBytesPerToken=1200), samples_fixture())
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'FAIL'))

    def test_bounded_state_that_moves_with_context_is_caught(self):
        s = samples_fixture()
        s['logical_state_far']['dsv41']['bounded'] *= 4
        f = judge_kv_budget(dsv41_fixture(kvBytesPerToken=890), s)
        self.assertIn('KV_BOUNDED_INVARIANT', codes(f, 'FAIL'))

class MhcTests(unittest.TestCase):
    def test_the_coefficient_count_follows_from_the_stream_count(self):
        f = judge_mhc(dsv41_fixture())
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('MHC_COEFF_COUNT', codes(f, 'PASS'))

    def test_a_stream_count_inconsistent_with_the_residual_mode_is_caught(self):
        f = judge_mhc(dsv41_fixture(streams=1))
        self.assertIn('MHC_COEFF_COUNT', codes(f, 'FAIL'))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: FAIL with `ImportError: cannot import name 'judge_kv_budget'`.

- [ ] **Step 3: Add the second ledger sample**

In `tools/export-verification.mjs`, beside the existing `logical_state` sample, add one at a different context so the bounded invariant is checkable:

```js
 logical_state_far:Object.fromEntries(Object.keys(MODELS).map(id=>[id,cacheLedger(id,{...DEFAULT_STATE,context:131072})])),
```

- [ ] **Step 4: Implement the checks**

In `verification/ced.py`:

```python
FP4_BYTES_PER_CHANNEL=0.5+1/16   # E2M1 plus one E4M3 scale per 16 channels, §2.4
FP4_FLOOR,FP8_CEILING=0.5,1.0    # any real KV precision lies in this range

def derived_main_kv_bytes_per_token(m:dict)->float:
    """Independent re-derivation of the ledger's arithmetic. Only Full-mode layers store
    main KV, each at its own band's compression ratio."""
    total=0.0
    for l in m['kvSourceLayers']:
        ratio=m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder']
        total+=(m['headDim']/ratio)*FP4_BYTES_PER_CHANNEL
    return total

def indexer_channels_per_token(m:dict)->float:
    """Indexer K is stored by the same Full-mode layers, at index_head_dim."""
    return sum(m['indexer']/(m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder'])
               for l in m['kvSourceLayers'])
```

`judge_kv_budget(models, samples)` then, per `ced-csa2` model:

- `KV_BUDGET_DISAGREEMENT` — `samples['logical_state'][id]['mainKV'] / (B*S)` equals `derived_main_kv_bytes_per_token(m)`. Compare with a tolerance of `1e-9`, not `==`, because both sides are floats.
- `KV_RESIDUAL_IMPLAUSIBLE` — `residual = m['kvBytesPerToken'] - derived`; require `residual >= 0` and `FP4_FLOOR <= residual/indexer_channels_per_token(m) <= FP8_CEILING`. The message must state the derived figure, the reported figure and the implied bytes per channel, so a reader sees the arithmetic rather than a verdict.
- `KV_BOUNDED_INVARIANT` — `samples['logical_state'][id]['bounded'] == samples['logical_state_far'][id]['bounded']`. Run this for **every** model, not only `ced-csa2`, since Task 4 made `bounded` a shared field.

`judge_mhc(models)`: `MHC_COEFF_COUNT` — for models with `residual == 'mhc'`, require `streams > 1` and `(2+streams)*streams == 24`. The remedy names `Block.__init__`'s `mix_hc` expression as the authority.

- [ ] **Step 5: Run the tests**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: PASS.

- [ ] **Step 6: Wire in and run the gate**

Add to `verification/schematics.py:run`:

```python
    inv_models,inv_samples=inventory(root)
    findings.extend(judge_architecture(inv_models,inv_samples))
    findings.extend(judge_kv_budget(inv_models,inv_samples))
    findings.extend(judge_mhc(inv_models))
```

`inventory(root)` already returns `(models, samples)` from Task 7, so no new reader is needed. Keep the exporter to **one** run per gate; shelling out twice is wasteful and can race the build.

Run: `npm run verify:schematics`
Expected: PASS, with `KV_BUDGET_DISAGREEMENT`, `KV_RESIDUAL_IMPLAUSIBLE`, `KV_BOUNDED_INVARIANT` and `MHC_COEFF_COUNT` all in the passed-codes table.

- [ ] **Step 7: Commit**

```bash
git add verification/ced.py verification/schematics.py verification/tests/test_ced.py tools/export-verification.mjs
git commit -m "Cross-check the KV budget in two implementations

The ledger computes main KV in JavaScript; the verifier re-derives it in Python
from the pinned config and compares. Neither is checked against itself.

KV_RESIDUAL_IMPLAUSIBLE holds the reported 890 B/token accountable without
inventing the indexer precision the report omits: the unexplained residual must
imply between 0.5 and 1.0 bytes per channel. Today it implies 0.53. The bound is
not narrowed to fit, because a tighter bound would be a guess wearing a check's
clothing.

KV_BOUNDED_INVARIANT runs for all nine models, which is what makes Task 4's
reclassification verifiable rather than cosmetic."
```

---

### Task 9: Verify the transcription against publisher bytes

**Files:**
- Modify: `verification/ced.py`, `verification/schematics.py`
- Test: `verification/tests/test_ced.py`

**Interfaces:**
- Consumes: the acquired `dsv41-config` cache entry from Task 1; `MODELS.dsv41` from Task 2.
- Produces: `judge_publisher_agreement(models, root, catalog) -> list[Finding]`.

**Context you need:** This is the strongest check in the plan. `MODELS.dsv41` is a hand transcription of `config.json`; transcription error is the dominant failure mode when adding an architecture and nothing in the repo currently catches it.

`.verification-cache/` is **git-ignored by design**, so CI holds no publisher bytes. Use the established skip pattern rather than letting the check pass vacuously — a prior session found `test_the_shipped_values_agree_with_upstream` reporting agreement having compared nothing, a false green inside a green suite. When the cache is absent, emit a **BLOCKED** finding naming the acquire command, never a PASS.

Do not weaken this into a comparison against committed values. That would be self-consistency, not verification.

- [ ] **Step 1: Write the failing test**

```python
from verification.ced import judge_publisher_agreement, FIELD_MAP

class PublisherAgreementTests(unittest.TestCase):
    def test_every_transcribed_field_is_mapped_to_a_config_pointer(self):
        """A field with no mapping is a field nobody checks. Adding one to the model must
        force a conscious decision here."""
        mapped = set(FIELD_MAP)
        self.assertIn('layers', mapped)
        self.assertIn('kvSourceLayers', mapped)
        self.assertIn('engramTables', mapped)
        self.assertIn('streams', mapped)

    def test_a_mistranscribed_field_is_caught(self):
        config = {'text_config': {'num_hidden_layers': 40, 'hidden_size': 5120}}
        f = judge_publisher_agreement(dsv41_fixture(layers=39), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'FAIL'))

    def test_a_faithful_transcription_passes(self):
        config = {'text_config': {'num_hidden_layers': 40, 'hidden_size': 5120}}
        f = judge_publisher_agreement(dsv41_fixture(), config=config)
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'PASS'))

    def test_an_absent_cache_blocks_rather_than_passing(self):
        f = judge_publisher_agreement(dsv41_fixture(), config=None)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertEqual(codes(f, 'PASS'), set(), 'never report agreement having compared nothing')
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: FAIL with `ImportError: cannot import name 'judge_publisher_agreement'`.

- [ ] **Step 3: Implement**

In `verification/ced.py`. `FIELD_MAP` maps each transcribed field to a JSON pointer into `config.json` plus an optional transform, so an unmapped field is visible:

```python
# Every transcribed dsv41 field and where it comes from in config.json. A field absent from
# this map is a field nothing verifies, so adding one to MODELS must be a conscious decision
# here too. Derived fields carry a callable instead of a pointer.
FIELD_MAP={
 'layers':'/text_config/num_hidden_layers',
 'd':'/text_config/hidden_size',
 'heads':'/text_config/num_attention_heads',
 'kvHeads':'/text_config/num_key_value_heads',
 'headDim':'/text_config/head_dim',
 'ropeDim':'/text_config/qk_rope_head_dim',
 'qRank':'/text_config/q_lora_rank',
 'oRank':'/text_config/o_lora_rank',
 'oGroups':'/text_config/o_groups',
 'experts':'/text_config/n_routed_experts',
 'shared':'/text_config/n_shared_experts',
 'topK':'/text_config/num_experts_per_tok',
 'expertFF':'/text_config/moe_intermediate_size',
 'vocab':'/text_config/vocab_size',
 'window':'/text_config/sliding_window',
 'kvSourceLayers':'/text_config/kv_source_layer_ids',
 'indexSourceLayers':'/text_config/index_source_layer_ids',
 'indexer':'/text_config/index_head_dim',
 'indexerHeads':'/text_config/index_n_heads',
 'indexerBudget':'/text_config/index_topk',
 'candidateBlocks':'/text_config/candidate_topk_blocks',
 'candidateBlockSize':'/text_config/candidate_block_size',
 'candidateSource':'/text_config/candidate_source_layer_id',
 'streams':'/text_config/hc_mult',
 'sinkhornIters':'/text_config/hc_sinkhorn_iters',
 'engramLayers':'/text_config/engram_layer_ids',
 'engramTables':'/text_config/engram_num_embeddings',
 'engramNgram':'/text_config/engram_max_ngram_size',
 'engramHeads':'/text_config/engram_n_heads',
 'engramHeadDim':'/text_config/engram_head_dim',
 'mtp':'/text_config/num_nextn_predict_layers',
 'dsparkBlock':'/text_config/dspark_block_size',
 'dsparkExperts':'/text_config/dspark_n_routed_experts',
 'dsparkTopK':'/text_config/dspark_num_experts_per_tok',
 'dsparkTargets':'/text_config/dspark_target_layer_ids',
 'maxContext':'/text_config/max_position_embeddings',
 'routeScale':'/text_config/routed_scaling_factor',
 'swigluLimit':'/text_config/swiglu_limit',
 'visionLayers':'/vision_config/num_hidden_layers',
 'visionD':'/vision_config/hidden_size',
 'visionHeads':'/vision_config/num_attention_heads',
 'visionFF':'/vision_config/intermediate_size',
 'visionPatch':'/vision_config/patch_size',
 # derived, not transcribed: checked against an expression rather than a pointer
 'nopeDim':lambda c:c['text_config']['head_dim']-c['text_config']['qk_rope_head_dim'],
 'encoderLayers':lambda c:c['text_config']['num_hidden_layers']//2,
 'swaLayers':lambda c:sum(1 for r in c['text_config']['compress_ratios'][:c['text_config']['num_hidden_layers']] if r==0),
 'compressEncoder':lambda c:c['text_config']['compress_ratios'][c['text_config']['num_hidden_layers']//2-1],
 'compressDecoder':lambda c:c['text_config']['compress_ratios'][c['text_config']['num_hidden_layers']//2],
 'ngram':lambda c:sum(c['text_config']['engram_num_embeddings'])*c['text_config']['engram_head_dim'],
}
```

`judge_publisher_agreement(models, *, config=None, root=None, catalog=None)`:

- When `config is None` and no cache is resolvable, return one BLOCKED finding per `ced-csa2` model whose message names the exact acquire command from Task 1 Step 7 and states that `.verification-cache/` is git-ignored by design.
- Otherwise, for each mapped field present in the config, compare the transcribed value against the resolved one; collect **all** mismatches into one FAIL finding listing field, transcribed value and publisher value, so one run shows every error rather than only the first.
- Emit a PASS finding naming how many fields were compared. A comparison count of zero must be BLOCKED, not PASS.
- When `root` and `catalog` are given, resolve `config` by reading the `cache` path of the `dsv41-config` record via `evidence.safe_path`, so path traversal is impossible.

- [ ] **Step 4: Run the tests**

Run: `uv run --frozen python -m unittest verification.tests.test_ced -v`
Expected: PASS, four tests.

- [ ] **Step 5: Wire in and run both ways**

Add to `verification/schematics.py:run`, passing `root` and the loaded `sources.json`.

Run with the cache present: `npm run verify:schematics`
Expected: PASS, `CED_PUBLISHER_AGREEMENT` in passed codes with a non-zero comparison count.

Now prove the skip path is honest:

```bash
mv .verification-cache .verification-cache.bak && npm run verify:schematics; mv .verification-cache.bak .verification-cache
```

Expected: exit 2, `CED_PUBLISHER_AGREEMENT` **BLOCKED**, never PASS. Restore the cache immediately.

- [ ] **Step 6: Commit**

```bash
git add verification/ced.py verification/schematics.py verification/tests/test_ced.py
git commit -m "Verify the dsv41 transcription against the publisher's own bytes

MODELS.dsv41 is a hand transcription of config.json and nothing checked it.
This maps every transcribed field to a config pointer -- or, for derived ones,
to an expression -- and compares. An unmapped field is visible in FIELD_MAP,
so adding a field forces a decision about how it gets verified.

.verification-cache/ is git-ignored by design, so CI has no publisher bytes.
The check reports BLOCKED there and names the acquire command. It never reports
agreement having compared nothing, which is the exact false green a previous
audit found in the upstream-comparison tests."
```

---

### Task 10: Draw the CSA2 block and the mHC residual

> **Controller ruling T2a — this task runs immediately after Task 2+3, before the ledger tasks.**
> `buildModel`'s dispatch at `src/diagram-model.mjs:217` is
> `if(kind==='mla')…else if(kind==='gqa')…else if(kind==='qsa')…else recurrent(g,m,s,kind)`, so
> all four new `dsv41` kinds fall through to `recurrent()` — the KDA/GatedDeltaNet renderer. A
> `csa2-full` layer currently draws `decay-gate`, `short-conv`, `state-prev` and `attention-gate`,
> identical to `kimi3`'s KDA layer. `dsv41` has none of those. That is a fabricated architecture,
> which `CLAUDE.md` forbids ("No branch invented to fit a template"), and it must not persist
> across six commits while the ledger tasks run.
>
> **This task therefore also owns** `verification/tests/test_schematics.py:101`'s hardcoded
> `assertEqual(len(layers),504)` → `544`; the failure in
> `test_core_operations_are_pixel_identical_across_every_preset` (caused by `recurrent()`'s
> different geometry — note this is *not* `test_row_alignment_across_presets`, which is green);
> and restoring `VERIFICATION.md`'s Engineering row from a fresh report. It closes all three
> failures Task 2+3 left open.
>
> **Controller ruling T2g — make the dispatch exhaustive.** A second reviewer on a different
> engine identified the fall-through as a **latent, model-independent defect that predates this
> model**: `git log -S` dates `else recurrent(g,m,s,kind)` to `b6cb9d7`, the 3.0.0 baseline. Every
> `layerKind` value except three has always silently rendered as recurrent attention — no error,
> no warning, a confident wrong diagram. `dsv41` did not create that hole, it fell into it.
>
> So replace the catch-all with an exhaustive dispatch: `kda` and `delta` explicitly to
> `recurrent()`, `swa` and the three `csa2-*` kinds to `csa2Attention()`, and **`throw new
> RangeError` for anything else**. Add a negative test proving an unknown kind cannot silently
> render as recurrence. The `RangeError` and `csa2Attention` must land in the same commit — a
> `RangeError` before the renderer exists would abort scene export and take `check_coverage` with
> it, which is why the guard could not have been added earlier.
>
> **Also delete-and-reauthor, not rebind**, the scene claim ids that durably name motifs `dsv41`
> lacks (`decay-gate`, `short-conv`, `state-prev`, `attention-gate` and the like). A rebind would
> keep an id asserting a component the model does not have.

**Files:**
- Modify: `src/diagram-model.mjs` (add `csa2Attention`, extend `residuals`, `feedForward`, `buildModel`)
- Modify: `src/diagram-meta.mjs:37` (`buildResidual`)
- Test: `tests/scene.test.mjs`

**Interfaces:**
- Consumes: `layerKind` values from Task 3.
- Produces: a `buildModel` scene for every `dsv41` layer; `buildResidual` handles `residual === 'mhc'`.

**Context you need:** `buildModel` composes `stackOverview`, `addGroup('attention-region',258,174,498,704)`, `addGroup('ffn-region',874,174,516,702)`, then dispatches on `layerKind`. Model `csa2Attention` on the existing `sparseAttention` at `src/diagram-model.mjs:138-154` — it is the closest analogue and its coordinates are already known to satisfy the geometry contract.

`structuralReport` requires every node inside `scene.width`/`scene.height`, strictly orthogonal edge segments, and resolvable endpoints. `npm run verify:schematics` and `npm run verify:aesthetics` are the arbiters of clearance. **When labels collide, expand lanes and boxes — never reduce font size**; the canvas type scale is a frozen inventory.

`residuals` at `src/diagram-model.mjs:30-58` already branches on `m.residual` for `'hyper'` and `'attnres'`. mHC needs its own branch: the distinguishing fact is that **each sublayer's mixing coefficients are consumed by the next sublayer**, so the coefficients entering attention were produced by the previous block's FFN.

**Semantic content required** (the gates check meaning, not decoration):

- Two KV paths converging into one attention call, because `Attention.forward` concatenates the sliding-window KV and the compressed KV into a single `sparse_attn`. Drawing them as two separate attention operations would misrepresent the mechanism.
- The window path present on **every** layer, at `m.window` positions, derived from this layer's own hidden state.
- The compressed path present only when the layer's mode is not `swa`, and labelled with whether this layer **owns** the cache (`csa2-full`), **re-scores** it (`csa2-reindex`), or **reads a prior selection** (`csa2-reuse`).
- One 512-wide latent KV serving all 64 query heads — `m.kvHeads` is 1, and the node subtitle must say so rather than implying per-head KV.
- The grouped output projection: `m.oGroups` groups at rank `m.oRank`, block-diagonal.
- A per-head attention sink.
- Clamped SwiGLU at `m.swigluLimit` in `feedForward`.
- `sqrtsoftplus` routing with a **selection-only** correction bias: the bias picks experts, the weights come from the unbiased scores.

- [ ] **Step 1: Write the failing test**

Append to `tests/scene.test.mjs`:

```js
test('every dsv41 layer draws a contract-clean block', () => {
  for (let layer = 1; layer <= MODELS.dsv41.layers; layer++) {
    const scene = buildScene({ model: 'dsv41', view: 'model', stage: 'block', layer });
    assert.deepEqual(structuralReport(scene), [], `layer ${layer} geometry`);
  }
});

test('dsv41 attention shows two KV paths merging into one call', () => {
  const full = buildScene({ model: 'dsv41', view: 'model', stage: 'block', layer: 3 });
  const ids = new Set(full.nodes.map(n => n.id));
  assert.ok(ids.has('window-kv'), 'the sliding window is on every layer');
  assert.ok(ids.has('compress-kv'), 'a Full-mode layer owns the compressed cache');
  assert.ok(ids.has('attention'), 'one attention call, not two');
  const intoAttention = full.edges.filter(e => e.target === 'attention');
  assert.ok(intoAttention.length >= 3, 'query, window KV and compressed KV all reach it');
});

test('an SWA-only dsv41 layer draws no compressed path', () => {
  const swa = buildScene({ model: 'dsv41', view: 'model', stage: 'block', layer: 1 });
  const ids = new Set(swa.nodes.map(n => n.id));
  assert.ok(ids.has('window-kv'));
  assert.ok(!ids.has('compress-kv'), 'layers 0-1 have compress_ratio 0');
});

test('dsv41 states its single latent KV rather than implying per-head KV', () => {
  const scene = buildScene({ model: 'dsv41', view: 'model', stage: 'block', layer: 3 });
  const kv = scene.nodes.find(n => n.id === 'window-kv');
  assert.match(kv.sub + kv.name, /1 KV|one latent|single latent/i);
});

test('the mHC residual is not drawn as an ordinary sum', () => {
  const scene = buildScene({ model: 'dsv41', view: 'model', stage: 'residual' });
  assert.deepEqual(structuralReport(scene), []);
  const text = JSON.stringify(scene);
  assert.match(text, /4 streams|four streams/i);
  assert.match(text, /Sinkhorn/i, 'the doubly stochastic mix is the distinguishing property');
});
```

- [ ] **Step 2: Run it and watch it fail**

Run: `node --test --test-name-pattern 'dsv41' tests/scene.test.mjs`
Expected: FAIL — `buildModel` has no branch for these layer kinds, so attention nodes are absent.

- [ ] **Step 3: Add `csa2Attention`**

Add a `csa2Attention(g,m,s)` beside `sparseAttention`, and dispatch to it from `buildModel` when `kind` starts with `'csa2'` or equals `'swa'`. Take `sparseAttention`'s coordinates as the starting layout: `qkv-proj` at `270,236,282,48`, indexer column at `x=588`, cache row near `y=474`, `attention` at `270,600,468,58`, then `finishAttention(g,m,s,658,m.heads*m.headDim)`.

Gate the compressed column on `kind!=='swa'` and vary only its labelling by mode. Add the grouped output projection and the attention sink as their own nodes.

Extend `buildModel`'s kind label at `src/diagram-model.mjs:213` so `csa2-full` renders as `CSA2 · FULL` rather than `CSA2-FULL`.

- [ ] **Step 4: Add the mHC branches**

In `residuals`, add `m.residual==='mhc'` to the existing ternaries, with stream text `'4 streams'` and a body stating that the mixing coefficients a sublayer computes are consumed by the next one and that the combination matrix is made doubly stochastic by Sinkhorn iteration. In `buildResidual` (`src/diagram-meta.mjs:37`), add the `mhc` case to `ar`'s ternaries.

- [ ] **Step 5: Add the mHC scene invariant**

A 4-stream residual drawn as an ordinary sum is exactly the misrepresentation this project
forbids, so it is checked as a graph property rather than trusted.

Add `judge_mhc_streams(scene, m)` to `verification/ced.py`, called per-scene from
`schematics.py`'s loop for models whose `residual` is `'mhc'`. `MHC_STREAM_BALANCE` asserts that
the scene carries a residual rail whose stream count is stated as `m['streams']` and that no
`kind:'sum'` node sits on the residual path — mHC collapses and re-expands through learned
mixing, and a bare summation node would assert arithmetic the model does not do.

Negative test in `verification/tests/test_ced.py`:

```python
    def test_a_bare_sum_on_an_mhc_residual_path_is_caught(self):
        scene = {'nodes':[{'id':'merge-attn','kind':'sum','name':'+'}],
                 'edges':[],'groups':[],'labels':[{'text':'residual stream · x'}]}
        f = judge_mhc_streams(scene, dsv41_fixture()['dsv41'])
        self.assertIn('MHC_STREAM_BALANCE', codes(f, 'FAIL'))
```

- [ ] **- [ ] **Step 6: Run tests and both visual gates: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.src/diagram-model.mjs`, `file.src/diagram-meta.mjs`, plus roughly 360 **new** per-scene
claims for the `dsv41` block scenes, each depending on `MODELS.dsv41.d` alone

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. Scene claims are `visualization` domain, so `required_roles` returns `()` and they need no roles. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 7: Run tests and both visual gates**

```bash
node --test tests/scene.test.mjs
npm run verify:schematics
npm run verify:aesthetics
```

Expected: tests PASS; schematics PASS. Aesthetics may report contrast or clearance defects in states never previously measured — **fix the tokens or the geometry, never the policy**.

- [ ] **Step 8: Commit**

```bash
git add src/diagram-model.mjs src/diagram-meta.mjs verification/ced.py verification/schematics.py verification/tests/test_ced.py tests/scene.test.mjs
git commit -m "Draw the CSA2 block and the mHC residual

The two KV paths merge into one attention call, because Attention.forward
concatenates the window and compressed caches into a single sparse_attn.
Drawing two attention operations would misrepresent the mechanism.

Mode changes only the labelling of the compressed column -- owns, re-scores, or
reads a prior selection -- because that is the only thing that differs.

The KV node states one latent serving all 64 query heads rather than implying
per-head KV, which kvHeads:1 would otherwise suggest."
```

---

### Task 11: The CED page

**Files:**
- Modify: `src/diagram-meta.mjs` (add `buildCED`), `src/diagrams.mjs:5`, `src/core.mjs:7`, `src/app.mjs:11,19`
- Modify: `verification/ced.py`, `verification/schematics.py`
- Test: `tests/scene.test.mjs`, `verification/tests/test_ced.py`

**Interfaces:**
- Consumes: `layerKind`, `layerPartition`, `servingLedger` phases.
- Produces: `buildCED(s)` scene; `stage:'ced'`; `judge_ced_topology(scene) -> list[Finding]`.

**Context you need:** `stage` is whitelisted in `sanitizeState`'s `choices` at `src/core.mjs:7`; a value absent from it is silently replaced by the default, so the page will never render until `'ced'` is added there. `src/app.mjs:19` builds the stage segments and must offer the page when `m.encoderLayers` is set. `src/app.mjs:11`'s model-switch guard must reset `stage` to `'block'` when moving to a model without `encoderLayers`, exactly as it already does for `ple`, `residual` and `inputs` — omitting it leaves a stale stage rendering the wrong model.

`buildScene` dispatch at `src/diagrams.mjs:5` gates focal stages on a model predicate; follow the `ple` pattern: `if(s.stage==='ced'&&m.encoderLayers)return buildCED(s);`

**The page must contain**, per Figure 3: text and vision embedding feeding a Single-Pass mHC expansion into 4 streams; a Causal Encoder group holding the SWA pair and three `[Full, Reuse×5]` groups; an encoder-output node; a single **CED** edge from it into the decoder's Full layer; a Decoder group holding `[Full, Reuse×3]` and four `[Reindex, Reuse×3]` groups; an Engram node fanning out to layers 1 and 14; a Candidate Pool sourced from the decoder Full layer and feeding the Reindex layers; and a DSpark node reading the attention inputs of layers 37-39.

Band tile counts, the CED edge and the fan-outs come from `MODELS.dsv41`, never from literals.

- [ ] **Step 1: Write the failing tests**

```js
test('the CED page is contract-clean and only offered where it applies', () => {
  const scene = buildScene({ model: 'dsv41', view: 'model', stage: 'ced' });
  assert.deepEqual(structuralReport(scene), []);
  // a model without an encoder/decoder split falls back rather than rendering a foreign page
  const fallback = buildScene({ model: 'ds3', view: 'model', stage: 'ced' });
  assert.notDeepEqual(fallback.nodes.map(n => n.id), scene.nodes.map(n => n.id));
  assert.deepEqual(structuralReport(fallback), []);
});

test('the CED page shows a one-way handoff, not a shared stack', () => {
  const scene = buildScene({ model: 'dsv41', view: 'model', stage: 'ced' });
  const ids = new Set(scene.nodes.map(n => n.id));
  for (const id of ['enc-out', 'dec-full', 'engram', 'candidate-pool', 'dspark']) {
    assert.ok(ids.has(id), `${id} is part of the architecture`);
  }
  const ced = scene.edges.filter(e => e.source === 'enc-out' && e.target === 'dec-full');
  assert.equal(ced.length, 1, 'exactly one CED handoff edge');
});
```

Python side, in `test_ced.py`:

```python
    def test_an_edge_from_the_decoder_back_into_the_encoder_is_caught(self):
        """CED is a one-way handoff. A back edge would make it a shared-KV Transformer."""
        scene = {'nodes':[{'id':'enc-l5'},{'id':'dec-l3'}],
                 'edges':[{'id':'w1','source':'dec-l3','target':'enc-l5','points':[[0,0],[1,0]]}],
                 'groups':[{'id':'encoder'},{'id':'decoder'}]}
        f = judge_ced_topology(scene, dsv41_fixture()['dsv41'])
        self.assertIn('CED_SCENE_TOPOLOGY', codes(f, 'FAIL'))
```

- [ ] **Step 2: Run and watch fail**

Run: `node --test --test-name-pattern 'CED page' tests/scene.test.mjs`
Expected: FAIL — `sanitizeState` clamps `stage:'ced'` back to `'block'`, so the two scenes are identical.

- [ ] **Step 3: Wire the state**

Add `'ced'` to `choices.stage` in `sanitizeState`. In `src/app.mjs:19`, append `if(m.encoderLayers)stages.push(['ced','Encoder / decoder']);`. In `src/app.mjs:11`'s guard, add `if(state.stage==='ced'&&!MODELS[state.model].encoderLayers)state.stage='block';`. In `src/diagrams.mjs:5`, add the dispatch before the `buildModel` fallback.

- [ ] **Step 4: Implement `buildCED`**

Add to `src/diagram-meta.mjs`, following `buildResidual`'s construction. Derive every count from the model: band tiles from `layerKind` over `1..m.layers`, Engram fan-out from `m.engramLayers`, Reindex targets from `m.indexSourceLayers` filtered to the decoder band, DSpark sources from `m.dsparkTargets`.

- [ ] **Step 5: Implement `judge_ced_topology`**

In `verification/ced.py`, called per-scene from `schematics.py`'s loop for entries whose key names the `ced` stage. Assert: encoder band tile count equals `encoderLayers`; decoder band equals `layers - encoderLayers`; exactly one `enc-out → dec-full` edge; Candidate Pool out-degree equals the decoder-band Reindex count; Engram out-degree equals `len(engramLayers)`; and **no edge whose source is a decoder node and whose target is an encoder node**.

- [ ] **- [ ] **Step 6: Run everything: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.src/diagrams.mjs`, `file.src/app.mjs`, `file.src/core.mjs`, `file.src/diagram-meta.mjs`,
plus roughly 45 new per-scene claims for the CED page

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 7: Run everything**

```bash
node --test tests/scene.test.mjs
uv run --frozen python -m unittest verification.tests.test_ced -v
npm run verify:schematics && npm run verify:aesthetics
```

Expected: PASS throughout, `CED_SCENE_TOPOLOGY` in the passed-codes table.

- [ ] **Step 8: Commit**

```bash
git add src/diagram-meta.mjs src/diagrams.mjs src/core.mjs src/app.mjs \
        verification/ced.py verification/schematics.py tests/scene.test.mjs verification/tests/test_ced.py
git commit -m "Add the CED page and check its topology

A whole-network page, because the block template draws one block and this
architecture's story is the handoff between two twenty-layer halves.

Every count is derived from the model: band tiles from layerKind, Engram
fan-out from engramLayers, Reindex targets from indexSourceLayers.

CED_SCENE_TOPOLOGY asserts the one-way handoff as a graph property. A back edge
from decoder to encoder would make this an ordinary shared-KV Transformer, and
that is checked rather than asserted in a caption."
```

---

### Task 12: The innovation coverage

**Files:**
- Create: `src/innovations.mjs`
- Modify: `build.mjs:3` (`files` array), `src/app.mjs`, `src/styles.css`
- Modify: `verification/ced.py`, `verification/schematics.py`
- Test: `tests/core.test.mjs`, `verification/tests/test_ced.py`

**Interfaces:**
- Consumes: `cacheLedger`, `parameterLedger`, `servingLedger`, `layerPartition`, `SOURCES`.
- Produces: `INNOVATIONS` (array of records) and `innovationRows(m)` from `src/innovations.mjs`; `innovationCoverage()` in `src/app.mjs`; `judge_coverage(models) -> list[Finding]`.

**Context you need:** `build.mjs` has a **hardcoded** module-order array at line 3: `['models','core','scene','diagram-model','diagram-systems','diagram-meta','diagrams','app']`. Insert `'innovations'` between `'diagrams'` and `'app'`, because it imports from `core.mjs` and `app.mjs` imports from it.

`build.mjs` supports **only** single-line `import {…} from './x.mjs';` and `export const|let|function|class`. A multi-line import list or a default export throws `Unsupported module syntax`.

`comparisonTable()` at `src/app.mjs:64` is the pattern to follow: a `.table-scroll` wrapper around a `.comparison-table` with real `<thead>`/`<th>`, every cell derived from a ledger. Reuse those classes so no new type sizes or colour tokens enter the frozen scales.

**The rule that makes this verifiable:** `effect` is a **function** of the model, never a string. A literal cannot drift from the data because a literal cannot be written. This is what stops the exposition becoming a second, unverified copy of the architecture.

This is reading-size chrome prose, held to **AAA (7:1)** by the aesthetics gate, and `presentations.py` checks heading order and accessible names.

- [ ] **Step 1: Write the failing tests**

```js
test('every innovation record derives its effect rather than stating it', () => {
  for (const rec of INNOVATIONS) {
    assert.equal(typeof rec.effect, 'function', `${rec.id} must derive its effect`);
    assert.ok(rec.sources.length > 0, `${rec.id} must cite a source`);
    for (const sid of rec.sources) assert.ok(SOURCES[sid], `${rec.id} cites unknown source ${sid}`);
    assert.ok(rec.title && rec.mechanism && rec.replaces, `${rec.id} needs all three prose fields`);
  }
});

test('the coverage names every innovation the architecture actually has', () => {
  const ids = new Set(INNOVATIONS.map(r => r.id));
  for (const id of ['ced', 'csa2', 'hier-indexer', 'mhc', 'engram', 'dspark', 'fp4-kv', 'swa-replay']) {
    assert.ok(ids.has(id), `${id} is a published innovation and must be covered`);
  }
});

test('rendered effects contain no hardcoded architectural number', () => {
  const m = MODELS.dsv41;
  const rows = innovationRows(m);
  assert.equal(rows.length, INNOVATIONS.length);
  // the CED row must move when the model does, proving it is derived
  const shallow = innovationRows({ ...m, encoderLayers: 10 });
  const cedBefore = rows.find(r => r.id === 'ced').effect;
  const cedAfter = shallow.find(r => r.id === 'ced').effect;
  assert.notEqual(cedBefore, cedAfter, 'the CED effect must follow encoderLayers');
});
```

- [ ] **Step 2: Run and watch fail**

Run: `node --test --test-name-pattern 'innovation' tests/core.test.mjs`
Expected: FAIL — `INNOVATIONS` is not defined.

- [ ] **Step 3: Create `src/innovations.mjs`**

```js
/** What is new in the Causal Encoder-Decoder family, and where each claim comes from.
 *  `effect` is a function so no architectural number can be retyped into prose. */
import {cacheLedger,servingLedger,layerPartition,parameterLedger} from './core.mjs';
export const INNOVATIONS=[
 {id:'ced',title:'Causal Encoder-Decoder',
  replaces:'Every layer deriving global KV from its own hidden state',
  mechanism:'The bottom half is a causal encoder. Decoder layers do not derive global KV from their own hidden states; it is projected from the final encoder hidden state with layer-dependent weights, so prefill computes only the encoder.',
  effect:m=>`prefill computes ${servingLedger(m,{phase:'prefill'}).phaseLayers} of ${m.layers} layers`,
  sources:['dsv41-code','dsv41-report']},
 {id:'csa2',title:'Compressed Sparse Attention 2',
  replaces:'Every attention layer keeping its own KV and its own selection',
  mechanism:'Each layer takes one of three static modes. Full computes main KV and indexer K; Reindex reuses both but re-scores them for a fresh selection; Reuse takes the previous selection unchanged.',
  effect:m=>{const p=layerPartition(m);return `${p['csa2-full']} Full · ${p['csa2-reindex']} Reindex · ${p['csa2-reuse']} Reuse`;},
  sources:['dsv41-config','dsv41-code','dsv41-report']},
 // …hier-indexer, mhc, engram, dspark, fp4-kv, swa-replay
];
export function innovationRows(m){return INNOVATIONS.map(r=>({id:r.id,title:r.title,replaces:r.replaces,mechanism:r.mechanism,effect:r.effect(m),sources:r.sources}));}
```

Complete the remaining six with the same shape. Required derivations, all from the model or a ledger:

| id | title | `effect` derives from |
| --- | --- | --- |
| `hier-indexer` | Hierarchical Sparse Indexer | `m.candidateBlocks * m.candidateBlockSize` candidate positions narrowing to `m.indexerBudget` |
| `mhc` | Single-Pass mHC | `m.streams` streams, `m.sinkhornIters` Sinkhorn iterations |
| `engram` | Engram conditional memory | `parameterLedger(m).ngram` and `m.engramLayers.length` injection points |
| `dspark` | DSpark speculative decoding | `m.dsparkBlock`-token blocks, `m.dsparkExperts`/`m.dsparkTopK` draft routing |
| `fp4-kv` | FP4 main KV cache | `cacheLedger(m,{context:1,batch:1}).mainKV` bytes per token against `m.kvBytesPerToken` reported |
| `swa-replay` | SWA Bounded Replay | `cacheLedger(m,{context:1048576}).bounded` staying equal to the short-context value |

For `fp4-kv`, the effect must present **both** the derived figure and the reported one, because they differ by the unexplained indexer residual. Do not print only the reported number.

- [ ] **Step 4: Register the module and render it**

Add `'innovations'` to `build.mjs`'s `files` array between `'diagrams'` and `'app'`.

In `src/app.mjs`, add `innovationCoverage()` mirroring `comparisonTable()`: an `<h3>` heading continuing the existing order, an intro paragraph, then a `.table-scroll` / `.comparison-table` with `<th>` cells `Innovation`, `Replaces`, `Mechanism`, `Measured effect`, `Evidence`, and a row per `innovationRows(getModel(state.model))` entry. Render source ids as the existing citation buttons so they stay clickable provenance. Show it only when `m.encoderLayers` is set and the stage is `'ced'`.

- [ ] **Step 5: Add the Python checks**

In `verification/ced.py`, `judge_coverage(models)` reads the rendered rows out of the inventory. Add the rows to `tools/export-verification.mjs` under `samples.innovations` (claim-free, as established in Task 8):

- `COVERAGE_UNCITED` — every record has at least one source id and every id resolves in `SOURCES`.
- `COVERAGE_DERIVED` — the `fp4-kv` row's derived bytes-per-token equals `derived_main_kv_bytes_per_token(m)` from Task 8, and the `ced` row's layer count equals `m['encoderLayers']`. Reuses the Task 8 arithmetic, so the exposition is checked against the same independent computation the ledger is.

Negative tests: a record with an empty `sources` list trips `COVERAGE_UNCITED`; a row whose `fp4-kv` figure disagrees with the Python derivation trips `COVERAGE_DERIVED`.

- [ ] **- [ ] **Step 6: Run everything: Rebind the claims your bytes invalidate**

Your edit moves these claimed consumers, and leaving any unrebound raises `CONSUMER_VALUE`
(**FAIL**), which flips the audit from BLOCKED to FAIL and breaks
`test_cold_cache_is_blocked_not_certified`:

`file.build.mjs`, `file.src/app.mjs`, `file.src/styles.css` (`presentation`, mechanical), and a
**new** `file.src/innovations.mjs` claim, which must also join the 24 all-models claims and gain
every `MODELS.<id>.d` dependency

Re-derive each digest from the working tree — `node tools/export-verification.mjs` prints the
current values — and update `expected` in `verification/manifests/observatory.json`. A new
fingerprint is not an approval: these claims stay `REVIEW_REQUIRED` BLOCKED, which is correct.
Confirm with `npm run verify:science 2>&1 | tail -3` that no `CONSUMER_VALUE` remains.

- [ ] **Step 7: Run everything**

```bash
node --test && npm run build
uv run --frozen python -m unittest verification.tests.test_ced -v
npm run verify:schematics && npm run verify:aesthetics && npm run verify:presentations
```

Expected: PASS. If `build.mjs` throws `Unsupported module syntax`, the new module used a form the bundler does not support — fix the module, not the bundler. If aesthetics reports a contrast failure on the new prose, remember chrome prose is held to AAA and fix the token.

- [ ] **Step 8: Commit**

```bash
git add src/innovations.mjs src/app.mjs src/styles.css build.mjs \
        verification/ced.py verification/schematics.py tools/export-verification.mjs \
        tests/core.test.mjs verification/tests/test_ced.py
git commit -m "Add the innovation coverage, with its numbers derived

Prose that asserts an architectural fact needs the same evidence chain as a
number in a diagram. Two rules make that enforceable: every effect is a
function of the model rather than a string, so a literal cannot be written in
the first place; and every record cites sources that must resolve.

COVERAGE_DERIVED reuses the same Python KV arithmetic the ledger is checked
against, so the exposition cannot become a second unverified copy of the
architecture. The FP4 row shows the derived figure beside the reported one
rather than only the headline number."
```

---

### Task 13: Author the claim manifest

**Files:**
- Modify: `verification/manifests/observatory.json`
- Modify: `docs/verifiers/ANCHOR_RESIDUE.md`
- Test: existing verifier suite

**Interfaces:**
- Consumes: everything above; the acquired source cache from Task 1.
- Produces: a claim per new inventory consumer, so `COVERAGE_MISSING` returns to zero.

**Context you need:** `check_coverage` in `verification/evidence.py:261-278` emits `COVERAGE_MISSING` **FAIL** for every inventory `fields`/`functions`/`scenes` entry with no claim. So the science gate goes from BLOCKED (exit 2, honest) to FAIL (exit 1, contradiction) until this task lands. That is why this task cannot be deferred to a later branch.

Expect roughly **110 field claims** (about 70 `dsv41` fields, plus the five new `common` fields across the eight existing models) and **360-400 scene claims**, including the CED page. `samples` and `models` are not claim-bearing.

Claim classification, and the rules that decide it:

- Fields whose last path segment is `id`, `name`, `family`, `sources` or `blurb` are `presentation` and need no roles.
- The five new `common` defaults on existing models assert the **absence** of a mechanism. Use the established statement verbatim: "Common defaults for absent mechanisms are not claims that the corresponding checkpoint has those operators." Per the earlier ruling on absence-asserting values, these must **not** receive a config pointer anchor — a pointer would imply the publisher documented the absence.
- Everything else in the `model` domain earns `CLAIM_ROLE` and `MODEL_DUAL_CODE` BLOCKED, because no framework implements this model. **This is the expected output.** Do not classify a shape claim as `assumed` to silence it; `required_roles` returns `()` for `assumed`, which is precisely the hole that would launder it.

- [ ] **Step 1: Measure the gap**

```bash
node tools/export-verification.mjs > /tmp/inv.json
npm run verify:science 2>&1 | tail -5
```

Record the `COVERAGE_MISSING` count. This is the work list and the number that must reach zero.

- [ ] **Step 2: Anchor the config-derived claims**

```bash
uv run --frozen python tools/anchor-configs.py
```

This resolves `json_pointer` anchors against the acquired `config.json` bytes. Review the diff: a pointer must select the value the claim asserts. **Value equality alone never justifies a pointer** — a prior audit found 91 false matches where Python's `0 == False` made `mtp: 0` match `attention_bias: false`.

- [ ] **Step 3: Hand-anchor the code-derived claims**

Claims about mechanism rather than shape anchor to `python_symbol` selectors in `inference/model.py`: `Attention`, `Compressor`, `Indexer`, `Block`, `Gate`, `Engram`, `DSparkAttention`, `Transformer`. Claims resting on the report's prose have no pinnable source, because the PDF fails the UTF-8 decode — leave those `CLAIM_ANCHOR` BLOCKED and record a decision for each in `docs/verifiers/ANCHOR_RESIDUE.md`, consulting the existing 128 class decisions for precedent first.

- [ ] **Step 4: Re-acquire so anchors resolve**

```bash
uv run --frozen python -m verification.sources acquire \
  --candidate .verification-cache/candidates/dsv41-anchors.json
```

`acquire` also resolves claim anchors and writes their `sha256` and `resolved_span` into the candidate. Promote those into `observatory.json`. Any anchor reported `BLOCKED: Anchor unresolved` did not select what it claimed — fix the selector, never the recorded hash.

- [ ] **Step 5: Drive `COVERAGE_MISSING` to zero**

```bash
npm run verify:science 2>&1 | tail -5
```

Expected: exit 2, `COVERAGE_MISSING` **0**, no `CONSUMER_MISSING`, no `CONSUMER_DUPLICATE`, no `CONSUMER_VALUE`, no `COVERAGE_EMPTY`. `CLAIM_ROLE` and `MODEL_DUAL_CODE` BLOCKED counts will have risen by roughly 100 — that is correct and expected.

**If any FAIL code appears, stop.** FAIL is a contradiction, not missing evidence, and no amount of BLOCKED is a reason to accept one.

- [ ] **Step 6: Commit**

```bash
git add verification/manifests/observatory.json docs/verifiers/ANCHOR_RESIDUE.md
git commit -m "Claim every input the ninth model introduces

COVERAGE_MISSING is a FAIL, so a partial add turns the science gate from an
honest BLOCKED into a contradiction. The claims land with the model.

Config-derived claims are anchored to json_pointers in the acquired bytes;
mechanism claims to python_symbols in the publisher implementation. Claims
resting only on the tech report have no pinnable source, because the PDF fails
the strict UTF-8 decode, and each is recorded in ANCHOR_RESIDUE.md rather than
given an invented quote.

The five new common defaults assert absence and deliberately carry no config
pointer: a pointer would imply the publisher documented the absence.

CLAIM_ROLE and MODEL_DUAL_CODE rise by about 100 BLOCKED because no framework
implements this model. Nothing is reclassified to hide that."
```

---

### Task 14: Baselines, previews, public record, and the closing gates

**Files:**
- Modify: `verification/baselines/aesthetics/composition.json`
- Modify: `previews/manifest.json` and `previews/` assets
- Modify: `VERIFICATION.md`, `AGENTS.md`, `docs/verifiers/README.md`
- Modify: `verification/reports/*.json`, `verification/reports/*.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a green four-gate state and an honest scientific record.

**Context you need:** the composition baseline is a normalized signature per sampled scene; new scenarios mean new signatures. `--record` writes a **candidate**; promoting it is a deliberate act. A prior session lost two rounds to promoting a baseline that dropped `$comment` keys `--record` does not preserve — check for those before overwriting.

`verification/tests/test_reported_numbers.py` fails when `VERIFICATION.md` drifts from the reports. Every figure must come from a report generated on the **current** bytes; never copy one from this plan or a commit message.

Any edit under `src/`, `tools/`, `verification/`, workflows or manifests invalidates the engineering receipt, so `verify:engineering` runs **last**.

- [ ] **Step 1: Record and promote the composition baseline**

```bash
npm run verify:aesthetics -- --record
```

Review the candidate diff. Confirm each new signature belongs to a genuinely new scenario (`dsv41` layers, the CED page) and that no **existing** model's signature moved. An existing signature moving means a shared change leaked — investigate rather than promote. Preserve any `$comment` keys, then promote.

- [ ] **Step 2: Recapture previews**

```bash
npm run build && npm run capture
```

Expected: `previews/manifest.json` gains the artifact hash and the new SVG hashes. Previews must come from the current portable HTML — never hand-edit them.

- [ ] **Step 3: Regenerate every report**

```bash
npm run verify:schematics
npm run verify:aesthetics
npm run verify:presentations
npm run verify:science || true    # exit 2 BLOCKED is expected
npm run verify:runtime || true    # exit 2 BLOCKED is expected
```

- [ ] **Step 4: Update the public record from those reports**

Update `VERIFICATION.md` with figures read out of `verification/reports/`: the new model count (9), decoder layers (544), claim total, the finding counts by code, and the four-gate status.

Add to the known-and-unclosed section, in these words or closer to the evidence:

- No framework implements `deepseek_v41`, so every `dsv41` model-domain claim is BLOCKED for want of independent corroboration. `deepseek_v4` exists in transformers and SGLang and is a different model.
- The 170 B residual between the derived 720 B/token main KV and the reported 890 B/token is indexer K at a precision the publisher does not state. It is bounded, not derived.
- No DSpark parameter count is published.
- The tech report PDF cannot be byte-pinned; `sources.py` decodes strictly as UTF-8.
- `CED_PUBLISHER_AGREEMENT` cannot run in CI, because `.verification-cache/` is git-ignored by design. It reports BLOCKED there rather than passing vacuously.

Correct `docs/verifiers/README.md` and `AGENTS.md` wherever they state a model count, a layer count, or the gate inventory.

- [ ] **Step 5: Run the engineering gate last**

```bash
npm run verify:engineering
```

Expected: PASS, a fresh receipt, 544 decoder layers, 0 geometry defects. Report the state count, the browser used, and the artifact identity.

- [ ] **Step 6: Confirm the whole fail-closed suite**

```bash
npm run verify; echo "exit=$?"
```

Expected: `exit=2`. Engineering, schematics, aesthetics and presentations PASS; publication BLOCKED. **An exit of 1 means a contradiction was introduced — do not accept it.** An exit of 0 would mean publication passed, which cannot honestly happen while no independent reviewer has signed.

- [ ] **Step 7: Verify the reported numbers agree with the reports**

```bash
uv run --frozen python -m unittest verification.tests.test_reported_numbers -v
```

Expected: PASS. A failure means `VERIFICATION.md` disagrees with a report — fix the prose, never the report.

- [ ] **Step 8: Commit**

```bash
git add verification/baselines/aesthetics/composition.json previews VERIFICATION.md \
        AGENTS.md docs/verifiers/README.md verification/reports
git commit -m "Record the ninth model in the baselines, previews and public record

Every figure in VERIFICATION.md comes from a report generated on these bytes.

Records what adding this model does not achieve: no framework corroborates it,
so its model claims are BLOCKED; the 170 B indexer residual in the 890 B/token
figure is bounded rather than derived; no DSpark parameter count is published;
the tech report cannot be byte-pinned; and the publisher-agreement check cannot
run in CI because the cache is git-ignored by design.

npm run verify exits 2. Publication is BLOCKED until an independent reviewer
signs, and a green publication job would be a lie."
```

## Verification

```sh
npm test                                   # Node ledgers, layer kinds, scene contracts
uv run --frozen python -m unittest discover -s verification/tests   # incl. test_ced
npm run verify:schematics                  # 544 layers + every §5 architecture check
npm run verify:aesthetics                  # measured legibility, promoted baseline
npm run verify:presentations               # CED page + coverage, file:// origin, AAA prose
npm run verify:engineering                 # last: fresh receipt over the current bytes
npm run verify:science                     # exit 2 BLOCKED, COVERAGE_MISSING 0, no FAIL
npm run verify:runtime                     # exit 2 BLOCKED, unchanged
```

**Success is:** nine models, 544 decoder layers, four software gates PASS, `COVERAGE_MISSING` at zero, and science at exit 2 BLOCKED **with no new FAIL codes**. `CLAIM_ROLE` and `MODEL_DUAL_CODE` rise by roughly 100 BLOCKED; that is the correct report, not a defect to clear.

**Failure that must not be accepted:** any exit code 1, any `SOURCE_AUTHORITY`, `CONSUMER_VALUE` left unresolved, `COVERAGE_MISSING` above zero, a green publication gate, or a `CED_PUBLISHER_AGREEMENT` PASS with a zero comparison count.
