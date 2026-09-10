"""Render the residue reports as a decision table, by class.

Residue totals 2,987 citations. Written one row per line that is a ~3,000-line document
nobody reads, which is a silent discard wearing a table's clothes. So the unit of decision
here is the CLASS -- one decision per (source, reason, field-class) group, each carrying its
own row count, and the counts reconcile to the residue total exactly.

Three resolutions are allowed and a fourth is not:

  re-pointed   the citation now resolves against bytes that state the fact
  dropped      the citation is removed, so CLAIM_ROLE/CLAIM_ANCHOR report the gap honestly
  reclassified within the existing PRESENTATION_SUFFIXES whitelist at verification/runner.py

A row may also stand as `unresolved` with a stated reason. Inventing a quote or a pointer is
not a resolution.

Default mode is read-only: it writes docs/verifiers/ANCHOR_RESIDUE.md and nothing else.
`--apply-drops` performs the one manifest edit this record implies -- deleting exactly the
evidence entries classified `dropped` -- so that the record and the manifest cannot drift
apart. It refuses to delete an entry that carries an anchor, and refuses any deletion that
would strip a claim of the last referent carrying a role its domain requires.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification.evidence import loads                                    # noqa: E402
from verification.evidence import CONFIG_ONLY_PUBLISHERS                    # noqa: E402
from verification.runner import PRESENTATION_SUFFIXES, required_roles      # noqa: E402

REPORTS = ('anchor-residue-config.json', 'anchor-residue-documents.json')

# --- Field classes -----------------------------------------------------------------------
# src/models.mjs:51 `common` splits into two halves. These fields default to a value that
# asserts an ABSENCE (no low-rank projection, no experts, no linear path, no vision). RULING
# (L/N): such a field is never anchored. A config key holding a falsy value does not document
# an absence, and a paper does not document an absence by failing to mention it.
ABSENCE_DEFAULTS = {'qRank', 'kvRank', 'ropeDim', 'nopeDim', 'kvHeads', 'experts', 'shared',
                    'topK', 'expertFF', 'denseLayers', 'latent', 'linearHeads',
                    'linearKeyHeads', 'gate', 'mtp', 'ngram', 'vision', 'indexer', 'pleLayers'}
# The other half are GENUINE VALUES that happen to be shared by most checkpoints. They may be
# anchored wherever an upstream key names the same quantity -- the test is semantic, never
# whether the value arrived via `common`.
GENUINE_DEFAULTS = {'headDim', 'vDim', 'linearHeadDim', 'convKernel', 'qkNorm', 'residual',
                    'maxContext'}
TUNING_GUIDES = {'hopper-tuning', 'blackwell-tuning'}
# The CUDA tuning guides do address these five, in prose, per architecture.
TUNING_GUIDE_SUBJECTS = {'arch', 'hbmGB', 'hbmTBps', 'nvGBps', 'nvGen'}
# Per-GPU figures the product/system pages state only as a rack or chassis aggregate.
AGGREGATE_ONLY = {'hbmGB', 'hbmTBps', 'nvGBps'}

# --- Resolutions -------------------------------------------------------------------------
# RULING (L/N) refinement: a genuine-value default IS anchorable where the upstream key names
# the same quantity. Each entry below was read against the cached bytes before being added.
RE_POINTED = {
    # Qwen3 GQA: transformers sizes q, k AND v projections from the single `head_dim` key, so
    # that key names the V head dimension as much as the Q head dimension.
    ('MODELS.q38.vDim', 'q38-config'),
    ('MODELS.q330.vDim', 'q330-config'),
    ('MODELS.q3235.vDim', 'q3235-config'),
    ('MODELS.q3827.vDim', 'q3827-config'),
    ('MODELS.q38flash.vDim', 'q38flash-config'),
    # Qwen3.8 names the linear-attention value head width and short-conv width outright.
    ('MODELS.q3827.linearHeadDim', 'q3827-config'),
    ('MODELS.q38flash.linearHeadDim', 'q38flash-config'),
    ('MODELS.q3827.convKernel', 'q3827-config'),
    ('MODELS.q38flash.convKernel', 'q38flash-config'),
    # Kimi K3 puts both inside the linear_attn_config block.
    ('MODELS.kimi3.linearHeadDim', 'hf-config'),
    ('MODELS.kimi3.convKernel', 'hf-config'),
    # The Qwen3 report states the native pre-training sequence length in one sentence that is
    # explicitly about "all models", so the shared (source, field) quote key is safe here.
    ('MODELS.q38.maxContext', 'q3-paper'),
    ('MODELS.q330.maxContext', 'q3-paper'),
    ('MODELS.q3235.maxContext', 'q3-paper'),
}
# The Qwen3 report's only vocabulary figure is 151,669 -- the tokenizer's BBPE vocabulary --
# against the displayed 151,936, which is the padded embedding-matrix size and is anchored to
# the publisher config. Two different quantities; leaving the citation would imply the paper
# corroborates a number it does not state.
QUANTITY_MISMATCH = {('MODELS.q38.vocab', 'q3-paper'),
                     ('MODELS.q330.vocab', 'q3-paper'),
                     ('MODELS.q3235.vocab', 'q3-paper')}
# Anchored in Task 2 against '/text_config/output_gate_type' = "swish" while the app displays
# 'SiLU'. SiLU and Swish-1 are the same function under two names, so the anchor stands; the
# relation is deliberately left `documents`, because `equal` would assert that the two strings
# are the same bytes, which they are not.
ACTIVATION_ALIAS = {('MODELS.q3827.linearGate', 'q3827-config')}
# A (source, field) quote key that several checkpoints share. Where the value diverges per
# model the tools could author no quote at all; this is a tool cap, not a gap in the source.
SHARED_QUOTE_KEY = {'q3-paper', 'hopper-tuning', 'blackwell-tuning'}

DECISIONS = {
    'scene-signature': ('unresolved', 'scene'),
    'code-digest': ('dropped', 'code-digest'),
    'contract-semantics': ('unresolved', 'contract'),
    'presentation-label': ('dropped', 'presentation'),
    'authored-prose': ('unresolved', 'not-whitelisted'),
    'repo-identifier': ('unresolved', 'not-whitelisted'),
    'unpublished-figure': ('unresolved', 'absence'),
    'absence-assertion': ('unresolved', 'absence'),
    'genuine-default': ('unresolved', 'unnamed-quantity'),
    'genuine-default-named': ('re-pointed', 'named-quantity'),
    'quantity-mismatch': ('dropped', 'quantity-mismatch'),
    'activation-alias': ('re-pointed', 'activation-alias'),
    'per-model-fact-shared-key': ('unresolved', 'shared-key'),
    'per-model-fact-unmapped': ('unresolved', 'unmapped'),
    'chassis-fact': ('dropped', 'wrong-document'),
    'architecture-fact': ('unresolved', 'shared-key'),
    'per-gpu-derived': ('unresolved', 'aggregate-only'),
    'sku-fact': ('unresolved', 'unauthored'),
}

RATIONALE = {
    'scene': (
        'A `scene.*` claim is `derived` and its `expected` is a sha256 of our own scene-contract '
        'signature. No span of an upstream document states that digest, and the upstream facts the '
        'diagram depicts are already claimed by the `dependencies` the claim declares '
        '(`MODELS.<model>.<field>`, `file.src/*.mjs`), each of which carries its own anchor or its own '
        'decision below. The citation is corroborating provenance for what is drawn, not a selector, '
        'and it is not false -- so it is kept, and it stays BLOCKED under `CLAIM_ANCHOR`, which is the '
        'correct fail-closed report. Anchoring it would mean hand-authoring one quote per node, 2,346 '
        'of them, adding no information the dependencies do not already carry.'),
    'code-digest': (
        'A sha256 of `src/core.mjs` or of `parameterLedger` is a fingerprint of code we wrote. The Kimi '
        'K3 paper does not document it, and these claims depend on all eight checkpoints, so citing one '
        "model's paper is arbitrary as well as false. Dropped; the publisher/framework code citations "
        'on the same claims are untouched.'),
    'contract': (
        "The claim's last segment is a SKU, fabric or library name (`H100-SXM`, `RoCEv2-lossless`), not "
        'a field name, and `expected` is null -- there is no value a quote could corroborate. The '
        'authoring tools key on `(source, field)`, so no key can address these. Left unresolved; '
        'contract records need per-contract quote authoring, which the current tool shape cannot express.'),
    'presentation': (
        '`id`, `name`, `family` and `sources` are our own labels and are already classified '
        '`presentation` (they sit on the `PRESENTATION_SUFFIXES` whitelist at `verification/runner.py:17`, '
        'so no reclassification is available or needed). No upstream document states our internal key '
        "`b200` or our source list. Dropped: the citation was blanket provenance, and a presentation "
        'claim requires no source role.'),
    'not-whitelisted': (
        '`MODELS.*.summary` is prose we wrote; `MODELS.*.repo` is the identifier of the source itself '
        'rather than a fact any source asserts. Reclassifying them would need `PRESENTATION_SUFFIXES` '
        'widened, and that is a policy-root change explicitly out of scope here. Left unresolved so the '
        'decision stays visible rather than being smuggled through the whitelist.'),
    'absence': (
        'The displayed value asserts an ABSENCE -- numeric 0, empty list, false, or a null where no '
        'figure is published. RULING (L/N) and the project\'s own non-negotiable ("Do not turn missing '
        'numeric evidence into zero") forbid anchoring these: a config key holding a falsy value does not '
        'document an absence, and a paper does not document one by failing to mention it. Permanently '
        'unresolved by policy, not by omission.'),
    'unnamed-quantity': (
        'A genuine-value default (`headDim`, `vDim`, `linearHeadDim`, `convKernel`, `qkNorm`, `residual`, '
        '`maxContext`) for which this source carries no key or sentence naming the same quantity. MLA '
        'checkpoints publish `qk_nope_head_dim`/`v_head_dim` and no single `head_dim`; no configuration '
        'names a residual style or a QK-Norm switch; `max_position_embeddings` is the position-embedding '
        'budget (40,960 for Qwen3, 163,840 for DeepSeek), which is neither the native context the app '
        'displays nor the extended one. Unresolved.'),
    'named-quantity': (
        'RULING (L/N) refinement applied: the value is genuine and the upstream key names the same '
        'quantity, so it is anchorable regardless of having arrived via the `common` block. Re-pointed '
        'and now carrying a resolved anchor in the manifest.'),
    'quantity-mismatch': (
        'The source states a different quantity from the one displayed, so the citation would imply '
        'corroboration that does not exist. Dropped; the citation that does state the displayed value '
        'is unaffected.'),
    'activation-alias': (
        'The publisher configuration says `"swish"` where the app displays `SiLU`. These are the same '
        'activation function under two names (SiLU is Swish-1), so the pointer is a true anchor and it '
        'stands. The relation is deliberately left `documents` rather than upgraded to `equal`: `equal` '
        'runs a byte comparison of the upstream value against the displayed one, and asserting that '
        '`"swish"` and `"SiLU"` are the same bytes would be false. The naming difference is a real, '
        'recorded fact about the two vocabularies, not a defect to be smoothed over.'),
    'unmapped': (
        'A single-model fact that no tool mapped, for one of three reasons, none of which is the source '
        'refusing to say it. (a) *Not published anywhere upstream*: `total` and `active` are parameter '
        'counts the configuration does not contain and the report states only in round figures; '
        '`architecture` and `generation` are our own taxonomy labels. (b) *Nameable but unmapped*: '
        "Kimi K3's `linearHeads` and `linearKeyHeads` (96) are named outright by "
        '`/text_config/linear_attn_config/num_heads`, and `POINTERS` simply has no entry. That is an '
        'anchorable row left unanchored, recorded here rather than quietly fixed: Task 4 is authorised '
        'to revisit only the genuine-value defaults RULING (L/N) named, and re-auditing every `POINTERS` '
        'gap is a different, larger exercise. (c) *Quotable but unauthored*: the DeepSeek and Kimi papers '
        'state `ff`, `kvHeads`, `scoring`, `kvRank`, `qRank`, `nopeDim`, `ropeDim` and `attnResBlock` in '
        'prose, and Task 3 stopped before them. Authoring a quote is anchoring, not one of the three '
        'resolutions this task may apply.'),
    'shared-key': (
        'Blocked by a known cap in the authoring tools, not by the source. `QUOTES` in '
        '`tools/anchor-documents.py` keys on `(source, field)` and not on model, so a source cited by '
        'several checkpoints can only carry facts identical across all of them. `q3-paper` backs q38, '
        'q330 and q3235 at once; `hopper-tuning` backs h100 and h200; `blackwell-tuning` backs b200 and '
        'gb300. Where the value diverges per model -- layer counts, hidden sizes, expert counts, per-GPU '
        'HBM -- one quote would silently back another model\'s different number, so no quote was '
        'authored. Unresolved, and deliberately not fixed here: re-keying `QUOTES` on '
        '`(source, model, field)` is out of this task\'s scope. Estimated unlock if it were re-keyed: '
        'about **24 of these 52 rows**. Tables 1 and 2 of the Qwen3 report give layers, Q/KV head counts '
        'and expert counts per model (13 rows), and the tuning guides name their architecture, NVLink '
        'generation, per-GPU HBM capacity and NVLink bandwidth per architecture (11 rows). The other 28 '
        'would then read honestly as "this source does not state this figure" -- the Qwen3 report never '
        'gives hidden size, FFN width, per-model parameter counts or expert FFN width, and the tuning '
        'guides state a "up to 3 TB/s" architectural ceiling rather than the 3.35 and 4.8 TB/s the '
        'H100 and H200 datasheets carry.'),
    'wrong-document': (
        'The CUDA tuning guides are programming guides for an architecture. They contain a Streaming '
        'Multiprocessor section, a memory-system section and an NVLink section, and they state nothing '
        'about a DGX chassis: no CPU model, no NIC, no GPU count, no board power, no peak throughput, no '
        'system name. Citing them for those fields is false provenance, so the citation is dropped. Every '
        'affected claim still cites the product page and the system guide, both `vendor_document`, so no '
        'required role is lost.'),
    'aggregate-only': (
        'The product and system pages state a rack or chassis aggregate -- "1,440 GB total GPU memory" '
        'across eight B200, "640GB" across eight H100 -- and never the per-GPU figure the app displays. '
        'Dividing by the GPU count is a derivation, and this manifest does not model it as one: the claim '
        'is `documented`, with no `dependencies`. Unresolved. Modelling it properly means restating these '
        'as `derived` claims that depend on the aggregate and on `gpus`, which is a manifest-shape change '
        'beyond this task.'),
    'unauthored': (
        'The page does state the fact and a quote is authorable from the cached bytes; Task 3 authored 30 '
        'vendor-document quotes and stopped before these. Authoring a quote is anchoring, not one of the '
        'three resolutions this task may apply, so these stand unresolved. They are the highest-value '
        'remaining anchoring work in the residue.'),
}

FINDINGS = """
## Findings recorded individually

Six questions were put to this task by name. Each is answered here and each is carried by a class
in the table below, so the answer and the count cannot drift apart.

**1. `MODELS.q3827.linearGate`: the app shows `SiLU`, Qwen's configuration says `"swish"`.**
Same function, two names -- SiLU is Swish-1 (`x * sigmoid(x)`), and Qwen3.8 selects it through
`output_gate_type`. The anchor at `/text_config/output_gate_type` stands and is correct. Its
relation stays `documents` and was deliberately not upgraded to `equal`, because `equal` compares
the upstream bytes with the displayed value and `"swish"` is not `"SiLU"`. Upgrading it would make
the gate assert something false in order to look greener.

**2. The Qwen3 report states a 151,669-token vocabulary; the app shows 151,936.**
Both numbers are right and they measure different things. The report's sentence is about the
tokenizer -- "Qwen3 models utilize Qwen's tokenizer, which implements byte-level byte-pair encoding
(BBPE ...) with a vocabulary size of 151,669" -- while 151,936 is `vocab_size` in the shipped
configuration, the padded height of the embedding matrix (151,669 rounded up to a multiple of 128,
plus the reserved special tokens). The app displays the padded figure, which is the one that costs
parameters and HBM, and anchors it to the configuration with an `equal` pointer. The paper citation
on those three claims is **dropped**: leaving it would imply the report corroborates 151,936, and
it does not mention that number at all.

**3. Qwen3-8B: the report says 128K context, the app shows 32,768. Which is right?**
Native versus extended, and the app is right for the quantity it names. `app.mjs` calls this value
"the native context used by this preset", and the report states the native figure in its own
sentence: "All models are pre-trained on hundreds of billions of tokens with a sequence length of
32,768 tokens." The 128K in Tables 1 and 2 is the deployment length reached afterwards with YaRN
and Dual Chunk Attention, which the report introduces explicitly "to enable length extrapolation";
the shipped configuration carries a third number again, `max_position_embeddings` = 40,960, which
is the position-embedding budget and is neither of the other two. Because the report's sentence is
about "all models", one quote is safe across q38, q330 and q3235, all of which display 32,768, so
these three citations are **re-pointed** onto it. The configuration citations at the same field stay
unresolved: no key in `config.json` names the native context.

**4. `hopper-tuning` and `blackwell-tuning` were left out of quote authoring entirely.**
Two different problems were hiding under one exclusion, and they get different answers. The guides
are CUDA programming documents: a Streaming Multiprocessor section, a memory-system section, an
NVLink section, and nothing whatever about a DGX chassis. For CPU model, NIC, GPU count, board
power, peak throughput, switch count and system name they are simply the wrong document, so those
42 citations are **dropped**. For architecture, per-GPU HBM capacity, HBM bandwidth and NVLink
bandwidth and generation they *are* the right document -- the Hopper guide states "capacity up to
80 GB" and "900 GB/s bidirectional bandwidth", the Blackwell guide states "capacity up to 180 GB"
-- but each guide is cited by two SKUs with diverging values (80 GB against H200's 141 GB, 180 GB
against GB300's 288 GB), and one quote per `(source, field)` would silently back the wrong one.
Those 20 stay **unresolved** under the tool cap in finding 6.

**5. `gb300` and `b200` per-GPU HBM and NVLink GB/s: the vendor pages state only aggregates.**
Confirmed. The DGX B200 pages give "1,440 GB total GPU memory" across eight GPUs and the GB300
page gives rack totals; the per-GPU 180 and 288 the app displays appear on neither. Dividing the
aggregate by the GPU count is a derivation, and these claims are `documented` with no
`dependencies`, so the manifest does not model it as one. Left **unresolved**. The honest repair is
to restate them as `derived` claims depending on the aggregate and on `gpus`, which is a
manifest-shape change beyond this task. Note the cross-reference: the Blackwell tuning guide does
state B200's 180 GB per GPU directly, and it is finding 4 -- not the absence of evidence -- that
keeps that row unanchored.

**6. Tool cap: `QUOTES` keys on `(source, field)` and not on model.**
A source cited by several checkpoints can therefore only carry facts that are identical across all
of them. This is why per-model paper facts stayed unanchored: `q3-paper` backs q38, q330 and q3235
at once, `hopper-tuning` backs h100 and h200, `blackwell-tuning` backs b200 and gb300. 52 residue
rows sit on such a shared key. Re-keying on `(source, model, field)` would unlock roughly **24** of
them -- 13 from Tables 1 and 2 of the Qwen3 report (layers, Q/KV head counts, expert counts) and 11
from the tuning guides (architecture, NVLink generation, per-GPU HBM, NVLink bandwidth) -- and would
turn the other 28 into an honest "this source does not state this figure". It is recorded here and
deliberately **not fixed**: it is out of scope for this task.

## What was changed in the manifest

| edit | citations | effect |
|---|---:|---|
| Anchors added as JSON pointers, under the RULING (L/N) refinement | 11 | `vDim` for the five Qwen GQA/hybrid checkpoints; `linearHeadDim` and `convKernel` for q3827, q38flash and kimi3 |
| Anchor added as a text quote, on the Qwen3 report's native-context sentence | 3 | `maxContext` for q38, q330, q3235 |
| Citations removed | 148 | 88 presentation labels, 42 chassis facts cited to a CUDA tuning guide, 15 code digests cited to the Kimi K3 paper, 3 vocabulary citations that state a different quantity |

`tools/anchor-residue.py --apply-drops` performs the removals and refuses any that would strip a
claim of the last referent carrying a role its domain requires, so a drop always reports a real
gap and never manufactures one. `CLAIM_ANCHOR` moved from 5,769 to 5,607 BLOCKED; no `CLAIM_ROLE`,
`MODEL_DUAL_CODE` or `DOCUMENTED_NO_SOURCE` finding appeared, and the publication gate is still
BLOCKED with no FAIL.
"""


def field_class(row: dict) -> str:
    """Classify one residue row. Decision-homogeneous by construction."""
    cid = row['claim']
    namespace, suffix = cid.split('.')[0], cid.split('.')[-1]
    key = (cid, row['source'])
    if key in RE_POINTED:
        return 'genuine-default-named'
    if key in QUANTITY_MISMATCH:
        return 'quantity-mismatch'
    if key in ACTIVATION_ALIAS:
        return 'activation-alias'
    if namespace == 'scene':
        return 'scene-signature'
    if namespace in ('function', 'file'):
        return 'code-digest'
    if namespace == 'contract-claim':
        return 'contract-semantics'
    if suffix in PRESENTATION_SUFFIXES:
        return 'presentation-label'
    if suffix == 'summary':
        return 'authored-prose'
    if suffix == 'repo':
        return 'repo-identifier'
    if row['expected'] is None:
        return 'unpublished-figure'
    if namespace == 'HARDWARE':
        if row['source'] in TUNING_GUIDES:
            return 'architecture-fact' if suffix in TUNING_GUIDE_SUBJECTS else 'chassis-fact'
        return 'per-gpu-derived' if suffix in AGGREGATE_ONLY else 'sku-fact'
    if suffix in ABSENCE_DEFAULTS and row['expected'] in (0, [], False):
        return 'absence-assertion'
    if suffix in GENUINE_DEFAULTS:
        return 'genuine-default'
    return 'per-model-fact-shared-key' if row['source'] in SHARED_QUOTE_KEY else 'per-model-fact-unmapped'


def decide(row: dict) -> tuple[str, str]:
    """(decision, rationale key) for one residue row."""
    return DECISIONS[field_class(row)]


def residue_rows(root: Path) -> list[dict]:
    rows = []
    for name in REPORTS:
        path = root/'verification/reports'/name
        if path.exists():
            rows.extend(loads(path.read_text()))
    return rows


HEADER = """# Anchor residue decisions

Citations that no tool could anchor, and what was decided about each. Three resolutions are
allowed: **re-pointed** to a source that states the fact, **dropped** so `CLAIM_ROLE` reports the
gap honestly, or **reclassified** within the existing `PRESENTATION_SUFFIXES` whitelist. A row may
also stand as **unresolved** with a stated reason. Inventing a quote is not a resolution.

The unit of decision is the **class**, not the row. Writing 2,987 rows one per line produces a
document nobody reads, and an unread table discards the same evidence a missing table would --
only less visibly. Each row below is one `(source, reason, field-class)` group with its own count,
and the counts reconcile to the residue total exactly. `verification/tests/test_anchor_tools.py`
enforces both halves of that: every residue row falls inside a recorded class, and no class may
overstate or understate its own count.

This file is generated by `tools/anchor-residue.py` from
`verification/reports/anchor-residue-config.json` and `anchor-residue-documents.json`. Those two
reports are the **census** these decisions were made against, exactly as Tasks 2 and 3 produced
them; they are deliberately not regenerated, so that the counts below describe the population that
was actually judged.
"""


def render(rows: list[dict]) -> str:
    groups = Counter((r['source'], r['reason'], field_class(r)) for r in rows)
    by_decision = Counter()
    for r in rows:
        by_decision[decide(r)[0]] += 1
    out = [HEADER, '\n## Outcome\n', '\n| Resolution | Citations |', '|---|---:|']
    for name in ('re-pointed', 'dropped', 'reclassified', 'unresolved'):
        out.append(f'| {name} | {by_decision.get(name, 0)} |')
    out.append(f'| **total** | **{len(rows)}** |')

    out.append(FINDINGS)
    out.append('\n## Why each class was decided that way\n')
    seen = set()
    for cls, (decision, why) in sorted(DECISIONS.items()):
        count = sum(n for k, n in groups.items() if k[2] == cls)
        if why in seen:
            out.append(f'\n### `{cls}` — {decision} ({count})\n\nSame reasoning as above.')
            continue
        seen.add(why)
        out.append(f'\n### `{cls}` — {decision} ({count})\n\n{RATIONALE[why]}')

    out.append('\n## Decisions by class\n')
    out.append('\n| source | reason | field-class | rows | decision |')
    out.append('|---|---|---|---:|---|')
    for (source, reason, cls), count in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        out.append(f'| `{source}` | `{reason}` | `{cls}` | {count} | {DECISIONS[cls][0]} |')
    out.append(f'\n**{sum(groups.values())} citations in {len(groups)} classes.**\n')
    return '\n'.join(out)


def apply_drops(root: Path) -> tuple[int, list[str]]:
    """Delete exactly the evidence entries this record classifies `dropped`.

    Refuses to remove an entry that already carries an anchor, and refuses any removal that
    would leave a claim without a referent carrying a role its domain requires -- dropping a
    citation must report a real gap, never manufacture one by accident.
    """
    manifest = root/'verification/manifests/observatory.json'
    target = loads(manifest.read_text())
    catalog = loads((root/'verification/manifests/sources.json').read_text())
    sources = {s['id']: s for s in catalog['sources']}
    doomed = {(r['claim'], r['source']) for r in residue_rows(root) if decide(r)[0] == 'dropped'}
    removed, refusals = 0, []
    for claim in target['claims']:
        keep = []
        for entry in claim['evidence']:
            if (claim['id'], entry['source']) not in doomed:
                keep.append(entry); continue
            if entry.get('anchor'):
                refusals.append(f"{claim['id']}/{entry['source']}: carries an anchor"); keep.append(entry); continue
            removed += 1
        if len(keep) != len(claim['evidence']):
            referents = [sources[e['source']] for e in keep if e['source'] in sources]
            roles = {s['role'] for s in referents}
            # Mirror runner.audit: a publisher that ships only configuration is judged against
            # publisher_config, not publisher_code. Getting this wrong here would refuse an
            # honest drop, which is its own kind of dishonesty.
            repos = {s.get('repo') for s in referents if s.get('role') == 'publisher_config'}
            ships_code = not (repos and repos <= CONFIG_ONLY_PUBLISHERS)
            missing = [r for r in required_roles(claim, publisher_ships_code=ships_code) if r not in roles]
            if missing:
                refusals.append(f"{claim['id']}: dropping would strip required role(s) {missing}")
                removed -= len(claim['evidence']) - len(keep)
                continue
            claim['evidence'] = keep
    manifest.write_text(json.dumps(target, indent=2) + '\n')
    return removed, refusals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply-drops', action='store_true',
                   help='remove the evidence entries classified `dropped` from the target manifest')
    p.add_argument('--root', default=str(ROOT))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    rows = residue_rows(root)
    if a.apply_drops:
        removed, refusals = apply_drops(root)
        for line in refusals:
            print(f'REFUSED {line}', file=sys.stderr)
        print(f'{removed} dropped citations removed from observatory.json; {len(refusals)} refusals')
        return 1 if refusals else 0
    record = root/'docs/verifiers/ANCHOR_RESIDUE.md'
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(render(rows))
    print(f'{len(rows)} residue rows recorded in {record.relative_to(root)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
