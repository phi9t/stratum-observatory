# Evidence Anchors and Dependency Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every atomic evidence citation a selector that resolves against real cached bytes, and make derived claims carry provenance through verified dependencies instead of copied citations.

**Architecture:** Three source-anchoring tools write *candidate* bundles that a human promotes; none of them edits an accepted lock in place. Then `verification/runner.py` moves its role evaluation after the dependency fixpoint so a derived claim's required roles are satisfied by the union of roles across its **verified** dependencies, after which the copied unanchorable citations are removed from the 2,520 `scene.*` claims.

**Tech Stack:** Python 3.13.12 under uv (`uv run --frozen`), stdlib only — no new dependencies. Node 20+ for the application layer. `unittest` for verifier tests.

**Spec:** `docs/superpowers/specs/2026-09-09-evidence-anchors-and-review-ceremony-design.md`

## Global Constraints

- Python is uv-managed. Run everything as `uv run --frozen python ...`. **Never** `pip install` into an ambient interpreter.
- No new third-party dependency. `uv.lock` is inside `toolchain_digest`; changing it invalidates every receipt.
- **Never invent an anchor, quote, hash, reviewer key or signature.** A tool may only write an anchor it has proven resolves against the cached bytes.
- **Never overwrite accepted source bytes, reviewed subjects or trust roots to silence drift.** All source changes go through `verification/sources.py`'s candidate path, then a human promotion step.
- Source budget: 8 MiB per source (`MAX_BYTES` in `sources.py:12`, `SOURCE_SIZE` in `evidence.py`).
- Every phase edits `verification/**` and therefore invalidates `verifier_sha256`. Re-run `npm run verify:engineering` before quoting any science or runtime number.
- Exit codes: `0` PASS, `1` contradiction/malformed input, `2` BLOCKED or NOT_RUN. BLOCKED is not failing.
- Success means **no new `FAIL` codes** and `MODEL_DUAL_CODE` staying at 0. A reduced BLOCKED count that introduces a FAIL is a regression.
- Write the failing test first, every time.

## File Structure

**Created:**
- `tools/anchor-configs.py` — proposes `json_pointer` anchors for `MODELS.*` citations of `publisher_config` sources. Resolves every candidate before writing; never writes an unresolved one.
- `tools/anchor-documents.py` — proposes `text_quote` anchors for `paper` and `vendor_document` citations. Same resolve-before-write rule.
- `tools/anchor-residue.py` — reports citations whose source provably cannot support the claim. Read-only; writes a report, changes no manifest.
- `verification/tests/test_anchor_tools.py` — behavioural tests for all three tools.
- `verification/tests/test_runner_order.py` — characterization tests locking the runner's current evaluation order **before** it is restructured.
- `docs/verifiers/ANCHOR_RESIDUE.md` — the decision record for every citation resolved by re-pointing, dropping or reclassifying.

**Modified:**
- `verification/manifests/sources.json` — three paper pins move to full-text HTML; `version` strings corrected.
- `verification/manifests/observatory.json` — anchors filled in; copied citations removed from `scene.*` claims.
- `verification/runner.py:47-56` (`required_roles`) and `verification/runner.py:176-232` (the claim loop and dependency fixpoint) — role evaluation moves after the fixpoint.
- `verification/tests/test_runner.py` — new negative tests for dependency-carried roles.
- `VERIFICATION.md`, `AGENTS.md` — refreshed from the actual reports.

Each anchoring tool is its own file because each has a different selector kind, a different failure mode, and a different residue class. They share only `verification.evidence.select_anchor`, which already exists.

---

### Task 1: Re-pin the three arXiv papers to full text

`ds2-paper`, `ds3-paper` and `q3-paper` point at arXiv `/abs/` landing pages. Those pages carry the abstract only: layer counts and head dimensions are absent, and terms that *are* present (`671B`, `MLA`) occur 3-4 times, which fails `text_quote`'s uniqueness rule.

Verified available on 2026-09-09: `ds2` → `https://arxiv.org/html/2405.04434v5` (727 KB), `ds3` → `https://arxiv.org/html/2412.19437v2` (550 KB), `q3` → `https://arxiv.org/html/2505.09388v1` (986 KB; v2 and v3 both 404).

Confirmed safe: **zero** existing evidence entries anchor into these three sources, so no anchor sha256 is invalidated by the re-pin.

The URL edit and the byte acquisition must land in **one commit**. An intermediate state where `url` names the full text while `sha256`/`cache` still describe the abstract page is a source that passes `check_source` while lying about its own identity.

**Files:**
- Modify: `verification/manifests/sources.json` (the `ds2-paper`, `ds3-paper`, `q3-paper` entries)
- Modify: `verification/manifests/observatory.json` (rewritten by the promotion step)
- Create: `verification/candidates/sources-fulltext.json`
- Test: `verification/tests/test_observatory.py`

**Interfaces:**
- Consumes: `verification.sources.acquire(root, catalog, target, candidate, accepted, *, only=None)` — already exists at `verification/sources.py:41`.
- Produces: three source entries whose `url` path begins `/html/`, with matching `sha256`, `cache` and `version`.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_observatory.py`:

```python
    def test_paper_sources_are_pinned_to_full_text(self):
        """An /abs/ landing page carries the abstract only; architecture facts are not on it."""
        catalog = json.loads((ROOT/'verification/manifests/sources.json').read_text())
        papers = [s for s in catalog['sources'] if s['role'] == 'paper']
        self.assertTrue(papers, 'catalog has no paper sources')
        for s in papers:
            with self.subTest(source=s['id']):
                self.assertIn('/html/', s['url'], f"{s['id']} is pinned to a landing page, not the paper")

    def test_paper_source_versions_are_not_pending(self):
        """Bytes are locked, so a version reading 'pending' is a stale lie about the snapshot."""
        catalog = json.loads((ROOT/'verification/manifests/sources.json').read_text())
        for s in catalog['sources']:
            if s.get('kind') == 'document' and s.get('sha256'):
                with self.subTest(source=s['id']):
                    self.assertNotIn('pending', s['version'].lower())
```

If `ROOT` and `json` are not already imported in that module, add `import json` and
`ROOT = Path(__file__).resolve().parents[2]` next to the existing imports.

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_observatory -v
```

Expected: FAIL — three paper sources contain `/abs/`, and several document sources have `pending` versions.

- [ ] **Step 3: Point the three sources at the full text**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
p = pathlib.Path('verification/manifests/sources.json')
catalog = json.loads(p.read_text())
NEW = {
    'ds2-paper': ('https://arxiv.org/html/2405.04434v5', 'arXiv:2405.04434v5 full-text HTML rendering'),
    'ds3-paper': ('https://arxiv.org/html/2412.19437v2', 'arXiv:2412.19437v2 full-text HTML rendering'),
    'q3-paper':  ('https://arxiv.org/html/2505.09388v1', 'arXiv:2505.09388v1 full-text HTML rendering'),
}
for s in catalog['sources']:
    if s['id'] in NEW:
        s['url'], s['version'] = NEW[s['id']]
p.write_text(json.dumps(catalog, indent=2) + '\n')
print('repinned', sorted(NEW))
PY
```

- [ ] **Step 4: Acquire the bytes into a candidate**

```bash
uv run --frozen python -m verification.sources acquire \
  --candidate verification/candidates/sources-fulltext.json \
  --only ds2-paper --only ds3-paper --only q3-paper
```

Expected: exit 0, three `"status": "PASS"` results. Exit 2 with `Acquisition unavailable` means no network — stop and report; do not hand-write a hash.

- [ ] **Step 5: Review the candidate before promoting it**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
b = json.loads(pathlib.Path('verification/candidates/sources-fulltext.json').read_text())
for s in b['catalog']['sources']:
    if s['id'] in {'ds2-paper', 'ds3-paper', 'q3-paper'}:
        print(f"{s['id']:12s} {s['sha256'][:16]} {s['url']}")
print('blocked results:', [r for r in b['results'] if r['status'] != 'PASS'])
PY
```

Read the three cached files and confirm each is the paper body, not an error page or a
"no HTML available" stub. A 404 stub is ~7.7 KB; a real paper body is 500 KB or more.

- [ ] **Step 6: Promote the candidate**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
b = json.loads(pathlib.Path('verification/candidates/sources-fulltext.json').read_text())
assert all(r['status'] == 'PASS' for r in b['results']), 'refusing to promote a non-PASS candidate'
pathlib.Path('verification/manifests/sources.json').write_text(json.dumps(b['catalog'], indent=2) + '\n')
pathlib.Path('verification/manifests/observatory.json').write_text(json.dumps(b['target'], indent=2) + '\n')
print('promoted')
PY
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
uv run --frozen python -m unittest verification.tests.test_observatory -v
uv run --frozen python -m unittest discover -s verification/tests
```

Expected: PASS. If any *other* document source still reports `pending`, correct its `version`
string to name the snapshot it actually holds — do not weaken the test.

- [ ] **Step 8: Confirm the gate did not regress**

```bash
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

Expected: engineering exit 0; science exit 2 with **no** `SOURCE_HASH`, `SOURCE_URL` or any other
FAIL code. `CLAIM_ANCHOR` is unchanged at this point — this task only makes the papers anchorable.

- [ ] **Step 9: Commit**

```bash
git add verification/manifests/sources.json verification/manifests/observatory.json \
        verification/candidates/sources-fulltext.json verification/tests/test_observatory.py
git commit -m "Re-pin the DeepSeek and Qwen papers to full-text arXiv HTML

The /abs/ landing pages carry the abstract only, so the architecture facts
they were cited for are not in the bytes. Acquired through the candidate
path; no accepted lock was edited in place."
```

---

### Task 2: Anchor configuration citations as JSON pointers

605 unanchored entries hang off `MODELS.*` claims, 277 of them on `publisher_config` sources. Where a field has a counterpart in the publisher's own configuration, the anchor is a `json_pointer` — the same kind as the 86 pointer anchors that already exist.

The mapping is per model family, because the same fact has different key names upstream: DeepSeek-V2 calls the routed-expert count `n_routed_experts`, Qwen3-235B calls it `num_experts`.

**The tool must never write an anchor it has not resolved.** It also proposes upgrading `relation` from `documents` to `equal` when the resolved value canonically equals the claim's `expected` — `relation: equal` triggers the `UPSTREAM_VALUE` comparison in `runner.py:194`, so this strictly strengthens the claim. Where the value does not match, the entry keeps `documents` and goes into the residue report for a human decision; the tool never silences a mismatch.

**Files:**
- Create: `tools/anchor-configs.py`
- Create: `verification/tests/test_anchor_tools.py`
- Modify: `verification/manifests/observatory.json` (via candidate)

**Interfaces:**
- Consumes: `verification.evidence.select_anchor(raw: bytes, anchor: dict) -> tuple[str, list[int]]`, `verification.evidence.canonical(obj) -> bytes`, `verification.evidence.loads(text) -> object`, `verification.evidence.safe_path(root, relative) -> Path`.
- Produces: `tools.anchor_configs.propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]` returning `(new_target, residue)` where each residue row is `{'claim': str, 'source': str, 'reason': str, 'pointer': str | None, 'upstream': str | None, 'expected': object}`.

- [ ] **Step 1: Write the failing test**

Create `verification/tests/test_anchor_tools.py`:

```python
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    """tools/*.py are scripts, not a package; load them by path."""
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnchorConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tool = load_tool('anchor-configs')
        config = {'hidden_size': 5120, 'num_hidden_layers': 60}
        raw = json.dumps(config).encode()
        (self.root/'config.json').write_bytes(raw)
        self.sources = {'ds2-config': {'id': 'ds2-config', 'role': 'publisher_config',
                                       'repo': 'deepseek-ai/DeepSeek-V2', 'cache': 'config.json',
                                       'sha256': hashlib.sha256(raw).hexdigest()}}

    def target(self, claim_id, expected):
        return {'claims': [{'id': claim_id, 'domain': 'model', 'classification': 'documented',
                            'expected': expected, 'statement': 's', 'dependencies': [],
                            'evidence': [{'source': 'ds2-config', 'anchor': None, 'relation': 'documents'}]}]}

    def test_matching_value_becomes_an_equal_pointer(self):
        new, residue = self.tool.propose(self.target('MODELS.ds2.d', 5120), self.sources, self.root)
        entry = new['claims'][0]['evidence'][0]
        self.assertEqual(entry['anchor']['kind'], 'json_pointer')
        self.assertEqual(entry['anchor']['pointer'], '/hidden_size')
        self.assertEqual(entry['relation'], 'equal')
        self.assertEqual(residue, [])

    def test_mismatched_value_is_reported_and_not_upgraded(self):
        new, residue = self.tool.propose(self.target('MODELS.ds2.d', 4096), self.sources, self.root)
        self.assertEqual(new['claims'][0]['evidence'][0]['relation'], 'documents')
        self.assertEqual(len(residue), 1)
        self.assertEqual(residue[0]['upstream'], '5120')

    def test_unmapped_field_writes_no_anchor(self):
        new, residue = self.tool.propose(self.target('MODELS.ds2.summary', 'a summary'), self.sources, self.root)
        self.assertIsNone(new['claims'][0]['evidence'][0]['anchor'])
        self.assertEqual(residue[0]['reason'], 'no mapped pointer')

    def test_pointer_absent_from_upstream_writes_no_anchor(self):
        self.tool.POINTERS[('deepseek-ai/DeepSeek-V2', 'vocab')] = '/vocab_size'
        new, residue = self.tool.propose(self.target('MODELS.ds2.vocab', 102400), self.sources, self.root)
        self.assertIsNone(new['claims'][0]['evidence'][0]['anchor'])
        self.assertEqual(residue[0]['reason'], 'pointer does not resolve')

    def test_existing_anchor_is_never_overwritten(self):
        t = self.target('MODELS.ds2.d', 5120)
        t['claims'][0]['evidence'][0]['anchor'] = {'kind': 'json_pointer', 'pointer': '/num_hidden_layers',
                                                   'sha256': 'f'*64}
        new, _ = self.tool.propose(t, self.sources, self.root)
        self.assertEqual(new['claims'][0]['evidence'][0]['anchor']['pointer'], '/num_hidden_layers')
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
```

Expected: FAIL — `tools/anchor-configs.py` does not exist, so `load_tool` raises `FileNotFoundError`.

- [ ] **Step 3: Write the tool**

Create `tools/anchor-configs.py`:

```python
"""Propose json_pointer anchors for MODELS.* citations of publisher configuration.

Writes a candidate bundle only. Every anchor is resolved against the cached bytes before it
is written; an anchor that does not resolve is reported as residue and never emitted.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.evidence import canonical, loads, safe_path, select_anchor

# (repo, MODELS field) -> JSON pointer in that publisher's config.json.
# Keyed by repo because the same fact has different upstream names per family.
POINTERS = {}
for _repo in ('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V3', 'moonshotai/Kimi-K3'):
    POINTERS.update({
        (_repo, 'd'): '/hidden_size',
        (_repo, 'layers'): '/num_hidden_layers',
        (_repo, 'heads'): '/num_attention_heads',
        (_repo, 'vocab'): '/vocab_size',
        (_repo, 'maxContext'): '/max_position_embeddings',
        (_repo, 'kvRank'): '/kv_lora_rank',
        (_repo, 'qRank'): '/q_lora_rank',
        (_repo, 'nopeDim'): '/qk_nope_head_dim',
        (_repo, 'ropeDim'): '/qk_rope_head_dim',
        (_repo, 'denseLayers'): '/first_k_dense_replace',
        (_repo, 'shared'): '/n_shared_experts',
        (_repo, 'experts'): '/n_routed_experts',
        (_repo, 'topk'): '/num_experts_per_tok',
        (_repo, 'ffn'): '/intermediate_size',
        (_repo, 'moeFfn'): '/moe_intermediate_size',
    })
for _repo in ('Qwen/Qwen3-235B-A22B', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-8B',
              'Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-Flash-Next'):
    POINTERS.update({
        (_repo, 'd'): '/hidden_size',
        (_repo, 'layers'): '/num_hidden_layers',
        (_repo, 'heads'): '/num_attention_heads',
        (_repo, 'kvHeads'): '/num_key_value_heads',
        (_repo, 'headDim'): '/head_dim',
        (_repo, 'vocab'): '/vocab_size',
        (_repo, 'maxContext'): '/max_position_embeddings',
        (_repo, 'experts'): '/num_experts',
        (_repo, 'topk'): '/num_experts_per_tok',
        (_repo, 'ffn'): '/intermediate_size',
        (_repo, 'moeFfn'): '/moe_intermediate_size',
    })


def propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]:
    """Return (new_target, residue). Never writes an anchor that does not resolve."""
    out = copy.deepcopy(target)
    residue: list[dict] = []
    for claim in out.get('claims', []):
        cid = claim.get('id', '')
        if not cid.startswith('MODELS.'):
            continue
        field = cid.split('.')[-1]
        for entry in claim.get('evidence', []):
            if entry.get('anchor'):
                continue
            source = sources.get(entry.get('source'))
            if not source or source.get('role') != 'publisher_config':
                continue
            row = {'claim': cid, 'source': source['id'], 'expected': claim.get('expected'),
                   'pointer': None, 'upstream': None, 'reason': ''}
            pointer = POINTERS.get((source.get('repo'), field))
            if not pointer:
                row['reason'] = 'no mapped pointer'
                residue.append(row)
                continue
            row['pointer'] = pointer
            anchor = {'kind': 'json_pointer', 'pointer': pointer}
            try:
                raw = safe_path(root, source['cache']).read_bytes()
                selected, _ = select_anchor(raw, anchor)
            except (ValueError, KeyError, OSError):
                row['reason'] = 'pointer does not resolve'
                residue.append(row)
                continue
            row['upstream'] = selected
            anchor['sha256'] = hashlib.sha256(selected.encode()).hexdigest()
            entry['anchor'] = anchor
            if canonical(loads(selected)) == canonical(claim.get('expected')):
                entry['relation'] = 'equal'
            else:
                row['reason'] = 'upstream value differs from expected'
                residue.append(row)
    return out, residue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--residue', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    catalog = loads((root/'verification/manifests/sources.json').read_text())
    sources = {s['id']: s for s in catalog['sources']}
    new, residue = propose(target, sources, root)
    candidate = safe_path(root, a.candidate)
    if candidate.exists():
        raise SystemExit('Candidate already exists; use a fresh candidate path')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(
        {'schema': 'stratum-acquisition-candidate/v1', 'catalog': catalog, 'target': new,
         'results': [], 'review_policy': 'Candidate is NOT an accepted lock. Read every proposed '
                                         'pointer against the cached bytes before promoting.'},
        indent=2) + '\n')
    safe_path(root, a.residue).write_text(json.dumps(residue, indent=2) + '\n')
    print(f'{a.candidate}: proposed anchors; {len(residue)} residue rows -> {a.residue}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
```

Expected: PASS, five tests.

- [ ] **Step 5: Run the tool and read its residue before promoting anything**

```bash
uv run --frozen python tools/anchor-configs.py \
  --candidate verification/candidates/anchors-config.json \
  --residue verification/reports/anchor-residue-config.json
uv run --frozen python - <<'PY'
import collections, json, pathlib
rows = json.loads(pathlib.Path('verification/reports/anchor-residue-config.json').read_text())
print(collections.Counter(r['reason'] for r in rows).most_common())
for r in rows:
    if r['reason'] == 'upstream value differs from expected':
        print(f"MISMATCH {r['claim']:34s} local={r['expected']!r} upstream={r['upstream']}")
PY
```

**Stop and report if any `upstream value differs from expected` row appears.** That is a real
contradiction between the diagram and the publisher's own configuration, and it is a diagram
defect to fix separately — not something this task resolves by loosening a relation.

- [ ] **Step 6: Promote the candidate**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
b = json.loads(pathlib.Path('verification/candidates/anchors-config.json').read_text())
pathlib.Path('verification/manifests/observatory.json').write_text(json.dumps(b['target'], indent=2) + '\n')
print('promoted')
PY
uv run --frozen python -m unittest discover -s verification/tests
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

Expected: engineering exit 0; science exit 2 with `CLAIM_ANCHOR` reduced and **no** `UPSTREAM_VALUE`
FAIL. A `UPSTREAM_VALUE` FAIL means step 5 was skipped or overridden.

- [ ] **Step 7: Commit**

```bash
git add tools/anchor-configs.py verification/tests/test_anchor_tools.py \
        verification/manifests/observatory.json verification/candidates/anchors-config.json \
        verification/reports/anchor-residue-config.json
git commit -m "Anchor configuration citations as resolved JSON pointers

Every pointer is resolved against the cached bytes before it is written, and
a citation whose upstream value matches the displayed one is upgraded to
relation: equal so UPSTREAM_VALUE actually compares them."
```

---

### Task 3: Anchor paper and vendor-document citations as quotes

209 unanchored entries hang off `HARDWARE.*` claims across 11 fields each, plus the paper citations that Task 1 made anchorable. `text_quote` already exists at `evidence.py:194` and requires both a **unique** quote and explicit `contains` tokens.

Quotes are authored, not generated — a quote is a claim about what a document says, so the mapping table is written by hand and the tool's job is to prove each one resolves uniquely.

**Files:**
- Create: `tools/anchor-documents.py`
- Modify: `verification/tests/test_anchor_tools.py`
- Modify: `verification/manifests/observatory.json` (via candidate)

**Interfaces:**
- Consumes: same `verification.evidence` helpers as Task 2.
- Produces: `tools.anchor_documents.propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]`, same residue row shape as Task 2 with `'quote'` in place of `'pointer'`.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_anchor_tools.py`:

```python
class AnchorDocumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tool = load_tool('anchor-documents')
        raw = b'<p>The H100 SXM offers 80GB of HBM3 at 3.35TB/s.</p>\n<p>NVLink is fast. NVLink is fast.</p>\n'
        (self.root/'h100.html').write_bytes(raw)
        self.sources = {'h100': {'id': 'h100', 'role': 'vendor_document', 'kind': 'document',
                                 'cache': 'h100.html', 'sha256': hashlib.sha256(raw).hexdigest()}}

    def target(self, claim_id):
        return {'claims': [{'id': claim_id, 'domain': 'gpu', 'classification': 'documented',
                            'expected': 80, 'statement': 's', 'dependencies': [],
                            'evidence': [{'source': 'h100', 'anchor': None, 'relation': 'documents'}]}]}

    def test_unique_quote_becomes_a_text_quote_anchor(self):
        self.tool.QUOTES[('h100', 'hbmGB')] = ('80GB of HBM3', ['80GB'])
        new, residue = self.tool.propose(self.target('HARDWARE.h100.hbmGB'), self.sources, self.root)
        anchor = new['claims'][0]['evidence'][0]['anchor']
        self.assertEqual(anchor['kind'], 'text_quote')
        self.assertEqual(anchor['quote'], '80GB of HBM3')
        self.assertEqual(anchor['contains'], ['80GB'])
        self.assertEqual(residue, [])

    def test_ambiguous_quote_writes_no_anchor(self):
        self.tool.QUOTES[('h100', 'nvGen')] = ('NVLink is fast.', ['NVLink'])
        new, residue = self.tool.propose(self.target('HARDWARE.h100.nvGen'), self.sources, self.root)
        self.assertIsNone(new['claims'][0]['evidence'][0]['anchor'])
        self.assertEqual(residue[0]['reason'], 'quote does not resolve uniquely')

    def test_absent_quote_writes_no_anchor(self):
        self.tool.QUOTES[('h100', 'arch')] = ('compute capability 9.0', ['9.0'])
        new, residue = self.tool.propose(self.target('HARDWARE.h100.arch'), self.sources, self.root)
        self.assertIsNone(new['claims'][0]['evidence'][0]['anchor'])
        self.assertEqual(residue[0]['reason'], 'quote does not resolve uniquely')

    def test_unmapped_field_is_reported_as_unsupported(self):
        new, residue = self.tool.propose(self.target('HARDWARE.h100.powerW'), self.sources, self.root)
        self.assertIsNone(new['claims'][0]['evidence'][0]['anchor'])
        self.assertEqual(residue[0]['reason'], 'no authored quote')
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
```

Expected: FAIL — `tools/anchor-documents.py` does not exist.

- [ ] **Step 3: Write the tool**

Create `tools/anchor-documents.py`:

```python
"""Propose text_quote anchors for paper and vendor-document citations.

Quotes are authored by hand in QUOTES, because a quote is an assertion about what a document
says. This tool's job is to prove each authored quote resolves uniquely against the cached
bytes; one that does not is reported as residue and never written.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.evidence import loads, safe_path, select_anchor

# (source id, claim field) -> (exact quote, required semantic tokens)
# Start empty: every row is authored against bytes that were read first.
QUOTES: dict[tuple[str, str], tuple[str, list[str]]] = {}


def propose(target: dict, sources: dict, root: Path) -> tuple[dict, list[dict]]:
    out = copy.deepcopy(target)
    residue: list[dict] = []
    for claim in out.get('claims', []):
        cid = claim.get('id', '')
        field = cid.split('.')[-1]
        for entry in claim.get('evidence', []):
            if entry.get('anchor'):
                continue
            source = sources.get(entry.get('source'))
            if not source or source.get('kind') != 'document':
                continue
            row = {'claim': cid, 'source': source['id'], 'expected': claim.get('expected'),
                   'quote': None, 'upstream': None, 'reason': ''}
            authored = QUOTES.get((source['id'], field))
            if not authored:
                row['reason'] = 'no authored quote'
                residue.append(row)
                continue
            quote, contains = authored
            row['quote'] = quote
            anchor = {'kind': 'text_quote', 'quote': quote, 'contains': list(contains)}
            try:
                selected, span = select_anchor(safe_path(root, source['cache']).read_bytes(), anchor)
            except (ValueError, KeyError, OSError):
                row['reason'] = 'quote does not resolve uniquely'
                residue.append(row)
                continue
            row['upstream'] = selected
            anchor['sha256'] = hashlib.sha256(selected.encode()).hexdigest()
            anchor['resolved_span'] = span
            entry['anchor'] = anchor
    return out, residue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--residue', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    catalog = loads((root/'verification/manifests/sources.json').read_text())
    new, residue = propose(target, {s['id']: s for s in catalog['sources']}, root)
    candidate = safe_path(root, a.candidate)
    if candidate.exists():
        raise SystemExit('Candidate already exists; use a fresh candidate path')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(
        {'schema': 'stratum-acquisition-candidate/v1', 'catalog': catalog, 'target': new,
         'results': [], 'review_policy': 'Candidate is NOT an accepted lock. Read every quote in '
                                         'its surrounding context before promoting.'}, indent=2) + '\n')
    safe_path(root, a.residue).write_text(json.dumps(residue, indent=2) + '\n')
    print(f'{a.candidate}: proposed quotes; {len(residue)} residue rows -> {a.residue}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
```

Expected: PASS, nine tests total in the module.

- [ ] **Step 5: Author the quotes against bytes you have actually read**

`QUOTES` ships empty. Fill it by reading each cached document and copying an exact, unique
sentence. This helper prints candidate contexts; **it does not choose the quote**:

```bash
uv run --frozen python - <<'PY'
import json, pathlib, re, sys
catalog = json.loads(pathlib.Path('verification/manifests/sources.json').read_text())
sources = {s['id']: s for s in catalog['sources']}
sid, needle = sys.argv[1] if len(sys.argv) > 1 else 'h100', sys.argv[2] if len(sys.argv) > 2 else '80GB'
text = pathlib.Path(sources[sid]['cache']).read_text()
for m in re.finditer(re.escape(needle), text):
    print(f'--- occurrence at offset {m.start()} ---')
    print(text[max(0, m.start()-200):m.end()+200].replace('\n', ' '))
print(f'total occurrences: {text.count(needle)}')
PY
```

A quote is usable only when `total occurrences: 1`. Add each authored row to `QUOTES` in
`tools/anchor-documents.py` with the semantic tokens it must contain.

- [ ] **Step 6: Run the tool and read the residue**

```bash
uv run --frozen python tools/anchor-documents.py \
  --candidate verification/candidates/anchors-documents.json \
  --residue verification/reports/anchor-residue-documents.json
uv run --frozen python - <<'PY'
import collections, json, pathlib
rows = json.loads(pathlib.Path('verification/reports/anchor-residue-documents.json').read_text())
print(collections.Counter(r['reason'] for r in rows).most_common())
print(collections.Counter(r['source'] for r in rows).most_common(15))
PY
```

- [ ] **Step 7: Promote and verify**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
b = json.loads(pathlib.Path('verification/candidates/anchors-documents.json').read_text())
pathlib.Path('verification/manifests/observatory.json').write_text(json.dumps(b['target'], indent=2) + '\n')
print('promoted')
PY
uv run --frozen python -m unittest discover -s verification/tests
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

Expected: engineering exit 0; science exit 2, `CLAIM_ANCHOR` further reduced, no new FAIL code.

- [ ] **Step 8: Commit**

```bash
git add tools/anchor-documents.py verification/tests/test_anchor_tools.py \
        verification/manifests/observatory.json verification/candidates/anchors-documents.json \
        verification/reports/anchor-residue-documents.json
git commit -m "Anchor paper and vendor-document citations as unique quotes

Quotes are authored against bytes that were read first; the tool proves each
one resolves uniquely and refuses to write one that does not."
```

---

### Task 4: Record the residue decisions

Some citations point at a source that provably does not contain the fact. An NVIDIA product page carries `80GB` and `3.35TB/s` exactly once each, but says nothing about compute capability, PCI device IDs or MIG mode.

Exactly three honest resolutions, and no fourth:

1. **Re-point** to a source that does state it — the `docs.nvidia.com` tuning and system guides are HTML and already in the catalog.
2. **Drop the citation** and let `CLAIM_ROLE` fail loudly, which is a true statement about the evidence.
3. **Reclassify**, but only within the existing whitelist at `runner.py:17` — `PRESENTATION_SUFFIXES = {'id','name','family','sources','blurb'}`. `MODELS.*.summary` and `MODELS.*.repo` are our own labels but are **not** on that whitelist, and widening it is a policy-root change that is out of scope here.

**Files:**
- Create: `docs/verifiers/ANCHOR_RESIDUE.md`
- Create: `tools/anchor-residue.py`
- Modify: `verification/manifests/observatory.json` (only for rows resolved by re-pointing or dropping)

**Interfaces:**
- Consumes: the two residue JSON reports from Tasks 2 and 3.
- Produces: `docs/verifiers/ANCHOR_RESIDUE.md`, one row per unresolved citation with its decision.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_anchor_tools.py`:

```python
class ResidueRecordTests(unittest.TestCase):
    """Every citation left unanchored must carry a recorded decision, not silence."""

    def test_every_residue_row_has_a_decision(self):
        record = ROOT/'docs/verifiers/ANCHOR_RESIDUE.md'
        self.assertTrue(record.exists(), 'residue decisions are not recorded')
        text = record.read_text()
        for name in ('anchor-residue-config.json', 'anchor-residue-documents.json'):
            path = ROOT/'verification/reports'/name
            if not path.exists():
                continue
            for row in json.loads(path.read_text()):
                with self.subTest(claim=row['claim'], source=row['source']):
                    self.assertIn(row['claim'], text)

    def test_no_residue_row_was_resolved_by_inventing_a_quote(self):
        """A decision must name one of the three allowed resolutions."""
        record = ROOT/'docs/verifiers/ANCHOR_RESIDUE.md'
        if not record.exists():
            self.skipTest('record not yet written')
        allowed = ('re-pointed', 'dropped', 'reclassified', 'unresolved')
        for line in record.read_text().splitlines():
            if line.startswith('| `') and 'claim' not in line:
                with self.subTest(line=line):
                    self.assertTrue(any(word in line for word in allowed), line)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
```

Expected: FAIL — `docs/verifiers/ANCHOR_RESIDUE.md` does not exist.

- [ ] **Step 3: Generate the skeleton record**

Create `tools/anchor-residue.py`:

```python
"""Render the residue reports as a decision table. Read-only; changes no manifest."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = """# Anchor residue decisions

Citations that no tool could anchor, and what was decided about each. Three resolutions are
allowed: **re-pointed** to a source that states the fact, **dropped** so `CLAIM_ROLE` reports the
gap honestly, or **reclassified** within the existing `PRESENTATION_SUFFIXES` whitelist. A row may
also stand as **unresolved** with a stated reason. Inventing a quote is not a resolution.

| claim | source | reason | decision |
|---|---|---|---|
"""


def main():
    rows = []
    for name in ('anchor-residue-config.json', 'anchor-residue-documents.json'):
        path = ROOT/'verification/reports'/name
        if path.exists():
            rows.extend(json.loads(path.read_text()))
    out = [HEADER]
    for r in sorted(rows, key=lambda r: (r['source'], r['claim'])):
        out.append(f"| `{r['claim']}` | `{r['source']}` | {r['reason']} | unresolved — decide |\n")
    (ROOT/'docs/verifiers/ANCHOR_RESIDUE.md').write_text(''.join(out))
    print(f'{len(rows)} residue rows written')


if __name__ == '__main__':
    sys.exit(main())
```

```bash
uv run --frozen python tools/anchor-residue.py
```

- [ ] **Step 4: Decide each row and edit the record**

Replace every `unresolved — decide` with an actual decision naming one of the four allowed words
and the reason. Where the decision is **re-pointed** or **dropped**, edit
`verification/manifests/observatory.json` to match — a record that disagrees with the manifest is
worse than no record.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run --frozen python -m unittest verification.tests.test_anchor_tools -v
uv run --frozen python -m unittest discover -s verification/tests
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

Expected: PASS; science exit 2 with no new FAIL code.

- [ ] **Step 6: Commit**

```bash
git add tools/anchor-residue.py docs/verifiers/ANCHOR_RESIDUE.md \
        verification/tests/test_anchor_tools.py verification/manifests/observatory.json
git commit -m "Record a decision for every citation that could not be anchored

Three resolutions are allowed and a fourth is not: no quote was invented to
make a check pass."
```

---

### Task 5: Lock the runner's current evaluation order

`verification/runner.py`'s claim loop is the load-bearing evaluation order of the entire gate. Task 6 moves role evaluation after the dependency fixpoint. Characterization tests go in **first**, and stay afterwards.

**Files:**
- Create: `verification/tests/test_runner_order.py`

**Interfaces:**
- Consumes: `verification.runner.audit(root, target, inventory, catalog, *, profile='publication', reviews=None, receipt=None, runtime=None) -> dict` — the report dict with a `findings` list.
- Produces: `verification.tests.test_runner_order.OrderTests.minimal_target()` — a two-claim fixture reused by Task 6's negative tests.

- [ ] **Step 1: Write the characterization tests**

Create `verification/tests/test_runner_order.py`:

```python
"""Characterization of the runner's evaluation order.

These tests describe what the runner does today. Task 6 changes role resolution; these must
keep passing except where a test is deliberately and visibly updated with its reason.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verification import runner


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'verification/trust').mkdir(parents=True)
        (self.root/'verification/trust/reviewers.json').write_text('{"reviewers": {}}')

    def source(self, sid, role, repo, text):
        raw = text.encode()
        (self.root/f'{sid}.txt').write_bytes(raw)
        return {'id': sid, 'role': role, 'repo': repo, 'kind': 'git', 'provider': 'github',
                'path': f'{sid}.txt', 'revision': 'a'*40,
                'url': f'https://raw.githubusercontent.com/{repo}/{"a"*40}/{sid}.txt',
                'cache': f'{sid}.txt', 'sha256': hashlib.sha256(raw).hexdigest(),
                'license': 'test fixture', 'fixture': True}

    def minimal_target(self, *, dependent_evidence):
        """An input claim carrying both code roles, and a derived claim that depends on it."""
        return {
            'schema': 'stratum-target/v1', 'id': 'test', 'description': 'order fixture',
            'claims': [
                {'id': 'MODELS.x.d', 'domain': 'model', 'classification': 'documented',
                 'consumer': 'MODELS.x.d', 'expected': 5120, 'statement': 'input', 'dependencies': [],
                 'evidence': [{'source': 'pub', 'anchor': None, 'relation': 'implements'},
                              {'source': 'fw', 'anchor': None, 'relation': 'implements'}]},
                {'id': 'scene.x.a', 'domain': 'model', 'classification': 'derived',
                 'consumer': 'scene.x.a', 'expected': 'digest', 'statement': 'derived',
                 'dependencies': ['MODELS.x.d'], 'evidence': dependent_evidence},
            ],
            'contracts': [], 'runtime_obligations': [],
        }

    def audit(self, target):
        catalog = {'schema': 'stratum-sources/v1', 'sources': [
            self.source('pub', 'publisher_code', 'deepseek-ai/DeepSeek-V2', 'class A: pass'),
            self.source('fw', 'framework_code', 'sgl-project/sglang', 'class B: pass'),
        ]}
        inventory = {'fields': {'MODELS.x.d': 5120, 'scene.x.a': 'digest'}, 'functions': {}}
        return runner.audit(self.root, target, inventory, catalog, reviews={})

    def codes(self, report, claim):
        return {f['code'] for f in report['findings'] if f['subject'] == claim}

    def test_copied_citations_satisfy_the_role_check_today(self):
        """Today a copied citation satisfies the role check regardless of whether the input verifies."""
        copied = [{'source': 'pub', 'anchor': None, 'relation': 'implements'},
                  {'source': 'fw', 'anchor': None, 'relation': 'implements'}]
        report = self.audit(self.minimal_target(dependent_evidence=copied))
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))
        self.assertNotIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.a'))

    def test_a_derived_claim_without_evidence_fails_roles_today(self):
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))
        self.assertIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.a'))

    def test_dependency_blocked_only_fires_on_an_otherwise_passing_claim(self):
        """DEPENDENCY_BLOCKED is a demotion of PASS, not an independent check."""
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        blocked = [f for f in report['findings'] if f['code'] == 'DEPENDENCY_BLOCKED']
        self.assertEqual(blocked, [], 'a claim already blocked is not demoted again')

    def test_unresolvable_anchors_block_the_input_claim(self):
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        self.assertIn('CLAIM_ANCHOR', self.codes(report, 'MODELS.x.d'))
```

- [ ] **Step 2: Run them to verify they pass against today's code**

```bash
uv run --frozen python -m unittest verification.tests.test_runner_order -v
```

Expected: PASS, four tests. **A characterization test that fails on unmodified code is describing
the wrong thing** — fix the test, not the runner.

- [ ] **Step 3: Commit**

```bash
git add verification/tests/test_runner_order.py
git commit -m "Characterize the runner's evaluation order before restructuring it"
```

---

### Task 6: Carry roles through verified dependencies

The policy-root change. For a claim classified `derived`, `required_roles` and `MODEL_DUAL_CODE` are satisfied by the union of roles across its **verified** dependencies — verified meaning `claim_status == PASS`.

This is strictly harder to fool than today. Right now a copied citation satisfies the role check on a scene claim whether or not the input it was copied from ever verifies (Task 5 pinned that behaviour in `test_copied_citations_satisfy_the_role_check_today`).

Role evaluation currently sits in the first pass at `runner.py:206-213`, before the fixpoint at `runner.py:224-231`. It has to move after it.

**Files:**
- Modify: `verification/runner.py:176-232`
- Modify: `verification/tests/test_runner_order.py`
- Modify: `verification/tests/test_runner.py`

**Interfaces:**
- Consumes: `runner.required_roles(c, *, publisher_ships_code=True) -> tuple[str, ...]`, `evidence.dual_code_roles(sources, *, publisher_ships_code=True) -> bool`, `OrderTests.minimal_target(*, dependent_evidence)` from Task 5.
- Produces: `runner.roles_from_dependencies(claim, claim_status, claim_sources) -> list[dict]` — the source dicts contributed by dependencies whose status is `PASS`.

- [ ] **Step 1: Write the failing negative tests**

Append to `verification/tests/test_runner_order.py`:

```python
class DependencyRoleTests(OrderTests):
    """Roles may flow from a verified dependency, and from nothing else."""

    def test_unverified_dependency_cannot_lend_a_role(self):
        """MODELS.x.d is blocked on anchors, so it lends nothing to its dependent."""
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))

    def test_a_role_no_dependency_holds_never_appears(self):
        target = self.minimal_target(dependent_evidence=[])
        target['claims'][0]['evidence'] = [{'source': 'fw', 'anchor': None, 'relation': 'implements'}]
        report = self.audit(target)
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))

    def test_dependency_cycle_cannot_manufacture_verification(self):
        target = self.minimal_target(dependent_evidence=[])
        target['claims'][0]['dependencies'] = ['scene.x.a']
        report = self.audit(target)
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.a'))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'MODELS.x.d'))

    def test_derived_claim_still_needs_dependencies(self):
        target = self.minimal_target(dependent_evidence=[])
        target['claims'][1]['dependencies'] = []
        report = self.audit(target)
        self.assertIn('DERIVATION_INPUTS', self.codes(report, 'scene.x.a'))

    def test_two_repos_of_one_org_still_refused_through_dependencies(self):
        target = self.minimal_target(dependent_evidence=[])
        target['claims'][0]['evidence'] = [{'source': 'fw', 'anchor': None, 'relation': 'implements'},
                                           {'source': 'fw2', 'anchor': None, 'relation': 'implements'}]
        report = self.audit(target)
        self.assertIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.a'))
```

Extend `OrderTests.audit` so the catalog also carries a second framework source from the same
organization:

```python
            self.source('fw2', 'framework_code', 'sgl-project/sglang', 'class C: pass'),
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run --frozen python -m unittest verification.tests.test_runner_order -v
```

Expected: `test_unverified_dependency_cannot_lend_a_role` and
`test_a_role_no_dependency_holds_never_appears` FAIL — today a derived claim with no evidence at
all still gets `CLAIM_ROLE`, so those two pass for the wrong reason; confirm by checking that
`test_copied_citations_satisfy_the_role_check_today` still passes. The genuinely new failures
appear once step 3 lands and the copied-citation test is updated.

- [ ] **Step 3: Restructure the runner**

In `verification/runner.py`, add above `audit`:

```python
def roles_from_dependencies(claim, claim_status, claim_sources):
    """Source dicts contributed by dependencies that are themselves verified.

    A derivation is only as good as its inputs, so a dependency lends its evidence roles to its
    dependent exactly when that dependency has passed. An unverified input lends nothing, which
    is what makes this stricter than copying its citations into the dependent claim.
    """
    out = []
    for dep in claim.get('dependencies', []):
        if claim_status.get(dep) == 'PASS':
            out.extend(claim_sources.get(dep, []))
    return out
```

Inside `audit`, record each claim's referents as they are gathered — immediately after the
`for e in evid:` loop completes, add:

```python
        claim_sources[cid] = referents
```

and initialise `claim_sources={}` next to `claim_status={}` at `runner.py:167`.

Then **remove** the role and dual-code block from the first pass (`runner.py:206-213`, the
`for role in required_roles(...)` loop and the `if domain=='model' ... MODEL_DUAL_CODE` statement),
and re-run it after the dependency fixpoint. Immediately after the fixpoint `for _ in range(len(claims))`
loop, insert:

```python
    # Roles are resolved after the fixpoint so a derived claim can inherit them from inputs that
    # actually verified. Doing this in the first pass would let an unverified input lend a role.
    for c in claims:
        cid=c.get('id');domain=c.get('domain');start=len(findings)
        referents=list(claim_sources.get(cid,[]))
        if c.get('classification')=='derived':referents+=roles_from_dependencies(c,claim_status,claim_sources)
        roles={s['role'] for s in referents}
        publisher_repos={s.get('repo') for s in referents if s.get('role')=='publisher_config'}
        ships_code=not (publisher_repos and publisher_repos<=CONFIG_ONLY_PUBLISHERS)
        for role in required_roles(c,publisher_ships_code=ships_code):
            if role not in roles:add('CLAIM_ROLE',domain,'BLOCKED',cid,f'Required {role} anchor is missing.')
        if domain=='model' and c.get('classification') not in ('presentation','assumed') and not dual_code_roles(referents,publisher_ships_code=ships_code):
            add('MODEL_DUAL_CODE','model','BLOCKED',cid,
                'Requires publisher implementation AND independent framework implementation.' if ships_code else
                'Publisher ships configuration only, so this needs independent implementations from at least two organizations.')
        if findings[start:]:
            claim_status[cid]='BLOCKED'
            for verified in [f for f in findings if f.code=='CLAIM_VERIFIED' and f.subject==str(cid)]:
                findings.remove(verified)
```

A claim demoted here may be the input of a claim the earlier fixpoint already marked PASS, so the
fixpoint has to run **again** after the role pass. Move the existing
`for _ in range(len(claims))` propagation loop into a local function and call it both before and
after the role pass:

```python
    def propagate():
        for _ in range(len(claims)):
            changed=False
            for c in claims:
                cid=c.get('id')
                if claim_status.get(cid)=='PASS' and any(claim_status.get(d)!='PASS' for d in c.get('dependencies',[])):
                    claim_status[cid]='BLOCKED';add('DEPENDENCY_BLOCKED',c['domain'],'BLOCKED',cid,'A derivation input is unverified.');changed=True
            if not changed:break
    propagate()   # resolve inputs so roles can be inherited from claims that actually passed
    # ... the role pass above ...
    propagate()   # a claim demoted by the role pass must demote its dependents too
```

Without the second call, a derived claim can be reported PASS while an input the role pass just
blocked sits underneath it — exactly the laundering this task exists to prevent.

- [ ] **Step 4: Update the one characterization test this deliberately changes**

`test_copied_citations_satisfy_the_role_check_today` now describes the old behaviour. Rename and
invert it, keeping the reason visible:

```python
    def test_copied_citations_no_longer_satisfy_roles_when_the_input_is_unverified(self):
        """Changed deliberately in Task 6: a copied citation used to satisfy the role check even
        when the claim it was copied from never verified. Roles now come from verified inputs."""
        copied = [{'source': 'pub', 'anchor': None, 'relation': 'implements'},
                  {'source': 'fw', 'anchor': None, 'relation': 'implements'}]
        report = self.audit(self.minimal_target(dependent_evidence=copied))
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))
```

- [ ] **Step 5: Run the full verifier suite**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
```

Expected: PASS. Any other test that fails is describing behaviour this change alters — read it,
and either fix the runner or update the test with its reason stated in a docstring. Never delete
a test to make the suite green.

- [ ] **Step 6: Verify the gate**

```bash
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
uv run --frozen python - <<'PY'
import collections, json, pathlib
d = json.loads(pathlib.Path('verification/reports/publication.json').read_text())
print(collections.Counter(f['code'] for f in d['findings']).most_common())
PY
```

Expected: engineering exit 0; science exit 2. `MODEL_DUAL_CODE` **must be 0**. `CLAIM_ROLE` will
rise for now, because scene claims currently inherit from unverified inputs — Task 7 removes the
copied citations and the count settles.

- [ ] **Step 7: Commit**

```bash
git add verification/runner.py verification/tests/test_runner_order.py verification/tests/test_runner.py
git commit -m "Resolve claim roles after the dependency fixpoint

A derived claim inherits evidence roles from dependencies that actually
verified. A copied citation previously satisfied the role check whether or
not the claim it was copied from ever passed."
```

---

### Task 7: Remove the copied citations from scene claims

With roles carried by verified dependencies, the 5,000 unanchorable copied entries on the 2,520 `scene.*` claims are removed. Anchored code entries stay — those are genuine.

**Files:**
- Modify: `verification/manifests/observatory.json`
- Modify: `verification/tests/test_observatory.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: a manifest where no `scene.*` claim carries an evidence entry with `anchor: null`.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_observatory.py`:

```python
    def test_scene_claims_carry_no_unanchored_citations(self):
        """A derived contract's provenance is its dependencies, not a copy of their citations."""
        target = json.loads((ROOT/'verification/manifests/observatory.json').read_text())
        offenders = [c['id'] for c in target['claims'] if c['id'].startswith('scene.')
                     and any(e['anchor'] is None for e in c['evidence'])]
        self.assertEqual(offenders[:5], [], f'{len(offenders)} scene claims carry copied citations')

    def test_every_scene_claim_still_declares_dependencies(self):
        target = json.loads((ROOT/'verification/manifests/observatory.json').read_text())
        for c in target['claims']:
            if c['id'].startswith('scene.'):
                with self.subTest(claim=c['id']):
                    self.assertTrue(c['dependencies'], 'a derived claim with no inputs proves nothing')
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_observatory -v
```

Expected: FAIL — roughly 2,520 scene claims still carry unanchored citations.

- [ ] **Step 3: Strip them**

```bash
uv run --frozen python - <<'PY'
import json, pathlib
p = pathlib.Path('verification/manifests/observatory.json')
target = json.loads(p.read_text())
removed = 0
for c in target['claims']:
    if not c['id'].startswith('scene.'):
        continue
    before = len(c['evidence'])
    c['evidence'] = [e for e in c['evidence'] if e['anchor'] is not None]
    removed += before - len(c['evidence'])
    assert c['dependencies'], f"{c['id']} would be left with no provenance at all"
p.write_text(json.dumps(target, indent=2) + '\n')
print(f'removed {removed} copied citations')
PY
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
```

- [ ] **Step 5: Verify the gate and record the real numbers**

```bash
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
uv run --frozen python - <<'PY'
import collections, json, pathlib
d = json.loads(pathlib.Path('verification/reports/publication.json').read_text())
c = collections.Counter(f['code'] for f in d['findings'])
print('total findings:', len(d['findings']))
print(c.most_common())
assert c['MODEL_DUAL_CODE'] == 0, 'dual-code corroboration regressed'
assert not [f for f in d['findings'] if f['status'] == 'FAIL'], 'a FAIL code appeared'
PY
```

Expected: engineering exit 0; science exit 2; `CLAIM_ANCHOR` on `scene.*` claims at 0; no FAIL
codes; `MODEL_DUAL_CODE` at 0.

- [ ] **Step 6: Commit**

```bash
git add verification/manifests/observatory.json verification/tests/test_observatory.py
git commit -m "Let derived scene claims carry provenance through their dependencies

Their citation lists were copies of their inputs' citations with the
unanchorable entries left null. The dependency edges were always the real
carrier, and the fixpoint enforces them."
```

---

### Task 8: Refresh the reported numbers

`VERIFICATION.md` has drifted from the reports it summarises: it says 15,737 BLOCKED findings (now 9,013 before this plan), 5,631 unanchored entries (now 5,899), `Accepted raw-byte snapshots | 0` (now 60 of 62 verified), and still states Qwen claims are "correctly BLOCKED on `MODEL_DUAL_CODE` and `CLAIM_ROLE`" — both are 0. It also disagrees with itself on aesthetics scenarios, 156 in the table against 112 in the prose.

**Files:**
- Modify: `VERIFICATION.md`
- Modify: `AGENTS.md` (the `CLAUDE.md` symlink target)
- Create: `verification/tests/test_reported_numbers.py`

**Interfaces:**
- Consumes: `verification/reports/publication.json`, `verification/reports/engineering.json`.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `verification/tests/test_reported_numbers.py`:

```python
"""VERIFICATION.md must agree with the reports it claims to summarise."""
import collections
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ReportedNumberTests(unittest.TestCase):
    def setUp(self):
        report = ROOT/'verification/reports/publication.json'
        if not report.exists():
            self.skipTest('no publication report; run npm run verify:science')
        self.report = json.loads(report.read_text())
        self.counts = collections.Counter(f['code'] for f in self.report['findings'])
        self.text = (ROOT/'VERIFICATION.md').read_text()

    def test_blocked_total_is_current(self):
        blocked = len([f for f in self.report['findings'] if f['status'] == 'BLOCKED'])
        self.assertIn(f'{blocked:,}', self.text, f'VERIFICATION.md does not state {blocked:,} BLOCKED findings')

    def test_claim_anchor_count_is_current(self):
        n = self.counts['CLAIM_ANCHOR']
        self.assertIn(f'{n:,}', self.text, f'VERIFICATION.md does not state {n:,} CLAIM_ANCHOR findings')

    def test_verified_source_count_is_current(self):
        n = self.report['coverage']['verified_sources']
        self.assertIn(str(n), self.text)

    def test_no_stale_dual_code_claim(self):
        if self.counts['MODEL_DUAL_CODE'] == 0:
            self.assertNotIn('correctly BLOCKED on `MODEL_DUAL_CODE`', self.text)

    def test_aesthetics_scenario_count_is_self_consistent(self):
        counts = set(re.findall(r'(\d+) scenarios', self.text))
        self.assertLessEqual(len(counts), 1, f'VERIFICATION.md states conflicting scenario counts: {counts}')
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_reported_numbers -v
```

Expected: FAIL on the stale totals and the conflicting scenario counts.

- [ ] **Step 3: Refresh `VERIFICATION.md` from the actual reports**

```bash
uv run --frozen python - <<'PY'
import collections, json, pathlib
d = json.loads(pathlib.Path('verification/reports/publication.json').read_text())
c = collections.Counter(f['code'] for f in d['findings'])
print('BLOCKED total:', len([f for f in d['findings'] if f['status'] == 'BLOCKED']))
print('coverage:', json.dumps(d['coverage'], indent=1))
print('codes:', c.most_common())
PY
```

Update every number in `VERIFICATION.md` from that output. Rewrite the Qwen paragraph to say the
dual-code requirement is now **met** by independent implementations from three organizations
rather than blocked. Reconcile the aesthetics scenario count against
`verification/reports/aesthetics.json`. Re-derive the three identity hashes in the
**Artifact identity** section from the fresh engineering receipt.

Add to **Known and unclosed**: PDF sources cannot currently be pinned, because `sources.py:30`
and `evidence.py:140` decode strictly as UTF-8; supporting them needs a pinned extractor inside
`toolchain_digest` and a `pdf_text_quote` anchor kind, and any future paper without an arXiv HTML
rendering will force that decision.

- [ ] **Step 4: Update `AGENTS.md`**

Correct any statement about gate status or `CLAIM_ANCHOR` that this plan changed. `CLAUDE.md` is a
symlink to `AGENTS.md`; edit `AGENTS.md` only.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

- [ ] **Step 6: Commit**

```bash
git add VERIFICATION.md AGENTS.md verification/tests/test_reported_numbers.py
git commit -m "Refresh the reported numbers and pin them to the reports

The document had drifted from the reports it summarises. A test now fails
when it drifts again."
```

---

## Final verification

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
npm test
npm run verify:engineering     # exit 0
npm run verify:schematics      # exit 0
npm run verify:aesthetics      # exit 0
npm run verify:presentations   # exit 0
npm run verify:science         # exit 2, BLOCKED
npm run verify:runtime         # exit 2, BLOCKED
```

Done means:

- No `FAIL` code anywhere in either report. `MODEL_DUAL_CODE` at 0.
- `CLAIM_ANCHOR` reduced to the residue recorded in `docs/verifiers/ANCHOR_RESIDUE.md`, and every
  remaining row has a stated decision.
- Engineering, schematics, aesthetics and presentations PASS at exit 0.
- Science and runtime BLOCKED at exit 2, dominated by `REVIEW_REQUIRED`.
- Every anchor in the manifest resolves against real cached bytes.
