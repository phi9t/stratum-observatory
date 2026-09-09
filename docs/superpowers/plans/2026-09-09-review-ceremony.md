# Review Ceremony Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 3,056 `REVIEW_REQUIRED` findings answerable by a real human — readable evidence packets, one signature able to cover a declared class of subjects, and a signing tool the reviewer runs themselves.

**Architecture:** `verification/evidence.py` gains a scope-review primitive: a signed body naming a selector, a subject count and a `scope_root`, where the verifier recomputes the root from **the live manifest** rather than trusting the body. `verification/review_packets.py` renders each subject with its resolved evidence text so it can be reviewed without the repository, and `verification/sign_review.py` builds and signs the canonical body from a key path the reviewer supplies.

**Tech Stack:** Python 3.13.12 under uv (`uv run --frozen`), stdlib only. Ed25519 via `openssl pkeyutl`, located by the existing `evidence.ed25519_openssl()` probe — macOS resolves `openssl` to LibreSSL, which has no Ed25519.

**Spec:** `docs/superpowers/specs/2026-09-09-evidence-anchors-and-review-ceremony-design.md` (Phase 3)

## Global Constraints

- Python is uv-managed. Run everything as `uv run --frozen python ...`. **Never** `pip install` into an ambient interpreter.
- No new third-party dependency. `uv.lock` is inside `toolchain_digest`.
- **Never invent a reviewer key, identity or signature, and never copy a signature across changed subjects.** The tooling generates no private key and signs nothing on its own initiative.
- **The reviewer runs the signing command.** A signature produced by the agent that authored the claims carries no information, whatever the `author` field says.
- The private key never enters the repository. Only a public key and its fingerprint are committed, under `verification/trust/`, which `.github/CODEOWNERS` already assigns to `@phissenschaft`.
- A missing Ed25519-capable OpenSSL must report `REVIEW_TOOLCHAIN` BLOCKED, never contradict an honest reviewer (`evidence.py:291-293`).
- Exit codes: `0` PASS, `1` contradiction/malformed input, `2` BLOCKED or NOT_RUN.
- Success means **no new `FAIL` codes**. Publication stays BLOCKED regardless of this work — the five stub contracts, two redirect-blocked sources and `RECEIPT_UNATTESTED` are untouched by it.
- Write the failing test first, every time.

## File Structure

**Created:**
- `verification/review_packets.py` — renders a reviewable packet per subject, including the resolved text each anchor selects. Read-only against the manifests.
- `verification/sign_review.py` — builds the canonical review body and invokes `openssl pkeyutl -sign` with an operator-supplied key. Writes a fragment for a human to merge; never edits `reviews.json`.
- `docs/verifiers/REVIEW_CEREMONY.md` — key generation, enrollment, signing, merging, revocation.
- `verification/tests/test_scope_reviews.py` — scope primitive and its negative cases.
- `verification/tests/test_review_packets.py` — packet contents.
- `verification/tests/test_sign_review.py` — end-to-end sign-then-verify with a test key.

**Modified:**
- `verification/evidence.py` — `verify_signed_body`, `normalize_reviews`, `scope_root`, `matching_subjects`, `check_scope_review`; `check_review` refactored onto the shared signature path.
- `verification/runner.py:147-320` — build the subject-digest map up front, verify scope reviews once, and let a valid scope satisfy the per-subject review check.
- `package.json` — `review:packets` and `review:sign` scripts.
- `VERIFICATION.md` — the honesty record.

Three separate files because they have three different trust postures: `evidence.py` verifies, `review_packets.py` only reads, and `sign_review.py` is the only thing that touches a private key.

---

### Task 1: Share one signature-verification path

`check_review` (`evidence.py:280-309`) does identity lookup, verdict, independence, expiry and signature verification inline. Scope reviews need all of it, and duplicating it would let the two paths drift — the exact way a weaker check gets introduced by accident.

**Files:**
- Modify: `verification/evidence.py:280-309`
- Modify: `verification/tests/test_reviews.py`

**Interfaces:**
- Produces: `evidence.verify_signed_body(body: dict, signature_b64: str, root: Path, trust: dict) -> None` — returns `None` on success, raises `ValueError` with a reviewer-facing reason otherwise, and raises `LookupError` when the reviewer is unknown or revoked (which callers turn into BLOCKED rather than FAIL).
- Produces: `evidence.normalize_reviews(reviews: dict) -> tuple[dict, list[dict]]` — `(by_subject, scopes)`, accepting both the legacy flat map and `{'schema': 'stratum-reviews/v1', 'claims': {...}, 'scopes': [...]}`.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_reviews.py`:

```python
class SharedSignaturePathTests(SignedReviewTests):
    def test_verify_signed_body_accepts_a_real_signature(self):
        from verification.evidence import verify_signed_body
        signed = self.sign(self.body)['c']
        self.assertIsNone(verify_signed_body(signed['body'], signed['signature'], self.root, self.trust))

    def test_verify_signed_body_rejects_a_tampered_body(self):
        from verification.evidence import verify_signed_body
        signed = self.sign(copy.deepcopy(self.body))['c']
        signed['body']['author'] = 'changed'
        with self.assertRaises(ValueError):
            verify_signed_body(signed['body'], signed['signature'], self.root, self.trust)

    def test_verify_signed_body_reports_unknown_reviewer_separately(self):
        from verification.evidence import verify_signed_body
        signed = self.sign(self.body)['c']
        with self.assertRaises(LookupError):
            verify_signed_body(signed['body'], signed['signature'], self.root, {'reviewers': {}})


class NormalizeReviewTests(unittest.TestCase):
    def test_legacy_flat_map_is_accepted(self):
        from verification.evidence import normalize_reviews
        by_subject, scopes = normalize_reviews({'claim.a': {'body': {}, 'signature': 'x'}})
        self.assertEqual(list(by_subject), ['claim.a'])
        self.assertEqual(scopes, [])

    def test_versioned_shape_splits_claims_and_scopes(self):
        from verification.evidence import normalize_reviews
        by_subject, scopes = normalize_reviews({
            'schema': 'stratum-reviews/v1',
            'claims': {'claim.a': {'body': {}, 'signature': 'x'}},
            'scopes': [{'body': {'scope': {'selector': 'scene.*'}}, 'signature': 'y'}]})
        self.assertEqual(list(by_subject), ['claim.a'])
        self.assertEqual(len(scopes), 1)

    def test_empty_reviews_are_not_an_error(self):
        from verification.evidence import normalize_reviews
        self.assertEqual(normalize_reviews({}), ({}, []))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_reviews -v
```

Expected: FAIL — `verify_signed_body` and `normalize_reviews` do not exist.

- [ ] **Step 3: Extract the shared path**

In `verification/evidence.py`, add:

```python
def normalize_reviews(reviews: dict) -> tuple[dict, list[dict]]:
    """Accept the legacy flat map and the versioned shape without preferring either."""
    if reviews.get('schema') == 'stratum-reviews/v1':
        return dict(reviews.get('claims', {})), list(reviews.get('scopes', []))
    return {k: v for k, v in reviews.items() if k != 'schema'}, []


def verify_signed_body(body: dict, signature_b64: str, root: Path, trust: dict) -> None:
    """Identity, verdict, independence, expiry and signature. Raises, or returns None.

    LookupError means the reviewer is not enrolled or is revoked -- a gap in the trust root,
    which callers report as BLOCKED. ValueError means the review itself is wrong, which is a
    contradiction and is reported as FAIL.
    """
    reviewer = body.get('reviewer')
    key = (trust or {}).get('reviewers', {}).get(reviewer)
    if not key:
        raise LookupError('Reviewer public key is not in the protected trust root.')
    if key.get('revoked'):
        raise LookupError('Reviewer key has been revoked in the protected trust root.')
    openssl = ed25519_openssl()
    if openssl is None:
        raise LookupError('No OpenSSL with Ed25519 support is available to verify review signatures.')
    if body.get('verdict') != 'accept':
        raise ValueError('Review verdict is not acceptance')
    if not body.get('author') or body.get('author') == reviewer:
        raise ValueError('Independent reviewer is required')
    exp = datetime.fromisoformat(str(body.get('expires_at', '')).replace('Z', '+00:00'))
    if exp.tzinfo is None or exp <= datetime.now(timezone.utc):
        raise ValueError('Review expired or lacks timezone')
    public = safe_path(root, key['public_key'])
    sig = base64.b64decode(signature_b64, validate=True)
    if len(sig) != 64:
        raise ValueError('Ed25519 signature must be 64 bytes')
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp)/'body.json'; payload.write_bytes(canonical(body))
        sf = Path(tmp)/'signature'; sf.write_bytes(sig)
        run = subprocess.run([openssl, 'pkeyutl', '-verify', '-pubin', '-inkey', str(public),
                              '-rawin', '-in', str(payload), '-sigfile', str(sf)],
                             capture_output=True, timeout=10)
        if run.returncode:
            raise ValueError('Review signature verification failed')
```

Rewrite `check_review`'s body to call it, preserving today's finding codes exactly:

```python
def check_review(claim: dict, subject_digest: str, reviews: dict, root: Path, trust: dict | None=None) -> Finding:
    cid=claim.get('id','<unknown>')
    by_subject,_=normalize_reviews(reviews)
    review=by_subject.get(cid)
    blocked=lambda msg: Finding('REVIEW_REQUIRED','evidence','BLOCKED',cid,msg,'Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.')
    if not review:return blocked('No signed semantic review. A reviewed boolean is not evidence.')
    if trust is None:
        try:trust=loads((root/'verification/trust/reviewers.json').read_text())
        except (OSError,ValueError):trust={}
    body=review.get('body',{})
    # Probe the toolchain before comparing digests, exactly as the original did: a host that
    # cannot verify Ed25519 must block, never contradict an honest reviewer whose subject also moved.
    if ed25519_openssl() is None:
        return Finding('REVIEW_TOOLCHAIN','evidence','BLOCKED',cid,'No OpenSSL with Ed25519 support is available to verify review signatures.','Install OpenSSL 3.x (LibreSSL cannot verify Ed25519) or point STRATUM_OPENSSL at a capable binary.')
    try:
        if body.get('subject_sha256')!=subject_digest:raise ValueError('Review does not bind the current evidence/consumer digest')
        verify_signed_body(body,review.get('signature',''),root,trust)
    except LookupError as exc:
        code='REVIEW_TOOLCHAIN' if 'OpenSSL' in str(exc) else 'REVIEW_REQUIRED'
        remedy=('Install OpenSSL 3.x (LibreSSL cannot verify Ed25519) or point STRATUM_OPENSSL at a capable binary.'
                if code=='REVIEW_TOOLCHAIN' else 'Enroll the reviewer in verification/trust/reviewers.json.')
        return Finding(code,'evidence','BLOCKED',cid,str(exc),remedy)
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
        return Finding('REVIEW_INVALID','evidence','FAIL',cid,str(exc),'Re-review; do not copy signatures across changed claims.')
    return Finding('REVIEW_VALID','evidence','PASS',cid,'Independent signature binds the current claim and evidence.')
```

- [ ] **Step 4: Run the whole review suite**

```bash
uv run --frozen python -m unittest verification.tests.test_reviews -v
uv run --frozen python -m unittest discover -s verification/tests
```

Expected: PASS, including every pre-existing test in `test_reviews.py`. Those tests are the
guarantee that this refactor changed no behaviour — if one of them fails, the extraction is wrong.

- [ ] **Step 5: Commit**

```bash
git add verification/evidence.py verification/tests/test_reviews.py
git commit -m "Share one signature-verification path between review kinds

Scope reviews need the same identity, independence, expiry and signature
checks. Two copies would drift, and the weaker copy would win."
```

---

### Task 2: The scope-review primitive

One signature covers a declared class of subjects. The body names a `selector`, a `count` and a `scope_root`; the verifier recomputes the matching set **from the live manifest** and rejects the review if either the count or the root disagrees. Adding a subject that matches the selector, or changing any member's evidence, changes the root.

`scope_root` is a flat SHA-256 over the canonical sorted `(subject_id, subject_sha256)` list, not a Merkle tree. Inclusion proofs answer "is this member in a set I cannot see", and the verifier can always see the whole set, so a tree would add machinery without adding a check.

`subject_sha256` on a scope review is required to equal the `scope_root`, so the signature binds the set through the same field a per-subject review binds its one subject.

**Files:**
- Modify: `verification/evidence.py`
- Create: `verification/tests/test_scope_reviews.py`

**Interfaces:**
- Consumes: `evidence.verify_signed_body`, `evidence.canonical`.
- Produces:
  - `evidence.scope_root(pairs: list[tuple[str, str]]) -> str`
  - `evidence.matching_subjects(selector: str, subject_ids) -> list[str]`
  - `evidence.check_scope_review(review: dict, subjects: dict[str, str], root: Path, trust: dict | None=None) -> tuple[Finding, set[str]]` — the finding plus the subject ids the review validly covers (empty when it does not).

- [ ] **Step 1: Write the failing tests**

Create `verification/tests/test_scope_reviews.py`:

```python
import base64
import copy
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from verification.evidence import (canonical, check_scope_review, ed25519_openssl,
                                   matching_subjects, scope_root)


class ScopeReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        openssl = ed25519_openssl()
        if not openssl:
            self.skipTest('No OpenSSL with Ed25519; install OpenSSL 3.x or set STRATUM_OPENSSL.')
        self.openssl = openssl
        subprocess.run([openssl, 'genpkey', '-algorithm', 'Ed25519', '-out', str(self.root/'private.pem')],
                       check=True, capture_output=True)
        subprocess.run([openssl, 'pkey', '-in', str(self.root/'private.pem'), '-pubout',
                        '-out', str(self.root/'reviewer.pem')], check=True, capture_output=True)
        self.trust = {'reviewers': {'independent': {'public_key': 'reviewer.pem'}}}
        self.subjects = {'scene.ds2.a': 'a'*64, 'scene.ds2.b': 'b'*64, 'scene.ds3.a': 'c'*64}

    def body(self, selector='scene.ds2.*', **over):
        covered = matching_subjects(selector, self.subjects)
        root = scope_root([(cid, self.subjects[cid]) for cid in covered])
        body = {'author': 'implementer', 'reviewer': 'independent', 'verdict': 'accept',
                'subject_sha256': root,
                'scope': {'selector': selector, 'count': len(covered), 'scope_root': root},
                'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()}
        body.update(over)
        return body

    def sign(self, body):
        (self.root/'body').write_bytes(canonical(body))
        subprocess.run([self.openssl, 'pkeyutl', '-sign', '-inkey', str(self.root/'private.pem'),
                        '-rawin', '-in', str(self.root/'body'), '-out', str(self.root/'signature')],
                       check=True, capture_output=True)
        return {'body': body, 'signature': base64.b64encode((self.root/'signature').read_bytes()).decode()}

    def check(self, review):
        return check_scope_review(review, self.subjects, self.root, self.trust)

    def test_valid_scope_covers_exactly_its_members(self):
        finding, covered = self.check(self.sign(self.body()))
        self.assertEqual(finding.status, 'PASS')
        self.assertEqual(covered, {'scene.ds2.a', 'scene.ds2.b'})

    def test_a_claim_outside_the_selector_is_not_covered(self):
        _, covered = self.check(self.sign(self.body()))
        self.assertNotIn('scene.ds3.a', covered)

    def test_adding_a_matching_subject_invalidates_the_review(self):
        review = self.sign(self.body())
        self.subjects['scene.ds2.c'] = 'd'*64
        finding, covered = self.check(review)
        self.assertEqual(finding.status, 'FAIL')
        self.assertEqual(covered, set())

    def test_changing_a_member_digest_invalidates_the_review(self):
        review = self.sign(self.body())
        self.subjects['scene.ds2.a'] = 'e'*64
        finding, covered = self.check(review)
        self.assertEqual(finding.status, 'FAIL')
        self.assertEqual(covered, set())

    def test_declared_count_must_match_the_live_set(self):
        body = self.body()
        body['scope']['count'] = 99
        finding, covered = self.check(self.sign(body))
        self.assertEqual(finding.status, 'FAIL')
        self.assertEqual(covered, set())

    def test_root_not_bound_by_subject_sha256_is_refused(self):
        body = self.body()
        body['subject_sha256'] = 'f'*64
        finding, _ = self.check(self.sign(body))
        self.assertEqual(finding.status, 'FAIL')

    def test_self_review_refused_at_scope_too(self):
        finding, _ = self.check(self.sign(self.body(author='independent')))
        self.assertEqual(finding.status, 'FAIL')

    def test_tampered_scope_body_refused(self):
        review = self.sign(self.body())
        review['body'] = copy.deepcopy(review['body'])
        review['body']['scope']['selector'] = 'scene.*'
        finding, _ = self.check(review)
        self.assertEqual(finding.status, 'FAIL')

    def test_unknown_reviewer_blocks_rather_than_fails(self):
        finding, _ = check_scope_review(self.sign(self.body()), self.subjects, self.root, {'reviewers': {}})
        self.assertEqual(finding.status, 'BLOCKED')

    def test_selector_matches_whole_ids_only(self):
        self.assertEqual(matching_subjects('scene.ds2', self.subjects), [])
        self.assertEqual(matching_subjects('scene.ds2.*', self.subjects), ['scene.ds2.a', 'scene.ds2.b'])

    def test_scope_root_is_order_independent(self):
        a = scope_root([('x', 'a'*64), ('y', 'b'*64)])
        b = scope_root([('y', 'b'*64), ('x', 'a'*64)])
        self.assertEqual(a, b)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run --frozen python -m unittest verification.tests.test_scope_reviews -v
```

Expected: FAIL — `check_scope_review`, `matching_subjects` and `scope_root` do not exist.

- [ ] **Step 3: Implement the primitive**

Add `import fnmatch` to `verification/evidence.py`'s imports, then:

```python
def scope_root(pairs) -> str:
    """SHA-256 over the canonical sorted (subject_id, subject_sha256) list.

    Flat, not a Merkle tree. An inclusion proof answers 'is this member of a set I cannot see';
    the verifier recomputes the whole set from the live manifest, so there is no such question.
    """
    return hashlib.sha256(canonical(sorted([list(p) for p in pairs]))).hexdigest()


def matching_subjects(selector: str, subject_ids) -> list[str]:
    """Whole-id glob match. A selector is a declaration, so it never matches a prefix by accident."""
    return sorted(sid for sid in subject_ids if fnmatch.fnmatchcase(sid, selector))


def check_scope_review(review: dict, subjects: dict, root: Path, trust: dict | None=None) -> tuple[Finding, set[str]]:
    """Verify one scope review against the live subject set.

    Returns the finding and the subjects it validly covers. The declared selector, count and
    root are all recomputed here: a review is a statement about a set that existed when it was
    signed, and any change to that set retires it.
    """
    body = review.get('body', {}); scope = body.get('scope') or {}
    selector = scope.get('selector')
    subject = str(selector or '<no selector>')
    fail = lambda msg: (Finding('SCOPE_REVIEW_INVALID', 'evidence', 'FAIL', subject, msg,
                                'Re-review the current set; never widen a selector to absorb new subjects.'), set())
    if trust is None:
        try: trust = loads((root/'verification/trust/reviewers.json').read_text())
        except (OSError, ValueError): trust = {}
    if not isinstance(selector, str) or not selector:
        return fail('Scope review names no selector')
    covered = matching_subjects(selector, subjects)
    computed = scope_root([(sid, subjects[sid]) for sid in covered])
    try:
        if scope.get('count') != len(covered):
            raise ValueError(f'Scope covered {scope.get("count")} subjects when signed; it now matches {len(covered)}')
        if scope.get('scope_root') != computed:
            raise ValueError('A subject in this scope changed since the review was signed')
        if body.get('subject_sha256') != computed:
            raise ValueError('Scope review does not bind its own scope root')
        verify_signed_body(body, review.get('signature', ''), root, trust)
    except LookupError as exc:
        code = 'REVIEW_TOOLCHAIN' if 'OpenSSL' in str(exc) else 'SCOPE_REVIEW_REQUIRED'
        return Finding(code, 'evidence', 'BLOCKED', subject, str(exc),
                       'Enroll the reviewer in verification/trust/reviewers.json.'), set()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        return fail(str(exc))
    return (Finding('SCOPE_REVIEW_VALID', 'evidence', 'PASS', subject,
                    f'Independent signature covers {len(covered)} subjects matching {selector}.'), set(covered))
```

- [ ] **Step 4: Run them to verify they pass**

```bash
uv run --frozen python -m unittest verification.tests.test_scope_reviews -v
```

Expected: PASS, eleven tests.

- [ ] **Step 5: Commit**

```bash
git add verification/evidence.py verification/tests/test_scope_reviews.py
git commit -m "Add scope reviews: one signature over a recomputed subject set

The selector, count and root are recomputed from the live manifest, so adding
a matching subject or changing a member's evidence retires the review."
```

---

### Task 3: Let a valid scope satisfy the per-subject review check

`runner.audit` calls `check_review` in three places — claims (`runner.py:221`), contracts (`runner.py:251`) and runtime obligations (`runner.py:307`). Scope coverage has to be resolved before any of them, which means building the subject-digest map up front.

**Files:**
- Modify: `verification/runner.py:147-320`
- Modify: `verification/tests/test_runner.py`

**Interfaces:**
- Consumes: `evidence.normalize_reviews`, `evidence.check_scope_review`.
- Produces: `runner.subject_digests(claims, contracts, obligations, sources, runtime) -> dict[str, str]` — every reviewable subject id mapped to the digest a review must bind.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_runner.py`:

```python
class ScopeCoverageTests(unittest.TestCase):
    """A valid scope review answers REVIEW_REQUIRED for the subjects it covers, and no others."""

    def test_subject_digests_cover_claims_contracts_and_obligations(self):
        from verification.runner import subject_digests
        claims = [{'id': 'c.1', 'domain': 'model', 'classification': 'derived',
                   'statement': 's', 'evidence': [], 'dependencies': []}]
        contracts = [{'id': 'model.x', 'domain': 'model', 'record': {}, 'claims': []}]
        obligations = [{'id': 'runtime.x', 'domain': 'model', 'description': 'd'}]
        digests = subject_digests(claims, contracts, obligations, {}, {'runtime.x': {'exit_code': 0}})
        self.assertEqual(set(digests), {'c.1', 'model.x', 'runtime.x'})
        for value in digests.values():
            self.assertRegex(value, r'^[0-9a-f]{64}$')

    def test_obligation_without_a_receipt_is_not_a_reviewable_subject(self):
        from verification.runner import subject_digests
        obligations = [{'id': 'runtime.x', 'domain': 'model', 'description': 'd'}]
        self.assertNotIn('runtime.x', subject_digests([], [], obligations, {}, {}))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_runner -v
```

Expected: FAIL — `subject_digests` does not exist.

- [ ] **Step 3: Build the subject map and resolve scopes once**

In `verification/runner.py`, add above `audit`:

```python
def subject_digests(claims,contracts,obligations,sources,runtime):
    """Every reviewable subject mapped to the digest a review must bind.

    A runtime obligation is only reviewable once it has a receipt: without one there is no
    execution for a reviewer to have an opinion about.
    """
    out={}
    for c in claims:
        if c.get('id'):out[c['id']]=source_subject(c,sources)
    for rec in contracts:
        if rec.get('id'):out[rec['id']]=digest(rec)
    for o in obligations:
        r=runtime.get(o.get('id'))
        if o.get('id') and r:out[o['id']]=digest({'obligation':o,'receipt':r})
    return out
```

Inside `audit`, immediately after `sources` is built (`runner.py:166`), add:

```python
    by_subject,scope_reviews=normalize_reviews(reviews)
    subjects=subject_digests(claims,contracts,target.get('runtime_obligations',[]),sources,runtime)
    scope_covered=set()
    for scope_review in scope_reviews:
        finding,covered=check_scope_review(scope_review,subjects,root)
        findings.append(finding);scope_covered|=covered
```

Then change each of the three `check_review` call sites to skip a subject a valid scope already
covers. At `runner.py:221`:

```python
        if cid in scope_covered:
            findings.append(Finding('REVIEW_VALID','evidence','PASS',str(cid),'Covered by an independent scope review binding the current subject set.'))
        else:
            findings.append(check_review(c,source_subject(c,sources),reviews,root))
```

Apply the same two-branch shape at the contract site (subject `rec['id']`, digest `digest(rec)`)
and the runtime obligation site (subject `rid`, digest `digest({'obligation':o,'receipt':r})`).

Add `normalize_reviews,check_scope_review` to the `from .evidence import (...)` list at
`runner.py:8`, and `Finding` is already imported from `.model` at `runner.py:7`.

- [ ] **Step 4: Write the integration test**

Append to `verification/tests/test_runner.py`:

```python
class ScopeCoverageIntegrationTests(unittest.TestCase):
    """End-to-end: one signed scope clears REVIEW_REQUIRED for its members and nothing else."""

    def setUp(self):
        import subprocess, tempfile
        from pathlib import Path
        from verification.evidence import ed25519_openssl
        self.openssl = ed25519_openssl()
        if not self.openssl:
            self.skipTest('No OpenSSL with Ed25519')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'verification/trust').mkdir(parents=True)
        subprocess.run([self.openssl, 'genpkey', '-algorithm', 'Ed25519',
                        '-out', str(self.root/'private.pem')], check=True, capture_output=True)
        subprocess.run([self.openssl, 'pkey', '-in', str(self.root/'private.pem'), '-pubout',
                        '-out', str(self.root/'verification/trust/reviewer.pem')],
                       check=True, capture_output=True)
        (self.root/'verification/trust/reviewers.json').write_text(
            json.dumps({'reviewers': {'reviewer-a': {'public_key': 'verification/trust/reviewer.pem'}}}))

    def target(self):
        claim = lambda cid: {'id': cid, 'domain': 'visualization', 'classification': 'presentation',
                             'consumer': cid, 'expected': 1, 'statement': 's',
                             'evidence': [], 'dependencies': []}
        return {'schema': 'stratum-target/v1', 'id': 'test', 'description': 'scope fixture',
                'claims': [claim('scene.a.1'), claim('scene.b.1')],
                'contracts': [], 'runtime_obligations': []}

    def sign_scope(self, selector, subjects):
        import base64, subprocess
        from datetime import datetime, timedelta, timezone
        from verification.evidence import canonical, matching_subjects, scope_root
        covered = matching_subjects(selector, subjects)
        root_hash = scope_root([(sid, subjects[sid]) for sid in covered])
        body = {'author': 'author-a', 'reviewer': 'reviewer-a', 'verdict': 'accept',
                'subject_sha256': root_hash,
                'scope': {'selector': selector, 'count': len(covered), 'scope_root': root_hash},
                'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()}
        (self.root/'body').write_bytes(canonical(body))
        subprocess.run([self.openssl, 'pkeyutl', '-sign', '-inkey', str(self.root/'private.pem'),
                        '-rawin', '-in', str(self.root/'body'), '-out', str(self.root/'sig')],
                       check=True, capture_output=True)
        return {'body': body, 'signature': base64.b64encode((self.root/'sig').read_bytes()).decode()}

    def test_scope_clears_review_required_for_members_only(self):
        from verification.runner import audit, subject_digests
        target = self.target()
        catalog = {'schema': 'stratum-sources/v1', 'sources': []}
        inventory = {'fields': {'scene.a.1': 1, 'scene.b.1': 1}, 'functions': {}}
        subjects = subject_digests(target['claims'], [], [], {}, {})
        reviews = {'schema': 'stratum-reviews/v1', 'claims': {},
                   'scopes': [self.sign_scope('scene.a.*', subjects)]}
        report = audit(self.root, target, inventory, catalog, reviews=reviews)
        codes = lambda cid: {f['code'] for f in report['findings'] if f['subject'] == cid}
        self.assertNotIn('REVIEW_REQUIRED', codes('scene.a.1'))
        self.assertIn('REVIEW_REQUIRED', codes('scene.b.1'))
        self.assertIn('SCOPE_REVIEW_VALID', {f['code'] for f in report['findings']})
```

- [ ] **Step 5: Run the whole suite**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
```

Expected: PASS. Science still exit 2 with `REVIEW_REQUIRED` unchanged — `reviews.json` is still
`{}`, so nothing is covered yet. **A drop in `REVIEW_REQUIRED` at this step means the skip branch
is firing without a valid scope**; find out why before continuing.

- [ ] **Step 6: Commit**

```bash
git add verification/runner.py verification/tests/test_runner.py
git commit -m "Let a valid scope review answer REVIEW_REQUIRED for its members

Subject digests for claims, contracts and runtime obligations are built up
front so scope coverage is resolved once, against the live set."
```

---

### Task 4: Packets a reviewer can actually read

`review_requests.py` emits `{id, kind, subject_sha256, statement}`. A reviewer holding that would have to reconstruct every piece of evidence from the repository. The packet must carry the **resolved text each anchor selects**, so the review can happen away from the repo.

**Files:**
- Create: `verification/review_packets.py`
- Create: `verification/tests/test_review_packets.py`

**Interfaces:**
- Consumes: `evidence.select_anchor`, `evidence.safe_path`, `runner.source_subject`, `runner.subject_digests`.
- Produces: `review_packets.build_packet(claim: dict, sources: dict, root: Path) -> dict` with keys `id`, `kind`, `subject_sha256`, `statement`, `domain`, `classification`, `consumer`, `expected`, `scope`, `units`, `dependencies`, `evidence`; each evidence row is `{'source': str, 'role': str, 'url': str, 'revision': str | None, 'relation': str, 'anchor': dict | None, 'selected': str | None, 'unresolved': str | None}`.

- [ ] **Step 1: Write the failing test**

Create `verification/tests/test_review_packets.py`:

```python
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verification import review_packets


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = json.dumps({'hidden_size': 5120}).encode()
        (self.root/'config.json').write_bytes(raw)
        self.sources = {'cfg': {'id': 'cfg', 'role': 'publisher_config', 'kind': 'git',
                                'repo': 'deepseek-ai/DeepSeek-V2', 'revision': 'a'*40,
                                'url': 'https://huggingface.co/x', 'cache': 'config.json',
                                'sha256': hashlib.sha256(raw).hexdigest()}}
        self.claim = {'id': 'MODELS.ds2.d', 'domain': 'model', 'classification': 'documented',
                      'consumer': 'MODELS.ds2.d', 'expected': 5120, 'statement': 'the width',
                      'dependencies': [], 'scope': 'checkpoint', 'units': 'dimensions',
                      'evidence': [{'source': 'cfg', 'relation': 'equal',
                                    'anchor': {'kind': 'json_pointer', 'pointer': '/hidden_size',
                                               'sha256': 'x'*64}}]}

    def test_packet_carries_the_resolved_upstream_text(self):
        packet = review_packets.build_packet(self.claim, self.sources, self.root)
        self.assertEqual(packet['evidence'][0]['selected'], '5120')
        self.assertIsNone(packet['evidence'][0]['unresolved'])

    def test_packet_names_the_source_identity_not_just_its_id(self):
        packet = review_packets.build_packet(self.claim, self.sources, self.root)
        row = packet['evidence'][0]
        self.assertEqual(row['role'], 'publisher_config')
        self.assertEqual(row['revision'], 'a'*40)
        self.assertEqual(row['url'], 'https://huggingface.co/x')

    def test_unanchored_evidence_says_so_rather_than_being_omitted(self):
        claim = json.loads(json.dumps(self.claim))
        claim['evidence'][0]['anchor'] = None
        packet = review_packets.build_packet(claim, self.sources, self.root)
        self.assertIsNone(packet['evidence'][0]['selected'])
        self.assertEqual(packet['evidence'][0]['unresolved'], 'no selector')

    def test_unresolvable_anchor_reports_the_reason(self):
        claim = json.loads(json.dumps(self.claim))
        claim['evidence'][0]['anchor'] = {'kind': 'json_pointer', 'pointer': '/absent', 'sha256': 'x'*64}
        packet = review_packets.build_packet(claim, self.sources, self.root)
        self.assertIsNone(packet['evidence'][0]['selected'])
        self.assertIn('pointer', packet['evidence'][0]['unresolved'].lower())

    def test_packet_carries_the_statement_and_the_declared_scope(self):
        packet = review_packets.build_packet(self.claim, self.sources, self.root)
        self.assertEqual(packet['statement'], 'the width')
        self.assertEqual(packet['scope'], 'checkpoint')
        self.assertEqual(packet['units'], 'dimensions')
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_review_packets -v
```

Expected: FAIL — `verification/review_packets.py` does not exist.

- [ ] **Step 3: Write the module**

Create `verification/review_packets.py`:

```python
"""Render reviewable packets. Read-only: creates no approval, key or signature.

A packet carries the text each anchor actually selects, so a reviewer can read the evidence
without the repository and without re-running the verifier.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .evidence import loads, safe_path, select_anchor
from .runner import source_subject


def build_packet(claim: dict, sources: dict, root: Path) -> dict:
    rows = []
    for entry in claim.get('evidence', []):
        source = sources.get(entry.get('source'), {})
        row = {'source': entry.get('source'), 'role': source.get('role'), 'url': source.get('url'),
               'revision': source.get('revision'), 'relation': entry.get('relation'),
               'anchor': entry.get('anchor'), 'selected': None, 'unresolved': None}
        anchor = entry.get('anchor')
        if not anchor:
            row['unresolved'] = 'no selector'
        else:
            try:
                row['selected'], _ = select_anchor(safe_path(root, source['cache']).read_bytes(), anchor)
            except (ValueError, KeyError, OSError) as exc:
                row['unresolved'] = str(exc)
        rows.append(row)
    return {'id': claim.get('id'), 'kind': 'claim',
            'subject_sha256': source_subject(claim, sources),
            'statement': claim.get('statement'), 'domain': claim.get('domain'),
            'classification': claim.get('classification'), 'consumer': claim.get('consumer'),
            'expected': claim.get('expected'), 'scope': claim.get('scope'),
            'units': claim.get('units'), 'dependencies': claim.get('dependencies', []),
            'evidence': rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='directory to write packets into')
    p.add_argument('--selector', help='glob over claim ids; omit for every claim')
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    sources = {s['id']: s for s in loads((root/'verification/manifests/sources.json').read_text())['sources']}
    out = safe_path(root, a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite an existing packet directory')
    out.mkdir(parents=True, exist_ok=True)
    from fnmatch import fnmatchcase
    written = []
    for claim in target['claims']:
        if a.selector and not fnmatchcase(claim['id'], a.selector):
            continue
        packet = build_packet(claim, sources, root)
        (out/f"{claim['id']}.json").write_text(json.dumps(packet, indent=2) + '\n')
        written.append({'id': claim['id'], 'subject_sha256': packet['subject_sha256']})
    (out/'index.json').write_text(json.dumps(
        {'schema': 'stratum-review-packets/v1', 'selector': a.selector,
         'warning': 'Packets are NOT approvals. Read the selected bytes, the operand semantics and '
                    'the declared scope before signing anything.',
         'subjects': written}, indent=2) + '\n')
    print(f'Wrote {len(written)} packets to {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run --frozen python -m unittest verification.tests.test_review_packets -v
```

Expected: PASS, five tests.

- [ ] **Step 5: Generate a real packet set and read one**

```bash
uv run --frozen python -m verification.review_packets \
  --out verification/packets/models --selector 'MODELS.ds2.*'
cat verification/packets/models/MODELS.ds2.d.json
```

The packet must show the actual upstream text in `selected`. If every row says
`"unresolved": "no selector"`, the anchor plan has not run yet — that is Plan A's Task 2.

- [ ] **Step 6: Commit**

```bash
git add verification/review_packets.py verification/tests/test_review_packets.py
git commit -m "Render review packets carrying the text each anchor selects

The previous request format was an id and a statement, which is not something
a reviewer can form an opinion about."
```

Do **not** commit `verification/packets/` — packets are generated working material. Add
`verification/packets/` to `.gitignore` in this commit.

---

### Task 5: The signing tool

Builds the canonical body and invokes `openssl pkeyutl -sign`. It writes a **fragment** for a human to merge into `reviews.json`; it never edits the manifest, and it never generates a key.

**Files:**
- Create: `verification/sign_review.py`
- Create: `verification/tests/test_sign_review.py`
- Modify: `package.json`

**Interfaces:**
- Consumes: `evidence.canonical`, `evidence.ed25519_openssl`, `evidence.scope_root`, `evidence.matching_subjects`, `runner.subject_digests`.
- Produces: `sign_review.build_body(*, author: str, reviewer: str, subject_sha256: str, days: int, scope: dict | None=None) -> dict` and `sign_review.sign(body: dict, key_path: Path) -> str` (base64 signature).

- [ ] **Step 1: Write the failing test**

Create `verification/tests/test_sign_review.py`:

```python
import subprocess
import tempfile
import unittest
from pathlib import Path

from verification import sign_review
from verification.evidence import check_review, ed25519_openssl


class SignReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        openssl = ed25519_openssl()
        if not openssl:
            self.skipTest('No OpenSSL with Ed25519')
        subprocess.run([openssl, 'genpkey', '-algorithm', 'Ed25519', '-out', str(self.root/'private.pem')],
                       check=True, capture_output=True)
        subprocess.run([openssl, 'pkey', '-in', str(self.root/'private.pem'), '-pubout',
                        '-out', str(self.root/'reviewer.pem')], check=True, capture_output=True)
        self.trust = {'reviewers': {'reviewer-a': {'public_key': 'reviewer.pem'}}}

    def test_signed_body_verifies_through_check_review(self):
        """The signing tool and the verifier must agree, or the ceremony produces nothing."""
        body = sign_review.build_body(author='author-a', reviewer='reviewer-a',
                                      subject_sha256='a'*64, days=7)
        signature = sign_review.sign(body, self.root/'private.pem')
        finding = check_review({'id': 'c'}, 'a'*64,
                               {'c': {'body': body, 'signature': signature}}, self.root, self.trust)
        self.assertEqual(finding.status, 'PASS')

    def test_self_review_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            sign_review.build_body(author='same', reviewer='same', subject_sha256='a'*64, days=7)

    def test_expiry_is_timezone_aware(self):
        body = sign_review.build_body(author='a', reviewer='b', subject_sha256='a'*64, days=7)
        self.assertTrue(body['expires_at'].endswith('+00:00'))

    def test_scope_body_binds_its_own_root(self):
        scope = {'selector': 'scene.x.*', 'count': 2, 'scope_root': 'b'*64}
        body = sign_review.build_body(author='a', reviewer='b', subject_sha256='b'*64, days=7, scope=scope)
        self.assertEqual(body['subject_sha256'], body['scope']['scope_root'])

    def test_scope_body_refuses_a_root_that_does_not_match_the_subject(self):
        scope = {'selector': 'scene.x.*', 'count': 2, 'scope_root': 'c'*64}
        with self.assertRaises(ValueError):
            sign_review.build_body(author='a', reviewer='b', subject_sha256='b'*64, days=7, scope=scope)

    def test_tool_never_generates_a_private_key(self):
        source = (Path(__file__).resolve().parents[1]/'sign_review.py').read_text()
        self.assertNotIn('genpkey', source)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_sign_review -v
```

Expected: FAIL — `verification/sign_review.py` does not exist.

- [ ] **Step 3: Write the tool**

Create `verification/sign_review.py`:

```python
"""Build and sign a review body with a key the operator supplies.

This tool generates no key and edits no manifest. It writes a fragment for a human to merge
into verification/manifests/reviews.json after reading what they are attesting to.
"""
from __future__ import annotations
import argparse, base64, json, subprocess, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .evidence import canonical, ed25519_openssl, loads, matching_subjects, safe_path, scope_root
from .runner import subject_digests


def build_body(*, author: str, reviewer: str, subject_sha256: str, days: int, scope: dict | None=None) -> dict:
    if not author or not reviewer or author == reviewer:
        raise ValueError('A review needs an author and a different reviewer')
    if scope is not None and scope.get('scope_root') != subject_sha256:
        raise ValueError('A scope review must bind its own scope root as its subject')
    body = {'author': author, 'reviewer': reviewer, 'subject_sha256': subject_sha256,
            'verdict': 'accept',
            'expires_at': (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()}
    if scope is not None:
        body['scope'] = scope
    return body


def sign(body: dict, key_path: Path) -> str:
    openssl = ed25519_openssl()
    if openssl is None:
        raise SystemExit('No OpenSSL with Ed25519. Install OpenSSL 3.x or set STRATUM_OPENSSL.')
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp)/'body.json'; payload.write_bytes(canonical(body))
        out = Path(tmp)/'signature'
        subprocess.run([openssl, 'pkeyutl', '-sign', '-inkey', str(key_path), '-rawin',
                        '-in', str(payload), '-out', str(out)], check=True, capture_output=True)
        return base64.b64encode(out.read_bytes()).decode()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', required=True, help='path to the reviewer private key, outside the repository')
    p.add_argument('--author', required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--out', required=True, help='fragment to write; merge it yourself after reading it')
    p.add_argument('--subject', help='a single claim/contract/obligation id')
    p.add_argument('--selector', help='glob over subject ids for a scope review')
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args(argv)
    if bool(a.subject) == bool(a.selector):
        raise SystemExit('Give exactly one of --subject or --selector')
    root = Path(a.root).resolve()
    target = loads((root/'verification/manifests/observatory.json').read_text())
    sources = {s['id']: s for s in loads((root/'verification/manifests/sources.json').read_text())['sources']}
    runtime = {}
    runtime_path = root/'verification/manifests/runtime.json'
    if runtime_path.exists():
        runtime = loads(runtime_path.read_text())
    subjects = subject_digests(target['claims'], target['contracts'],
                               target.get('runtime_obligations', []), sources, runtime)
    if a.subject:
        if a.subject not in subjects:
            raise SystemExit(f'Unknown or unreviewable subject: {a.subject}')
        body = build_body(author=a.author, reviewer=a.reviewer,
                          subject_sha256=subjects[a.subject], days=a.days)
        fragment = {'schema': 'stratum-reviews/v1',
                    'claims': {a.subject: {'body': body, 'signature': sign(body, Path(a.key))}},
                    'scopes': []}
        print(f'Signed 1 subject: {a.subject}')
    else:
        covered = matching_subjects(a.selector, subjects)
        if not covered:
            raise SystemExit(f'Selector matches no subject: {a.selector}')
        root_hash = scope_root([(sid, subjects[sid]) for sid in covered])
        body = build_body(author=a.author, reviewer=a.reviewer, subject_sha256=root_hash, days=a.days,
                          scope={'selector': a.selector, 'count': len(covered), 'scope_root': root_hash})
        fragment = {'schema': 'stratum-reviews/v1', 'claims': {},
                    'scopes': [{'body': body, 'signature': sign(body, Path(a.key))}]}
        print(f'Signed a scope over {len(covered)} subjects matching {a.selector}')
    out = safe_path(root, a.out)
    if out.exists():
        raise SystemExit('Refusing to overwrite an existing fragment')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fragment, indent=2) + '\n')
    print(f'Wrote {a.out}. Merge it into verification/manifests/reviews.json yourself.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

```bash
uv run --frozen python -m unittest verification.tests.test_sign_review -v
```

Expected: PASS, six tests. `test_signed_body_verifies_through_check_review` is the important one —
it proves the signer and the verifier agree on the canonical form.

- [ ] **Step 5: Add the npm scripts**

In `package.json`, next to the existing `verify:*` scripts:

```json
    "review:packets": "uv run --frozen python -m verification.review_packets",
    "review:sign": "uv run --frozen python -m verification.sign_review",
```

- [ ] **Step 6: Commit**

```bash
git add verification/sign_review.py verification/tests/test_sign_review.py package.json
git commit -m "Add the review signing tool

Builds the canonical body and signs it with an operator-supplied key. It
generates no key, edits no manifest, and a test asserts it never calls genpkey."
```

---

### Task 6: The enrollment ceremony

**Files:**
- Create: `docs/verifiers/REVIEW_CEREMONY.md`
- Modify: `AGENTS.md`

**Interfaces:** none — documentation.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_sign_review.py`:

```python
class CeremonyDocTests(unittest.TestCase):
    def test_ceremony_is_documented(self):
        doc = Path(__file__).resolve().parents[2]/'docs/verifiers/REVIEW_CEREMONY.md'
        self.assertTrue(doc.exists())
        text = doc.read_text()
        for required in ('genpkey -algorithm Ed25519', 'verification/trust/reviewers.json',
                         'revoked', 'never enters the repository'):
            with self.subTest(required=required):
                self.assertIn(required, text)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_sign_review -v
```

- [ ] **Step 3: Write the ceremony**

Create `docs/verifiers/REVIEW_CEREMONY.md` covering, in order:

1. **Generate a key.** `openssl genpkey -algorithm Ed25519 -out ~/.stratum/reviewer-<name>.pem`,
   then `openssl pkey -in ... -pubout -out reviewer-<name>.pem`. The private key **never enters the
   repository**; state the `chmod 600` and the fact that macOS `openssl` is LibreSSL and cannot do
   this, so `STRATUM_OPENSSL` may be needed.
2. **Enroll the public key.** Commit `verification/trust/reviewer-<name>.pem` and add the entry to
   `verification/trust/reviewers.json`. Record the fingerprint
   (`openssl pkey -pubin -in reviewer-<name>.pem -outform DER | sha256sum`) in the commit message.
   `.github/CODEOWNERS` already assigns `/verification/trust/` to `@phissenschaft`; note that this
   is only enforced with branch protection on `main`.
3. **Generate packets.** `npm run review:packets -- --out verification/packets/<name> --selector '<glob>'`.
4. **Read them.** What a reviewer is attesting to: the selected bytes say what the claim says they
   say; the operand semantics are right; the declared scope and units are honest.
5. **Sign.** `npm run review:sign -- --key ~/.stratum/reviewer-<name>.pem --author <author>
   --reviewer <name> --selector '<glob>' --out verification/candidates/review-<name>.json`.
6. **Merge.** Merge the fragment into `verification/manifests/reviews.json` by hand, then
   `npm run verify:science` and confirm the covered subjects report `REVIEW_VALID` or
   `SCOPE_REVIEW_VALID`.
7. **Revoke.** Set `"revoked": true` on the entry in `reviewers.json`. Every review signed by that
   key blocks immediately. Never delete the entry — a deleted reviewer is indistinguishable from
   one who never existed.
8. **What retires a review.** Changing a claim's evidence, consumer value or expected value changes
   its subject digest; adding a subject matching a scope's selector changes the scope root. Both
   retire the review, which is the intended behaviour and never a reason to widen a selector.

- [ ] **Step 4: Reference it from `AGENTS.md`**

Add a line to the Commands section for `review:packets` and `review:sign`, and point at
`docs/verifiers/REVIEW_CEREMONY.md`. `CLAUDE.md` is a symlink; edit `AGENTS.md` only.

- [ ] **Step 5: Run the tests and commit**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
git add docs/verifiers/REVIEW_CEREMONY.md AGENTS.md verification/tests/test_sign_review.py
git commit -m "Document the review enrollment, signing and revocation ceremony"
```

---

### Task 7: Record what the independence property actually is

The reviewer identity will be held by the same human who authors the claims, under a distinct identity. That satisfies `author != reviewer` as written, and `VERIFICATION.md` must say so — the difference between a gate that reports its own strength honestly and one that launders a mechanism into a guarantee.

**Files:**
- Modify: `VERIFICATION.md`
- Modify: `verification/tests/test_reported_numbers.py` (created by Plan A Task 8; create it here if Plan A has not run)

**Interfaces:** none.

- [ ] **Step 1: Write the failing test**

Append to `verification/tests/test_reported_numbers.py`:

```python
    def test_review_independence_is_described_honestly(self):
        """If any enrolled reviewer shares a human with the claim author, the document must say so."""
        trust = json.loads((ROOT/'verification/trust/reviewers.json').read_text())
        if not trust.get('reviewers'):
            self.skipTest('no reviewer enrolled yet')
        self.assertIn('procedural', self.text.lower(),
                      'VERIFICATION.md does not state what the independence property actually is')
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run --frozen python -m unittest verification.tests.test_reported_numbers -v
```

Expected: skip while `reviewers.json` is empty; FAIL once a reviewer is enrolled and the document
is silent.

- [ ] **Step 3: Write the record**

In `VERIFICATION.md`'s **Known and unclosed** section, replace the `REVIEW_REQUIRED` bullet with an
accurate account: the review mechanism enforces that the reviewer is enrolled in the protected
trust root, holds the private key, and is named differently from the author; it **cannot** enforce
that they are a different person. State that author and reviewer are currently the same human under
distinct identities, so **independence here is procedural, not actual**, and that a second party
holding their own key is what would make it actual.

Also record the count of subjects covered by scope reviews versus per-subject reviews, so a reader
can see how much of the review surface one signature covers.

- [ ] **Step 4: Run the tests and commit**

```bash
uv run --frozen python -m unittest discover -s verification/tests -v
npm run verify:engineering && npm run verify:science; echo "science exit: $?"
git add VERIFICATION.md verification/tests/test_reported_numbers.py
git commit -m "Record that review independence is procedural, not actual

The mechanism enforces a different name and a separately held key. It cannot
enforce a different person, and the document should not imply that it does."
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

- A reviewer can generate a key, enroll it, read packets containing real upstream text, sign a
  scope, merge the fragment, and watch `REVIEW_REQUIRED` fall for exactly the covered subjects.
- No `FAIL` code appears anywhere. `SCOPE_REVIEW_INVALID` firing means a scope no longer matches
  the manifest, which is the mechanism working.
- Publication remains BLOCKED on the five stub contracts, the two redirect-blocked sources and
  `RECEIPT_UNATTESTED` — none of which this plan addresses.
- `VERIFICATION.md` states what the independence property is and is not.
