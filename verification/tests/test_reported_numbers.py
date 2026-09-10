"""`VERIFICATION.md` must agree with the reports it claims to summarise.

The document is the project's public record. It drifted once already -- it quoted 15,737 BLOCKED
findings long after the number was 5,160, and it disagreed with itself about how many aesthetics
scenarios ran. These tests pin every load-bearing figure to the report that produced it, and pin
the narrative corrections that are not figures, so the next drift fails a gate instead of being
read as fact.

Nothing here is inside `toolchain_digest`: `VERIFICATION.md` is not a verifier input, so updating
it to satisfy these tests cannot invalidate the receipt they read.
"""
import collections
import hashlib
import json
import re
import unittest
from pathlib import Path

from verification.engineering import toolchain_digest
from verification.evidence import digest, loads

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'verification/reports'


def load(name):
    path = REPORTS/f'{name}.json'
    return json.loads(path.read_text()) if path.exists() else None


class RecordTestCase(unittest.TestCase):
    """Common setup: the record itself, plus whichever reports exist."""

    def setUp(self):
        self.text = (ROOT/'VERIFICATION.md').read_text()

    def require(self, name, command):
        report = load(name)
        if report is None:
            self.skipTest(f'no {name} report; run {command}')
        return report

    def assertStated(self, value, why):
        """A figure must appear in the record, written the way the record writes figures."""
        forms = {str(value)}
        if isinstance(value, int):
            forms.add(f'{value:,}')
        if not any(form in self.text for form in forms):
            self.fail(f'VERIFICATION.md does not state {sorted(forms)[-1]} ({why})')


class PublicationNumberTests(RecordTestCase):
    def setUp(self):
        super().setUp()
        self.report = self.require('publication', 'npm run verify:science')
        self.counts = collections.Counter(f['code'] for f in self.report['findings'])
        self.statuses = collections.Counter(f['status'] for f in self.report['findings'])

    def test_blocked_total_is_current(self):
        self.assertStated(self.statuses['BLOCKED'], 'BLOCKED findings in the publication audit')

    def test_pass_total_is_current(self):
        self.assertStated(self.statuses['PASS'], 'PASS findings in the publication audit')

    def test_no_fail_findings_are_reported_as_present(self):
        self.assertEqual(self.statuses['FAIL'], 0, 'publication gate has FAIL findings')

    def test_claim_anchor_count_is_current(self):
        self.assertStated(self.counts['CLAIM_ANCHOR'], 'CLAIM_ANCHOR findings')

    def test_review_required_count_is_current(self):
        self.assertStated(self.counts['REVIEW_REQUIRED'], 'REVIEW_REQUIRED findings')

    def test_verified_source_count_is_current(self):
        coverage = self.report['coverage']
        self.assertStated(coverage['verified_sources'], 'verified source snapshots')
        self.assertStated(coverage['sources'], 'candidate source entries')

    def test_no_stale_accepted_snapshot_zero(self):
        """`Accepted raw-byte snapshots | 0` was false once 60 of 62 sources verified."""
        if self.report['coverage']['verified_sources'] > 0:
            self.assertNotRegex(self.text, r'Accepted raw-byte snapshots \|\s*0\b')

    def test_no_stale_dual_code_claim(self):
        """Saying a code BLOCKS is a claim about the report, and the report says it does not."""
        if self.counts['MODEL_DUAL_CODE'] == 0:
            self.assertNotRegex(self.text, r'BLOCKED on [^.]*`MODEL_DUAL_CODE`')
            self.assertRegex(self.text, r'`MODEL_DUAL_CODE`[^.]*\b0\b',
                             'the record does not state that MODEL_DUAL_CODE is 0')

    def test_claim_role_is_not_reported_as_blocking(self):
        if self.counts['CLAIM_ROLE'] == 0:
            self.assertNotRegex(self.text, r'BLOCKED on [^.]*`CLAIM_ROLE`')


class RuntimeNumberTests(RecordTestCase):
    def setUp(self):
        super().setUp()
        self.report = self.require('runtime', 'npm run verify:runtime')
        self.statuses = collections.Counter(f['status'] for f in self.report['findings'])

    def test_runtime_is_blocked_not_failed(self):
        self.assertEqual(self.report['status'], 'BLOCKED')
        self.assertEqual(self.statuses['FAIL'], 0, 'runtime gate has FAIL findings')

    def test_not_run_obligation_count_is_current(self):
        self.assertStated(self.statuses['NOT_RUN'], 'NOT_RUN runtime obligations')

    def test_receipt_staleness_is_recorded_as_operational_fact(self):
        """A manifest edit invalidates runtime receipts; they must be re-probed, not re-stamped."""
        self.assertRegex(self.text, r're-?probe', 'the record does not say receipts are re-probed')
        self.assertIn('RUNTIME_STALE', self.text)


class SoftwareGateNumberTests(RecordTestCase):
    def test_aesthetics_scenario_count_is_self_consistent(self):
        counts = set(re.findall(r'(\d+) scenarios', self.text))
        self.assertLessEqual(len(counts), 1, f'VERIFICATION.md states conflicting scenario counts: {counts}')

    def test_aesthetics_figures_are_current(self):
        report = self.require('aesthetics', 'npm run verify:aesthetics')
        self.assertStated(report['scenarios'], 'aesthetics scenarios')
        self.assertStated(report['counts'].get('PASS', 0), 'aesthetics checks')

    def test_schematics_figures_are_current(self):
        report = self.require('schematics', 'npm run verify:schematics')
        self.assertStated(report['scenes'], 'schematics scenes')
        self.assertStated(report['counts'].get('PASS', 0), 'schematics checks')
        self.assertStated(report['decoder_layers'], 'decoder layers')

    def test_engineering_test_counts_are_current(self):
        report = self.require('engineering', 'npm run verify:engineering')
        if report['verifier_sha256'] != toolchain_digest(ROOT):
            self.skipTest('receipt predates the current verifier; runner.py reports VERIFIER_STALE')
        self.assertStated(report['test_counts']['verifier'], 'verifier tests in the receipt')
        self.assertStated(report['test_counts']['atlas'], 'model/graph tests in the receipt')

    def test_artifact_identity_names_the_current_bytes(self):
        """Re-derived from the working tree, not copied from the receipt, so it cannot go stale.

        This converges: `VERIFICATION.md` is not a verifier input, so writing the current digest
        into it does not move the digest. Edit the verifier last, then paste what this reports.
        """
        self.assertIn(toolchain_digest(ROOT), self.text,
                      'VERIFICATION.md quotes a stale verifier identity; paste the current '
                      'toolchain_digest into "Artifact identity" after your last verifier edit')
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        catalog = loads((ROOT/'verification/manifests/sources.json').read_text())
        self.assertIn(digest({'target': target, 'catalog': catalog}), self.text,
                      'VERIFICATION.md quotes a stale scientific lock')
        artifact = ROOT/'dist/index.html'
        if not artifact.exists():
            self.skipTest('no built artifact; run npm run build')
        self.assertIn(hashlib.sha256(artifact.read_bytes()).hexdigest(), self.text,
                      'VERIFICATION.md quotes a stale portable-HTML digest')


class ResidueNarrativeTests(RecordTestCase):
    """The corrections that are prose rather than a single figure."""

    def setUp(self):
        super().setUp()
        self.report = self.require('publication', 'npm run verify:science')

    def scene_residue(self):
        """Recompute the kept scene citations straight from the report."""
        rows = [f for f in self.report['findings']
                if f['code'] == 'CLAIM_ANCHOR' and f['subject'].startswith('scene.')]
        by_source = collections.Counter()
        claims = collections.defaultdict(set)
        for row in rows:
            match = re.search(r'selector for ([^;]+);', row['message'])
            source = match[1] if match else '?'
            by_source[source] += 1
            claims[source].add(row['subject'])
        return rows, by_source, claims

    def test_kept_scene_citations_are_stated_and_explained(self):
        rows, by_source, claims = self.scene_residue()
        gpu = {'h100', 'h200', 'b200', 'gb300'}
        comm = {'nccl-api', 'nvshmem-api', 'nixl-api', 'nccl', 'nvshmem'}
        self.assertStated(len(rows), 'kept scene citations no dependency supplies')
        self.assertStated(sum(by_source[s] for s in gpu), 'GPU vendor-document citations kept')
        self.assertStated(len(set().union(*(claims[s] for s in gpu))), 'claims carrying them')
        self.assertStated(sum(by_source[s] for s in comm), 'communication citations kept')
        self.assertStated(len(set().union(*(claims[s] for s in comm))), 'claims carrying them')
        self.assertIn('ANCHOR_RESIDUE.md', self.text, 'the record does not point at the residue decisions')

    def test_residue_document_totals_are_stated(self):
        residue = ROOT/'docs/verifiers/ANCHOR_RESIDUE.md'
        table = residue.read_text().split('## Decisions by class')[1]
        classes = [line for line in table.splitlines() if line.startswith('| `')]
        total = sum(int(line.strip('|').split('|')[3]) for line in classes)
        self.assertStated(len(classes), 'residue decision classes')
        self.assertStated(total, 'residue rows with a recorded decision')

    def test_the_withdrawn_anti_laundering_claim_is_not_restated(self):
        """The dependency fixpoint already carried that property; the role change did not add it."""
        self.assertNotIn('strictly harder to fool', self.text)
        self.assertNotIn('strictly harder to fool', (ROOT/'AGENTS.md').read_text())

    def test_the_anti_laundering_correction_is_recorded(self):
        self.assertIn('fixpoint', self.text)
        self.assertRegex(self.text, r'more permissive')

    def test_pdf_limitation_is_recorded_with_its_cause(self):
        self.assertIn('sources.py:30', self.text)
        self.assertIn('evidence.py:140', self.text)
        self.assertIn('pdf_text_quote', self.text)
        self.assertIn('toolchain_digest', self.text)

    def test_pdf_limitation_cites_lines_that_still_decode_strictly(self):
        """The cited line numbers must still be the strict UTF-8 decode they are blamed on."""
        for path, line in (('verification/sources.py', 30), ('verification/evidence.py', 140)):
            source = (ROOT/path).read_text().splitlines()[line - 1]
            self.assertIn("decode('utf-8','strict')", source, f'{path}:{line} is no longer the decode')

    def test_data_versus_source_discrepancies_are_recorded(self):
        for figure in ('151,669', '151,936', '32,768'):
            self.assertIn(figure, self.text, f'the record does not state {figure}')
        self.assertIn('swish', self.text)
        self.assertIn('SiLU', self.text)

    def test_quote_authoring_cap_is_recorded(self):
        self.assertIn('anchor-documents.py', self.text)
        self.assertRegex(self.text, r'`\(source, field\)`')
