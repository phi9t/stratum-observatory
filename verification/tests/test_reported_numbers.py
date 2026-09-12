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

    def assertStated(self, value, why, unit=None):
        """A figure must appear in the record as a whole number, not inside a longer one.

        Word-boundary matched rather than substring matched. Raw `in` let a stated 113 satisfy an
        assertion about 3,113, and it is the third-order cause of three drifts this file was
        meant to catch. Pass `unit` -- the word the record writes after the figure -- wherever the
        same integer could plausibly stand for a different quantity; that is the part word
        boundaries alone cannot fix, because "28 PASS" and "28 of 31 obligations" are both whole
        words and a bare 28 matches either.
        """
        forms = {str(value)}
        if isinstance(value, int):
            forms.add(f'{value:,}')
        # A digit, thousands comma or decimal point on either side means the match sits inside a
        # larger number: a stated 3,113 must not satisfy an assertion about 113, which plain \b
        # still allows because a comma is a word boundary.
        before, after = r'(?<![\d,.])', r'(?![\d,])(?!\.\d)'
        suffix = rf'\s+{re.escape(unit)}\b' if unit else after
        if not any(re.search(rf'{before}{re.escape(form)}{suffix}', self.text) for form in forms):
            wanted = f'{sorted(forms)[-1]}{" " + unit if unit else ""}'
            self.fail(f'VERIFICATION.md does not state {wanted} ({why})')


class RecordMatcherTests(RecordTestCase):
    """The matcher itself. Three drifts have hidden behind it, so its behaviour is tested rather
    than assumed: it is the tool every remaining task uses to keep the public record honest."""

    def test_a_figure_inside_a_longer_number_is_not_stated(self):
        self.text = 'Schematics ran 3,113 checks over 1,447 states.'
        self.assertStated(3113, 'the figure the record actually writes')
        for hidden in (113, 447, 13):
            with self.assertRaises(AssertionError, msg=hidden):
                self.assertStated(hidden, 'a figure that is only a substring')

    def test_a_figure_inside_a_decimal_is_not_stated(self):
        self.text = 'The residual implies 0.531 bytes per channel.'
        with self.assertRaises(AssertionError):
            self.assertStated(531, 'a figure that is only a decimal fraction')

    def test_a_figure_ending_a_sentence_is_still_stated(self):
        self.text = 'The audit reports 28.'
        self.assertStated(28, 'a figure followed by a full stop')

    def test_the_unit_separates_two_quantities_that_share_a_number(self):
        """What word boundaries alone cannot do, and what actually hid the 26-against-28 drift:
        both readings are whole words, so only the unit tells them apart."""
        self.text = 'Runtime: 28 of 31 obligations NOT_RUN. Publication: 26 PASS.'
        self.assertStated(28, 'unrun obligations')
        with self.assertRaises(AssertionError):
            self.assertStated(28, 'publication PASS findings', unit='PASS')
        self.assertStated(26, 'publication PASS findings', unit='PASS')


class PublicationNumberTests(RecordTestCase):
    def setUp(self):
        super().setUp()
        self.report = self.require('publication', 'npm run verify:science')
        self.counts = collections.Counter(f['code'] for f in self.report['findings'])
        self.statuses = collections.Counter(f['status'] for f in self.report['findings'])

    def test_the_committed_publication_report_describes_these_bytes(self):
        """Offline-only is only honest if the offline report is proven current by a gate that runs.

        Completeness cannot run in CI, so the committed `publication.json` is the authoritative
        completeness result and this record states its figures. Without this check, "offline"
        quietly becomes "stale": the report could describe a lock and a consumer inventory three
        edits old while every figure here still agreed with it. The report binds both digests;
        they are re-derived from the working tree, exactly as `runner.py` does for a receipt.
        """
        import subprocess
        from verification.evidence import canonical
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        catalog = loads((ROOT/'verification/manifests/sources.json').read_text())
        self.assertEqual(self.report['lock_sha256'], digest({'target': target, 'catalog': catalog}),
                         'the committed publication report was generated on a different scientific lock; '
                         'run npm run verify:science on a trusted machine and commit the report')
        inv = loads(subprocess.run(['node', 'tools/export-verification.mjs'], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout)
        self.assertEqual(self.report['consumer_sha256'], digest(inv['functions']),
                         'the committed publication report was generated on different application bytes; '
                         'run npm run verify:science on a trusted machine and commit the report')

    def test_blocked_total_is_current(self):
        self.assertStated(self.statuses['BLOCKED'], 'BLOCKED findings in the publication audit',
                          unit='BLOCKED')

    def test_pass_total_is_current(self):
        self.assertStated(self.statuses['PASS'], 'PASS findings in the publication audit',
                          unit='PASS')

    def test_no_fail_findings_are_reported_as_present(self):
        self.assertEqual(self.statuses['FAIL'], 0, 'publication gate has FAIL findings')

    def test_transcription_figures_are_current(self):
        """The 54-of-79 field comparison, read out of the audit that actually performed it."""
        found = {x['code']: x for x in self.report['findings']}
        total = re.search(r'All (\d+) transcribed fields', found['CED_FIELD_UNMAPPED']['message'])
        self.assertStated(int(total.group(1)), 'transcribed fields in the model table')
        compared = re.search(r'(\d+) transcribed fields re-derived',
                             found['CED_PUBLISHER_AGREEMENT']['message'])
        if not compared:
            self.skipTest('the audit compared no publisher field; acquire the evidence cache')
        self.assertStated(int(compared.group(1)), 'transcribed fields compared')

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
        """With a unit, because a bare NOT_RUN count is exactly the collision `unit=` exists for.

        On these bytes 29 is the NOT_RUN obligation count, the number of published previews, the
        number of function/file identities and a citation count. A bare `assertStated(29)` passes
        on whichever of those it reaches first, which is a test that agrees with the record by
        coincidence -- the same failure mode as a check written against a fixed set of eight. The
        total is read from the manifest so the unit cannot itself go stale.
        """
        total = len(loads((ROOT/'verification/manifests/observatory.json').read_text())
                    ['runtime_obligations'])
        self.assertStated(self.statuses['NOT_RUN'], 'NOT_RUN runtime obligations',
                          unit=f'of {total} obligations')

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

    def test_kv_budget_figures_are_current(self):
        """Re-derived from the working tree rather than copied, for the same reason the artifact
        identity below is: the gate's report keeps only non-PASS messages, and a figure re-derived
        by the very functions the gate runs cannot go stale against them."""
        from verification.ced import (derived_main_kv_bytes_per_token, indexer_channels_per_token,
                                      inventory)
        models,_ = inventory(ROOT)
        m = models['dsv41']
        main = derived_main_kv_bytes_per_token(m)
        self.assertStated(int(main), 'derived main KV bytes per token')
        per_channel = (m['kvBytesPerToken']-main)/indexer_channels_per_token(m)
        self.assertIn(f'{per_channel:.3f}', self.text,
                      'VERIFICATION.md does not state the implied bytes per indexer channel')

    def test_parameter_band_figures_are_current(self):
        """Re-derived from the working tree, like the KV figures above and for the same reason.

        `PARAM_BAND_RATIO` is a differential, so the record quotes a quantity that appears in no
        report field: the gate report keeps only non-PASS messages. Re-deriving it with the very
        function the gate runs is the only form of this check that cannot go stale.
        """
        from verification.ced import (derived_compressor_gate_parameters,
                                      derived_main_kv_bytes_per_token, inventory)
        models, _ = inventory(ROOT)
        m = models['dsv41']
        self.assertStated(derived_compressor_gate_parameters(m), 'compressor gate parameters',
                          unit='parameters')
        residual = int(m['kvBytesPerToken'] - derived_main_kv_bytes_per_token(m))
        self.assertStated(residual, 'the unexplained FP4 KV residual', unit='B')

    def test_the_exposition_checks_are_reported_as_having_run(self):
        """The record says the written exposition is checked. That is a claim about the gate, so
        the codes must be in the gate's own passed set, not only in the prose."""
        report = self.require('schematics', 'npm run verify:schematics')
        for code in ('COVERAGE_UNCITED', 'COVERAGE_DERIVED', 'PARAM_BAND_RATIO'):
            self.assertIn(code, report['passed_codes'],
                          f'{code} did not run in the schematics report the record summarises')
            self.assertIn(code, self.text, f'the record does not name {code}')

    def test_browser_figures_are_current(self):
        """The browser figures had no reconciliation at all, and one of them was wrong.

        `device_scale_states` was the literal `32` in a dict whose every sibling is derived --
        `len(cases)`, `len(responsive)`, `sum(...)`. The loop behind it is
        `2 ratios x models x 2 themes`, so a ninth checkpoint made the true value 36 while the
        literal kept reporting the eight-model figure. It reached `tests/browser-results.json`,
        the signed engineering receipt and the published record, and nothing here looked: this
        file reconciled aesthetics scenarios, schematics scenes, decoder layers and both receipt
        test counts, and not one browser figure. Deriving the literal closes that number; this
        test closes the mechanism, which is the half that keeps the next one from landing.
        """
        report = self.require('engineering', 'npm run verify:engineering')
        if report['verifier_sha256'] != toolchain_digest(ROOT):
            self.skipTest('receipt predates the current verifier; runner.py reports VERIFIER_STALE')
        browser = report.get('browser') or {}
        for key, why in (('geometry_states', 'rendered browser states'),
                         ('responsive_states', 'responsive viewport combinations'),
                         ('device_scale_states', 'device-scale cases')):
            self.assertIn(key, browser, f'the receipt carries no {key}')
            self.assertStated(browser[key], why)

    def test_device_scale_states_matches_the_cases_the_run_recorded(self):
        """The receipt must agree with the case file the same run wrote.

        This is the check that fails on a transcribed count while every gate stays green: the
        states are enumerated in `tests/browser-cases.json` with their labels, so the figure has
        a source of truth sitting beside it that nothing was comparing against.
        """
        report = self.require('engineering', 'npm run verify:engineering')
        if report['verifier_sha256'] != toolchain_digest(ROOT):
            self.skipTest('receipt predates the current verifier; runner.py reports VERIFIER_STALE')
        cases = ROOT/'tests/browser-cases.json'
        if not cases.exists():
            self.skipTest('no browser case file; run npm run verify:engineering')
        recorded = sum(1 for c in json.loads(cases.read_text())['states']
                       if str(c.get('label', '')).startswith('device-scale-'))
        self.assertEqual(report['browser']['device_scale_states'], recorded,
                         'the receipt states a device-scale count the run did not record')

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

    def residue_buckets(self, seen):
        """Partition the residue sources by what the application and the manifests say they are.

        Both sets were literals -- `{'h100','h200','b200','gb300'}` and five communication ids --
        and the failure mode was quiet in a specific way: a source that joined the residue and
        belonged to neither bucket would grow `len(rows)`, which is stated, while dropping out of
        both subtotals, which are also stated. All three figures would then agree with a record
        written from the same stale assumption. So the buckets are derived, and the partition is
        closed: a source in neither is a failure, not a silent omission.
        """
        import subprocess
        inv = json.loads(subprocess.run(['node', 'tools/export-verification.mjs'], cwd=ROOT,
                                        capture_output=True, text=True, check=True).stdout)
        gpu = set(inv['hardware']) & seen
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        libraries = {r['id'].split('.', 1)[1].lower() for r in target.get('contracts', [])
                     if r.get('domain') == 'communication'}
        comm = {s for s in seen if s in libraries or s.removesuffix('-api') in libraries}
        unclassified = seen - gpu - comm
        self.assertFalse(unclassified,
                         f'residue sources in neither the GPU nor the communication bucket: '
                         f'{sorted(unclassified)}. They are counted in the stated total and in '
                         f'neither stated subtotal; classify them or widen a bucket deliberately.')
        return gpu, comm

    def test_kept_scene_citations_are_stated_and_explained(self):
        rows, by_source, claims = self.scene_residue()
        gpu, comm = self.residue_buckets(set(by_source))
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

class AccountingNumberTests(RecordTestCase):
    def test_accounting_report_is_current_and_classified(self):
        import subprocess
        report=self.require('accounting','npm run verify:accounting')
        exported=json.loads(subprocess.check_output(['node','tools/export-accounting.mjs'],cwd=ROOT))
        self.assertEqual(report['consumer_sha256'],exported['consumer_sha256'])
        self.assertEqual(report['function_sha256'],exported['function_sha256'])
        self.assertEqual(report['methods_sha256'],hashlib.sha256((ROOT/'verification/manifests/accounting-methods.json').read_bytes()).hexdigest())
        self.assertEqual(report['case_count'],len(exported['cases']))
        counts=collections.Counter(f['classification'] for f in report['findings'])
        self.assertTrue(set(counts)<={'agreement','explained_difference','unsupported','contradiction'})
        self.assertEqual(counts['contradiction'],0)
        self.assertEqual(report['status'],'PASS')
        for name,count in report['counts'].items():
            self.assertEqual(count,counts[name])
            self.assertStated(count,'accounting '+name,unit=name)
        self.assertStated(report['case_count'],'accounting live scenarios',unit='accounting')
