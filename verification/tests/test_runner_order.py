"""Characterization of the runner's evaluation order.

These tests describe what the runner does today. Task 6 changes role resolution; these must
keep passing except where a test is deliberately and visibly updated with its reason.
"""
import base64
import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from verification import runner
from verification.evidence import canonical, ed25519_openssl


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'verification/trust').mkdir(parents=True)
        (self.root/'verification/trust/reviewers.json').write_text('{"reviewers": {}}')

    def source(self, sid, role, repo, text):
        # Deliberately NOT `fixture: True`. `runner.audit()` calls `check_source`/`check_anchor`
        # with their default `allow_fixtures=False` and never threads an override through --
        # there is no way to reach the anchor-resolution path via the public `audit()` entry
        # point with a fixture-flagged source. A fixture-flagged source here would fail with
        # SOURCE_FIXTURE and short-circuit every evidence entry into CLAIM_SOURCE_BLOCKED before
        # the anchor is ever inspected, which hides the exact behaviour these tests pin. These
        # sources are still synthetic (temp-dir bytes, not real upstream pins); they simply
        # satisfy check_source's structural checks (authorised repo/role, valid pinned URL,
        # matching sha256) well enough to reach CLAIM_ROLE/MODEL_DUAL_CODE/CLAIM_ANCHOR.
        raw = text.encode()
        (self.root/f'{sid}.txt').write_bytes(raw)
        return {'id': sid, 'role': role, 'repo': repo, 'kind': 'git', 'provider': 'github',
                'path': f'{sid}.txt', 'revision': 'a'*40,
                'url': f'https://raw.githubusercontent.com/{repo}/{"a"*40}/{sid}.txt',
                'cache': f'{sid}.txt', 'sha256': hashlib.sha256(raw).hexdigest(),
                'license': 'test fixture'}

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

    def audit(self, target, *, extra_fields=None):
        catalog = {'schema': 'stratum-sources/v1', 'sources': [
            self.source('pub', 'publisher_code', 'deepseek-ai/DeepSeek-V2', 'class A: pass'),
            self.source('fw', 'framework_code', 'sgl-project/sglang', 'class B: pass'),
            self.source('fw2', 'framework_code', 'sgl-project/sglang', 'class C: pass'),
        ]}
        inventory = {'fields': {'MODELS.x.d': 5120, 'scene.x.a': 'digest', **(extra_fields or {})},
                     'functions': {}}
        return runner.audit(self.root, target, inventory, catalog, reviews={})

    def codes(self, report, claim):
        return {f['code'] for f in report['findings'] if f['subject'] == claim}

    def test_a_dependent_cannot_verify_while_its_input_is_unverified(self):
        """Replaced test_copied_citations_satisfy_the_role_check_today in Task 6.

        That test pinned the copied-citation behaviour and an inverted version of it was
        specified, then withdrawn (ruling S): whether a copied citation satisfies the role check
        is not what protects the gate. This is. Copied citations still satisfy roles, and the
        dependency fixpoint still refuses to let the dependent verify over an unverified input,
        which is the property that carries the load.
        """
        copied = [{'source': 'pub', 'anchor': None, 'relation': 'implements'},
                  {'source': 'fw', 'anchor': None, 'relation': 'implements'}]
        report = self.audit(self.minimal_target(dependent_evidence=copied))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'MODELS.x.d'))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.a'))

    def test_a_derived_claim_with_no_citations_of_its_own_inherits_its_input_roles(self):
        """Changed by ruling S in Task 6, and this is the change that makes Task 7 possible.

        It previously asserted CLAIM_ROLE and MODEL_DUAL_CODE on a derived claim carrying no
        evidence at all. Under that behaviour, removing the ~5,000 citations a scene claim only
        ever copied from its inputs would have manufactured a role gap on every one of them and
        reported a missing publisher implementation where the citation is simply one level up.
        Roles now flow along the declared dependency edge. What still holds is that the claim
        cannot verify -- the fixpoint sees the unverified input and says so under its own cause.
        """
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))
        self.assertNotIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.a'))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.a'))

    def test_dependency_blocked_only_fires_on_an_otherwise_passing_claim(self):
        """DEPENDENCY_BLOCKED is a demotion of PASS, not an independent check."""
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        blocked = [f for f in report['findings'] if f['code'] == 'DEPENDENCY_BLOCKED']
        self.assertEqual(blocked, [], 'a claim already blocked is not demoted again')

    def test_unresolvable_anchors_block_the_input_claim(self):
        report = self.audit(self.minimal_target(dependent_evidence=[]))
        self.assertIn('CLAIM_ANCHOR', self.codes(report, 'MODELS.x.d'))


class DependencyRoleTests(OrderTests):
    """Roles may flow from a verified dependency, and from nothing else."""

    def test_inheritance_is_one_level_and_reads_what_a_dependency_cites(self):
        """Specified in the Task 6 brief as `test_unverified_dependency_cannot_lend_a_role` and
        withdrawn by ruling S -- an unverified input is refused by the fixpoint, not by the role
        check. Repurposed to pin what `roles_from_dependencies` actually reads: a dependency's
        OWN citations, never the set it inherited in turn. So a grandchild derivation whose only
        input is itself a citation-free derivation does report CLAIM_ROLE.

        This is sufficient for the real target, where all 2,520 derived scene claims depend on a
        `documented` claim as well as a derived one, and those documented inputs alone supply
        both code roles for all 1,737 model scene claims. If that ever stops being true, this
        test is the one that will say so.
        """
        target = self.minimal_target(dependent_evidence=[])
        target['claims'].append(
            {'id': 'scene.x.g', 'domain': 'model', 'classification': 'derived',
             'consumer': 'scene.x.g', 'expected': 'grandchild', 'statement': 'derived on derived',
             'dependencies': ['scene.x.a'], 'evidence': []})
        report = self.audit(target, extra_fields={'scene.x.g': 'grandchild'})
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.a'))
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.g'))

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


class VerifiedDependencyTests(OrderTests):
    """A fixture in which claims genuinely reach PASS, so demotion is observable at all.

    Every other fixture in this module carries `anchor: None` everywhere, so no claim ever
    reaches PASS -- which makes any assertion about a demotion vacuously true. This one
    resolves real anchors and carries real Ed25519-signed reviews, so claims actually reach
    PASS. PASS is the only state DEPENDENCY_BLOCKED can fire from, and the only state from
    which a role failure discovered after the dependency fixpoint can demote anything.
    """

    def setUp(self):
        super().setUp()
        openssl = ed25519_openssl()
        if not openssl:
            self.skipTest('No OpenSSL with Ed25519; install OpenSSL 3.x or set STRATUM_OPENSSL.')
        self.openssl = openssl
        subprocess.run([openssl, 'genpkey', '-algorithm', 'Ed25519', '-out', str(self.root/'private.pem')],
                       check=True, capture_output=True)
        subprocess.run([openssl, 'pkey', '-in', str(self.root/'private.pem'), '-pubout',
                        '-out', str(self.root/'reviewer.pem')], check=True, capture_output=True)
        (self.root/'verification/trust/reviewers.json').write_text(
            json.dumps({'reviewers': {'independent': {'public_key': 'reviewer.pem'}}}))

    def document(self, sid, text):
        """A vendor document source: no repo authority, a trusted host, a pinned edition."""
        raw = text.encode()
        (self.root/f'{sid}.txt').write_bytes(raw)
        return {'id': sid, 'role': 'vendor_document', 'kind': 'document', 'version': '2025.1',
                'path': f'{sid}.txt', 'url': f'https://docs.nvidia.com/{sid}.txt',
                'cache': f'{sid}.txt', 'sha256': hashlib.sha256(raw).hexdigest(),
                'license': 'test fixture'}

    @staticmethod
    def symbol_anchor(symbol, selected):
        return {'kind': 'python_symbol', 'symbol': symbol, 'contains': [f'class {symbol}'],
                'sha256': hashlib.sha256(selected.encode()).hexdigest()}

    @staticmethod
    def quote_anchor(quote, selected):
        return {'kind': 'text_quote', 'quote': quote, 'contains': ['B200'],
                'sha256': hashlib.sha256(selected.encode()).hexdigest()}

    def catalog(self):
        return {'schema': 'stratum-sources/v1', 'sources': [
            self.source('pub', 'publisher_code', 'deepseek-ai/DeepSeek-V2', 'class A: pass'),
            self.source('fw', 'framework_code', 'sgl-project/sglang', 'class B: pass'),
            self.document('doc', 'The B200 SXM module provides 180 GB of HBM3e.\n'),
        ]}

    def code_evidence(self):
        return [{'source': 'pub', 'anchor': self.symbol_anchor('A', 'class A: pass'), 'relation': 'implements'},
                {'source': 'fw', 'anchor': self.symbol_anchor('B', 'class B: pass'), 'relation': 'implements'}]

    def target(self):
        """GPU.x.v and MODELS.x.d verify. scene.x.b inherits only a vendor_document role from
        GPU.x.v and so must be demoted by the role pass. scene.x.c inherits both code roles from
        the verified MODELS.x.d, so the role pass leaves it PASS -- and it also depends on
        scene.x.b, which the same pass demotes, so only re-running the fixpoint afterwards can
        catch it. scene.x.i is the positive case: roles inherited from a verified input."""
        doc = [{'source': 'doc', 'anchor': self.quote_anchor('B200 SXM module',
                                                             'The B200 SXM module provides 180 GB of HBM3e.'),
                'relation': 'documents'}]
        return {'schema': 'stratum-target/v1', 'id': 'test', 'description': 'demotion fixture',
                'claims': [
                    {'id': 'GPU.x.v', 'domain': 'gpu', 'classification': 'documented',
                     'consumer': 'GPU.x.v', 'expected': 180, 'statement': 'gpu input',
                     'dependencies': [], 'evidence': doc},
                    {'id': 'MODELS.x.d', 'domain': 'model', 'classification': 'documented',
                     'consumer': 'MODELS.x.d', 'expected': 5120, 'statement': 'model input',
                     'dependencies': [], 'evidence': self.code_evidence()},
                    {'id': 'scene.x.c', 'domain': 'model', 'classification': 'derived',
                     'consumer': 'scene.x.c', 'expected': 'digest-c', 'statement': 'derived on b',
                     'dependencies': ['MODELS.x.d', 'scene.x.b'], 'evidence': []},
                    {'id': 'scene.x.b', 'domain': 'model', 'classification': 'derived',
                     'consumer': 'scene.x.b', 'expected': 'digest-b', 'statement': 'derived on gpu',
                     'dependencies': ['GPU.x.v'], 'evidence': []},
                    {'id': 'scene.x.i', 'domain': 'model', 'classification': 'derived',
                     'consumer': 'scene.x.i', 'expected': 'digest-i', 'statement': 'derived on model',
                     'dependencies': ['MODELS.x.d'], 'evidence': []},
                    {'id': 'MODELS.x.u', 'domain': 'model', 'classification': 'documented',
                     'consumer': 'MODELS.x.u', 'expected': 4096, 'statement': 'unanchored input',
                     'dependencies': [], 'evidence': [
                         {'source': 'pub', 'anchor': None, 'relation': 'implements'},
                         {'source': 'fw', 'anchor': None, 'relation': 'implements'}]},
                    {'id': 'scene.x.u', 'domain': 'model', 'classification': 'derived',
                     'consumer': 'scene.x.u', 'expected': 'digest-u', 'statement': 'derived on unanchored',
                     'dependencies': ['MODELS.x.u'], 'evidence': []},
                ],
                'contracts': [], 'runtime_obligations': []}

    def sign(self, target, sources):
        reviews = {}
        for c in target['claims']:
            body = {'author': 'implementer', 'reviewer': 'independent',
                    'subject_sha256': runner.source_subject(c, sources), 'verdict': 'accept',
                    'expires_at': (datetime.now(timezone.utc)+timedelta(days=7)).isoformat()}
            (self.root/'body').write_bytes(canonical(body))
            subprocess.run([self.openssl, 'pkeyutl', '-sign', '-inkey', str(self.root/'private.pem'),
                            '-rawin', '-in', str(self.root/'body'), '-out', str(self.root/'signature')],
                           check=True, capture_output=True)
            reviews[c['id']] = {'body': body,
                                'signature': base64.b64encode((self.root/'signature').read_bytes()).decode()}
        return reviews

    def run_audit(self, target=None):
        target = target or self.target()
        catalog = self.catalog()
        sources = {s['id']: s for s in catalog['sources']}
        inventory = {'fields': {c['consumer']: c['expected'] for c in target['claims']}, 'functions': {}}
        return runner.audit(self.root, target, inventory, catalog, reviews=self.sign(target, sources))

    def test_the_fixture_is_not_vacuous_and_claims_do_reach_pass(self):
        """Without this, every demotion assertion below would be trivially true."""
        report = self.run_audit()
        self.assertIn('CLAIM_VERIFIED', self.codes(report, 'GPU.x.v'))
        self.assertIn('CLAIM_VERIFIED', self.codes(report, 'MODELS.x.d'))

    def test_a_derived_claim_inherits_roles_from_a_verified_dependency(self):
        report = self.run_audit()
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.i'))
        self.assertNotIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.i'))
        self.assertIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.i'))

    def test_the_role_pass_demotes_a_claim_that_had_already_reached_pass(self):
        """scene.x.b has no evidence of its own and its only input is a vendor document, so the
        code roles it needs can never be inherited. It reaches PASS in the first pass and must
        be taken back."""
        report = self.run_audit()
        self.assertIn('CLAIM_ROLE', self.codes(report, 'scene.x.b'))
        self.assertIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.b'))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.b'))

    def test_dependency_blocked_fires_on_a_genuinely_passing_dependent(self):
        """scene.x.c inherits both code roles from the verified MODELS.x.d, so the role pass
        leaves it PASS even though its other input, scene.x.b, is demoted by that same pass.
        Only re-running the dependency fixpoint afterwards can catch it."""
        report = self.run_audit()
        blocked = {f['subject'] for f in report['findings'] if f['code'] == 'DEPENDENCY_BLOCKED'}
        self.assertIn('scene.x.c', blocked, 'a PASS claim over a role-blocked input was not demoted')
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.c'))
        self.assertEqual(report['coverage']['verified_scientific_claims'], 3,
                         'exactly GPU.x.v, MODELS.x.d and scene.x.i may be counted as verified')

    def test_an_unverified_input_lends_its_roles_but_never_its_verification(self):
        """The property ruling S makes load-bearing, in one test.

        MODELS.x.u cites both code roles but neither citation is anchored, so it is BLOCKED.
        scene.x.u derives from it and cites nothing of its own. Its roles come through the
        dependency edge, so the gate does NOT accuse it of missing a publisher or framework
        implementation -- the citations exist, one level up, and Task 7 removing the copy from
        scene.x.u changes nothing here. What it must never do is verify: the fixpoint demotes it
        because its input did not verify, and DEPENDENCY_BLOCKED names that actual cause.
        """
        report = self.run_audit()
        self.assertIn('CLAIM_ANCHOR', self.codes(report, 'MODELS.x.u'))
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'MODELS.x.u'),
                         'the lending input must really be unverified, or this proves nothing')
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.u'))
        self.assertNotIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.u'))
        blocked = {f['subject'] for f in report['findings'] if f['code'] == 'DEPENDENCY_BLOCKED'}
        self.assertIn('scene.x.u', blocked)
        self.assertNotIn('CLAIM_VERIFIED', self.codes(report, 'scene.x.u'))

    def test_a_verified_dependency_cannot_lend_two_repos_of_one_org(self):
        """The config-only corroboration path needs two organizations, and inheritance from a
        claim that genuinely PASSED must not launder that.

        TRAIN.x.t is a training-domain claim, so it requires no roles and is not dual-code
        checked: it reaches PASS while citing a config-only publisher and two framework sources
        from the SAME organization. A model claim derived from it therefore inherits a role set
        that looks complete -- publisher_config plus framework_code -- from a verified input.
        `dual_code_roles` must still refuse it, because one organization read the architecture
        once, not twice.
        """
        target = self.target()
        target['claims'].append(
            {'id': 'TRAIN.x.t', 'domain': 'training', 'classification': 'documented',
             'consumer': 'TRAIN.x.t', 'expected': 7, 'statement': 'training input',
             'dependencies': [], 'evidence': [
                 {'source': 'cfg', 'anchor': {'kind': 'json_pointer', 'pointer': '/a',
                                              'sha256': hashlib.sha256(b'1').hexdigest()},
                  'relation': 'consumes'},
                 {'source': 'fw', 'anchor': self.symbol_anchor('B', 'class B: pass'),
                  'relation': 'implements'},
                 {'source': 'fw2', 'anchor': self.symbol_anchor('C', 'class C: pass'),
                  'relation': 'implements'}]})
        target['claims'][4]['dependencies'] = ['TRAIN.x.t']
        catalog = self.catalog()
        catalog['sources'] += [self.source('cfg', 'publisher_config', 'Qwen/Qwen3-8B', '{"a": 1}'),
                               self.source('fw2', 'framework_code', 'sgl-project/sglang', 'class C: pass')]
        sources = {s['id']: s for s in catalog['sources']}
        inventory = {'fields': {c['consumer']: c['expected'] for c in target['claims']}, 'functions': {}}
        report = runner.audit(self.root, target, inventory, catalog, reviews=self.sign(target, sources))
        self.assertIn('CLAIM_VERIFIED', self.codes(report, 'TRAIN.x.t'),
                      'the laundering input must really have verified, or this proves nothing')
        self.assertNotIn('CLAIM_ROLE', self.codes(report, 'scene.x.i'),
                         'the config-only role pair was inherited, so the role check is satisfied')
        self.assertIn('MODEL_DUAL_CODE', self.codes(report, 'scene.x.i'))
