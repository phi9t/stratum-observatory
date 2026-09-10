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
