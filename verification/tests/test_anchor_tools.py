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
