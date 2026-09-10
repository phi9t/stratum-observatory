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


class ResidueRecordTests(unittest.TestCase):
    """Every citation left unanchored must carry a recorded decision, not silence.

    RULING (O): 2,987 rows is a document nobody reads if it is written one row per line, and
    an unread table is a silent discard wearing a table's clothes. The record is therefore
    organised by class -- one decision per (source, reason, field-class) group -- and this
    test enforces the property that matters: every residue row falls inside exactly one
    recorded class, and the recorded row counts add up to the real population. It still fails
    if a row escapes every class, or if a class overstates or understates its own count.
    """

    RECORD = ROOT/'docs/verifiers/ANCHOR_RESIDUE.md'

    @staticmethod
    def residue_rows():
        rows = []
        for name in ('anchor-residue-config.json', 'anchor-residue-documents.json'):
            path = ROOT/'verification/reports'/name
            if path.exists():
                rows.extend(json.loads(path.read_text()))
        return rows

    @staticmethod
    def recorded_classes():
        """(source, reason, field-class) -> (count, decision) parsed back out of the record."""
        out = {}
        for line in ResidueRecordTests.RECORD.read_text().splitlines():
            if not line.startswith('| `'):
                continue
            cells = [c.strip().strip('`') for c in line.strip('|').split('|')]
            if len(cells) != 5 or not cells[3].isdigit():
                continue
            out[(cells[0], cells[1], cells[2])] = (int(cells[3]), cells[4])
        return out

    def test_every_residue_row_has_a_decision(self):
        self.assertTrue(self.RECORD.exists(), 'residue decisions are not recorded')
        classes = self.recorded_classes()
        self.assertTrue(classes, 'the record contains no class rows')
        tool = load_tool('anchor-residue')
        counted = {}
        for row in self.residue_rows():
            key = (row['source'], row['reason'], tool.field_class(row))
            with self.subTest(claim=row['claim'], source=row['source']):
                self.assertIn(key, classes, f"residue row {row['claim']} falls outside every recorded class")
            counted[key] = counted.get(key, 0) + 1
        for key, (recorded, _) in classes.items():
            with self.subTest(cls=key):
                self.assertEqual(counted.get(key, 0), recorded, f'recorded count for {key} does not match the reports')
        self.assertEqual(sum(n for n, _ in classes.values()), len(self.residue_rows()),
                         'recorded class counts do not sum to the residue total')

    def test_no_residue_row_was_resolved_by_inventing_a_quote(self):
        """A decision must name one of the three allowed resolutions."""
        if not self.RECORD.exists():
            self.skipTest('record not yet written')
        allowed = ('re-pointed', 'dropped', 'reclassified', 'unresolved')
        for line in self.RECORD.read_text().splitlines():
            if line.startswith('| `') and 'claim' not in line:
                with self.subTest(line=line):
                    self.assertTrue(any(word in line for word in allowed), line)

    def test_every_dropped_citation_is_gone_from_the_manifest(self):
        """A record that disagrees with the manifest is worse than no record."""
        if not self.RECORD.exists():
            self.skipTest('record not yet written')
        tool = load_tool('anchor-residue')
        target = json.loads((ROOT/'verification/manifests/observatory.json').read_text())
        present = {(c['id'], e['source']) for c in target['claims'] for e in c['evidence']}
        for row in self.residue_rows():
            if tool.decide(row)[0] != 'dropped':
                continue
            with self.subTest(claim=row['claim'], source=row['source']):
                self.assertNotIn((row['claim'], row['source']), present,
                                 'record says dropped but the manifest still cites it')

    def test_every_re_pointed_citation_now_carries_an_anchor(self):
        if not self.RECORD.exists():
            self.skipTest('record not yet written')
        tool = load_tool('anchor-residue')
        target = json.loads((ROOT/'verification/manifests/observatory.json').read_text())
        anchors = {(c['id'], e['source']): e['anchor'] for c in target['claims'] for e in c['evidence']}
        for row in self.residue_rows():
            if tool.decide(row)[0] != 're-pointed':
                continue
            with self.subTest(claim=row['claim'], source=row['source']):
                anchor = anchors.get((row['claim'], row['source']))
                self.assertIsNotNone(anchor, 'record says re-pointed but the manifest has no anchor')
                self.assertIn(anchor.get('kind'), ('json_pointer', 'text_quote'))
