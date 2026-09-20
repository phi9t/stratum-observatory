"""The delivered trace must be reproducible by current CPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_worked', ROOT / 'tools/export-worked.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class WorkedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fresh = exporter.compute()
        cls.saved = json.loads(exporter.OUTPUT.read_text())

    def test_saved_intermediates_and_outputs_match_current_cpu(self):
        exporter.check(self.saved, self.fresh)

    def test_changed_intermediate_cannot_pass_even_with_updated_digest(self):
        changed = copy.deepcopy(self.saved)
        changed['examples']['query']['steps'][0]['values'][0][0] += 1
        changed['payload'] = json.dumps(changed['examples'], separators=(',', ':'))
        changed['payload_sha256'] = exporter.hashlib.sha256(changed['payload'].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'CPU result differs'):
            exporter.check(changed, self.fresh)

    def test_changed_source_fingerprint_fails_before_numbers(self):
        changed = copy.deepcopy(self.saved)
        changed['source_sha256']['examples/deepseek_v41/config.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Stale CPU'):
            exporter.check(changed, self.fresh)
