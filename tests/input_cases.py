"""Offline integrity and negative contracts for publisher-derived input fixtures."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('input_cases_export', ROOT / 'tools/export-input-cases.py')


class InputCasesTests(unittest.TestCase):
    def test_fixture_contract_and_tampering(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        fixture = json.loads((ROOT / 'examples/deepseek_v41/input-cases.json').read_text())
        module.validate_fixture(fixture)
        for mutation in ('merge', 'serialized', 'type', 'patch'):
            damaged = copy.deepcopy(fixture)
            case = damaged['cases'][1]
            if mutation == 'merge':
                case['image']['merged'][0]['token_index'] += 1
            elif mutation == 'serialized':
                case['serialized_token_ids'][0] += 1
            elif mutation == 'type':
                case['token_types'][case['image']['span'][0]] = 1
            else:
                case['image']['patches'][0]['merged_index'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.validate_fixture(damaged)

    def test_exact_sources_and_generator_current(self):
        fixture = json.loads((ROOT / 'examples/deepseek_v41/input-cases.json').read_text())
        sources = {s['id']: s for s in json.loads((ROOT / 'verification/manifests/sources.json').read_text())['sources']}
        for source in fixture['sources']:
            for key, value in source.items():
                self.assertEqual(value, sources[source['id']][key])
        for path, digest in fixture['recipe_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_text_image_distinction(self):
        fixture = json.loads((ROOT / 'examples/deepseek_v41/input-cases.json').read_text())
        text, chart = fixture['cases']
        self.assertEqual([m['role'] for m in text['messages']], ['system', 'user', 'assistant', 'user'])
        self.assertIsNone(text['image'])
        self.assertTrue(all(t == -1 for t in text['token_types']))
        self.assertIn('Which bar is tallest?', chart['serialized_prompt'])
        start, end = chart['image']['span']
        self.assertEqual(chart['token_types'][start], 0)
        self.assertEqual(chart['token_types'][end - 1], 3)
        self.assertEqual(len(set(chart['token_ids'][start:end])), 1)
        self.assertGreater(chart['image']['preprocessing_args']['vision_min_pixels'], 0)
        self.assertEqual(chart['image']['publisher_default_grid'][2], chart['image']['patch_grid'][0] * chart['image']['patch_size'])

    def test_cached_source_hashes_when_available(self):
        fixture = json.loads((ROOT / 'examples/deepseek_v41/input-cases.json').read_text())
        sources = {s['id']: s for s in json.loads((ROOT / 'verification/manifests/sources.json').read_text())['sources']}
        for source in fixture['sources']:
            path = ROOT / sources[source['id']]['cache']
            if not path.exists():
                self.skipTest('Pinned upstream cache unavailable; live reproduction NOT_RUN')
            self.assertEqual(source['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
