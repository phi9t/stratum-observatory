"""Portable config drift checks, not acceptance of publisher source bytes.

The live model table is an independent local consumer of the published checkpoint.
Values absent from that table require publication anchors, not copied test literals.
These checks never load the ignored source cache or allocate checkpoint weights.
"""
from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

from examples.deepseek_v41.config import V41_CONFIG


ROOT = Path(__file__).resolve().parents[1]
MODEL_FIELDS = {
    'hidden_dim': 'd',
    'streams': 'streams',
    'num_layers': 'layers',
    'encoder_layers': 'encoderLayers',
    'vocab_size': 'vocab',
    'sinkhorn_iterations': 'sinkhornIters',
    'expert_hidden_dim': 'expertFF',
    'num_experts': 'experts',
    'experts_per_token': 'topK',
    'route_scale': 'routeScale',
    'activation_limit': 'swigluLimit',
    'attention_heads': 'heads',
    'head_dim': 'headDim',
    'window_size': 'window',
    'encoder_compression_ratio': 'compressEncoder',
    'index_heads': 'indexerHeads',
    'index_head_dim': 'indexer',
    'index_topk': 'indexerBudget',
    'candidate_blocks': 'candidateBlocks',
    'candidate_block_size': 'candidateBlockSize',
    'engram_row_dim': 'engramHeadDim',
    'engram_heads': 'engramHeads',
    'engram_max_ngram': 'engramNgram',
    'dspark_block_size': 'dsparkBlock',
}
# Explicitly bounded coverage: do not invent a second numeric oracle for settings
# that the local model table does not expose. Publication owns their source checks.
SOURCE_ONLY_FIELDS = {
    'norm_epsilon', 'mixing_epsilon', 'router_temperature',
    'engram_score_floor', 'markov_rank',
}


class V41ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.check_output([
            'node', '--input-type=module', '-e',
            "import {MODELS,SOURCES} from './src/models.mjs';"
            "console.log(JSON.stringify({model:MODELS.dsv41,"
            "sources:Object.fromEntries(['dsv41-config','dsv41-code'].map(id=>[id,SOURCES[id]]))}));",
        ], cwd=ROOT, text=True)
        cls.live = json.loads(result)

    def test_released_dimensions_match_live_checkpoint(self):
        model = self.live['model']
        for field, model_field in MODEL_FIELDS.items():
            with self.subTest(field=field, model_field=model_field):
                self.assertEqual(getattr(V41_CONFIG, field), model[model_field])
        self.assertEqual(V41_CONFIG.engram_rows, model['engramTables'][0],
                         'the default excerpt uses the first Engram insertion table')
        self.assertEqual(V41_CONFIG.engram_hash_columns,
                         (model['engramNgram'] - 1) * model['engramHeads'])

    def test_every_config_field_has_an_explicit_check_scope(self):
        self.assertEqual({field.name for field in fields(V41_CONFIG)},
                         MODEL_FIELDS.keys() | SOURCE_ONLY_FIELDS | {'engram_rows'},
                         'map new fields to the live model or identify their source-only scope')

    def test_model_source_identity_matches_pinned_catalog(self):
        catalog = json.loads((ROOT / 'verification/manifests/sources.json').read_text())
        entries = {source['id']: source for source in catalog['sources']}
        for identity, source in self.live['sources'].items():
            with self.subTest(source=identity):
                pinned = entries[identity]
                self.assertEqual(pinned['repo'], self.live['model']['repo'])
                self.assertEqual(source['revision'], pinned['revision'])
                self.assertIn('/' + pinned['revision'] + '/', source['url'])
                self.assertTrue(source['url'].endswith('/' + pinned['path']))
                self.assertIn(identity, self.live['model']['sources'])

    def test_config_is_frozen_and_fixture_replacement_is_independent(self):
        original_width = V41_CONFIG.hidden_dim
        with self.assertRaises(FrozenInstanceError):
            V41_CONFIG.hidden_dim = original_width + 1
        fixture = replace(V41_CONFIG, hidden_dim=original_width + 1)
        self.assertEqual(V41_CONFIG.hidden_dim, original_width)
        self.assertNotEqual(fixture.hidden_dim, V41_CONFIG.hidden_dim)

    def test_config_import_needs_no_tensor_runtime_or_weights(self):
        # A fresh interpreter makes this meaningful even when another test has
        # already imported torch. Reject the allocation runtime before import.
        script = '''
import dataclasses
import importlib.abc
import sys

class RejectTensorRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise AssertionError('config import requested a tensor runtime: ' + fullname)

sys.meta_path.insert(0, RejectTensorRuntime())
from examples.deepseek_v41.config import V41_CONFIG
assert all(type(getattr(V41_CONFIG, field.name)) in (int, float)
           for field in dataclasses.fields(V41_CONFIG))
assert not any(name.split('.')[0] in {'torch', 'numpy'} for name in sys.modules)
'''
        subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True,
                       capture_output=True, text=True)
