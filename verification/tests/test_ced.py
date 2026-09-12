import subprocess
import unittest
import unittest.mock
from pathlib import Path

from verification import ced
from verification.ced import (FIELD_MAP, FP4_FLOOR, FP8_CEILING, NOT_IN_CONFIG, SCENE_JUDGES,
                              SCHEMATICS_JUDGES, band_id,
                              derived_main_kv_bytes_per_token, indexer_channels_per_token,
                              inventory, judge_active_ratio, judge_architecture, judge_architecture_tag,
                              judge_ced_topology,
                              judge_claim_model_coverage, judge_contract_model_coverage,
                              judge_kv_budget, judge_mhc,
                              judge_coverage, judge_mhc_streams, judge_parameter_bands,
                              judge_publisher_agreement, layer_bands,
                              derived_compressor_gate_parameters, gated_compressor_layers,
                              modes_from_config, modes_from_report_structure)
from verification.evidence import REPOS, loads
from verification import schematics
from verification.schematics import export_scenes

DSV41 = 'deepseek-ai/DeepSeek-V4.1-Flash'
ROOT = Path(__file__).resolve().parents[2]

#: A minimal well-formed model contract and its obligation, shaped like the shipped ones.
MODEL_CONTRACT = {'id': 'model.ds2', 'domain': 'model',
                  'bindings': {'layers': 'MODELS.ds2.layers'}, 'claims': ['MODELS.ds2.d']}
MODEL_OBLIGATION = {'id': 'runtime.model.ds2', 'domain': 'model', 'contract': 'model.ds2'}

class PublisherAuthorizationTests(unittest.TestCase):
    def test_the_publisher_is_authorized_for_the_roles_its_bytes_are_cited_under(self):
        """Six dsv41 sources are cited: one config and one card, four modeling files.
        An unlisted repo is SOURCE_AUTHORITY FAIL, so both roles must be authorized."""
        self.assertIn(DSV41, REPOS['publisher_config'])
        self.assertIn(DSV41, REPOS['publisher_code'])

    def test_no_framework_is_authorized_to_stand_in_for_this_model(self):
        """No framework implements deepseek_v41. deepseek_v4 is a different model and
        must never be cited as an independent reading of this one."""
        self.assertNotIn(DSV41, REPOS['framework_code'])


def dsv41_kinds():
    """The 40 layer kinds `core.layerKind` reports, written out rather than recomputed.

    Transcribed from the checkpoint's own pattern: two window-only layers, then a compressed-KV
    source every six layers to the encoder boundary, then a Full layer and four re-index layers
    each followed by three reuse layers.
    """
    encoder = ['swa', 'swa'] + ['csa2-full'] + ['csa2-reuse'] * 5 + ['csa2-full'] + ['csa2-reuse'] * 5 \
        + ['csa2-full'] + ['csa2-reuse'] * 5
    decoder = ['csa2-full'] + ['csa2-reuse'] * 3 + (['csa2-reindex'] + ['csa2-reuse'] * 3) * 4
    return encoder + decoder


def dsv41_fixture():
    """The `expect` record `tools/export-scenes.mjs` emits for a dsv41 scene."""
    return {'dsv41': {'model': 'dsv41', 'residual': 'mhc', 'streams': 4, 'layers': 40,
                      'encoderLayers': 20, 'layerKinds': dsv41_kinds(),
                      'engramLayers': [1, 14], 'dsparkTargets': [37, 38, 39],
                      'candidateSource': 20, 'indexSourceLayers': [2, 8, 14, 20, 24, 28, 32, 36],
                      'compressEncoder': 2, 'compressDecoder': 1}}


def codes(findings, status=None):
    return {f.code for f in findings if status is None or f.status == status}


class MHCResidualTests(unittest.TestCase):
    """A four-stream residual drawn as a '+' passes every geometric check there is."""

    def clean(self):
        return {'nodes': [{'id': 'merge-attn', 'kind': 'pill', 'name': 'Mixed stream write'}],
                'edges': [], 'groups': [],
                'labels': [{'text': 'residual stream · 4 streams'}]}

    def test_a_bare_sum_on_an_mhc_residual_path_is_caught(self):
        scene = {'nodes': [{'id': 'merge-attn', 'kind': 'sum', 'name': '+'}],
                 'edges': [], 'groups': [], 'labels': [{'text': 'residual stream · x'}]}
        f = judge_mhc_streams(scene, dsv41_fixture()['dsv41'])
        self.assertIn('MHC_STREAM_BALANCE', codes(f, 'FAIL'))

    def test_a_rail_that_understates_the_stream_count_is_caught(self):
        scene = self.clean()
        scene['labels'] = [{'text': 'residual stream · x'}]
        self.assertIn('MHC_STREAM_BALANCE', codes(judge_mhc_streams(scene, dsv41_fixture()['dsv41']), 'FAIL'))

    def test_a_correctly_drawn_mhc_residual_passes(self):
        self.assertEqual([f for f in judge_mhc_streams(self.clean(), dsv41_fixture()['dsv41']) if f.status != 'PASS'], [])

    def test_a_lens_without_a_residual_rail_is_skipped_not_failed(self):
        scene = {'nodes': [], 'edges': [], 'groups': [], 'labels': [{'text': 'COMPARE'}]}
        self.assertEqual(codes(judge_mhc_streams(scene, dsv41_fixture()['dsv41']), 'PASS'), {'MHC_SKIP'})

    def test_the_declared_stream_count_comes_from_the_model_not_the_drawing(self):
        """A model with no declared stream count cannot be satisfied by any label."""
        f = judge_mhc_streams(self.clean(), {'model': 'dsv41', 'residual': 'mhc', 'streams': 0})
        self.assertIn('MHC_STREAM_BALANCE', codes(f, 'FAIL'))

    def test_the_shipped_dsv41_block_scenes_satisfy_the_invariant(self):
        """Asserted over the real matrix, not only over a fixture."""
        data = export_scenes(Path(__file__).resolve().parents[2])
        judged = 0
        for entry in data['scenes']:
            if entry['expect'].get('residual') != 'mhc':
                continue
            bad = [f for f in judge_mhc_streams(entry['scene'], entry['expect']) if f.status != 'PASS']
            self.assertEqual(bad, [], f"{entry['key']}: {[f.message for f in bad]}")
            judged += 1
        self.assertGreater(judged, 0, 'no mhc scene was judged; the hook is not wired')


def clean_ced_scene(m=None):
    """The graph `buildCED` must produce, built from the declared model rather than copied.

    Every relation the judge asserts is derived here from the same `expect` record, so a test
    that mutates one relation is mutating the drawing and not the expectation.
    """
    m = m or dsv41_fixture()['dsv41']
    bands = {'enc': layer_bands(m['layerKinds'], 1, m['encoderLayers']),
             'dec': layer_bands(m['layerKinds'], m['encoderLayers'] + 1, m['layers'])}
    engram = [l + 1 for l in m['engramLayers']]
    dspark = [l + 1 for l in m['dsparkTargets']]
    nodes = [{'id': 'enc-out'}, {'id': 'candidate-pool'},
             {'id': 'engram', 'name': 'Engram n-gram memory', 'sub': f"→ layers {', '.join(map(str, engram))}"},
             {'id': 'dspark', 'name': 'DSpark draft head', 'sub': f"reads L{', '.join(map(str, dspark))}"}]
    edges = []

    def wire(source, target):
        edges.append({'id': f'w{len(edges)}', 'source': source, 'target': target, 'points': [[0, 0], [1, 0]]})

    def holding(layers):
        out = []
        for layer in layers:
            for side, declared in bands.items():
                for i, band in enumerate(declared):
                    if band[1] <= layer <= band[2]:
                        out.append(band_id(side, i, band[1], band[2]))
        return out

    for side, declared in bands.items():
        for i, band in enumerate(declared):
            node = {'id': band_id(side, i, band[1], band[2])}
            if band[0] == 'csa2-full':
                # `buildCED` states the band's compression ratio in the subtitle. Derived here
                # from the declared ratios, so a test that edits the text is editing the drawing.
                ratio = m['compressEncoder'] if band[1] - 1 < m['encoderLayers'] else m['compressDecoder']
                node['sub'] = (f"L{band[1]} writes "
                               + ('1 latent / position' if ratio == 1 else f'÷{ratio} latents')
                               + f' · {band[2] - band[1]} layers reuse')
            nodes.append(node)
    wire('enc-out', 'dec-full')
    wire('dec-full', 'candidate-pool')
    for i, band in enumerate(bands['dec']):
        if band[0] == 'csa2-reindex':
            wire('candidate-pool', band_id('dec', i, band[1], band[2]))
    for target in holding(engram):
        wire('engram', target)
    for source in set(holding(dspark)):
        wire(source, 'dspark')
    return {'nodes': nodes, 'edges': edges,
            'groups': [{'id': 'encoder'}, {'id': 'decoder'}], 'labels': []}


class CEDTopologyTests(unittest.TestCase):
    """A collision-free two-column drawing can still assert the wrong architecture."""

    def setUp(self):
        self.m = dsv41_fixture()['dsv41']

    def judge(self, scene):
        return judge_ced_topology(scene, self.m)

    def test_the_graph_the_builder_owes_passes(self):
        self.assertEqual([f for f in self.judge(clean_ced_scene()) if f.status != 'PASS'], [])

    # A minimal two-node scene with ids the page never draws was removed here: the judge FAILed
    # it for malformed input, not for the back edge, so it passed whether the rule existed or not.
    # The test below perturbs exactly one relation in a valid drawing instead.
    def test_a_back_edge_is_caught_in_an_otherwise_correct_drawing(self):
        """CED is a one-way handoff. A back edge would make it a shared-KV Transformer."""
        scene = clean_ced_scene()
        scene['edges'].append({'id': 'back', 'source': 'dec-full', 'target': 'enc-band-15-20',
                               'points': [[0, 0], [1, 0]]})
        bad = [f for f in self.judge(scene) if f.status == 'FAIL']
        self.assertEqual([f.code for f in bad], ['CED_SCENE_TOPOLOGY'])
        self.assertIn('back into encoder', bad[0].message)

    def test_a_lens_without_the_two_halves_is_skipped_not_failed(self):
        scene = {'nodes': [], 'edges': [], 'groups': [{'id': 'block-region'}], 'labels': []}
        self.assertEqual(codes(self.judge(scene), 'PASS'), {'CED_SKIP'})

    def test_a_dropped_band_leaves_the_halves_short(self):
        scene = clean_ced_scene()
        scene['nodes'] = [n for n in scene['nodes'] if n['id'] != 'enc-band-9-14']
        messages = ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL')
        self.assertIn('does not draw', messages)
        self.assertIn('cover 14 layers against 20 declared', messages)

    def test_a_band_the_layer_pattern_does_not_produce_is_caught(self):
        scene = clean_ced_scene()
        scene['nodes'].append({'id': 'dec-band-41-44'})
        self.assertIn('the layer pattern does not produce',
                      ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_second_handoff_edge_is_caught(self):
        scene = clean_ced_scene()
        scene['edges'].append({'id': 'w99', 'source': 'enc-out', 'target': 'dec-full', 'points': [[0, 0], [1, 0]]})
        self.assertIn('2 CED handoff edges', ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_missing_handoff_edge_is_caught(self):
        scene = clean_ced_scene()
        scene['edges'] = [e for e in scene['edges'] if not (e['source'] == 'enc-out' and e['target'] == 'dec-full')]
        self.assertIn('0 CED handoff edges', ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_pool_that_skips_a_reindex_band_is_caught(self):
        scene = clean_ced_scene()
        scene['edges'] = [e for e in scene['edges']
                          if not (e['source'] == 'candidate-pool' and e['target'] == 'dec-band-33-36')]
        self.assertIn('decoder-band Reindex layers',
                      ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_pool_that_feeds_an_encoder_band_is_caught(self):
        """The indexer pool is decoder-only; an encoder Full layer ranks the whole visible range."""
        scene = clean_ced_scene()
        for edge in scene['edges']:
            if edge['source'] == 'candidate-pool' and edge['target'] == 'dec-band-25-28':
                edge['target'] = 'enc-band-3-8'
        self.assertIn('CED_SCENE_TOPOLOGY', codes(self.judge(scene), 'FAIL'))

    def test_an_engram_fanout_that_under_counts_its_sites_is_caught(self):
        scene = clean_ced_scene()
        scene['edges'] = [e for e in scene['edges'] if e['source'] != 'engram'][:]
        scene['edges'].append({'id': 'e1', 'source': 'engram', 'target': 'enc-band-1-2', 'points': [[0, 0], [1, 0]]})
        self.assertIn('declared injection sites',
                      ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_fanout_node_that_never_names_its_layers_is_caught(self):
        scene = clean_ced_scene()
        for node in scene['nodes']:
            if node['id'] == 'dspark':
                node['sub'] = 'reads the last decoder block'
        self.assertIn('never states target layer',
                      ' '.join(f.message for f in self.judge(scene) if f.status == 'FAIL'))

    def test_a_preset_with_no_declared_split_cannot_be_satisfied_by_any_drawing(self):
        m = dict(self.m, encoderLayers=0)
        self.assertIn('CED_SCENE_TOPOLOGY', codes(judge_ced_topology(clean_ced_scene(), m), 'FAIL'))

    def test_the_band_rule_folds_only_reuse_and_window_layers(self):
        bands = layer_bands(self.m['layerKinds'], 1, self.m['layers'])
        self.assertEqual(bands[0], ('swa', 1, 2))
        self.assertEqual(bands[1], ('csa2-full', 3, 8))
        self.assertEqual([b[2] - b[1] + 1 for b in bands], [2, 6, 6, 6, 4, 4, 4, 4, 4])
        self.assertEqual(sum(b[2] - b[1] + 1 for b in bands), self.m['layers'])


class ShippedCEDSceneTests(unittest.TestCase):
    """Asserted over the real exported matrix, not only over a fixture."""

    @classmethod
    def setUpClass(cls):
        cls.data = export_scenes(ROOT)
        cls.entries = [e for e in cls.data['scenes'] if e['state'].get('stage') == 'ced']

    def test_the_ced_lens_is_exported_at_all(self):
        self.assertEqual([e['key'] for e in self.entries], ['dsv41/model/ced/4'],
                         'the ced stage must reach tools/export-scenes.mjs or no gate sees it')

    def test_the_shipped_ced_scene_satisfies_the_invariant(self):
        for entry in self.entries:
            bad = [f for f in judge_ced_topology(entry['scene'], entry['expect']) if f.status != 'PASS']
            self.assertEqual(bad, [], f"{entry['key']}: {[f.message for f in bad]}")

    def test_the_fixture_layer_kinds_match_the_application(self):
        """A hand-written fixture that drifts from core.mjs would make every negative test vacuous."""
        for entry in self.entries:
            self.assertEqual(entry['expect']['layerKinds'], dsv41_kinds())
            self.assertEqual(entry['expect']['encoderLayers'], dsv41_fixture()['dsv41']['encoderLayers'])

    def test_every_other_lens_is_skipped_rather_than_judged(self):
        for entry in self.data['scenes']:
            if entry['state'].get('stage') == 'ced':
                continue
            self.assertEqual(codes(judge_ced_topology(entry['scene'], entry['expect']), 'PASS'), {'CED_SKIP'},
                             entry['key'])


def dsv41_model(**overrides):
    """The transcribed shape, so a test can break one field at a time.

    Distinct from `dsv41_fixture`, which is the per-scene `expect` record: this is the
    `MODELS.dsv41` entry `judge_architecture` reads. `test_the_fixture_matches_the_shipped_model`
    below pins it to the application so a drifted literal cannot make the negatives vacuous.
    """
    m = {'id': 'dsv41', 'architecture': 'ced-csa2', 'layers': 40, 'encoderLayers': 20, 'swaLayers': 2,
         'kvSourceLayers': [2, 8, 14, 20], 'indexSourceLayers': [2, 8, 14, 20, 24, 28, 32, 36],
         'compressEncoder': 2, 'compressDecoder': 1, 'candidateSource': 20,
         'engramLayers': [1, 14], 'engramTables': [384006168, 384016682], 'engramHeadDim': 256,
         'engramNgram': 4, 'engramHeads': 8, 'ngram': 196613849600,
         'mtp': 3, 'dsparkTargets': [37, 38, 39], 'dsparkBlock': 5, 'dsparkExperts': 128,
         'dsparkTopK': 3, 'experts': 384, 'topK': 6, 'streams': 4, 'headDim': 512, 'd': 5120,
         'indexer': 128, 'kvBytesPerToken': 890, 'active': 8e9, 'activeDecode': 16e9,
         'indexerBudget': 512, 'candidateBlocks': 2048, 'candidateBlockSize': 8,
         'sinkhornIters': 20, 'residual': 'mhc'}
    m.update(overrides)
    return {'dsv41': m}


#: The backbone figure `parameterLedger` reports, and the gate weights the band-ratio rule owns:
#: three encoder bands above ratio 1 at d x head_dim. `test_the_parameter_fixture_matches_the
#: _shipped_ledger` pins both to the application.
PARAM_MAIN = 551_880_835_184
PARAM_GATES = 3 * 5120 * 512


def innovation_rows_fixture(derived=720, reported=890, encoder=20, depth=40,
                            blocks=2048, block_size=8, budget=512, streams=4, sinkhorn=20,
                            dspark_block=5, dspark_experts=128, dspark_topk=3, **row_overrides):
    """The shape `samples.innovations` carries: `effect` already rendered, citations resolved.

    Deliberately not the shipped prose. These tests are about what the judge does with a row,
    and a fixture copied from `src/innovations.mjs` would drift with the wording rather than with
    the figures the check is actually keyed to. It carries the five checked rows; the three the
    gate deliberately does not check are absent, which is itself asserted below.
    """
    def row(rid, effect, source):
        return {'id': rid, 'title': rid, 'replaces': 'a', 'mechanism': 'b', 'effect': effect,
                'sources': [source], 'citations': [f'https://example.invalid/{source}']}
    rows = [row('ced', f'prefill computes {encoder} of {depth} layers', 'dsv41-code'),
            row('fp4-kv', f'{derived} B/token derived against {reported} B/token reported',
                'dsv41-report'),
            row('hier-indexer',
                f'{blocks} blocks x {block_size} narrowed to a {budget}-position budget',
                'dsv41-config'),
            row('mhc', f'{streams} residual streams, {sinkhorn} Sinkhorn iterations',
                'dsv41-config'),
            row('dspark',
                f'{dspark_block}-token blocks, {dspark_topk} of {dspark_experts} draft experts',
                'dsv41-code')]
    for r in rows:
        r.update(row_overrides.get(r['id'], {}))
    return [r for r in rows if r.get('id') is not None]


def samples_fixture(main_per_token=720, bounded=128 * 40 * 512, S=32768,
                    param_main=PARAM_MAIN, param_unit=None, innovations=None):
    return {'partitions': {'dsv41': {'swa': 2, 'csa2-full': 4, 'csa2-reindex': 4, 'csa2-reuse': 30}},
            'logical_state': {'dsv41': {'mainKV': main_per_token * S, 'bounded': bounded,
                                        'contextLinear': main_per_token * S, 'B': 1, 'S': S}},
            'logical_state_far': {'dsv41': {'bounded': bounded, 'B': 1, 'S': S * 4}},
            'parameters': {'dsv41': {'main': param_main}},
            'parameters_unit_ratio': {'dsv41': {'main': param_main - PARAM_GATES
                                                if param_unit is None else param_unit}},
            'innovations': {'dsv41': innovation_rows_fixture() if innovations is None
                            else innovations},
            'source_ids': ['dsv41-code', 'dsv41-report', 'dsv41-config']}


class ArchitectureInvariantTests(unittest.TestCase):
    """Structural invariants a mis-transcription breaks, on the model table rather than a drawing."""

    def test_the_transcribed_architecture_satisfies_every_invariant(self):
        f = judge_architecture(dsv41_model(), samples_fixture())
        self.assertEqual(codes(f, 'FAIL'), set(), 'the real shape must be clean')
        self.assertIn('CED_MODE_DISAGREEMENT', codes(f, 'PASS'), 'the check must report that it ran')

    def test_the_two_mode_derivations_are_compared_and_can_disagree(self):
        """Move one Full layer in the config arrays only. The report-derived structure is
        unchanged, so the two must diverge."""
        f = judge_architecture(dsv41_model(kvSourceLayers=[2, 9, 14, 20]))
        self.assertIn('CED_MODE_DISAGREEMENT', codes(f, 'FAIL'))

    def test_the_report_derivation_reads_no_layer_ids(self):
        """If it did, the comparison would be a tautology. Proved by signature, not by reading."""
        import inspect
        names = set(inspect.signature(modes_from_report_structure).parameters)
        self.assertEqual(names, {'swa', 'encoder', 'total', 'encoder_group', 'decoder_group'})

    def test_a_full_layer_outside_the_index_sources_is_caught(self):
        f = judge_architecture(dsv41_model(indexSourceLayers=[8, 14, 20, 24, 28, 32, 36]))
        self.assertIn('CSA2_INDEX_SUBSET', codes(f, 'FAIL'))

    def test_a_reuse_layer_with_nothing_to_reuse_is_caught(self):
        """Drop the encoder's first Full layer: layers 3-7 then read a cache no layer wrote."""
        f = judge_architecture(dsv41_model(kvSourceLayers=[8, 14, 20]))
        self.assertIn('CED_REUSE_UNREACHABLE', codes(f, 'FAIL'))

    def test_a_decoder_deriving_its_own_kv_is_caught(self):
        """Two decoder KV sources means the decoder no longer projects solely from H20."""
        f = judge_architecture(dsv41_model(kvSourceLayers=[2, 8, 14, 20, 30]))
        self.assertIn('CED_KV_PROVENANCE', codes(f, 'FAIL'))

    def test_a_candidate_pool_built_after_its_consumers_is_caught(self):
        f = judge_architecture(dsv41_model(candidateSource=28))
        self.assertIn('CSA2_CANDIDATE_SOURCE', codes(f, 'FAIL'))

    def test_indistinguishable_band_ratios_are_caught(self):
        """Identical ratios would make the two halves the same mechanism drawn twice."""
        f = judge_architecture(dsv41_model(compressDecoder=2))
        self.assertIn('CED_BAND_RATIO', codes(f, 'FAIL'))

    def test_a_sub_unit_compression_ratio_is_caught(self):
        f = judge_architecture(dsv41_model(compressDecoder=0))
        self.assertIn('CED_BAND_RATIO', codes(f, 'FAIL'))

    def test_a_rounded_engram_total_is_caught(self):
        f = judge_architecture(dsv41_model(ngram=196_000_000_000))
        self.assertIn('ENGRAM_TOTAL_DERIVATION', codes(f, 'FAIL'))

    def test_mismatched_engram_tables_and_layers_are_caught(self):
        f = judge_architecture(dsv41_model(engramTables=[384006168]))
        self.assertIn('ENGRAM_LAYER_BOUNDS', codes(f, 'FAIL'))

    def test_a_wrong_hash_column_count_is_caught(self):
        f = judge_architecture(dsv41_model(engramNgram=1))
        self.assertIn('ENGRAM_HASH_COLUMNS', codes(f, 'FAIL'))

    def test_a_draft_head_without_one_target_per_layer_is_caught(self):
        f = judge_architecture(dsv41_model(dsparkTargets=[38, 39]))
        self.assertIn('DSPARK_TARGET_COUNT', codes(f, 'FAIL'))

    def test_a_trivial_draft_block_is_caught(self):
        f = judge_architecture(dsv41_model(dsparkBlock=1))
        self.assertIn('DSPARK_BLOCK_TRIVIAL', codes(f, 'FAIL'))

    def test_a_draft_moe_identical_to_the_backbone_is_caught(self):
        f = judge_architecture(dsv41_model(dsparkExperts=384, dsparkTopK=6))
        self.assertIn('DSPARK_MOE_DISTINCT', codes(f, 'FAIL'))

    def test_a_draft_layer_owning_a_compressed_cache_is_caught(self):
        """DSparkAttention.forward opens with `assert self.compress_ratio == 0`, so a draft layer
        in kv_source_layer_ids contradicts the publisher's own assertion."""
        f = judge_architecture(dsv41_model(kvSourceLayers=[2, 8, 14, 20, 40]))
        self.assertIn('CED_MTP_RATIO', codes(f, 'FAIL'))

    def test_a_phase_split_covering_every_layer_is_not_a_split(self):
        f = judge_architecture(dsv41_model(encoderLayers=40))
        self.assertIn('CED_PHASE_DEPTH', codes(f, 'FAIL'))

    def test_a_partition_that_does_not_tile_the_depth_is_caught(self):
        s = samples_fixture()
        s['partitions']['dsv41']['csa2-reuse'] = 29
        f = judge_architecture(dsv41_model(), s)
        self.assertIn('CED_PARTITION_SUM', codes(f, 'FAIL'))

    def test_an_absent_inventory_fails_the_partition_check(self):
        """Ruling P1 required this not be skipped silently; ruling T789b requires the software
        gate never to return BLOCKED. FAIL satisfies both, and more loudly: the inventory comes
        from our own exporter, so its absence is a defect in our wiring."""
        f = judge_architecture(dsv41_model())
        self.assertIn('CED_PARTITION_SUM', codes(f, 'FAIL'))
        self.assertNotIn('CED_PARTITION_SUM', codes(f, 'PASS'))

    def test_the_active_ratio_is_no_longer_judged_by_the_software_gate(self):
        """Ruling T789b: its honest outcome is BLOCKED, which is exit 2 on a gate the Pages
        deploy requires."""
        f = judge_architecture(dsv41_model(activeDecode=64e9), samples_fixture())
        self.assertNotIn('CED_ACTIVE_RATIO', codes(f))

    def test_models_without_the_architecture_are_left_alone(self):
        f = judge_architecture({'ds3': {'id': 'ds3', 'architecture': 'mla', 'layers': 61}})
        self.assertEqual(f, [], 'these invariants apply only to CED presets')

    def test_a_ced_model_with_a_different_tag_is_judged_not_skipped(self):
        """The hole the `architecture == 'ced-csa2'` guard left open.

        Every other consumer of this population selects on `encoderLayers`, so a checkpoint
        tagged `ced-csa3` would get its focal page, its scenes, its innovation rows and its
        previews. Under the old guard all six judges here matched nothing and returned an empty
        list, and `aggregate()` over no findings is PASS: a whole architecture verified by
        silence. It must now be selected, judged, and contradicted on the tag itself.
        """
        models = dsv41_model()
        models['dsv41']['architecture'] = 'ced-csa3'
        f = judge_architecture(models, samples_fixture())
        self.assertIn('CED_ARCH_TAG', codes(f, 'FAIL'),
                      'a CED checkpoint carrying a foreign architecture tag was not contradicted')
        self.assertTrue(f, 'the CED judges returned nothing at all for a CED checkpoint')

    def test_the_tag_invariant_passes_on_the_shipped_inventory(self):
        models, _ = inventory(ROOT)
        f = judge_architecture_tag(models)
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('CED_ARCH_TAG', codes(f, 'PASS'))

    def test_the_fixture_matches_the_shipped_model(self):
        """A drifted literal would make every negative test above a test of nothing."""
        models, _ = inventory(ROOT)
        shipped = models['dsv41']
        for key, value in dsv41_model()['dsv41'].items():
            self.assertEqual(shipped[key], value, key)

    def test_the_shipped_model_satisfies_every_invariant(self):
        models, samples = inventory(ROOT)
        bad = [f for f in judge_architecture(models, samples) if f.status == 'FAIL']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')


class ClaimModelCoverageTests(unittest.TestCase):
    def test_a_manifest_claim_omitting_a_supported_model_is_caught(self):
        """The all-models claims say they reconcile every supported model. One that names
        eight of nine is asserting completeness it does not have."""
        claims = [{'id': 'function.cacheLedger',
                   'dependencies': ['MODELS.ds2.d', 'MODELS.ds3.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'FAIL'))

    def test_a_single_model_scene_claim_is_not_required_to_cover_all_models(self):
        """The per-scene claims each depend on their own one model; that is correct."""
        claims = [{'id': 'scene.ds2.model.block.norm1.abc',
                   'dependencies': ['MODELS.ds2.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertEqual(codes(f, 'FAIL'), set())

    def test_a_complete_all_models_claim_passes(self):
        claims = [{'id': 'file.src/core.mjs',
                   'dependencies': ['MODELS.ds2.d', 'MODELS.ds3.d', 'MODELS.dsv41.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'PASS'))

    def test_a_dependency_naming_an_unsupported_model_is_caught(self):
        claims = [{'id': 'file.src/core.mjs',
                   'dependencies': ['MODELS.ds2.d', 'MODELS.ds3.d', 'MODELS.dsv41.d', 'MODELS.ghost.d']}]
        f = judge_claim_model_coverage(claims, {'ds2': {}, 'ds3': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'FAIL'))

    def test_a_manifest_with_no_multi_model_claim_blocks_rather_than_passing(self):
        """Zero comparisons is not agreement. The same rule as the absent publisher cache."""
        f = judge_claim_model_coverage([], {'ds2': {}, 'dsv41': {}})
        self.assertIn('CLAIM_MODEL_COVERAGE', codes(f, 'BLOCKED'))
        self.assertEqual(codes(f, 'PASS'), set())

    def test_the_shipped_manifest_covers_every_model(self):
        models, _ = inventory(ROOT)
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        bad = [f for f in judge_claim_model_coverage(target['claims'], models) if f.status != 'PASS']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')


class ContractModelCoverageTests(unittest.TestCase):
    """The tenth instance of the branch's recurring defect, at the contract layer.

    `observatory.json` carried one `model.*` contract per model for eight models and stayed at
    eight while a ninth was authored, and nothing in the audit could report it: `RUNTIME_MATRIX`
    excludes the `model` domain, `RUNTIME_COVERAGE` asks only that a domain have *some*
    obligation, and `DOMAIN_UNCOVERED` fires only on an empty domain. The public record then
    reported a 31/31 inventory over nine checkpoints.
    """

    def test_a_model_with_no_contract_is_caught(self):
        contracts = [MODEL_CONTRACT]
        f = judge_contract_model_coverage({'ds2': {}, 'dsv41': {}}, contracts, [MODEL_OBLIGATION])
        self.assertIn('CONTRACT_MODEL_COVERAGE', codes(f, 'FAIL'))
        self.assertIn('dsv41', {x.subject for x in f if x.status == 'FAIL'})

    def test_a_model_with_a_contract_but_no_runtime_obligation_is_caught(self):
        """A contract with no obligation is never NOT_RUN, so its absence of evidence is
        invisible rather than reported."""
        f = judge_contract_model_coverage({'ds2': {}}, [MODEL_CONTRACT], [])
        self.assertIn('CONTRACT_MODEL_COVERAGE', codes(f, 'FAIL'))

    def test_a_model_with_both_passes(self):
        f = judge_contract_model_coverage({'ds2': {}}, [MODEL_CONTRACT], [MODEL_OBLIGATION])
        self.assertIn('CONTRACT_MODEL_COVERAGE', codes(f, 'PASS'))
        self.assertEqual(codes(f, 'FAIL'), set())

    def test_the_contract_id_is_not_assumed_to_name_the_model(self):
        """`model.kimi-k3` speaks for `MODELS.kimi3`. A check keyed on `f'model.{mid}'` would
        report the one model whose ids differ as uncovered and prove nothing about the rest, so
        coverage is read off the contract's own bindings and claims."""
        contract = {'id': 'model.kimi-k3', 'domain': 'model',
                    'bindings': {'layers': 'MODELS.kimi3.layers'},
                    'claims': ['MODELS.kimi3.d']}
        obligation = {'id': 'runtime.model', 'domain': 'model', 'contract': 'model.kimi-k3'}
        f = judge_contract_model_coverage({'kimi3': {}}, [contract], [obligation])
        self.assertEqual(codes(f, 'FAIL'), set())

    def test_a_contract_in_another_domain_does_not_count_as_model_coverage(self):
        """A `gpu` contract that happens to bind a model field is not a model contract."""
        contract = {'id': 'gpu.B200', 'domain': 'gpu',
                    'bindings': {'d': 'MODELS.ds2.d'}, 'claims': []}
        f = judge_contract_model_coverage({'ds2': {}}, [contract], [])
        self.assertIn('CONTRACT_MODEL_COVERAGE', codes(f, 'FAIL'))

    def test_an_empty_inventory_blocks_rather_than_passing(self):
        """Zero comparisons is not agreement -- the same rule as the absent publisher cache."""
        f = judge_contract_model_coverage({}, [MODEL_CONTRACT], [MODEL_OBLIGATION])
        self.assertIn('CONTRACT_MODEL_COVERAGE', codes(f, 'BLOCKED'))
        self.assertEqual(codes(f, 'PASS'), set())

    def test_the_shipped_manifest_covers_every_shipped_model(self):
        models, _ = inventory(ROOT)
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        bad = [f for f in judge_contract_model_coverage(models, target['contracts'],
                                                        target['runtime_obligations'])
               if f.status != 'PASS']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')

    def test_the_shipped_model_contract_validates(self):
        """Authored, not fabricated: every bound field is the application's own constant and the
        layer groups partition the depth exactly once."""
        from verification.domains.contracts import validate
        from verification.runner import contract_record
        target = loads((ROOT/'verification/manifests/observatory.json').read_text())
        rec = next(c for c in target['contracts'] if c['id'] == 'model.dsv41')
        inv = loads(subprocess.run(['node', 'tools/export-verification.mjs'], cwd=ROOT,
                                   text=True, capture_output=True, check=True).stdout)
        record, unbound = contract_record(rec, inv['fields'])
        self.assertEqual(unbound, [], 'a binding names a core input that does not exist')
        bad = [f for f in validate('model', record) if f.status == 'FAIL']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')


class ReportStructureTests(unittest.TestCase):
    def test_the_report_group_structure_reproduces_the_published_assignment(self):
        modes = modes_from_report_structure(swa=2, encoder=20, total=40,
                                            encoder_group=6, decoder_group=4)
        self.assertEqual(len(modes), 40)
        self.assertEqual([i for i, m in enumerate(modes) if m == 'csa2-full'], [2, 8, 14, 20])
        self.assertEqual([i for i, m in enumerate(modes) if m == 'csa2-reindex'], [24, 28, 32, 36])
        self.assertEqual(modes[:2], ['swa', 'swa'])
        self.assertEqual(modes.count('csa2-reuse'), 30)

    def test_the_two_derivations_agree_on_the_shipped_model(self):
        models, _ = inventory(ROOT)
        m = models['dsv41']
        self.assertEqual(modes_from_config(m),
                         modes_from_report_structure(swa=m['swaLayers'], encoder=m['encoderLayers'],
                                                     total=m['layers'], encoder_group=6, decoder_group=4))

    def test_the_config_derivation_agrees_with_the_application(self):
        """A third reading: core.layerKind, exported per layer. If ced.py's transcription of the
        rule drifted from core.mjs, the two would part company here."""
        data = export_scenes(ROOT)
        models, _ = inventory(ROOT)
        entry = next(e for e in data['scenes'] if e['state'].get('stage') == 'ced')
        self.assertEqual(entry['expect']['layerKinds'], modes_from_config(models['dsv41']))


class BandRatioTests(unittest.TestCase):
    """The encoder/decoder ratio rule exists in three implementations: `ratioOf` in
    diagram-meta.mjs and the same expression inside `parameterLedger` and `cacheLedger`.
    Nothing compared them. This judges the *drawn* ratio against a Python re-derivation from
    the declared ratios, so `ratioOf` drifting from the model table is a FAIL.
    """

    def test_the_drawn_ratios_agree_with_the_declared_ones(self):
        f = judge_ced_topology(clean_ced_scene(), dsv41_fixture()['dsv41'])
        self.assertIn('CED_BAND_RATIO_DRAWN', codes(f, 'PASS'))
        self.assertEqual(codes(f, 'FAIL'), set())

    def test_a_band_stating_the_other_half_ratio_is_caught(self):
        """What an off-by-one in `ratioOf` would look like: the decoder Full band drawn with
        the encoder's ratio."""
        scene = clean_ced_scene()
        for node in scene['nodes']:
            if node['id'] == 'dec-full':
                node['sub'] = 'L21 writes ÷2 latents · 3 layers reuse'
        f = judge_ced_topology(scene, dsv41_fixture()['dsv41'])
        self.assertIn('CED_BAND_RATIO_DRAWN', codes(f, 'FAIL'))

    def test_a_band_that_states_no_ratio_is_caught(self):
        scene = clean_ced_scene()
        for node in scene['nodes']:
            if node['id'] == 'enc-band-3-8':
                node['sub'] = 'L3 writes a compressed cache · 5 layers reuse'
        f = judge_ced_topology(scene, dsv41_fixture()['dsv41'])
        self.assertIn('CED_BAND_RATIO_DRAWN', codes(f, 'FAIL'))

    def test_a_preset_declaring_no_ratios_cannot_be_judged(self):
        m = dict(dsv41_fixture()['dsv41'])
        del m['compressEncoder']
        f = judge_ced_topology(clean_ced_scene(), m)
        self.assertIn('CED_BAND_RATIO_DRAWN', codes(f, 'FAIL'))

    def test_the_shipped_ced_scene_states_its_derived_ratios(self):
        data = export_scenes(ROOT)
        entry = next(e for e in data['scenes'] if e['state'].get('stage') == 'ced')
        f = judge_ced_topology(entry['scene'], entry['expect'])
        self.assertIn('CED_BAND_RATIO_DRAWN', codes(f, 'PASS'))


class KvBudgetTests(unittest.TestCase):
    """The ledger computes main KV in JavaScript; this re-derives it in Python from the pinned
    config. Neither side is checked against itself."""

    def test_the_real_budget_agrees_between_python_and_the_ledger(self):
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=890), samples_fixture())
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('KV_BUDGET_DISAGREEMENT', codes(f, 'PASS'))
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'PASS'))
        self.assertIn('KV_BOUNDED_INVARIANT', codes(f, 'PASS'))

    def test_a_ledger_that_drifts_from_the_python_derivation_is_caught(self):
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=890), samples_fixture(main_per_token=640))
        self.assertIn('KV_BUDGET_DISAGREEMENT', codes(f, 'FAIL'))

    def test_a_reported_total_below_the_derived_main_kv_is_caught(self):
        """890 -> 600 would make the residual negative, i.e. main KV alone exceeds the total."""
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=600), samples_fixture())
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'FAIL'))

    def test_a_residual_implying_an_impossible_precision_is_caught(self):
        """1200 B leaves 480 B over 320 channels = 1.5 B/channel, wider than FP8."""
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=1200), samples_fixture())
        self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'FAIL'))

    def test_the_residual_bound_is_the_physical_range_not_one_fitted_to_890(self):
        """The report states no indexer precision, so the bound is FP4..FP8 and nothing narrower.

        The accepted window is therefore 880..1040 B/token, and the reported 890 sits near its
        low edge rather than at its centre -- which is what makes this a precision bound and not
        a tolerance drawn around the observed figure. Pinned so narrowing it breaks a test.
        """
        self.assertEqual((FP4_FLOOR, FP8_CEILING), (0.5, 1.0))
        for reported in (880, 1040):
            f = judge_kv_budget(dsv41_model(kvBytesPerToken=reported), samples_fixture())
            self.assertEqual(codes(f, 'FAIL'), set(), reported)
        for reported in (879, 1041):
            f = judge_kv_budget(dsv41_model(kvBytesPerToken=reported), samples_fixture())
            self.assertIn('KV_RESIDUAL_IMPLAUSIBLE', codes(f, 'FAIL'), reported)

    def test_bounded_state_that_moves_with_context_is_caught(self):
        s = samples_fixture()
        s['logical_state_far']['dsv41']['bounded'] *= 4
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=890), s)
        self.assertIn('KV_BOUNDED_INVARIANT', codes(f, 'FAIL'))

    def test_two_samples_at_the_same_context_cannot_prove_the_invariant(self):
        """Comparing a state against itself is not evidence that it is bounded. FAIL rather than
        BLOCKED per ruling T789b: the samples are ours, so one context is a wiring defect."""
        s = samples_fixture()
        s['logical_state_far']['dsv41']['S'] = s['logical_state']['dsv41']['S']
        f = judge_kv_budget(dsv41_model(kvBytesPerToken=890), s)
        self.assertIn('KV_BOUNDED_INVARIANT', codes(f, 'FAIL'))
        self.assertNotIn('KV_BOUNDED_INVARIANT', codes(f, 'PASS'))

    def test_the_bounded_invariant_runs_for_every_model_not_only_this_architecture(self):
        models = dsv41_model()
        models['ds3'] = {'id': 'ds3', 'architecture': 'mla', 'layers': 61}
        s = samples_fixture()
        s['logical_state']['ds3'] = {'bounded': 0, 'B': 1, 'S': 32768}
        s['logical_state_far']['ds3'] = {'bounded': 7, 'B': 1, 'S': 131072}
        f = judge_kv_budget(models, s)
        self.assertIn('KV_BOUNDED_INVARIANT', codes(f, 'FAIL'))
        self.assertEqual([x.subject for x in f if x.status == 'FAIL'], ['ds3'])

    def test_the_python_derivation_reproduces_the_published_figures(self):
        """720 B/token over 1,280 stored channels, and 320 indexer channels. Written out so a
        change to either expression has to be argued with, not just re-recorded."""
        m = dsv41_model()['dsv41']
        self.assertEqual(derived_main_kv_bytes_per_token(m), 720.0)
        self.assertEqual(indexer_channels_per_token(m), 320)

    def test_the_shipped_ledger_agrees_with_the_python_derivation(self):
        models, samples = inventory(ROOT)
        bad = [f for f in judge_kv_budget(models, samples) if f.status == 'FAIL']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')


class MhcTests(unittest.TestCase):
    def test_the_coefficient_count_follows_from_the_stream_count(self):
        f = judge_mhc(dsv41_model())
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('MHC_COEFF_COUNT', codes(f, 'PASS'))

    def test_a_stream_count_inconsistent_with_the_residual_mode_is_caught(self):
        f = judge_mhc(dsv41_model(streams=1))
        self.assertIn('MHC_COEFF_COUNT', codes(f, 'FAIL'))

    def test_a_stream_count_that_is_merely_wrong_is_caught(self):
        f = judge_mhc(dsv41_model(streams=3))
        self.assertIn('MHC_COEFF_COUNT', codes(f, 'FAIL'))

    def test_a_standard_residual_is_left_alone(self):
        self.assertEqual(judge_mhc({'ds3': {'id': 'ds3', 'residual': 'standard', 'streams': 0}}), [])

    def test_the_shipped_models_satisfy_the_coefficient_count(self):
        models, _ = inventory(ROOT)
        bad = [f for f in judge_mhc(models) if f.status != 'PASS']
        self.assertEqual(bad, [], f'{[f.message for f in bad]}')


class PublisherAgreementTests(unittest.TestCase):
    """MODELS.dsv41 is a hand transcription of config.json. Transcription error is the dominant
    failure mode when adding an architecture and nothing in this repo caught it."""

    def test_every_transcribed_field_is_accounted_for(self):
        """A field with no mapping is a field nobody checks. Read from the shipped model rather
        than a literal, so adding a field to MODELS forces a decision here."""
        models, _ = inventory(ROOT)
        for field in sorted(models['dsv41']):
            self.assertTrue(field in FIELD_MAP or field in NOT_IN_CONFIG,
                            f'{field} is in neither FIELD_MAP nor NOT_IN_CONFIG, so nothing '
                            f'verifies it and nothing records why not')
        self.assertEqual(set(FIELD_MAP) & set(NOT_IN_CONFIG), set(),
                         'a field cannot be both mapped and declared unmappable')

    def test_the_map_covers_the_fields_the_architecture_turns_on(self):
        for field in ('layers', 'kvSourceLayers', 'engramTables', 'streams', 'compressEncoder',
                      'encoderLayers', 'dsparkTargets', 'candidateSource'):
            self.assertIn(field, FIELD_MAP)

    def test_a_mistranscribed_field_is_caught(self):
        config = {'text_config': {'num_hidden_layers': 40, 'hidden_size': 5120}}
        f = judge_publisher_agreement(dsv41_model(layers=39), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'FAIL'))

    def test_every_mismatch_is_reported_not_only_the_first(self):
        config = {'text_config': {'num_hidden_layers': 40, 'hidden_size': 5120,
                                  'num_attention_heads': 64, 'hc_mult': 4}}
        f = judge_publisher_agreement(dsv41_model(layers=39, heads=32, streams=2), config=config)
        message = ' '.join(x.message for x in f if x.status == 'FAIL')
        for field in ('layers', 'heads', 'streams'):
            self.assertIn(field, message)

    def test_a_partial_config_cannot_report_agreement(self):
        """Two fields resolving out of fifty-four is not a faithful transcription verified; it is
        fifty-two fields nothing compared. A mapped field that stops resolving -- an upstream
        rename, a truncated document -- is the same defect class as an unmapped one."""
        config = {'text_config': {'num_hidden_layers': 40, 'hidden_size': 5120}}
        f = judge_publisher_agreement(dsv41_model(), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertNotIn('CED_PUBLISHER_AGREEMENT', codes(f, 'PASS'))
        self.assertIn('did not resolve', ' '.join(x.message for x in f))

    def test_one_unresolvable_field_is_enough_to_block(self):
        """Runs with no cache: the real config minus one key, so the guard is exercised where it
        matters rather than skipped there."""
        config = self.config()
        if config is None:
            self.skipTest('publisher cache absent')
        del config['text_config']['hc_mult']
        f = judge_publisher_agreement(dsv41_model(), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertIn('streams', ' '.join(x.message for x in f))

    def test_a_faithful_transcription_against_the_whole_config_passes(self):
        config = self.config()
        if config is None:
            self.skipTest('publisher cache absent')
        f = judge_publisher_agreement(dsv41_model(), config=config)
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'PASS'))
        self.assertIn('0 unresolved', ' '.join(x.message for x in f))

    def test_a_checkpoint_with_one_ratio_throughout_cannot_pin_its_boundary(self):
        """Why CED_BAND_RATIO's distinctness rule is load-bearing rather than aesthetic:
        _encoder_boundary derives encoderLayers as the index where the ratio changes, so with one
        ratio throughout the field becomes unverifiable -- unresolved, therefore BLOCKED, never
        guessed."""
        config = self.config()
        if config is None:
            self.skipTest('publisher cache absent')
        ratios = config['text_config']['compress_ratios']
        config['text_config']['compress_ratios'] = [0 if r == 0 else 2 for r in ratios]
        f = judge_publisher_agreement(dsv41_model(), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertIn('encoderLayers', ' '.join(x.message for x in f))

    def test_an_absent_cache_blocks_rather_than_passing(self):
        f = judge_publisher_agreement(dsv41_model(), config=None)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertNotIn('CED_PUBLISHER_AGREEMENT', codes(f, 'PASS'),
                         'never report agreement having compared nothing')
        self.assertIn('verification.sources acquire',
                      ' '.join(x.message + x.remedy for x in f), 'name the acquire command')

    def test_a_config_matching_no_mapped_field_blocks_rather_than_passing(self):
        """Zero comparisons is the exact false green a previous audit found here."""
        f = judge_publisher_agreement(dsv41_model(), config={'text_config': {}})
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'BLOCKED'))
        self.assertNotIn('CED_PUBLISHER_AGREEMENT', codes(f, 'PASS'))

    def test_an_unmapped_model_field_is_visible_in_the_report(self):
        config = {'text_config': {'num_hidden_layers': 40}}
        f = judge_publisher_agreement(dsv41_model(inventedField=7), config=config)
        self.assertIn('CED_FIELD_UNMAPPED', codes(f, 'FAIL'))
        self.assertIn('inventedField', ' '.join(x.message for x in f if x.status == 'FAIL'))

    def test_a_derived_field_is_checked_against_an_expression_not_a_pointer(self):
        """encoderLayers is our arithmetic over compress_ratios, not a published field. It still
        has to follow from the bytes."""
        config = self.config()
        if config is None:
            self.skipTest('publisher cache absent')
        f = judge_publisher_agreement(dsv41_model(encoderLayers=24), config=config)
        self.assertIn('CED_PUBLISHER_AGREEMENT', codes(f, 'FAIL'))
        self.assertIn('encoderLayers', ' '.join(x.message for x in f if x.status == 'FAIL'))

    def record(self):
        catalog = loads((ROOT/'verification/manifests/sources.json').read_text())
        return next(s for s in catalog['sources'] if s['id'] == 'dsv41-config')

    def config(self):
        return loads((ROOT/self.record()['cache']).read_text()) if self.cached() else None

    def cached(self):
        return (ROOT/self.record()['cache']).exists()

    def test_the_shipped_transcription_agrees_with_the_publisher_bytes(self):
        """The check this task exists for, run against the real acquired config.json."""
        if not self.cached():
            self.skipTest('publisher cache absent; run verification.sources acquire')
        models, _ = inventory(ROOT)
        catalog = loads((ROOT/'verification/manifests/sources.json').read_text())
        f = judge_publisher_agreement(models, root=ROOT, catalog=catalog)
        self.assertEqual([x for x in f if x.status != 'PASS'], [],
                         f'{[x.message for x in f if x.status != "PASS"]}')

    def test_every_mapped_pointer_resolves_against_the_acquired_config(self):
        """A pointer that silently fails to resolve is a field nothing compares, which is the
        same defect as an unmapped field wearing a mapping."""
        if not self.cached():
            self.skipTest('publisher cache absent; run verification.sources acquire')
        models, _ = inventory(ROOT)
        catalog = loads((ROOT/'verification/manifests/sources.json').read_text())
        f = judge_publisher_agreement(models, root=ROOT, catalog=catalog)
        passed = next(x for x in f if x.code == 'CED_PUBLISHER_AGREEMENT')
        self.assertIn('0 unresolved', passed.message, passed.message)


class GatePlacementTests(unittest.TestCase):
    """Ruling T789a: the transcription check is evidence work, not a software gate.

    `schematics` deploys the page. A software gate that BLOCKs because `.verification-cache/` is
    absent -- which it always is in CI, by design -- would stop the page shipping, which is the
    exact failure the amended Pages contract exists to prevent.
    """

    def test_the_transcription_check_reports_in_the_evidence_domain(self):
        f = judge_publisher_agreement(dsv41_model(), config={'text_config': {'num_hidden_layers': 40}})
        self.assertEqual({x.domain for x in f}, {'evidence'})

    def test_the_schematics_gate_does_not_carry_the_transcription_check(self):
        report = schematics.run(ROOT)
        self.assertNotIn('CED_PUBLISHER_AGREEMENT', report['passed_codes'])
        self.assertNotIn('CED_PUBLISHER_AGREEMENT', {f['code'] for f in report['findings']})

    def test_the_schematics_gate_cannot_report_blocked(self):
        """Its exit code gates a deployment, so BLOCKED must not be reachable from it. Every
        judge it runs reads only our own bytes."""
        report = schematics.run(ROOT)
        self.assertEqual(set(report['counts']) - {'PASS', 'FAIL'}, set(), report['counts'])

    def test_the_architecture_invariants_stay_in_the_schematics_gate(self):
        """Only the upstream-bytes check moved; these read the application alone."""
        report = schematics.run(ROOT)
        for code in ('CED_MODE_DISAGREEMENT', 'KV_BUDGET_DISAGREEMENT', 'MHC_COEFF_COUNT',
                     'CED_BAND_RATIO_DRAWN', 'CED_PARTITION_SUM'):
            self.assertIn(code, report['passed_codes'])
        for moved in ('CED_PUBLISHER_AGREEMENT', 'CED_ACTIVE_RATIO', 'CLAIM_MODEL_COVERAGE'):
            self.assertNotIn(moved, report['passed_codes'], f'{moved} belongs to the audit')


class ActiveRatioTests(unittest.TestCase):
    """Evidence-domain, in the audit: it compares published card figures against depth."""

    def test_the_published_actives_track_the_depth_split(self):
        f = judge_active_ratio(dsv41_model())
        self.assertIn('CED_ACTIVE_RATIO', codes(f, 'PASS'))
        self.assertEqual({x.domain for x in f}, {'evidence'})

    def test_a_ratio_far_from_the_depth_ratio_blocks_rather_than_failing(self):
        """The published actives are whole billions and the embedding and Engram terms are
        depth-independent, so a mismatch is weak evidence, not a contradiction."""
        f = judge_active_ratio(dsv41_model(activeDecode=64e9))
        self.assertIn('CED_ACTIVE_RATIO', codes(f, 'BLOCKED'))
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('0.125', ' '.join(x.message for x in f))

    def test_an_absent_decode_active_blocks(self):
        f = judge_active_ratio(dsv41_model(activeDecode=0))
        self.assertIn('CED_ACTIVE_RATIO', codes(f, 'BLOCKED'))

    def test_models_without_the_architecture_are_left_alone(self):
        self.assertEqual(judge_active_ratio({'ds3': {'architecture': 'mla'}}), [])

    def test_the_shipped_model_tracks_its_depth_split(self):
        models, _ = inventory(ROOT)
        self.assertEqual(codes(judge_active_ratio(models), 'BLOCKED'), set())


#: Model shapes that each trip at least one schematics-domain check. Used to exercise failure
#: branches rather than the happy path, because a judge that never fails proves nothing about
#: what it returns when it does.
SCHEMATICS_MUTATIONS = [
    {}, {'kvSourceLayers': [2, 9, 14, 20]}, {'indexSourceLayers': [8, 14, 20, 24, 28, 32, 36]},
    {'kvSourceLayers': [8, 14, 20]}, {'kvSourceLayers': [2, 8, 14, 20, 30]},
    {'candidateSource': 28}, {'compressDecoder': 2}, {'compressDecoder': 0}, {'compressEncoder': 0},
    {'ngram': 1}, {'engramTables': [1]}, {'engramNgram': 1}, {'engramLayers': [1, 99]},
    {'dsparkTargets': [38, 39]}, {'dsparkBlock': 1}, {'dsparkExperts': 384, 'dsparkTopK': 6},
    {'kvSourceLayers': [2, 8, 14, 20, 40]}, {'encoderLayers': 40}, {'encoderLayers': 0},
    {'streams': 1}, {'streams': 3}, {'residual': 'standard'},
    {'kvBytesPerToken': 600}, {'kvBytesPerToken': 1200}, {'headDim': 256}, {'indexer': 0},
    {'activeDecode': 64e9}, {'activeDecode': 0}, {'active': 0}, {'swaLayers': 0}, {'layers': 41},
]


class ParameterBandRatioTests(unittest.TestCase):
    """The third implementation of the encoder/decoder ratio rule, which nothing compared.

    `ratioOf` is checked by CED_BAND_RATIO against the drawing and `cacheLedger`'s copy by
    KV_BUDGET_DISAGREEMENT. `parameterLedger`'s copy was unchecked for a structural reason, not an
    oversight: only `function.parameterLedger`'s *source hash* reached the inventory, and a hash
    changes whether the edit was a fix or a regression. These tests are about the value.
    """

    def test_the_parameter_fixture_matches_the_shipped_ledger(self):
        """Otherwise every negative below is a test of a literal nobody ships."""
        _, samples = inventory(ROOT)
        self.assertEqual(samples['parameters']['dsv41']['main'], PARAM_MAIN)
        self.assertEqual(samples['parameters']['dsv41']['main']
                         - samples['parameters_unit_ratio']['dsv41']['main'], PARAM_GATES)

    def test_the_shipped_ledger_agrees_with_the_python_derivation(self):
        models, samples = inventory(ROOT)
        f = judge_parameter_bands(models, samples)
        self.assertEqual(codes(f, 'FAIL'), set())
        self.assertIn('PARAM_BAND_RATIO', codes(f, 'PASS'), 'the check must report that it ran')

    def test_a_ledger_that_ignores_the_band_ratio_is_caught(self):
        """The gate allocated unconditionally, or not at all: both make the two samples agree."""
        f = judge_parameter_bands(dsv41_model(), samples_fixture(param_unit=PARAM_MAIN))
        self.assertIn('PARAM_BAND_RATIO', codes(f, 'FAIL'))

    def test_a_ledger_that_gates_the_decoder_band_too_is_caught(self):
        """`l < encoderLayers` becoming `l <= encoderLayers` gates a fourth band. This is the
        exact off-by-one the rule exists to prevent, and it moves a value, not a hash."""
        four = PARAM_MAIN - 4 * 5120 * 512
        f = judge_parameter_bands(dsv41_model(), samples_fixture(param_unit=four))
        self.assertIn('PARAM_BAND_RATIO', codes(f, 'FAIL'))

    def test_the_derivation_follows_the_encoder_boundary_and_the_ratio(self):
        """The derivation itself, so the negatives above are not merely arithmetic on a constant."""
        self.assertEqual(gated_compressor_layers(dsv41_model()['dsv41']), [2, 8, 14])
        self.assertEqual(gated_compressor_layers(dsv41_model(compressDecoder=2)['dsv41']),
                         [2, 8, 14, 20], 'a compressed decoder band is gated too')
        self.assertEqual(gated_compressor_layers(dsv41_model(compressEncoder=1)['dsv41']), [],
                         'ratio 1 allocates no gate')
        self.assertEqual(derived_compressor_gate_parameters(dsv41_model()['dsv41']), PARAM_GATES)

    def test_an_absent_parameter_sample_fails_rather_than_passes(self):
        f = judge_parameter_bands(dsv41_model(), {})
        self.assertIn('PARAM_BAND_RATIO', codes(f, 'FAIL'))
        self.assertEqual(codes(f, 'PASS'), set(), 'an unrun check must not report agreement')


class InnovationCoverageTests(unittest.TestCase):
    """The written exposition, held to the evidence chain a diagram number already has.

    Prose is where an unsupported assertion is easiest to write and hardest to see, so both
    checks are exercised against rows that are wrong in exactly one way each.
    """

    def test_the_shipped_coverage_is_cited_and_derived(self):
        models, samples = inventory(ROOT)
        f = judge_coverage(models, samples)
        self.assertEqual(codes(f, 'FAIL'), set(), [x.message for x in f if x.status == 'FAIL'])
        self.assertEqual(codes(f, 'PASS'), {'COVERAGE_UNCITED', 'COVERAGE_DERIVED'})

    def test_the_shipped_coverage_covers_every_published_innovation(self):
        """The eight records `tests/core.test.mjs` requires must actually reach the inventory,
        or the Python checks below are judging a shorter list than the page renders."""
        _, samples = inventory(ROOT)
        ids = [r['id'] for r in samples['innovations']['dsv41']]
        self.assertEqual(sorted(ids), sorted(['ced', 'csa2', 'hier-indexer', 'mhc', 'engram',
                                              'dspark', 'fp4-kv', 'swa-replay']))

    def test_a_record_that_cites_nothing_is_caught(self):
        rows = innovation_rows_fixture(**{'ced': {'sources': [], 'citations': []}})
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=rows))
        self.assertIn('COVERAGE_UNCITED', codes(f, 'FAIL'))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'PASS'),
                      'the uncited record must trip its own check, not the other one')

    def test_a_citation_that_resolves_to_no_source_is_caught(self):
        """`citation()` in src/app.mjs renders nothing for an unknown id, so a typo becomes an
        uncited assertion that still looks cited in the record."""
        rows = innovation_rows_fixture(**{'fp4-kv': {'sources': ['dsv41-repot'], 'citations': []}})
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=rows))
        self.assertIn('COVERAGE_UNCITED', codes(f, 'FAIL'))
        self.assertIn('dsv41-repot', ' '.join(x.message for x in f))

    def test_a_row_that_quotes_only_the_published_total_is_caught(self):
        """The specific hazard: 890 B/token is the publisher's headline and 720 is what the
        Full-mode layers account for. Printing only the headline hides the residual."""
        rows = innovation_rows_fixture(**{'fp4-kv': {'effect': '890 B/token reported'}})
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=rows))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))
        self.assertIn('COVERAGE_UNCITED', codes(f, 'PASS'))

    def test_a_row_whose_derived_figure_disagrees_with_python_is_caught(self):
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=innovation_rows_fixture(derived=640)))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))

    def test_a_row_that_states_the_wrong_encoder_depth_is_caught(self):
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=innovation_rows_fixture(encoder=24)))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))

    def test_a_figure_inside_a_longer_number_does_not_satisfy_the_check(self):
        """Numbers are compared as numbers. A row writing 1720 must not satisfy 720, which a
        substring match would allow and which is the drift `assertStated` was written for."""
        rows = innovation_rows_fixture(**{'fp4-kv': {'effect': '1720 B/token against 8901 B'}})
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=rows))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))

    def test_a_thousands_separated_figure_still_satisfies_the_check(self):
        """The page formats with separators, so the matcher must read 1,280 as 1280 -- otherwise
        the check would fire on correct prose and be turned off."""
        self.assertIn(1280.0, ced.rendered_numbers('1,280 channels at 0.5625 B'))
        self.assertIn(0.5625, ced.rendered_numbers('1,280 channels at 0.5625 B'))

    def test_a_missing_row_is_not_read_as_agreement(self):
        rows = innovation_rows_fixture(**{'ced': {'id': None}})
        f = judge_coverage(dsv41_model(), samples_fixture(innovations=rows))
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))

    def test_absent_coverage_rows_fail_rather_than_pass(self):
        for samples in ({}, None, {'innovations': {}}):
            f = judge_coverage(dsv41_model(), samples)
            self.assertEqual(codes(f, 'FAIL'), {'COVERAGE_UNCITED', 'COVERAGE_DERIVED'}, samples)
            self.assertEqual(codes(f, 'PASS'), set(), 'an unrun check must not report agreement')

    def test_the_config_pinned_rows_are_actually_pinned(self):
        """The claim that three more rows are checked against *publisher bytes* rests entirely on
        FIELD_MAP mapping their fields. If it stopped, the check would silently become a
        comparison against our own model table, so the judge enforces the membership."""
        for rid, fields in ced.CONFIG_PINNED_FIGURES.items():
            for _label, field in fields:
                self.assertIn(field, FIELD_MAP, f'{rid} claims {field} is pinned to config bytes')

    def test_a_row_claimed_as_pinned_but_unmapped_is_caught(self):
        """Remove the mapping and the row must stop counting as evidence, not keep passing."""
        pinned = dict(ced.CONFIG_PINNED_FIGURES)
        mapped = dict(FIELD_MAP)
        mapped.pop('sinkhornIters')
        with unittest.mock.patch.dict(ced.FIELD_MAP, mapped, clear=True):
            f = judge_coverage(dsv41_model(), samples_fixture())
        self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'))
        self.assertIn('FIELD_MAP does not map it', ' '.join(x.message for x in f))
        self.assertEqual(pinned, ced.CONFIG_PINNED_FIGURES, 'the patch must not mutate the table')

    def test_every_row_is_either_checked_or_named_with_its_reason(self):
        """The 5-of-8 scope is stated, not implied: no row may be silently unchecked."""
        _, samples = inventory(ROOT)
        ids = {r['id'] for r in samples['innovations']['dsv41']}
        covered = set(ced.CONFIG_PINNED_FIGURES) | {'fp4-kv'}
        self.assertEqual(ids - covered, set(ced.UNPINNED_COVERAGE_ROWS),
                         'a row is neither checked nor recorded with the check that holds it')
        for reason in ced.UNPINNED_COVERAGE_ROWS.values():
            self.assertRegex(reason, r'[A-Z][A-Z_0-9]{6,}', 'the reason must name the check')

    def test_each_pinned_row_fails_on_its_own_figure(self):
        """One mutation per newly covered row, so none of the three is decorative."""
        for kwargs, rid in (({'budget': 256}, 'hier-indexer'), ({'blocks': 1024}, 'hier-indexer'),
                            ({'sinkhorn': 10}, 'mhc'), ({'streams': 2}, 'mhc'),
                            ({'dspark_experts': 384}, 'dspark'), ({'dspark_block': 4}, 'dspark')):
            f = judge_coverage(dsv41_model(),
                               samples_fixture(innovations=innovation_rows_fixture(**kwargs)))
            self.assertIn('COVERAGE_DERIVED', codes(f, 'FAIL'), f'{rid} / {kwargs}')
            self.assertIn('COVERAGE_UNCITED', codes(f, 'PASS'), f'{rid} / {kwargs}')

    def test_the_pass_message_states_the_scope(self):
        """A reader of the report must be told how much was checked, not just that it passed."""
        _, samples = inventory(ROOT)
        models, _ = inventory(ROOT)
        passed = [x for x in judge_coverage(models, samples)
                  if x.code == 'COVERAGE_DERIVED' and x.status == 'PASS']
        self.assertEqual(len(passed), 1)
        self.assertIn('5 of 8 rows', passed[0].message)
        for rid in ced.UNPINNED_COVERAGE_ROWS:
            self.assertIn(rid, passed[0].message)

    def test_the_check_reuses_the_kv_arithmetic_the_ledger_is_held_to(self):
        """Not a second derivation of the same quantity: the same function. If it were a copy,
        the exposition and the ledger could agree with each other and both be wrong."""
        import inspect
        src = inspect.getsource(ced._coverage_expectations)
        self.assertIn('derived_main_kv_bytes_per_token', src)


class SoftwareGateStatusTests(unittest.TestCase):
    """`aggregate` turns one BLOCKED into exit 2, and `deploy` needs `schematics`. So no
    reachable branch of any judge wired into that gate may return BLOCKED -- not on absent
    evidence, and not on a data condition either. Ruling T789b: this is exercised, not asserted.
    """

    def sample_variants(self):
        """Including the degenerate inventories, which are where BLOCKED used to come from."""
        broken = samples_fixture()
        broken['partitions']['dsv41']['csa2-reuse'] = 29
        same_s = samples_fixture()
        same_s['logical_state_far']['dsv41']['S'] = same_s['logical_state']['dsv41']['S']
        return [samples_fixture(), None, {}, {'partitions': {}, 'logical_state': {}},
                broken, same_s, samples_fixture(main_per_token=640)]

    def test_no_schematics_judge_can_emit_blocked(self):
        """Iterates `SCHEMATICS_JUDGES`, the same object `schematics.run` iterates, so a judge
        newly wired into the gate comes under this test without anyone adding it here."""
        self.assertTrue(SCHEMATICS_JUDGES, 'an empty judge list would make this vacuous')
        seen = set()
        for judge in SCHEMATICS_JUDGES:
            for overrides in SCHEMATICS_MUTATIONS:
                models = dsv41_model(**overrides)
                for samples in self.sample_variants():
                    statuses = {f.status for f in judge(models, samples)}
                    self.assertEqual(statuses - {'PASS', 'FAIL'}, set(),
                                     f'{judge.__name__} on {overrides} produced {statuses}')
                    seen |= statuses
        self.assertEqual(seen, {'PASS', 'FAIL'}, 'the sweep must exercise both outcomes')

    def test_every_wired_model_judge_takes_the_shared_signature(self):
        """The single wiring list only works if every member is callable the same way."""
        for judge in SCHEMATICS_JUDGES:
            self.assertEqual(judge(dsv41_model(), None), judge(dsv41_model(), samples=None))

    def test_no_scene_judge_can_emit_blocked(self):
        """Iterates `SCENE_JUDGES`, likewise shared with the gate, condition and all."""
        self.assertTrue(SCENE_JUDGES)
        m = dsv41_fixture()['dsv41']
        scenes = [clean_ced_scene(), {'nodes': [], 'edges': [], 'groups': [], 'labels': []}]
        broken = clean_ced_scene()
        broken['nodes'] = [n for n in broken['nodes'] if n['id'] != 'enc-band-9-14']
        for node in broken['nodes']:
            if node['id'] == 'dec-full':
                node['sub'] = 'L21 writes ÷2 latents'
        scenes.append(broken)
        seen = set()
        for scene in scenes:
            for _applies, judge in SCENE_JUDGES:
                statuses = {f.status for f in judge(scene, m)}
                self.assertEqual(statuses - {'PASS', 'FAIL'}, set(), f'{judge.__name__}: {statuses}')
                seen |= statuses
        self.assertEqual(seen, {'PASS', 'FAIL'})

    def test_the_scene_conditions_are_the_gate_s_own(self):
        """The conditions travel with the judges, so the gate cannot call one on a lens the
        property test never exercises."""
        ced_expect, ced_state = dsv41_fixture()['dsv41'], {'stage': 'ced'}
        applied = [judge.__name__ for applies, judge in SCENE_JUDGES if applies(ced_expect, ced_state)]
        self.assertEqual(applied, ['judge_mhc_streams', 'judge_ced_topology'])
        none = [j.__name__ for applies, j in SCENE_JUDGES
                if applies({'residual': 'standard'}, {'stage': 'block'})]
        self.assertEqual(none, [])

    def test_the_module_has_no_schematics_domain_blocked_constructor(self):
        """Deleted rather than left unused: constructing one must take new code, not a keyword."""
        import inspect
        self.assertFalse(hasattr(ced, '_blocked'))
        f = ced._evidence_blocked('X', 'y', 'm', 'r')
        self.assertEqual((f.domain, f.status), ('evidence', 'BLOCKED'),
                         'the only BLOCKED constructor in the module is evidence-domain')
        # The keyword-sized route back: `_one` used to take status='FAIL' as a default, so any
        # caller could pass 'BLOCKED' and rebuild the hazard without touching a constructor.
        self.assertEqual(set(inspect.signature(ced._one).parameters),
                         {'out', 'code', 'subject', 'ok', 'passed', 'failed', 'remedy'})

    def test_the_gate_report_carries_only_pass_and_fail(self):
        report = schematics.run(ROOT)
        self.assertEqual(set(report['counts']) - {'PASS', 'FAIL'}, set(), report['counts'])
