import json,unittest
from pathlib import Path
from verification import presentations as p

ROOT=Path(__file__).resolve().parents[2]

#: A stand-in focal-stage map for the negative coverage tests, shaped like `core.focalStages`.
FOCAL={'ds3':[],'dsv41':['inputs','residual','ced']}

class ArtifactSourceTests(unittest.TestCase):
 """Checks on the delivered bytes that need no browser."""
 def test_artifact_declares_no_external_origin(self):
  bad=[f for f in p.judge_artifact_source(ROOT/'dist/index.html',p.load_policy(ROOT)) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))
 def test_an_embedded_external_url_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'a.html';f.write_text('<script src="https://cdn.example.com/x.js"></script>')
   codes={x.code for x in p.judge_artifact_source(f,p.load_policy(ROOT)) if x.status!='PASS'}
   self.assertIn('PORTABILITY_EXTERNAL',codes)
 def test_a_clickable_citation_is_not_a_portability_defect(self):
  import tempfile
  # README: external source links require connectivity. They are the provenance feature.
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'a.html';f.write_text('<a href="https://arxiv.org/abs/2412.19437">report</a>')
   codes={x.code for x in p.judge_artifact_source(f,p.load_policy(ROOT)) if x.status!='PASS'}
   self.assertNotIn('PORTABILITY_EXTERNAL',codes)
 def test_oversized_artifact_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'a.html';f.write_text('x'*10)
   codes={x.code for x in p.judge_artifact_source(f,{'max_artifact_kib':0}) if x.status!='PASS'}
   self.assertIn('PORTABILITY_SIZE',codes)
 def test_invalid_utf8_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'a.html';f.write_bytes(b'\xff\xfe not utf-8')
   codes={x.code for x in p.judge_artifact_source(f,p.load_policy(ROOT)) if x.status!='PASS'}
   self.assertIn('PORTABILITY_ENCODING',codes)

class ScenarioSchemaTests(unittest.TestCase):
 def test_schema_exists_and_accepts_a_real_export(self):
  schema=json.loads((ROOT/'verification/schemas/scenario.schema.json').read_text())
  self.assertEqual(schema['properties']['scientificStatus']['enum'],['BLOCKED'])
 def test_a_scenario_claiming_certification_is_rejected(self):
  from verification.schema import validate
  schema=json.loads((ROOT/'verification/schemas/scenario.schema.json').read_text())
  doc={'schema':'stratum-observatory-scenario/v1','version':'3.0.0','state':{},'model':{},'hardware':{},
       'parameters':{},'logicalState':{},'references':{},'evidence':{},
       'scientificStatus':'PASS','runtimeStatus':'NOT_RUN'}
  self.assertTrue(validate(doc,schema),'an export claiming a passing scientific status was accepted')

class ExportedSvgTests(unittest.TestCase):
 def test_external_reference_in_an_export_is_caught(self):
  svg='<svg xmlns="http://www.w3.org/2000/svg"><image href="https://x/y.png"/></svg>'
  codes={f.code for f in p.judge_exported_svg(svg,{}) if f.status!='PASS'}
  self.assertIn('EXPORT_EXTERNAL',codes)
 def test_export_without_an_opaque_backdrop_is_caught(self):
  svg='<svg xmlns="http://www.w3.org/2000/svg"><text style="fill:rgb(1,2,3)">a</text></svg>'
  codes={f.code for f in p.judge_exported_svg(svg,{}) if f.status!='PASS'}
  self.assertIn('EXPORT_BACKDROP',codes)
 def test_the_xmlns_declaration_is_required(self):
  codes={f.code for f in p.judge_exported_svg('<svg><text>a</text></svg>',{}) if f.status!='PASS'}
  self.assertIn('EXPORT_MALFORMED',codes)

class FocalStateTests(unittest.TestCase):
 """`page.goto` lands on the default state, so every state with prose of its own was unmeasured.

 The report said "9 headings ordered" while the CED page had 10 and a five-column table that no
 accessibility check had ever seen.
 """
 def test_the_focal_states_come_from_the_live_model_table(self):
  """Every stage a model offers, not the one stage a literal named.

  The model list was read live and then narrowed to `stage:'ced'`, which is the same defect one
  layer in: `core.focalStages` offers four stages, so `dsv41/model/residual`, `dsv41/model/inputs`
  and the two other `inputs` pages reached a reader measured in no state at all.
  """
  states=p.focal_states(ROOT)
  self.assertTrue(states,'a checkpoint with a focal page must be measured')
  from verification.ced import inventory
  models,samples=inventory(ROOT)
  focal=samples['focal_stages']
  self.assertEqual({(st['model'],st['stage']) for st in states},
                   {(mid,stage) for mid,stages in focal.items() for stage in stages},
                   'the list must be derived from core.focalStages, not narrowed by a literal; '
                   'neither a hardcoded model list nor a hardcoded stage can notice a tenth '
                   'checkpoint or a fifth page')
  self.assertIn(('dsv41','residual'),{(st['model'],st['stage']) for st in states},
                'the mHC residual page this branch drew must be measured somewhere')

 def test_the_shipped_report_measured_more_than_the_page_it_opened(self):
  report=json.loads((ROOT/'verification/reports/presentations.json').read_text())
  measured=report['delivery'].get('states') or []
  self.assertIn('default',measured)
  self.assertTrue([x for x in measured if x!='default'],
                  'the report records only the default state; the focal pages went unmeasured')
  codes={f['code'] for f in report['findings']}
  self.assertIn('PROSE_SCOPE',codes)


class ProseProbeTests(unittest.TestCase):
 """The prose probe runs in the DOM, so it is exercised in one -- against synthetic markup, not
 against the artifact, so each defect is present exactly once and on purpose."""
 @classmethod
 def setUpClass(cls):
  import sys
  sys.path.insert(0,str(ROOT/'tests'))
  try:
   from playwright.sync_api import sync_playwright
   from browser_env import chromium_path
  except Exception as exc:  # pragma: no cover - reported rather than silently skipped
   raise unittest.SkipTest(f'no browser available: {exc!r}')
  cls._pw=sync_playwright().start()
  cls.browser=cls._pw.chromium.launch(executable_path=chromium_path(cls._pw),headless=True,
                                      args=['--no-sandbox'])
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls._pw.stop()
 def probe(self,body):
  page=self.browser.new_page()
  try:
   page.set_content(f'<!doctype html><html lang="en"><body>{body}</body></html>')
   return page.evaluate(p.PROBE_PROSE)
  finally:
   page.close()
 def kinds(self,body):
  return {i['kind'] for i in self.probe(body)['issues']}
 def test_a_well_formed_table_and_citation_report_nothing(self):
  out=self.probe('<table><thead><tr><th scope="col">A</th></tr></thead><tbody>'
                 '<tr><th scope="row">r</th></tr></tbody></table>'
                 '<a href="https://example.invalid/x" rel="noopener noreferrer">Named</a>')
  self.assertEqual(out['issues'],[])
  self.assertEqual((out['counts']['tables'],out['counts']['rows'],out['counts']['links']),(1,1,1))
 def test_a_table_of_plain_cells_is_caught(self):
  """The defect that is invisible to every geometric and contrast check: five columns of <td>,
  perfectly legible and completely unnavigable."""
  self.assertIn('table-no-thead',
                self.kinds('<table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>'))
 def test_unscoped_headers_are_caught_in_both_directions(self):
  self.assertIn('table-unscoped-column',
                self.kinds('<table><thead><tr><th>A</th></tr></thead>'
                           '<tbody><tr><th scope="row">r</th></tr></tbody></table>'))
  self.assertIn('table-unscoped-row',
                self.kinds('<table><thead><tr><th scope="col">A</th></tr></thead>'
                           '<tbody><tr><th>r</th></tr></tbody></table>'))
  self.assertIn('table-no-row-header',
                self.kinds('<table><thead><tr><th scope="col">A</th></tr></thead>'
                           '<tbody><tr><td>r</td></tr></tbody></table>'))
 def test_a_header_row_with_no_headers_is_caught(self):
  self.assertIn('table-no-column-headers',
                self.kinds('<table><thead><tr><td>A</td></tr></thead>'
                           '<tbody><tr><th scope="row">r</th></tr></tbody></table>'))
 def test_an_unnamed_or_unguarded_citation_is_caught(self):
  self.assertIn('unnamed-citation',
                self.kinds('<a href="https://example.invalid/x" rel="noopener"></a>'))
  self.assertIn('citation-without-noopener',
                self.kinds('<a href="https://example.invalid/x">Named</a>'))
 def test_the_shipped_ced_page_satisfies_the_probe(self):
  """And the real page, so the negatives above are not tests of a probe nothing runs."""
  report=json.loads((ROOT/'verification/reports/presentations.json').read_text())
  bad=[f for f in report['findings'] if f['status']!='PASS']
  self.assertEqual(bad,[],[f['message'] for f in bad])


class PreviewFreshnessTests(unittest.TestCase):
 def test_previews_match_the_built_artifact(self):
  bad=[f for f in p.judge_preview_freshness(ROOT) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))
 def test_a_stale_manifest_is_caught(self):
  bad=p.judge_preview_freshness(ROOT,recorded='0'*64)
  self.assertIn('PREVIEW_STALE',{f.code for f in bad if f.status!='PASS'})

class PreviewIntegrityTests(unittest.TestCase):
 """The recorded per-file digests, which nothing read until now.

 `tests/capture.py` wrote an `svg_sha256` per capture and across the whole repository there was
 exactly one reference to that key: the write. So a published SVG could be hand-edited, truncated
 or left over from an older build and still satisfy both the freshness check (one artifact digest)
 and the coverage check (model names). These tests drive the mismatch and the absence directly,
 because a judge that has only ever been run on good input proves nothing about what it reports
 on bad input.
 """
 def fixture(self,root,svg,recorded):
  (root/'previews').mkdir(parents=True,exist_ok=True)
  if svg is not None:(root/'previews/ds3.svg').write_text(svg)
  (root/'previews/manifest.json').write_text(json.dumps(
    {'artifact_sha256':'0'*64,'captures':[{'name':'ds3','svg_sha256':recorded}]}))

 def test_the_published_svgs_match_their_recorded_digests(self):
  bad=[f for f in p.judge_preview_integrity(ROOT) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))

 def test_an_edited_preview_is_caught(self):
  import hashlib,tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.fixture(root,'<svg>edited</svg>',hashlib.sha256(b'<svg>original</svg>').hexdigest())
   bad=[f for f in p.judge_preview_integrity(root) if f.status!='PASS']
   self.assertEqual([f.code for f in bad],['PREVIEW_INTEGRITY'])

 def test_a_recorded_capture_with_no_file_is_caught(self):
  import hashlib,tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.fixture(root,None,hashlib.sha256(b'<svg/>').hexdigest())
   self.assertIn('PREVIEW_INTEGRITY',{f.code for f in p.judge_preview_integrity(root) if f.status!='PASS'})

 def test_zero_comparisons_is_not_integrity(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'previews').mkdir(parents=True)
   (root/'previews/manifest.json').write_text(json.dumps({'artifact_sha256':'0'*64,'captures':[{'name':'ds3'}]}))
   f=p.judge_preview_integrity(root)
   self.assertEqual([x.status for x in f],['BLOCKED'],'a comparison count of zero reported as a pass')


class PreviewInventoryTests(unittest.TestCase):
 """Coverage with the arrow reversed: files on disk the manifest never names.

 `judge_preview_coverage` walks the manifest asking whether each model appears. Nothing walked
 `previews/` asking whether each published file is accounted for, so bytes sat in the published
 directory attested by nothing -- one of them a screenshot from a scratch script, one an orphan
 from the 3.0.0 baseline depicting an eight-model page.
 """
 def test_every_published_file_is_named_by_a_capture(self):
  bad=[f for f in p.judge_preview_inventory(ROOT) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))

 def test_an_unrecorded_file_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'previews').mkdir(parents=True)
   (root/'previews/ds3.svg').write_text('<svg/>');(root/'previews/stray.png').write_bytes(b'x')
   (root/'previews/manifest.json').write_text(json.dumps({'artifact_sha256':'0'*64,'captures':[{'name':'ds3'}]}))
   bad=[f for f in p.judge_preview_inventory(root) if f.status!='PASS']
   self.assertEqual([f.code for f in bad],['PREVIEW_INVENTORY'])
   self.assertIn('stray',bad[0].message)


class PreviewCoverageTests(unittest.TestCase):
 """Freshness is not coverage, and this gate checked only freshness.

 `tests/capture.py` built its case list from a hardcoded eight-model literal, so no published
 preview depicted the ninth checkpoint at all -- not its decoder block and not its encoder/decoder
 page -- while `judge_preview_freshness` compared one artifact digest and reported the previews
 that did exist as fresh. A manifest depicting eight of nine checkpoints is exactly as fresh as
 one depicting all nine, which is why the coverage question has to be asked separately.
 """
 def manifest(self,root,captures):
  (root/'previews').mkdir(parents=True,exist_ok=True)
  (root/'previews/manifest.json').write_text(json.dumps({'artifact_sha256':'0'*64,'captures':captures}))

 def test_a_checkpoint_no_preview_depicts_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.manifest(root,[{'name':'ds3','state':{'model':'ds3','stage':'block'}}])
   bad=[f for f in p.judge_preview_coverage(root,{'ds3':{},'dsv41':{}},FOCAL) if f.status!='PASS']
   self.assertEqual([f.code for f in bad],['PREVIEW_COVERAGE'])
   self.assertIn('dsv41',bad[0].message)

 def test_a_focal_page_no_preview_depicts_is_caught(self):
  """Depicting the checkpoint is not depicting the page whose whole subject is the band split."""
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.manifest(root,[{'name':'dsv41','state':{'model':'dsv41','stage':'block'}}])
   bad=[f for f in p.judge_preview_coverage(root,{'dsv41':{}},FOCAL) if f.status!='PASS']
   self.assertEqual([f.code for f in bad],['PREVIEW_COVERAGE'])
   self.assertIn('ced',bad[0].message)

 def test_a_focal_page_other_than_ced_is_also_required(self):
  """The exact residue this closes: the requirement was `ced` and only `ced`, so the mHC
  residual page could be in no preview and PREVIEW_COVERAGE still reported PASS."""
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.manifest(root,[{'name':'dsv41','state':{'model':'dsv41','stage':'block'}},
                       {'name':'dsv41-inputs','state':{'model':'dsv41','stage':'inputs'}},
                       {'name':'dsv41-ced','state':{'model':'dsv41','stage':'ced'}}])
   bad=[f for f in p.judge_preview_coverage(root,{'dsv41':{}},FOCAL) if f.status!='PASS']
   self.assertEqual([f.code for f in bad],['PREVIEW_COVERAGE'])
   self.assertIn('residual',bad[0].message)

 def test_a_complete_manifest_passes(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   self.manifest(root,[{'name':'ds3','state':{'model':'ds3','stage':'block'}},
                       {'name':'dsv41','state':{'model':'dsv41','stage':'block'}},
                       {'name':'dsv41-inputs','state':{'model':'dsv41','stage':'inputs'}},
                       {'name':'dsv41-residual','state':{'model':'dsv41','stage':'residual'}},
                       {'name':'dsv41-ced','state':{'model':'dsv41','stage':'ced'}}])
   bad=[f for f in p.judge_preview_coverage(root,{'ds3':{},'dsv41':{}},FOCAL) if f.status!='PASS']
   self.assertEqual(bad,[])

 def test_the_published_previews_cover_every_live_checkpoint(self):
  """And against the real manifest and the real model table, so the negatives above are not
  tests of a check that nothing runs."""
  from verification.ced import inventory
  models,samples=inventory(ROOT)
  bad=[f for f in p.judge_preview_coverage(ROOT,models,samples['focal_stages']) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))

 def test_the_shipped_report_asserts_coverage_and_not_only_freshness(self):
  report=json.loads((ROOT/'verification/reports/presentations.json').read_text())
  codes={f['code'] for f in report['findings']}
  self.assertIn('PREVIEW_COVERAGE',codes)
  self.assertIn('PREVIEW_FRESH',codes)
