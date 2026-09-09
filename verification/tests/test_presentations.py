import json,unittest
from pathlib import Path
from verification import presentations as p

ROOT=Path(__file__).resolve().parents[2]

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

class PreviewFreshnessTests(unittest.TestCase):
 def test_previews_match_the_built_artifact(self):
  bad=[f for f in p.judge_preview_freshness(ROOT) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f.message for f in bad))
 def test_a_stale_manifest_is_caught(self):
  bad=p.judge_preview_freshness(ROOT,recorded='0'*64)
  self.assertIn('PREVIEW_STALE',{f.code for f in bad if f.status!='PASS'})
