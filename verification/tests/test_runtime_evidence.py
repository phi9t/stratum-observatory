import json,unittest
from pathlib import Path
from verification.schema import validate as shape_validate
from verification.domains.contracts import validate as validate_domain

ROOT=Path(__file__).resolve().parents[2]

class RuntimeEvidenceTests(unittest.TestCase):
 """Receipts built from fleet probes must satisfy the schema and the domain validator."""
 @classmethod
 def setUpClass(cls):
  cls.candidate=ROOT/'verification/candidates/runtime.json'
  cls.schema=json.loads((ROOT/'verification/schemas/runtime.schema.json').read_text())
 def receipts(self):
  if not self.candidate.exists():self.skipTest('no runtime candidate built yet')
  return json.loads(self.candidate.read_text())
 def test_candidate_matches_the_runtime_schema(self):
  errors=shape_validate(self.receipts(),self.schema)
  self.assertEqual(errors,[],'\n'.join(errors[:8]))
 def test_every_receipt_record_satisfies_its_domain(self):
  for oid,r in self.receipts().items():
   domain=oid.split('.')[1] if oid.startswith('runtime.') else 'gpu'
   domain='gpu' if domain=='gpu' else domain
   bad=[f for f in validate_domain('gpu' if 'gpu' in oid else 'scale_out',r['record']) if f.status!='PASS']
   self.assertEqual(bad,[],f'{oid}: '+'; '.join(f.message for f in bad))
 def test_every_environment_fingerprint_names_a_checksummed_artifact(self):
  for oid,r in self.receipts().items():
   paths={a['path'] for a in r['artifacts']}
   for key,fp in r['environment'].items():
    self.assertTrue(fp.get('value'),f'{oid}.{key} has no value')
    if fp.get('method')=='probed':
     self.assertIn(fp.get('source_artifact'),paths,f'{oid}.{key} claims probed but names no artifact')
 def test_artifact_digests_match_the_files_on_disk(self):
  import hashlib
  for oid,r in self.receipts().items():
   for a in r['artifacts']:
    p=ROOT/a['path']
    self.assertTrue(p.exists(),f'{oid}: missing artifact {a["path"]}')
    self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),a['sha256'],f'{oid}: digest drift in {a["path"]}')
 def test_the_normalized_record_is_reproducible_from_a_checksummed_file(self):
  from verification.evidence import canonical,pointer,loads
  for oid,r in self.receipts().items():
   binding=r['record_artifact']
   doc=loads((ROOT/binding['path']).read_text())
   self.assertEqual(canonical(pointer(doc,binding['pointer'])),canonical(r['record']),
                    f'{oid}: summary is not derivable from its raw record artifact')
