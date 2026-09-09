import json,unittest
from pathlib import Path
from verification.schema import validate
class SchemaTests(unittest.TestCase):
 def test_wrong_type_is_rejected(self):self.assertTrue(validate(True,{'type':'integer'}))
 def test_unknown_fields_rejected(self):self.assertTrue(validate({'skip':True},{'type':'object','properties':{},'additionalProperties':False}))
 def test_null_pending_checksum_allowed(self):self.assertFalse(validate(None,{'type':['string','null'],'pattern':'^[a-f0-9]{64}$'}))
 def test_invalid_digest_rejected(self):self.assertTrue(validate('main',{'type':'string','pattern':'^[a-f0-9]{64}$'}))
 def test_required_and_min_items(self):self.assertTrue(validate([],{'type':'array','minItems':1}))
 def test_real_manifests_have_valid_shapes(self):
  root=Path(__file__).resolve().parents[2]
  for name,manifest in [('sources','sources'),('target','observatory')]:
   schema=json.loads((root/f'verification/schemas/{name}.schema.json').read_text());value=json.loads((root/f'verification/manifests/{manifest}.json').read_text())
   self.assertEqual(validate(value,schema),[])
