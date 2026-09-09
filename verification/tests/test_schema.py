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


class EvaluatorCoverageTests(unittest.TestCase):
 """A schema keyword that is silently ignored is worse than no schema at all."""
 def test_additional_properties_schema_is_enforced(self):
  schema={'type':'object','additionalProperties':{'type':'object','required':['value'],'properties':{'value':{'type':'string'}}}}
  self.assertEqual(validate({'a':{'value':'ok'}},schema),[])
  self.assertTrue(validate({'a':{'value':1}},schema),'value type was not checked')
  self.assertTrue(validate({'a':{}},schema),'required key was not checked')
  self.assertTrue(validate({'a':'not-an-object'},schema),'entry type was not checked')
 def test_anchored_patterns_reject_surrounding_text(self):
  # JSON Schema `pattern` is unanchored by design, so every pattern in
  # verification/schemas/ carries ^...$ and this locks that in: an unanchored
  # pattern here would silently accept a value that merely contains a match.
  schema={'type':'string','pattern':'^[0-9a-f]{64}$'}
  self.assertEqual(validate('a'*64,schema),[])
  self.assertTrue(validate('a'*64+' trailing',schema),'pattern was matched against a substring')
  self.assertTrue(validate('prefix '+'a'*64,schema))
 def test_unsupported_keywords_are_reported_not_ignored(self):
  # This evaluator is a subset. It must say so rather than quietly pass a schema
  # whose author believed the constraint was being applied.
  for keyword,value in [('oneOf',[{'type':'string'}]),('anyOf',[{'type':'string'}]),('allOf',[{'type':'string'}]),
                        ('not',{'type':'string'}),('patternProperties',{'^a':{'type':'string'}}),
                        ('minimum',1),('maximum',9),('uniqueItems',True)]:
   errors=validate({'a':1},{'type':'object',keyword:value})
   self.assertTrue(errors,f'{keyword} was silently ignored')
   self.assertIn('unsupported',errors[0])
