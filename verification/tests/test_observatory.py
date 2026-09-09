import json,subprocess,unittest
from pathlib import Path
from verification.domains.contracts import validate
from verification.evidence import check_source,check_coverage
ROOT=Path(__file__).resolve().parents[2]
class ObservatoryTests(unittest.TestCase):
 def dense(self):return {'layers':2,'layer_groups':{'gqa':[1,2]},'experts':0,'top_k':0,'shared':0,'d':4096,'latent':4096,'heads':32,'kv_heads':8,'head_dim':128,'index_base':1}
 def test_dense_model_is_not_forced_to_invent_experts(self):
  self.assertEqual([f for f in validate('model',self.dense()) if f.status!='PASS'],[])
 def test_dense_with_routed_topk_is_rejected(self):
  x=self.dense();x['top_k']=1
  self.assertIn('MODEL_ROUTING',[f.code for f in validate('model',x)])
 def qwen_reference(self,role):
  return {'id':'qwen-ref','role':role,'kind':'git','provider':'github','repo':'huggingface/transformers','path':'src/transformers/models/qwen3/modeling_qwen3.py','revision':'c93057d4835cd31752bb56f59989dd27696eb45b','url':'https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3/modeling_qwen3.py','cache':None,'sha256':None,'license':'No source redistributed.'}
 def test_registered_qwen_reference_is_code_not_a_config_substitute(self):
  # A real code reference is authorized evidence; only its bytes are missing.
  findings=check_source(self.qwen_reference('framework_code'),ROOT)
  self.assertNotIn('SOURCE_AUTHORITY',[f.code for f in findings]);self.assertIn('SOURCE_UNFETCHED',[f.code for f in findings])
 def test_qwen_has_no_publisher_implementation_to_register(self):
  # Qwen does not publish modeling code for these presets. transformers reimplements
  # it, so registering transformers as the publisher would let one third-party
  # implementation corroborate another and satisfy MODEL_DUAL_CODE.
  self.assertIn('SOURCE_AUTHORITY',[f.code for f in check_source(self.qwen_reference('publisher_code'),ROOT)])
 def test_live_inventory_covers_new_renderer_modules_and_all_presets(self):
  r=subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)
  inv=json.loads(r.stdout)
  self.assertEqual(len(inv['models']),8)
  self.assertEqual(inv['samples']['decoder_layers'],504)
  for path in ['src/models.mjs','src/diagram-model.mjs','src/diagram-systems.mjs','src/diagram-meta.mjs']:
   self.assertIn('file.'+path,inv['functions'])
  manifest=json.loads((ROOT/'verification/manifests/observatory.json').read_text())
  self.assertEqual(check_coverage(inv,manifest['claims']),[])
