import copy,json,subprocess,tempfile,unittest
from pathlib import Path
from verification.evidence import loads
from verification.runner import audit,source_subject
from verification.model import aggregate,Finding,exit_code
class IntegrationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.target=loads((cls.root/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((cls.root/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=cls.root,capture_output=True,text=True,check=True).stdout)
 def run_audit(self,target=None,inv=None,profile='publication'):
  # Explicitly absent offline snapshots test the cold-start gate, not production truth.
  cat=copy.deepcopy(self.catalog)
  for s in cat['sources']:s['cache']=None;s['sha256']=None
  return audit(self.root,target or self.target,inv or self.inv,cat,profile=profile)
 def test_cold_cache_is_blocked_not_certified(self):
  r=self.run_audit();self.assertEqual(r['status'],'BLOCKED');self.assertFalse(r['runtime_certified']);self.assertEqual(r['coverage']['verified_sources'],0)
 def test_changed_live_core_field_fails(self):
  x=copy.deepcopy(self.inv);x['fields']['MODELS.kimi3.layers']=94
  r=self.run_audit(inv=x);self.assertEqual(r['status'],'FAIL');self.assertIn('CONSUMER_VALUE',{f['code'] for f in r['findings']})
 def test_new_scientific_field_cannot_hide(self):
  x=copy.deepcopy(self.inv);x['fields']['MODELS.kimi3.hidden_gate']=64
  self.assertIn('COVERAGE_MISSING',{f['code'] for f in self.run_audit(inv=x)['findings']})
 def test_scientific_field_cannot_be_presentation(self):
  t=copy.deepcopy(self.target)
  for c in t['claims']:
   if c.get('consumer')=='MODELS.kimi3.layers':c['classification']='presentation'
  self.assertIn('CLAIM_CLASS',{f['code'] for f in self.run_audit(target=t)['findings']})
 def test_removing_verifier_domain_is_detected(self):
  t=copy.deepcopy(self.target);t['contracts']=[c for c in t['contracts'] if c['domain']!='gpu']
  self.assertIn('DOMAIN_UNCOVERED',{f['code'] for f in self.run_audit(target=t)['findings']})
 def test_runtime_requires_each_registered_sku_and_fabric(self):
  t=copy.deepcopy(self.target);t['runtime_obligations']=[o for o in t['runtime_obligations'] if o.get('contract')!='gpu.B200']
  self.assertIn('RUNTIME_MATRIX',{f['code'] for f in self.run_audit(target=t,profile='runtime')['findings']})
 def test_unrun_gpu_is_not_pass(self):
  r=self.run_audit(profile='runtime');self.assertTrue(any(f['code']=='RUNTIME_NOT_RUN' for f in r['findings']))
 def test_upstream_pin_change_invalidates_review_subject(self):
  c=next(c for c in self.target['claims'] if c.get('consumer')=='MODELS.kimi3.d');sources={s['id']:s for s in self.catalog['sources']}
  before=source_subject(c,sources);sources=copy.deepcopy(sources);sources['hf-model']['revision']='0'*40
  self.assertNotEqual(before,source_subject(c,sources))
 def test_empty_and_unrun_never_return_success_exit(self):
  self.assertNotEqual(exit_code(aggregate([])),0)
  self.assertNotEqual(exit_code(aggregate([Finding('x','gpu','NOT_RUN','a','not run')])),0)
 def test_publication_workflow_cannot_deploy_on_engineering_alone(self):
  text=(self.root/'.github/workflows/pages.yml').read_text()
  self.assertIn('needs: [engineering, publication]',text)
  self.assertIn('run: npm run verify:science',text)
  self.assertNotIn('continue-on-error: true',text)
