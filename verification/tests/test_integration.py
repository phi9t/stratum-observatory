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

class ToolchainDigestTests(unittest.TestCase):
 """The receipt must bind every file that decides what the tests actually did."""
 @classmethod
 def setUpClass(cls):cls.root=Path(__file__).resolve().parents[2]
 def digest_with(self,relative,content):
  """Copy the tree, change one file, and report whether the toolchain digest moved."""
  from verification.engineering import toolchain_digest
  before=toolchain_digest(self.root)
  with tempfile.TemporaryDirectory() as tmp:
   copy_root=Path(tmp)/'repo'
   subprocess.run(['cp','-R',str(self.root),str(copy_root)],check=True,capture_output=True)
   target=copy_root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
   return before,toolchain_digest(copy_root)
 def assert_covered(self,relative,content):
  before,after=self.digest_with(relative,content)
  self.assertNotEqual(before,after,f'{relative} is outside toolchain_digest; a receipt would survive changing it.')
 def test_python_lock_is_covered(self):self.assert_covered('uv.lock','# tampered\n')
 def test_pyproject_is_covered(self):self.assert_covered('pyproject.toml','[project]\nname = "tampered"\n')
 def test_python_version_pin_is_covered(self):self.assert_covered('.python-version','3.9\n')
 def test_browser_suite_is_covered(self):
  # This file produces the browser evidence the receipt attests to.
  self.assert_covered('tests/browser.test.py','# weakened\n')
 def test_codeowners_is_covered(self):self.assert_covered('.github/CODEOWNERS','* @nobody\n')
 def test_yaml_workflows_are_covered(self):self.assert_covered('.github/workflows/extra.yaml','on: push\n')

class ReceiptCountTests(unittest.TestCase):
 """A receipt that cannot count its own tests is not evidence that they ran."""
 def test_counts_are_scraped_from_both_node_reporter_formats(self):
  from verification.engineering import scrape_count
  # node 22 wrote "# tests 40"; node 24 writes "ℹ tests 40". Both must parse.
  self.assertEqual(scrape_count('# tests 40\n# pass 40\n',r'tests (\d+)'),40)
  self.assertEqual(scrape_count('ℹ tests 40\nℹ pass 40\n',r'tests (\d+)'),40)
  self.assertEqual(scrape_count('Ran 125 tests in 4s\n',r'Ran (\d+) tests'),125)
 def test_unparsable_output_returns_none_so_the_run_can_refuse(self):
  from verification.engineering import scrape_count
  self.assertIsNone(scrape_count('no counts here\n',r'tests (\d+)'))
