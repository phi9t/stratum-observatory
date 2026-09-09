import copy,json,re,subprocess,tempfile,unittest
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
 def test_drifted_presentation_claim_is_not_reported_as_verified(self):
  # Presentation claims skip review, so they are the only claims that can reach PASS --
  # and therefore the only place a bypassed coverage finding is observable.
  presentation=[c for c in self.target['claims'] if c.get('classification')=='presentation' and c.get('consumer','').startswith('file.')]
  self.assertTrue(presentation,'expected at least one presentation file claim')
  consumer=presentation[0]['consumer']
  x=copy.deepcopy(self.inv);x['functions'][consumer]={'sha256':'0'*64}
  r=self.run_audit(inv=x)
  verified={f['subject'] for f in r['findings'] if f['code']=='CLAIM_VERIFIED'}
  self.assertIn('CONSUMER_VALUE',{f['code'] for f in r['findings']})
  self.assertNotIn(presentation[0]['id'],verified,'a claim whose bound value drifted was still reported verified')
  drifted={f['subject'] for f in r['findings'] if f['code']=='CONSUMER_VALUE'}
  self.assertIn(presentation[0]['id'],drifted,'the drift was not attributed to the claim that owns it')
 def test_claim_with_missing_dependency_is_not_reported_as_verified(self):
  t=copy.deepcopy(self.target)
  presentation=next(c for c in t['claims'] if c.get('classification')=='presentation' and c.get('consumer','').startswith('file.'))
  presentation['dependencies']=['does.not.exist']
  r=self.run_audit(target=t)
  verified={f['subject'] for f in r['findings'] if f['code']=='CLAIM_VERIFIED'}
  self.assertIn('CLAIM_DEPENDENCY',{f['code'] for f in r['findings']})
  self.assertNotIn(presentation['id'],verified)
 def test_no_claim_is_exempt_from_review(self):
  # Presentation claims used to skip review entirely, so an entire stylesheet and the
  # HTML entry point were outside the gate -- and were the only two PASSing claims.
  r=self.run_audit()
  self.assertEqual(r['coverage']['passed_presentation_bindings'],0)
  required={f['subject'] for f in r['findings'] if f['code']=='REVIEW_REQUIRED'}
  for cid in ('file.src/styles.css','file.index.html'):
   self.assertIn(cid,required,f'{cid} is still review-exempt')
  self.assertFalse([f for f in r['findings'] if f['code']=='CLAIM_VERIFIED'])
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
  # Deploy must depend on publication, not on the software gate alone. Extra required
  # jobs are welcome; a missing publication dependency is the failure mode.
  deploy=re.search(r'^  deploy:.*?^    needs: \[([^\]]*)\]',text,re.S|re.M)
  self.assertIsNotNone(deploy,'deploy job declares no needs')
  required={j.strip() for j in deploy.group(1).split(',')}
  self.assertIn('publication',required)
  self.assertIn('engineering',required)
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

class ReviewSubjectTests(unittest.TestCase):
 """A review must bind source identity, not the bookkeeping of when it was fetched."""
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.claim={'id':'c','consumer':'x','expected':1,'evidence':[{'source':'s','relation':'documents'}]}
  cls.sources={'s':{'id':'s','kind':'git','role':'publisher_config','provider':'hf','repo':'a/b','path':'config.json',
                    'revision':'0'*40,'url':'https://huggingface.co/a/b/raw/'+'0'*40+'/config.json',
                    'sha256':'1'*64,'cache':'.verification-cache/sha256/'+'1'*64+'.txt',
                    'license':'apache-2.0','retrieved_at':'2026-09-06T00:00:00+00:00','etag':'W/"v1"','version':None,'fixture':False}}
 def subject(self,**overrides):
  s=copy.deepcopy(self.sources);s['s'].update(overrides)
  return source_subject(self.claim,s)
 def test_refetch_metadata_does_not_invalidate_review(self):
  # Re-running acquire on byte-identical upstream must not force every review to be re-signed.
  self.assertEqual(self.subject(),self.subject(retrieved_at='2026-09-08T00:00:00+00:00',etag='W/"v2"'))
 def test_changed_bytes_invalidate_review(self):
  self.assertNotEqual(self.subject(),self.subject(sha256='2'*64))
 def test_changed_revision_invalidates_review(self):
  self.assertNotEqual(self.subject(),self.subject(revision='1'*40))
 def test_changed_role_invalidates_review(self):
  self.assertNotEqual(self.subject(),self.subject(role='framework_code'))
 def test_identity_projection_classifies_every_schema_field(self):
  # Schema growth must force a conscious identity/volatile decision, never silent exclusion.
  from verification.runner import SOURCE_IDENTITY,SOURCE_VOLATILE
  schema=loads((self.root/'verification/schemas/sources.schema.json').read_text())
  declared=set(schema['properties']['sources']['items']['properties'])
  self.assertEqual(SOURCE_IDENTITY|SOURCE_VOLATILE,declared)
  self.assertEqual(SOURCE_IDENTITY&SOURCE_VOLATILE,set())
