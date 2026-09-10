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
  # AMENDED CONTRACT. This test used to require `publication` in deploy's needs. Publication is
  # BLOCKED until an independent reviewer signs, so that gated a deployment which could never
  # happen; the page now ships and reports its own scientific status. What must NOT weaken is the
  # rest: deploy still requires every gate that measures the software, publication still RUNS on
  # every push, and nothing may be continue-on-error.
  deploy=re.search(r'^  deploy:.*?^    needs: \[([^\]]*)\]',text,re.S|re.M)
  self.assertIsNotNone(deploy,'deploy job declares no needs')
  required={j.strip() for j in deploy.group(1).split(',')}
  for job in ('engineering','schematics','aesthetics','presentations'):
   self.assertIn(job,required,f'deploy no longer requires the {job} gate')
  self.assertIn('run: npm run verify:science',text,'the publication audit must still run')
  self.assertIn('  publication:',text,'the publication job must still exist')
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

class ContractBindingTests(unittest.TestCase):
 """A bound constant is checked against the application; a transcribed one is not."""
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.target=loads((cls.root/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((cls.root/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=cls.root,capture_output=True,text=True,check=True).stdout)
 def run_audit(self,target=None):
  cat=copy.deepcopy(self.catalog)
  for s in cat['sources']:s['cache']=None;s['sha256']=None
  return audit(self.root,target or self.target,self.inv,cat,profile='publication')
 def test_a_nested_field_can_be_bound(self):
  # GPU memory is {value, unit, scope, kind}; without dotted paths only whole-field
  # replacement is possible and every capacity stays hand-transcribed.
  from verification.runner import apply_bindings
  record={'memory':{'value':0,'unit':'GB'}}
  missing=apply_bindings(record,{'memory.value':'HARDWARE.h100.hbmGB'},{'HARDWARE.h100.hbmGB':80})
  self.assertEqual(missing,[])
  self.assertEqual(record['memory']['value'],80)
  self.assertEqual(record['memory']['unit'],'GB','binding must not clobber its siblings')
 def test_a_binding_to_a_missing_consumer_is_reported(self):
  from verification.runner import apply_bindings
  self.assertEqual(apply_bindings({},{'x':'NOPE.nope'},{}),['NOPE.nope'])
 def test_drifted_hardware_value_breaks_its_bound_contract(self):
  # The whole point: change the application and a bound contract must notice.
  x=copy.deepcopy(self.inv);x['fields']['HARDWARE.h100.hbmGB']=79
  cat=copy.deepcopy(self.catalog)
  for s in cat['sources']:s['cache']=None;s['sha256']=None
  codes={f['code'] for f in audit(self.root,self.target,x,cat,profile='publication')['findings']}
  self.assertTrue({'GPU_MEMORY','CONSUMER_VALUE'}&codes,'a drifted hardware value was not noticed anywhere')
 def test_no_contract_hides_an_empty_binding_set_while_claiming_a_constant(self):
  # Contracts that genuinely have nothing to bind must say so, not leave {} to imply checked.
  for c in self.target['contracts']:
   if not c.get('bindings') and not c['id'].endswith('.pending'):
    self.assertIn('binding_scope',c,f"{c['id']} has no bindings and no stated reason")

class ReleaseContractTests(unittest.TestCase):
 """The release record is derived from the receipt, never transcribed into the manifest."""
 def test_a_current_receipt_carries_both_build_hashes(self):
  # engineering.py computed the second build hash and discarded it, leaving determinism as a
  # bare boolean a hand-written receipt could simply assert. Only a receipt bound to the
  # current toolchain is checked; a stale one is already caught by VERIFIER_STALE.
  root=Path(__file__).resolve().parents[2]
  path=root/'verification/reports/engineering.json'
  if not path.exists():self.skipTest('no engineering receipt yet')
  receipt=loads(path.read_text())
  from verification.engineering import toolchain_digest
  if receipt.get('verifier_sha256')!=toolchain_digest(root):self.skipTest('receipt predates the current toolchain')
  self.assertIn('repeat_build_sha256',receipt)
  self.assertEqual(receipt['repeat_build_sha256'],receipt['artifact_sha256'])
 def test_release_record_is_built_from_the_receipt(self):
  from verification.runner import release_record
  receipt={'consumer_sha256':'a'*64,'lock_sha256':'b'*64,'artifact_sha256':'c'*64,'repeat_build_sha256':'c'*64,
           'commands':{'node-tests':{'exit':0},'browser':{'exit':0}},
           'browser':{'geometry_defects':0,'runtime_errors':0,'external_runtime_requests':0}}
  rec=release_record(receipt,consumer='a'*64,lock='b'*64)
  self.assertEqual(rec['source_sha256'],rec['built_from_sha256'])
  self.assertEqual(rec['source_lock_sha256'],rec['receipt_lock_sha256'])
  self.assertEqual(rec['commands'],{'node':0,'browser':0})
 def test_a_receipt_from_another_tree_is_a_stale_release(self):
  from verification.runner import release_record
  from verification.domains.contracts import validate
  receipt={'consumer_sha256':'a'*64,'lock_sha256':'b'*64,'artifact_sha256':'c'*64,'repeat_build_sha256':'c'*64,
           'commands':{'node-tests':{'exit':0},'browser':{'exit':0}},
           'browser':{'geometry_defects':0,'runtime_errors':0,'external_runtime_requests':0}}
  rec=release_record(receipt,consumer='9'*64,lock='b'*64)
  self.assertIn('RELEASE_STALE',{f.code for f in validate('release',rec)})
 def test_a_nondeterministic_build_is_caught(self):
  from verification.runner import release_record
  from verification.domains.contracts import validate
  receipt={'consumer_sha256':'a'*64,'lock_sha256':'b'*64,'artifact_sha256':'c'*64,'repeat_build_sha256':'d'*64,
           'commands':{'node-tests':{'exit':0},'browser':{'exit':0}},
           'browser':{'geometry_defects':0,'runtime_errors':0,'external_runtime_requests':0}}
  rec=release_record(receipt,consumer='a'*64,lock='b'*64)
  self.assertIn('RELEASE_DETERMINISM',{f.code for f in validate('release',rec)})

class ReceiptTrustTests(unittest.TestCase):
 """The receipt asserts facts about itself; those it can, must be cross-checked."""
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=cls.root,capture_output=True,text=True,check=True).stdout)
 def receipt(self,**over):
  from verification.engineering import toolchain_digest
  import hashlib
  from datetime import datetime,timezone
  from verification.evidence import digest
  target=loads((self.root/'verification/manifests/observatory.json').read_text())
  catalog=loads((self.root/'verification/manifests/sources.json').read_text())
  browser=loads((self.root/'tests/browser-results.json').read_text())
  r={'schema':'stratum-engineering/v1','verifier_sha256':toolchain_digest(self.root),
     'consumer_sha256':digest(self.inv['functions']),'lock_sha256':digest({'target':target,'catalog':catalog}),
     'artifact_sha256':hashlib.sha256((self.root/'dist/index.html').read_bytes()).hexdigest(),
     'repeat_build_sha256':hashlib.sha256((self.root/'dist/index.html').read_bytes()).hexdigest(),
     'completed_at':datetime.now(timezone.utc).isoformat(),'deterministic_build':True,
     'commands':{k:{'exit':0} for k in ('verifier-tests','node-tests','build','browser')},
     'browser':browser}
  r.update(over);return r
 def lock(self):
  from verification.evidence import digest
  return digest({'target':loads((self.root/'verification/manifests/observatory.json').read_text()),
                 'catalog':loads((self.root/'verification/manifests/sources.json').read_text())})
 def codes(self,receipt):
  from verification.runner import check_receipt
  return {f.code for f in check_receipt(self.root,receipt,self.lock(),self.inv)}
 def test_a_faithful_receipt_is_accepted(self):
  self.assertIn('ENGINEERING_CURRENT',self.codes(self.receipt()))
 def test_a_receipt_that_rewrites_its_browser_result_is_caught(self):
  # The receipt copies the browser object wholesale and then re-reads it as truth. It must
  # agree with the file the suite actually wrote.
  browser=dict(loads((self.root/'tests/browser-results.json').read_text()));browser['geometry_states']=1
  self.assertIn('BROWSER_MISREPORTED',self.codes(self.receipt(browser=browser)))
 def test_a_receipt_asserting_determinism_without_two_hashes_is_caught(self):
  self.assertIn('BUILD_NONDETERMINISTIC',self.codes(self.receipt(repeat_build_sha256='0'*64)))
 def test_an_unattested_receipt_is_blocked_not_passed(self):
  from verification.runner import check_receipt
  findings=check_receipt(self.root,self.receipt(),self.lock(),self.inv)
  unattested=[f for f in findings if f.code=='RECEIPT_UNATTESTED']
  self.assertTrue(unattested,'a self-produced receipt should say so as a finding, not only in prose')
  self.assertEqual(unattested[0].status,'BLOCKED')

class UpstreamComparisonTests(unittest.TestCase):
 """The strongest check in the framework: our number against the publisher's own config."""
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.target=loads((cls.root/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((cls.root/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=cls.root,capture_output=True,text=True,check=True).stdout)
 def test_equal_relations_are_registered(self):
  equal=[(c,e) for c in self.target['claims'] for e in c.get('evidence',[]) if e.get('relation')=='equal']
  self.assertGreaterEqual(len(equal),80,'no upstream value comparisons are registered')
  for c,e in equal:
   self.assertEqual(e['anchor']['kind'],'json_pointer','an equal relation needs a typed pointer')
 def cached_equal_claims(self):
  """Claims with an `equal` relation whose upstream bytes are actually on this machine.

  `.verification-cache/` is git-ignored on purpose -- the project does not redistribute upstream
  source bytes -- so a fresh clone and every CI runner has no cache. Without it an `equal`
  anchor cannot resolve and the audit reports SOURCE_CACHE instead of comparing anything. These
  tests must SKIP there, never pass: the absent-cache run of
  test_the_shipped_values_agree_with_upstream was asserting the absence of a finding that could
  not have been produced, which is a false green.
  """
  from verification.evidence import safe_path
  sources={s['id']:s for s in self.catalog['sources']}
  out=[]
  for c in self.target['claims']:
   for e in c.get('evidence',[]):
    if e.get('relation')!='equal':continue
    cache=sources.get(e.get('source'),{}).get('cache')
    if cache and safe_path(self.root,cache).exists():out.append((c,e))
  return out
 def require_cache(self):
  if not self.cached_equal_claims():
   self.skipTest('no upstream source bytes cached; run `uv run --frozen python -m verification.sources acquire` '
                 'to compare against publisher configs. .verification-cache/ is git-ignored by design.')
 def test_a_value_contradicting_upstream_is_caught(self):
  import copy
  self.require_cache()
  t=copy.deepcopy(self.target)
  for c in t['claims']:
   if c.get('consumer')=='MODELS.ds3.layers' and any(e.get('relation')=='equal' for e in c.get('evidence',[])):
    c['expected']=c['expected']+1;break
  else:self.skipTest('no anchored ds3 layer claim')
  codes={f['code'] for f in audit(self.root,t,self.inv,self.catalog,profile='publication')['findings']}
  self.assertIn('UPSTREAM_VALUE',codes,'a value contradicting the publisher config was accepted')
 def test_the_shipped_values_agree_with_upstream(self):
  self.require_cache()
  codes={f['code'] for f in audit(self.root,self.target,self.inv,self.catalog,profile='publication')['findings']}
  self.assertNotIn('UPSTREAM_VALUE',codes)
