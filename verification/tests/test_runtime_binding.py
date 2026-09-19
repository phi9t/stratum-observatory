"""What a runtime receipt is allowed to be retired by.

A receipt for a hardware-attribute obligation (`gpu`, `scale_up`, `scale_out`) attests what a
machine or a fabric IS. An H100 holding 80 GB of HBM does not depend on our claim set, on our
source files, or on how a diagram labels a wire, so such a receipt must not bind `lock_sha256`
or `consumer_sha256`. It binds the data model it was captured under, the contract record it
vouches for, and the attested hardware state it rests on.

Every other obligation attests what our own code DOES, and keeps binding the lock and the
consumer inventory exactly as before. The tests below are written in both directions on
purpose: the relaxation is only safe if it cannot leak into the behavioural set.
"""
import copy,subprocess,unittest,json,tempfile,hashlib
from unittest.mock import patch
from datetime import datetime,timezone
from pathlib import Path

from verification import hardware
from verification.evidence import loads
from verification.runner import audit,contract_sha256
from verification.tests.test_hardware import probe,FIXTURES

ROOT=Path(__file__).resolve().parents[2]
NOW=datetime.now(timezone.utc).isoformat()


class RuntimeBindingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.directory=tempfile.TemporaryDirectory(prefix="stratum-runtime-test-")
  cls.addClassCleanup(cls.directory.cleanup)
  cls.root=Path(cls.directory.name)
  cls.target=loads((ROOT/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((ROOT/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  # Synthetic captures live only in a temporary test root, never in accepted evidence.
  captured=probe.build_record(FIXTURES/'probe-baseline')
  dest=cls.root/cls.HARDWARE_ARTIFACT;dest.parent.mkdir(parents=True)
  dest.write_text(json.dumps(captured))
  contract=next(c for c in cls.target['contracts'] if c['id']=='gpu.H100-SXM')
  record=copy.deepcopy(contract['record'])
  record_path=dest.with_name('record.json');record_path.write_text(json.dumps(record))
  obligation=next(o for o in cls.target['runtime_obligations'] if o['id']=='runtime.gpu.H100-SXM')
  cls.accepted={'runtime.gpu.H100-SXM':{
   'executed_at':NOW,'exit_code':0,'lock_sha256':'0'*64,'consumer_sha256':'1'*64,
   'record':record,'record_artifact':{'path':str(record_path.relative_to(cls.root)),'pointer':''},
   'environment':{k:{'value':'synthetic-test-value','method':'fixture'} for k in obligation['environment']},
   'artifacts':[{'path':str(p.relative_to(cls.root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (dest,record_path)]}}
  # Only this isolated audit accepts fixture-marked captures. The production loader still
  # rejects them, covered independently by FixtureBoundaryTests.
  loader=patch('verification.runner.load_hardware',side_effect=lambda p:hardware.load(p,allow_fixtures=True))
  loader.start();cls.addClassCleanup(loader.stop)

 # -- helpers ------------------------------------------------------------------------------
 def findings(self,runtime,*,target=None,inv=None,subject=None):
  report=audit(self.root,target or self.target,inv or self.inv,self.catalog,profile='runtime',runtime=runtime)
  return [f for f in report['findings'] if subject is None or f['subject']==subject]

 def codes(self,runtime,**kw):
  return {f['code'] for f in self.findings(runtime,**kw)}

 def contract(self,target,cid):
  return next(c for c in target['contracts'] if c['id']==cid)

 HARDWARE_ARTIFACT='synthetic/hardware.json'

 def hardware_receipt(self,*,target=None,inv=None,**over):
  """Bind a synthetic test receipt to the current contract and fixture bytes."""
  target=target or self.target;inv=inv or self.inv
  r=copy.deepcopy(self.accepted['runtime.gpu.H100-SXM'])
  r['probe_schema_sha256']=hardware.schema_sha256()
  r['contract_sha256']=contract_sha256(self.contract(target,'gpu.H100-SXM'),inv.get('fields',{}))
  r['hardware_artifact']={'path':self.HARDWARE_ARTIFACT}
  r['hardware_sha256']=hardware.load(self.root/self.HARDWARE_ARTIFACT,allow_fixtures=True)['hardware_sha256']
  r.update(over)
  return {'runtime.gpu.H100-SXM':r}

 def minimal(self,oid,**over):
  r={'executed_at':NOW,'exit_code':0,'lock_sha256':'0'*64,'consumer_sha256':'1'*64,
     'environment':{},'record':{},'artifacts':[],'record_artifact':{'path':'x','pointer':''}}
  r.update(over)
  return {oid:r}

 def edited_manifest(self):
  """A claim-manifest edit that touches no contract -- the cosmetic change that kept
  retiring GPU receipts."""
  t=copy.deepcopy(self.target)
  for c in t['claims']:
   if c.get('classification')=='presentation':
    c['statement']=c['statement']+' (wording)';break
  else:self.fail('no presentation claim to edit')
  return t

 # -- 1. the entire point ------------------------------------------------------------------
 def test_a_hardware_receipt_survives_a_claim_manifest_edit(self):
  t=self.edited_manifest()
  codes=self.codes(self.hardware_receipt(target=t),target=t,subject='runtime.gpu.H100-SXM')
  self.assertNotIn('RUNTIME_STALE',codes,'a claim edit retired a hardware-attribute receipt')

 def test_a_hardware_receipt_survives_a_source_file_edit(self):
  """Relabelling a wire changes the function/file digests. It is not a hardware fact."""
  inv=copy.deepcopy(self.inv)
  inv['functions']['file.src/diagram-model.mjs']={'sha256':'f'*64}
  codes=self.codes(self.hardware_receipt(inv=inv),inv=inv,subject='runtime.gpu.H100-SXM')
  self.assertNotIn('RUNTIME_STALE',codes,'a source edit retired a hardware-attribute receipt')

 def test_a_completed_hardware_receipt_raises_no_binding_finding_at_all(self):
  """Survival is not merely the absence of RUNTIME_STALE: nothing about the binding may
  block or fail either, or the receipt is retired under a different name."""
  t=self.edited_manifest()
  bad={f['code'] for f in self.findings(self.hardware_receipt(target=t),target=t,subject='runtime.gpu.H100-SXM')
       if f['status'] not in ('PASS',)}
  self.assertEqual(bad-{'REVIEW_REQUIRED'},set(),'a complete hardware receipt still reports unresolved findings')

 # -- 2. what MUST still retire it ---------------------------------------------------------
 def test_a_changed_contract_record_fails_the_receipt(self):
  t=copy.deepcopy(self.target)
  # Not a bound field: the manifest itself now claims a measured capacity where the receipt
  # attested a nominal one.
  self.contract(t,'gpu.H100-SXM')['record']['memory']['kind']='allocatable_measured'
  findings=self.findings(self.hardware_receipt(),target=t,subject='runtime.gpu.H100-SXM')
  drift=[f for f in findings if f['code']=='RUNTIME_CONTRACT_DRIFT']
  self.assertTrue(drift,'a contract record changed under the receipt and the receipt still vouched for it')
  self.assertEqual(drift[0]['status'],'FAIL','contract drift is a contradiction, not missing evidence')

 def test_a_moved_bound_core_value_fails_the_receipt(self):
  """The digest covers the record the gate judges, bindings applied. Changing the HBM
  capacity in src/models.mjs changes what the contract asserts, so the probe that never saw
  that number must stop vouching for it."""
  inv=copy.deepcopy(self.inv)
  inv['fields']['HARDWARE.h100.hbmGB']=96
  codes=self.codes(self.hardware_receipt(),inv=inv,subject='runtime.gpu.H100-SXM')
  self.assertIn('RUNTIME_CONTRACT_DRIFT',codes)

 def test_a_superseded_data_model_blocks_and_names_the_re_probe(self):
  findings=self.findings(self.hardware_receipt(probe_schema_sha256='b'*64),subject='runtime.gpu.H100-SXM')
  schema=[f for f in findings if f['code']=='RUNTIME_SCHEMA']
  self.assertTrue(schema,'a receipt captured under a superseded hardware model was accepted')
  self.assertEqual(schema[0]['status'],'BLOCKED','a superseded data model is missing evidence, not a contradiction')
  self.assertRegex((schema[0]['message']+' '+schema[0]['remedy']).lower(),r'probe')

 def test_a_receipt_without_the_new_fields_is_blocked_not_passed_and_not_failed(self):
  """A receipt captured before this binding existed. That is missing evidence: it must block,
  and the gate must not crash on it or wave it through."""
  r=self.hardware_receipt()
  for field in ('probe_schema_sha256','contract_sha256','hardware_sha256','hardware_artifact'):
   r['runtime.gpu.H100-SXM'].pop(field,None)
  findings=self.findings(r,subject='runtime.gpu.H100-SXM')
  codes={f['code'] for f in findings}
  self.assertIn('RUNTIME_SCHEMA',codes)
  self.assertIn('RUNTIME_HARDWARE',codes)
  self.assertEqual([f['code'] for f in findings if f['status']=='FAIL'],[])

 def test_a_receipt_naming_no_captured_hardware_state_is_blocked(self):
  """A receipt with no attested-state digest names nothing to re-derive, which leaves a reader
  unable to say which hardware state it rests on. That is missing evidence, not a contradiction."""
  r=self.hardware_receipt();r['runtime.gpu.H100-SXM'].pop('hardware_sha256')
  named=[f for f in self.findings(r,subject='runtime.gpu.H100-SXM')
         if f['code']=='RUNTIME_HARDWARE' and 'hardware_sha256' in f['message']]
  self.assertTrue(named,'a hardware receipt with no attested-state digest was accepted')
  self.assertEqual(named[0]['status'],'BLOCKED')

 def test_a_receipt_naming_no_contract_record_is_blocked(self):
  r=self.hardware_receipt();r['runtime.gpu.H100-SXM'].pop('contract_sha256')
  named=[f for f in self.findings(r,subject='runtime.gpu.H100-SXM')
         if f['code']=='RUNTIME_HARDWARE' and 'contract_sha256' in f['message']]
  self.assertTrue(named,'a receipt that names no contract record was allowed to vouch for one')
  self.assertEqual(named[0]['status'],'BLOCKED')

 def test_a_hand_written_hardware_digest_is_caught(self):
  """`hardware_sha256` is re-derived from the capture the receipt names, not merely carried.
  A digest typed in by hand -- or kept after the capture was replaced -- is a contradiction:
  the receipt vouches for an attested state its own evidence does not produce."""
  drift=[f for f in self.findings(self.hardware_receipt(hardware_sha256='a'*64),subject='runtime.gpu.H100-SXM')
         if f['code']=='RUNTIME_HARDWARE_DRIFT']
  self.assertTrue(drift,'a receipt naming an attested state its capture does not produce was accepted')
  self.assertEqual(drift[0]['status'],'FAIL','a digest that does not re-derive is a contradiction')

 def test_a_receipt_naming_no_hardware_record_cannot_have_its_digest_checked(self):
  r=self.hardware_receipt();r['runtime.gpu.H100-SXM'].pop('hardware_artifact')
  named=[f for f in self.findings(r,subject='runtime.gpu.H100-SXM')
         if f['code']=='RUNTIME_HARDWARE' and 'hardware_artifact' in f['message']]
  self.assertTrue(named,'a carried digest with no capture behind it was accepted')
  self.assertEqual(named[0]['status'],'BLOCKED')

 def test_a_hardware_record_the_receipt_does_not_checksum_is_blocked(self):
  """Re-deriving from bytes is only worth anything if those bytes are pinned by the receipt."""
  r=self.hardware_receipt(hardware_artifact={'path':'evidence/runtime/hosts.json'})
  named=[f for f in self.findings(r,subject='runtime.gpu.H100-SXM')
         if f['code']=='RUNTIME_HARDWARE' and 'checksum' in f['message']]
  self.assertTrue(named,'an unchecksummed hardware record was accepted as the receipt\'s basis')
  self.assertEqual(named[0]['status'],'BLOCKED')

 def test_a_hardware_record_that_is_not_a_valid_capture_fails(self):
  import json,tempfile
  broken=json.loads((self.root/self.HARDWARE_ARTIFACT).read_text())
  broken['attested']['kernel']={'release':'6.6.0-never-probed','sources':['kernel.txt']}
  path=self.root/'verification/candidates/broken-hardware.json'
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(broken))
  try:
   r=self.hardware_receipt(hardware_artifact={'path':'verification/candidates/broken-hardware.json'})
   r['runtime.gpu.H100-SXM']['artifacts'].append(
    {'path':'verification/candidates/broken-hardware.json','sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest()})
   codes={f['code'] for f in self.findings(r,subject='runtime.gpu.H100-SXM')}
   self.assertIn('RUNTIME_HARDWARE_DRIFT',codes)
  finally:path.unlink()

 def test_expiry_still_retires_a_hardware_receipt(self):
  """Nothing between probes can detect a re-capped GPU or a swapped NIC. Kept deliberately."""
  codes=self.codes(self.hardware_receipt(executed_at='2025-01-01T00:00:00+00:00'),subject='runtime.gpu.H100-SXM')
  self.assertIn('RUNTIME_EXECUTION',codes)

 def test_artifact_tampering_is_still_caught_for_a_hardware_receipt(self):
  r=self.hardware_receipt()
  r['runtime.gpu.H100-SXM']['artifacts'][0]['sha256']='c'*64
  self.assertIn('RUNTIME_ARTIFACT',self.codes(r,subject='runtime.gpu.H100-SXM'))

 def test_a_hardware_receipt_cannot_stand_in_for_another_sku(self):
  r=self.hardware_receipt()
  r['runtime.gpu.H100-SXM']['record']=dict(r['runtime.gpu.H100-SXM']['record'],sku='B200')
  self.assertIn('RUNTIME_TARGET',self.codes(r,subject='runtime.gpu.H100-SXM'))

 # -- 3. the relaxation must not leak ------------------------------------------------------
 def test_only_hardware_attribute_domains_are_relaxed(self):
  """Every obligation, both directions: a stale lock and a stale consumer digest still
  retire a behavioural receipt, and only a hardware-attribute one is spared."""
  runtime={}
  for o in self.target['runtime_obligations']:runtime.update(self.minimal(o['id']))
  findings=self.findings(runtime)
  for o in self.target['runtime_obligations']:
   with self.subTest(obligation=o['id'],domain=o['domain']):
    codes={f['code'] for f in findings if f['subject']==o['id']}
    if o['domain'] in hardware.HARDWARE_DOMAINS:
     self.assertNotIn('RUNTIME_STALE',codes)
    else:
     self.assertIn('RUNTIME_STALE',codes,'a behavioural receipt escaped the lock/consumer binding')

 def test_a_behavioural_receipt_with_a_stale_lock_still_fails(self):
  o=next(o for o in self.target['runtime_obligations'] if o['domain']=='model')
  findings=self.findings(self.minimal(o['id'],consumer_sha256=self.current_consumer()),subject=o['id'])
  stale=[f for f in findings if f['code']=='RUNTIME_STALE']
  self.assertTrue(stale);self.assertEqual(stale[0]['status'],'FAIL')

 def test_a_behavioural_receipt_with_a_stale_consumer_digest_still_fails(self):
  o=next(o for o in self.target['runtime_obligations'] if o['domain']=='model')
  findings=self.findings(self.minimal(o['id'],lock_sha256=self.current_lock()),subject=o['id'])
  stale=[f for f in findings if f['code']=='RUNTIME_STALE']
  self.assertTrue(stale);self.assertEqual(stale[0]['status'],'FAIL')

 def test_the_hardware_domain_set_is_a_closed_subset_of_the_registered_domains(self):
  from verification.domains.contracts import DOMAINS
  self.assertEqual(hardware.HARDWARE_DOMAINS,frozenset({'gpu','scale_up','scale_out'}))
  self.assertTrue(hardware.HARDWARE_DOMAINS<=set(DOMAINS))
  behavioural=set(DOMAINS)-hardware.HARDWARE_DOMAINS
  self.assertEqual(behavioural,{'model','communication','training','serving','numerics','storage',
                                'measurement','visualization','release'})

 # -- current-state digests ----------------------------------------------------------------
 def current_lock(self):
  from verification.evidence import digest
  return digest({'target':self.target,'catalog':self.catalog})

 def current_consumer(self):
  from verification.evidence import digest
  return digest(self.inv['functions'])


class CapturedHardwareTests(unittest.TestCase):
 """The accepted hardware receipts, checked against the capture bytes they were built from.

 None of this is a shape check. Every digest is re-derived here from the artifacts on disk, so a
 receipt carrying a hand-written `hardware_sha256`, a superseded `probe_schema_sha256` or a
 `contract_sha256` for a record that has since moved fails in this file as well as in the gate.
 """
 @classmethod
 def setUpClass(cls):
  cls.target=loads((ROOT/'verification/manifests/observatory.json').read_text())
  cls.accepted=loads((ROOT/'verification/manifests/runtime.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  if not cls.accepted:
   raise unittest.SkipTest('Public distribution withholds private hardware receipts; binding logic is tested with isolated fixtures')

 def receipts(self):
  return [(o,self.accepted[o['id']]) for o in self.target['runtime_obligations']
          if o['domain'] in hardware.HARDWARE_DOMAINS and o['id'] in self.accepted]

 def test_the_repository_carries_accepted_hardware_receipts_to_check(self):
  # Otherwise every assertion below would pass over an empty list.
  self.assertTrue(self.receipts(),'no accepted receipt for any hardware-attribute obligation')

 def test_each_names_the_hardware_record_it_rests_on(self):
  for o,r in self.receipts():
   with self.subTest(obligation=o['id']):
    path=(r.get('hardware_artifact') or {}).get('path')
    self.assertTrue(path,'receipt names no hardware_artifact')
    self.assertIn(path,{a['path'] for a in r['artifacts']},
                  'the hardware record is not among the receipt\'s checksummed artifacts')

 def test_each_carried_hardware_digest_re_derives_from_the_capture(self):
  for o,r in self.receipts():
   with self.subTest(obligation=o['id']):
    record=hardware.load(ROOT/r['hardware_artifact']['path'])
    self.assertEqual(r['hardware_sha256'],hardware.hardware_sha256(record),
                     'the receipt names a hardware state the capture does not produce')

 def test_each_binds_the_current_hardware_data_model(self):
  for o,r in self.receipts():
   with self.subTest(obligation=o['id']):
    self.assertEqual(r.get('probe_schema_sha256'),hardware.schema_sha256())

 def test_each_binds_the_contract_record_it_attests(self):
  for o,r in self.receipts():
   with self.subTest(obligation=o['id']):
    contract=next(c for c in self.target['contracts'] if c['id']==o['contract'])
    self.assertEqual(r.get('contract_sha256'),contract_sha256(contract,self.inv.get('fields',{})))

 def test_no_accepted_receipt_rests_on_a_fixture_capture(self):
  for o,r in self.receipts():
   with self.subTest(obligation=o['id']):
    # hardware.load refuses a fixture unless asked; this states the intent explicitly.
    self.assertNotIn('fixture',loads((ROOT/r['hardware_artifact']['path']).read_text()))


if __name__=='__main__':unittest.main()
