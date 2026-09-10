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
import copy,subprocess,unittest
from datetime import datetime,timezone
from pathlib import Path

from verification import hardware
from verification.evidence import loads
from verification.runner import audit,contract_sha256

ROOT=Path(__file__).resolve().parents[2]
NOW=datetime.now(timezone.utc).isoformat()


class RuntimeBindingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=ROOT
  cls.target=loads((ROOT/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((ROOT/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  cls.accepted=loads((ROOT/'verification/manifests/runtime.json').read_text())

 # -- helpers ------------------------------------------------------------------------------
 def findings(self,runtime,*,target=None,inv=None,subject=None):
  report=audit(self.root,target or self.target,inv or self.inv,self.catalog,profile='runtime',runtime=runtime)
  return [f for f in report['findings'] if subject is None or f['subject']==subject]

 def codes(self,runtime,**kw):
  return {f['code'] for f in self.findings(runtime,**kw)}

 def contract(self,target,cid):
  return next(c for c in target['contracts'] if c['id']==cid)

 def hardware_receipt(self,*,target=None,inv=None,**over):
  """The accepted H100 receipt, completed with the fields the new binding requires.

  Based on the real receipt so its artifacts, digests and record all still check out; the
  three new fields are what Task C's capture will fill in for real.
  """
  target=target or self.target;inv=inv or self.inv
  r=copy.deepcopy(self.accepted['runtime.gpu.H100-SXM'])
  r['probe_schema_sha256']=hardware.schema_sha256()
  r['contract_sha256']=contract_sha256(self.contract(target,'gpu.H100-SXM'),inv.get('fields',{}))
  r['hardware_sha256']='a'*64
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
  """The accepted receipts predate this binding. That is missing evidence: they must block,
  and the gate must not crash on them or wave them through."""
  findings=self.findings(self.accepted,subject='runtime.gpu.H100-SXM')
  codes={f['code'] for f in findings}
  self.assertIn('RUNTIME_SCHEMA',codes)
  self.assertIn('RUNTIME_HARDWARE',codes)
  self.assertEqual([f['code'] for f in findings if f['status']=='FAIL'],[])

 def test_a_receipt_naming_no_captured_hardware_state_is_blocked(self):
  """Carried, not re-derived -- but a receipt that names no capture at all leaves a reader
  unable to say which hardware state it rests on, which is missing evidence."""
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


if __name__=='__main__':unittest.main()
