import copy,json,subprocess,tempfile,unittest
from pathlib import Path
from verification.evidence import loads
from verification.runner import audit

class RuntimeReceiptTests(unittest.TestCase):
 """A runtime receipt is the strongest claim in the framework; its shape must be checked."""
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parents[2]
  cls.target=loads((cls.root/'verification/manifests/observatory.json').read_text())
  cls.catalog=loads((cls.root/'verification/manifests/sources.json').read_text())
  cls.inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=cls.root,capture_output=True,text=True,check=True).stdout)
 def codes(self,runtime):
  return {f['code'] for f in audit(self.root,self.target,self.inv,self.catalog,profile='runtime',runtime=runtime)['findings']}
 def obligation(self,domain='gpu'):
  return next(o for o in self.target['runtime_obligations'] if o['domain']==domain)
 def test_string_exit_code_is_a_shape_error_not_a_silent_pass(self):
  o=self.obligation()
  # `exit_code != 0` is False for the string "0", so an unchecked shape lets it through.
  self.assertIn('MANIFEST_SHAPE',self.codes({o['id']:{'executed_at':'2026-09-08T00:00:00+00:00','exit_code':'0','record':{},'environment':{},'artifacts':[],'record_artifact':{'path':'x','pointer':''}}}))
 def test_boolean_environment_fingerprint_is_not_an_environment(self):
  o=self.obligation()
  env={k:True for k in o['environment']}
  codes=self.codes({o['id']:{'executed_at':'2026-09-08T00:00:00+00:00','exit_code':0,'record':{},'environment':env,'artifacts':[],'record_artifact':{'path':'x','pointer':''}}})
  self.assertIn('MANIFEST_SHAPE',codes)
 def test_declared_value_without_a_probe_artifact_is_blocked(self):
  o=self.obligation()
  env={k:{'value':'something','method':'probed'} for k in o['environment']}
  codes=self.codes({o['id']:{'executed_at':'2026-09-08T00:00:00+00:00','exit_code':0,'record':{},'environment':env,'artifacts':[],'record_artifact':{'path':'x','pointer':''}}})
  self.assertIn('RUNTIME_ENVIRONMENT',codes)
 def test_absent_runtime_manifest_still_reports_not_run(self):
  self.assertIn('RUNTIME_NOT_RUN',self.codes({}))
