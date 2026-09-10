import copy, json, subprocess, tempfile, unittest
from pathlib import Path

class RunnerTests(unittest.TestCase):
    def test_auditor_api_exists(self):
        from verification import runner
        self.assertTrue(callable(getattr(runner,'audit',None)))

    def test_inventory_is_derived_from_actual_core(self):
        root=Path(__file__).resolve().parents[2]
        result=subprocess.run(['node','tools/export-verification.mjs'],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        inv=json.loads(result.stdout)
        self.assertEqual(inv['fields']['MODELS.kimi3.layers'],93)
        self.assertIn('function.cacheLedger',inv['functions'])
        self.assertIn('file.src/core.mjs',inv['functions'])
        self.assertTrue(inv['scenes'])

    def test_empty_target_is_blocked(self):
        from verification.runner import audit
        with tempfile.TemporaryDirectory() as tmp:
            report=audit(Path(tmp),{}, {'fields':{},'functions':{},'scenes':{}}, {})
            self.assertNotEqual(report['status'],'PASS')
            self.assertTrue(report['findings'])

    def test_roles_flow_only_from_a_verified_dependency(self):
        """`roles_from_dependencies` is the policy root: PASS lends, anything else lends nothing."""
        from verification.runner import roles_from_dependencies
        claim={'id':'d','dependencies':['a','b','missing']}
        status={'a':'PASS','b':'BLOCKED','missing':'PASS'}
        sources={'a':[{'role':'publisher_code','repo':'deepseek-ai/DeepSeek-V2'}],
                 'b':[{'role':'framework_code','repo':'sgl-project/sglang'}]}
        self.assertEqual(roles_from_dependencies(claim,status,sources),
                         [{'role':'publisher_code','repo':'deepseek-ai/DeepSeek-V2'}])
        self.assertEqual(roles_from_dependencies({'id':'d'},status,sources),[])
        self.assertEqual(roles_from_dependencies(claim,{},sources),[])

    def test_unknown_profile_rejected(self):
        from verification.runner import audit
        with self.assertRaises(ValueError):audit(Path('.'),{}, {}, {}, profile='skip-science')

if __name__=='__main__':unittest.main()
