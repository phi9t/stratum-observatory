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

    def test_roles_flow_along_declared_dependency_edges(self):
        """Roles follow the declared edges. A dependency that cites nothing lends nothing, and a
        role no input holds cannot appear -- but the edge itself, not the input's verification
        status, is what carries provenance (the fixpoint refuses unverified inputs)."""
        from verification.runner import roles_from_dependencies
        pub={'role':'publisher_code','repo':'deepseek-ai/DeepSeek-V2'}
        fw={'role':'framework_code','repo':'sgl-project/sglang'}
        sources={'a':[pub],'b':[fw],'empty':[]}
        self.assertEqual(roles_from_dependencies({'id':'d','dependencies':['a','b']},sources),[pub,fw])
        self.assertEqual(roles_from_dependencies({'id':'d','dependencies':['empty','absent']},sources),[])
        self.assertEqual(roles_from_dependencies({'id':'d'},sources),[])

    def test_the_role_pass_does_not_soften_a_contradiction_into_missing_evidence(self):
        """A FAIL claim that also fails a role check stays FAIL, never BLOCKED."""
        from verification.runner import demoted_status
        self.assertEqual(demoted_status('FAIL'),'FAIL')
        self.assertEqual(demoted_status('PASS'),'BLOCKED')
        self.assertEqual(demoted_status('NOT_RUN'),'NOT_RUN')

    def test_unknown_profile_rejected(self):
        from verification.runner import audit
        with self.assertRaises(ValueError):audit(Path('.'),{}, {}, {}, profile='skip-science')

if __name__=='__main__':unittest.main()
