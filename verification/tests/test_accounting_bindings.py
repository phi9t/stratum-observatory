"""Method references reach the publication runner and fail closed on drift."""
import copy
import json
import subprocess
import unittest
from pathlib import Path
from verification.runner import audit

ROOT = Path(__file__).resolve().parents[2]

class AccountingBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(subprocess.check_output(['node', 'tools/export-verification.mjs'], cwd=ROOT, text=True))
        cls.target = json.loads((ROOT/'verification/manifests/observatory.json').read_text())
        cls.catalog = json.loads((ROOT/'verification/manifests/sources.json').read_text())
        cls.methods = json.loads((ROOT/'verification/manifests/accounting-methods.json').read_text())

    def subset(self):
        claims=[copy.deepcopy(c) for c in self.target['claims'] if c.get('consumer','').startswith('accounting.')]
        self.assertGreaterEqual(len(claims), len(self.methods['methods']))
        target={'schema':'stratum-target/v1','claims':claims,'contracts':[],'runtime_obligations':[]}
        inventory={'fields':{k:v for k,v in self.inventory['fields'].items() if k.startswith('accounting.')},'functions':{},'scenes':{}}
        return target,inventory

    def test_each_method_is_a_real_exact_consumer(self):
        target,inv=self.subset()
        for method in self.methods['methods']:
            key='accounting.method.'+method['id']
            self.assertEqual(inv['fields'][key],method)
            c=next(c for c in target['claims'] if c['consumer']==key)
            self.assertEqual(c['expected'],method)
            self.assertTrue(c['evidence'])
            self.assertTrue(all(e['anchor'].get('sha256') for e in c['evidence']))
        result=audit(ROOT,target,inv,self.catalog)
        self.assertFalse(any(f['code'] in {'EVIDENCE_RELATION','CLAIM_SOURCE','CONSUMER_VALUE','CONSUMER_MISSING'} for f in result['findings']))
        self.assertNotEqual(result['status'],'PASS')

    def test_mutated_method_fails_in_publication_runner(self):
        target,inv=self.subset(); key=next(iter(inv['fields']))
        inv=copy.deepcopy(inv);inv['fields'][key]['assumptions']=['mutated semantics']
        result=audit(ROOT,target,inv,self.catalog)
        self.assertTrue(any(f['code']=='CONSUMER_VALUE' and f['status']=='FAIL' for f in result['findings']))

    def test_unknown_method_source_fails_in_publication_runner(self):
        target,inv=self.subset();target['claims'][0]['evidence'][0]['source']='unknown-book'
        result=audit(ROOT,target,inv,self.catalog)
        self.assertTrue(any(f['code']=='CLAIM_SOURCE' and f['status']=='FAIL' for f in result['findings']))

class AccountingMigrationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runpy
        cls.bind=staticmethod(runpy.run_path(str(ROOT/'tools/bind-accounting.py'))['bind'])
        cls.inventory=json.loads(subprocess.check_output(['node','tools/export-verification.mjs'],cwd=ROOT,text=True))

    def migration(self,consumer,reviewed):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'verification/manifests').mkdir(parents=True)
            for name in ('sources','observatory','accounting-methods','reviews'):
                (root/f'verification/manifests/{name}.json').write_bytes((ROOT/f'verification/manifests/{name}.json').read_bytes())
            target_path=root/'verification/manifests/observatory.json'
            target=json.loads(target_path.read_text());claim=next(c for c in target['claims'] if c.get('consumer')==consumer)
            claim['expected']='deliberately stale binding'
            target_path.write_text(json.dumps(target))
            if reviewed:(root/'verification/manifests/reviews.json').write_text(json.dumps({claim['id']:{'signed_subject':'preserve'}}))
            before=target_path.read_bytes()
            # Isolate migration safety from fetching and source selection, covered separately.
            with patch('subprocess.check_output',return_value=json.dumps(self.inventory)),patch.dict(self.bind.__globals__,{'check_anchor':lambda *args,**kwargs:[]}):
                with self.assertRaisesRegex(ValueError,'Reviewed subject would change' if reviewed else 'outside reviewed accounting scope'):
                    self.bind(root,refresh=True)
            self.assertEqual(target_path.read_bytes(),before)

    def test_unexpected_consumer_drift_is_not_refreshed(self):
        self.migration('MODELS.kimi3.layers',False)

    def test_reviewed_subject_is_not_rewritten(self):
        self.migration('file.src/app.mjs',True)
