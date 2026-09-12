"""Negative contracts for independent arithmetic reconciliation."""
import copy
import unittest
from verification.accounting import audit_export, audit_examples

class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json, subprocess
        from pathlib import Path
        root=Path(__file__).resolve().parents[2]
        cls.export=json.loads(subprocess.check_output(['node','tools/export-accounting.mjs'],cwd=root))
    def run_mutation(self, method, mutate):
        data=copy.deepcopy(self.export)
        gqa=next(m['id'] for m in data['models'] if m['architecture']=='gqa')
        case=next(c for c in data['cases'] if c['method']==method and (method!='book-gqa-cache' or c['model']==gqa))
        mutate(case)
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
    def test_live(self):
        self.assertFalse([f for f in audit_export(self.export) if f['classification']=='contradiction'])
    def test_missing_gradient(self):
        self.run_mutation('book-training-state',lambda c:c['actual'].__setitem__('gradients',0))
    def test_double_kv(self):
        self.run_mutation('book-gqa-cache',lambda c:c['actual'].__setitem__('gqa',2*c['actual']['gqa']))
    def test_wrong_kv_heads(self):
        data=copy.deepcopy(self.export);next(m for m in data['models'] if m['architecture']=='gqa')['kvHeads']*=2
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
    def test_rank_direction(self):
        data=copy.deepcopy(self.export)
        c=next(c for c in data['cases'] if c['method']=='book-ring-allreduce' and c['settings']['ranks']>1)
        c['actual']['sent']*=2
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
    def test_precision(self):
        data=copy.deepcopy(self.export);data['hardware'][0]['peakFormat']='Sparse FP4'
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
    def test_case_inventory(self):
        for change in ('missing','extra','malformed'):
            data=copy.deepcopy(self.export)
            if change=='missing':data['cases'].pop()
            elif change=='extra':data['cases'].append(copy.deepcopy(data['cases'][0]))
            else:data['cases'][0]['actual']={'total':float('nan')}
            self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)),change)
    def test_unknown_architecture(self):
        data=copy.deepcopy(self.export);data['models'][0]['architecture']='new-architecture'
        findings=audit_export(data)
        self.assertTrue(any(f['classification']=='unsupported' and f.get('model')==data['models'][0]['id'] for f in findings))
    def test_duplicate_cache_owners(self):
        data=copy.deepcopy(self.export);m=next(m for m in data['models'] if m['architecture']=='ced-csa2');m['kvSourceLayers'].append(m['kvSourceLayers'][0])
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))

    def test_examples(self):
        import json
        from pathlib import Path
        manifest=json.loads((Path(__file__).resolve().parents[1]/'manifests/accounting-methods.json').read_text())
        self.assertFalse([f for f in audit_examples(manifest) if f['classification']=='contradiction'])
        for action in ('missing','extra','wrong','malformed'):
            data=copy.deepcopy(manifest)
            if action=='missing':data['examples'].pop()
            elif action=='extra':data['examples'].append(copy.deepcopy(data['examples'][0]))
            elif action=='wrong':data['examples'][0]['expected']['main']+=1
            else:data['examples'][0]['inputs']={}
            self.assertTrue(any(f['classification']=='contradiction' for f in audit_examples(data)),action)

    def test_tied_embeddings_and_norm_policy(self):
        from verification.accounting import gqa_parameters
        m=copy.deepcopy(next(m for m in self.export['models'] if m['architecture']=='gqa'))
        original=gqa_parameters(m)
        m['tieWordEmbeddings']=True;m['qkNorm']=False
        changed=gqa_parameters(m)
        self.assertEqual(original['resident']-changed['resident'],m['vocab']*m['d']+2*m['headDim']*m['layers'])

    def test_joint_axis_and_case_omission(self):
        data=copy.deepcopy(self.export)
        i=len(data['axes']['cache'])-1;data['axes']['cache'].pop()
        data['cases']=[c for c in data['cases'] if not(c['method']=='book-gqa-cache' and c['id'].endswith('/'+str(i)))]
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))

    def test_negative_settings_and_dimensions(self):
        for group,field in [('models','d'),('hardware','hbmTBps')]:
            data=copy.deepcopy(self.export);data[group][0][field]=-1
            self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
        data=copy.deepcopy(self.export);data['axes']['cache'][0]['context']=-1
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))

    def test_nonpositive_example_inputs(self):
        import json
        from pathlib import Path
        manifest=json.loads((Path(__file__).resolve().parents[1]/'manifests/accounting-methods.json').read_text())
        for value in (-1,0,float('nan')):
            data=copy.deepcopy(manifest);data['examples'][0]['inputs']['d']=value
            self.assertTrue(any(f['classification']=='contradiction' for f in audit_examples(data)))

    def test_main_writes_failure_reports(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from verification.accounting import main
        for content in (None,'{}','{"schema":"stratum-accounting-methods/v1","methods":[{"id":"book-gqa-cache"}]}'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'verification/manifests').mkdir(parents=True)
                if content is not None:(root/'verification/manifests/accounting-methods.json').write_text(content)
                with patch('verification.accounting.ROOT',root),patch('verification.accounting.subprocess.check_output',return_value=json.dumps(self.export).encode()),patch('builtins.print'):
                    self.assertEqual(main(),1)
                report=json.loads((root/'verification/reports/accounting.json').read_text())
                self.assertEqual(report['status'],'FAIL')
                self.assertGreater(report['counts']['contradiction'],0)
                self.assertTrue((root/'verification/reports/accounting.md').exists())

    def test_missing_identity(self):
        data=copy.deepcopy(self.export);del data['function_sha256']
        self.assertTrue(any(f['classification']=='contradiction' for f in audit_export(data)))
