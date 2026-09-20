import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from verification import evidence

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
    def source(self, role='publisher_code', text='class Model:\n    def __init__(self, config):\n        self.width = config.hidden_size\n'):
        path = role + ('.json' if role == 'publisher_config' else '.py')
        raw = text.encode(); (self.root/path).write_bytes(raw)
        repo = 'moonshotai/Kimi-K3' if role.startswith('publisher') else 'sgl-project/sglang'
        host = 'huggingface.co' if role.startswith('publisher') else 'raw.githubusercontent.com'
        rev = 'a'*40
        url = f'https://{host}/{repo}/'+('raw/' if host=='huggingface.co' else '')+f'{rev}/{path}'
        return dict(id=role, role=role, provider='huggingface' if role.startswith('publisher') else 'github', repo=repo, path=path,
                    revision=rev, url=url, cache=path, sha256=hashlib.sha256(raw).hexdigest(),kind='git',license='test fixture',fixture=True)
    def test_user_selected_dspark_publishers_are_exact_repository_scopes(self):
        for repo,provider in [('deepseek-ai/DeepSpec','github'),('deepseek-ai/DeepSeek-V4-Pro-DSpark','huggingface')]:
            s=self.source();s.update(repo=repo,provider=provider)
            base='https://raw.githubusercontent.com/' if provider=='github' else 'https://huggingface.co/'
            middle='/' if provider=='github' else '/raw/'
            s['url']=base+repo+middle+s['revision']+'/'+s['path']
            self.assertEqual(evidence.check_source(s,self.root,allow_fixtures=True),[])
            s['repo']=repo+'-unreviewed'
            s['url']=base+s['repo']+middle+s['revision']+'/'+s['path']
            self.assertIn('SOURCE_AUTHORITY',[f.code for f in evidence.check_source(s,self.root,allow_fixtures=True)])
    def test_pro_draft_configuration_has_exact_publisher_authority(self):
        s=self.source(role='publisher_config',text='{"n_mtp_layers":3}')
        s['repo']='deepseek-ai/DeepSeek-V4-Pro-DSpark'
        s['url']=f"https://huggingface.co/{s['repo']}/raw/{s['revision']}/{s['path']}"
        self.assertEqual(evidence.check_source(s,self.root,allow_fixtures=True),[])
        s['repo']+='-unreviewed'
        s['url']=f"https://huggingface.co/{s['repo']}/raw/{s['revision']}/{s['path']}"
        self.assertIn('SOURCE_AUTHORITY',[f.code for f in evidence.check_source(s,self.root,allow_fixtures=True)])
    def test_api_exists(self):
        self.assertTrue(callable(getattr(evidence,'check_source',None)))
    def test_hash_mismatch_is_failure(self):
        s=self.source(); (self.root/s['cache']).write_text('changed')
        self.assertIn('SOURCE_HASH', [f.code for f in evidence.check_source(s,self.root,allow_fixtures=True)])
    def test_missing_cache_is_blocked(self):
        s=self.source(); (self.root/s['cache']).unlink()
        self.assertEqual(evidence.check_source(s,self.root,allow_fixtures=True)[0].status,'BLOCKED')
    def test_mutable_git_revision_is_rejected(self):
        s=self.source();s['revision']='main'
        self.assertIn('SOURCE_PIN',[f.code for f in evidence.check_source(s,self.root,allow_fixtures=True)])
    def test_empty_checksum_is_not_accepted(self):
        s=self.source();s['sha256']=None
        self.assertTrue(any(f.status=='BLOCKED' for f in evidence.check_source(s,self.root,allow_fixtures=True)))
    def test_pin_must_match_url(self):
        s=self.source();s['url']=s['url'].replace('a'*40,'b'*40)
        self.assertIn('SOURCE_URL',[f.code for f in evidence.check_source(s,self.root,allow_fixtures=True)])
    def test_traversal_and_symlink_escape_rejected(self):
        for p in ['../outside','/etc/passwd']:
            with self.assertRaises(ValueError): evidence.safe_path(self.root,p)
        (self.root/'escape').symlink_to('/etc')
        with self.assertRaises(ValueError): evidence.safe_path(self.root,'escape/passwd')
    def test_fixture_cannot_be_production_evidence(self):
        self.assertIn('SOURCE_FIXTURE',[f.code for f in evidence.check_source(self.source(),self.root)])
    def test_python_anchor_resolves_real_qualified_symbol(self):
        s=self.source();a={'kind':'python_symbol','symbol':'Model.__init__','contains':['config.hidden_size']}
        text, span = evidence.select_anchor((self.root/s['cache']).read_bytes(),a)
        self.assertIn('config.hidden_size',text);self.assertEqual(span,[2,3])
    def test_comment_is_not_python_symbol(self):
        with self.assertRaises(ValueError): evidence.select_anchor(b'# class Model:\n',{'kind':'python_symbol','symbol':'Model'})
    def test_ambiguous_symbol_rejected(self):
        with self.assertRaises(ValueError): evidence.select_anchor(b'def f(): pass\ndef f(): pass\n',{'kind':'python_symbol','symbol':'f'})
    def test_pointer_escape_and_array(self):
        self.assertEqual(evidence.pointer({'a/b':[{'~x':4}]},'/a~1b/0/~0x'),4)
    def test_missing_pointer_rejected(self):
        with self.assertRaises(ValueError): evidence.pointer({},'/missing')
    def test_nonfinite_and_duplicate_json_rejected(self):
        for text in ['{"x":NaN}','{"x":1,"x":2}']:
            with self.assertRaises(ValueError): evidence.loads(text)
    def test_anchor_hash_catches_shifted_or_changed_span(self):
        s=self.source();a={'kind':'span','start':2,'end':3,'sha256':'0'*64,'contains':['config.hidden_size']}
        self.assertIn('ANCHOR_HASH',[f.code for f in evidence.check_anchor(s,a,self.root,allow_fixtures=True)])
    def test_config_is_not_code(self):
        self.assertFalse(evidence.dual_code_roles([{'role':'publisher_config','repo':'moonshotai/Kimi-K3'}, {'role':'framework_code','repo':'sgl-project/sglang'}]))
    def test_two_independent_code_references(self):
        self.assertTrue(evidence.dual_code_roles([{'role':'publisher_code','repo':'moonshotai/Kimi-K3'}, {'role':'framework_code','repo':'sgl-project/sglang'}]))
    def test_same_repository_cannot_fill_both_roles(self):
        self.assertFalse(evidence.dual_code_roles([{'role':'publisher_code','repo':'same/repo'},{'role':'framework_code','repo':'same/repo'}]))
    def test_evidence_cycle_rejected(self):
        self.assertIn('CLAIM_CYCLE',[f.code for f in evidence.check_dependencies([{'id':'a','dependencies':['b']},{'id':'b','dependencies':['a']}])])
    def test_missing_dependency_rejected(self):
        self.assertIn('CLAIM_DEPENDENCY',[f.code for f in evidence.check_dependencies([{'id':'a','dependencies':['missing']}])])
    def test_reviewed_true_is_not_a_signature(self):
        result=evidence.check_review({'id':'c','reviewed':True},'d'*64,{},self.root)
        self.assertEqual(result.status,'BLOCKED')
    def test_claim_digest_changes_with_local_implementation(self):
        self.assertNotEqual(evidence.digest({'local':'one'}),evidence.digest({'local':'two'}))
    def test_coverage_detects_new_local_field(self):
        inv={'fields':{'x':1,'new':2},'functions':{}}
        result=evidence.check_coverage(inv,[{'id':'x','consumer':'x','expected':1,'classification':'assumed','domain':'serving'}])
        self.assertIn('COVERAGE_MISSING',[f.code for f in result])
    def test_bound_value_change_detected(self):
        result=evidence.check_coverage({'fields':{'x':2},'functions':{}},[{'id':'x','consumer':'x','expected':1}])
        self.assertIn('CONSUMER_VALUE',[f.code for f in result])
    def test_duplicate_claim_and_orphan_detected(self):
        claims=[{'id':'a','consumer':'x','expected':1},{'id':'a','consumer':'z','expected':1}]
        codes=[f.code for f in evidence.check_coverage({'fields':{'x':1},'functions':{}},claims)]
        self.assertIn('CLAIM_DUPLICATE',codes);self.assertIn('CONSUMER_MISSING',codes)
    def test_empty_inventory_cannot_pass(self):
        self.assertIn('COVERAGE_EMPTY',[f.code for f in evidence.check_coverage({'fields':{},'functions':{}},[])])

if __name__=='__main__': unittest.main()


class DualCodeIndependenceTests(unittest.TestCase):
 """Two reimplementations are not a publisher and an independent check of it."""
 def test_transformers_is_not_a_publisher(self):
  from verification.evidence import REPOS
  # huggingface/transformers is a third-party reimplementation of every model in
  # this atlas. Accepting it as publisher_code lets transformers + sglang satisfy
  # "publisher implementation AND independent framework implementation".
  self.assertNotIn('huggingface/transformers',REPOS['publisher_code'])
  self.assertIn('huggingface/transformers',REPOS['framework_code'])
 def test_catalog_declares_no_third_party_publisher(self):
  from verification.evidence import REPOS
  root=Path(__file__).resolve().parents[2]
  catalog=evidence.loads((root/'verification/manifests/sources.json').read_text())
  for s in catalog['sources']:
   if s.get('role')=='publisher_code':
    self.assertIn(s['repo'],REPOS['publisher_code'],f"{s['id']} claims publisher_code for {s['repo']}")
 def test_two_repos_from_one_organization_are_not_independent(self):
  from verification.evidence import dual_code_roles
  same=[{'role':'publisher_code','repo':'acme/model'},{'role':'framework_code','repo':'acme/server'}]
  self.assertFalse(dual_code_roles(same),'same-organization repos were accepted as independent')
 def test_genuine_publisher_and_independent_framework_are_accepted(self):
  from verification.evidence import dual_code_roles
  ok=[{'role':'publisher_code','repo':'deepseek-ai/DeepSeek-V3'},{'role':'framework_code','repo':'sgl-project/sglang'}]
  self.assertTrue(dual_code_roles(ok))
 def test_framework_alone_is_not_dual_code(self):
  from verification.evidence import dual_code_roles
  self.assertFalse(dual_code_roles([{'role':'framework_code','repo':'sgl-project/sglang'},{'role':'framework_code','repo':'vllm-project/vllm'}]))


class AnchorSelfCertificationTests(unittest.TestCase):
 """An anchor must be checked against source bytes, never against a copy of itself."""
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  cache=self.root/'.verification-cache/sha256';cache.mkdir(parents=True)
  self.text='alpha\ndef routed_experts():\n    return 256\nomega\n'
  self.digest=hashlib.sha256(self.text.encode()).hexdigest()
  (cache/f'{self.digest}.txt').write_text(self.text)
  self.source={'id':'s','kind':'git','role':'framework_code','provider':'github','repo':'sgl-project/sglang',
               'path':'python/sglang/model.py','revision':'a'*40,
               'url':'https://raw.githubusercontent.com/sgl-project/sglang/'+'a'*40+'/python/sglang/model.py',
               'sha256':self.digest,'cache':f'.verification-cache/sha256/{self.digest}.txt','license':'apache-2.0'}
 def test_quote_hashed_against_itself_is_not_accepted(self):
  quote='return 256'
  anchor={'kind':'text_quote','quote':quote,'sha256':hashlib.sha256(quote.encode()).hexdigest(),'contains':['256']}
  codes=[f.code for f in evidence.check_anchor(self.source,anchor,self.root)]
  self.assertTrue(codes,'an anchor whose digest is a hash of its own quote was accepted')
 def test_quote_absent_from_the_source_is_rejected(self):
  quote='return 512'
  anchor={'kind':'text_quote','quote':quote,'sha256':hashlib.sha256(quote.encode()).hexdigest(),'contains':['512']}
  self.assertTrue([f.code for f in evidence.check_anchor(self.source,anchor,self.root)])


class SymbolIndexTests(unittest.TestCase):
 """Anchor resolution is memoised per file content; semantics must not change."""
 SRC=b'''
class Outer:
    def inner(self):
        return 1
def top():
    return 2
class Dup: pass
class Dup: pass
'''
 def anchor(self,symbol,**kw):
  a={'kind':'python_symbol','symbol':symbol};a.update(kw);return a
 def test_qualified_symbols_resolve(self):
  sel,span=evidence.select_anchor(self.SRC,self.anchor('Outer.inner'))
  self.assertIn('def inner',sel);self.assertEqual(len(span),2)
  sel,_=evidence.select_anchor(self.SRC,self.anchor('top'))
  self.assertIn('def top',sel)
 def test_ambiguous_symbol_is_rejected(self):
  with self.assertRaises(ValueError):evidence.select_anchor(self.SRC,self.anchor('Dup'))
 def test_missing_symbol_is_rejected(self):
  with self.assertRaises(ValueError):evidence.select_anchor(self.SRC,self.anchor('nope'))
 def test_repeated_resolution_is_identical(self):
  a=evidence.select_anchor(self.SRC,self.anchor('top'))
  b=evidence.select_anchor(self.SRC,self.anchor('top'))
  self.assertEqual(a,b)
 def test_changed_content_is_not_served_from_the_memo(self):
  first,_=evidence.select_anchor(self.SRC,self.anchor('top'))
  changed=self.SRC.replace(b'return 2',b'return 3')
  second,_=evidence.select_anchor(changed,self.anchor('top'))
  self.assertNotEqual(first,second,'a changed file was served from the memo')
 def test_contains_tokens_still_enforced(self):
  with self.assertRaises(ValueError):
   evidence.select_anchor(self.SRC,self.anchor('top',contains=['not-present']))


class IndependentCorroborationTests(unittest.TestCase):
 """Where a publisher ships no code, independent implementations must corroborate instead."""
 def sources(self,*specs):
  return [{'role':r,'repo':repo} for r,repo in specs]
 def test_publisher_plus_independent_framework_still_satisfies(self):
  s=self.sources(('publisher_code','deepseek-ai/DeepSeek-V3'),('framework_code','sgl-project/sglang'))
  self.assertTrue(evidence.dual_code_roles(s,publisher_ships_code=True))
 def test_publisher_path_still_rejects_a_single_organisation(self):
  s=self.sources(('publisher_code','acme/model'),('framework_code','acme/server'))
  self.assertFalse(evidence.dual_code_roles(s,publisher_ships_code=True))
 def test_config_only_publisher_is_corroborated_by_two_organisations(self):
  # Qwen ships configuration and no modeling code, so no publisher implementation can ever
  # exist. Two independent reimplementations agreeing is the available corroboration.
  s=self.sources(('framework_code','huggingface/transformers'),('framework_code','sgl-project/sglang'))
  self.assertTrue(evidence.dual_code_roles(s,publisher_ships_code=False))
 def test_config_only_publisher_still_needs_more_than_one_organisation(self):
  s=self.sources(('framework_code','huggingface/transformers'),('framework_code','huggingface/transformers'))
  self.assertFalse(evidence.dual_code_roles(s,publisher_ships_code=False))
 def test_the_relaxation_does_not_leak_to_publishers_that_do_ship_code(self):
  # DeepSeek publishes modeling code, so two frameworks must not substitute for reading it.
  s=self.sources(('framework_code','huggingface/transformers'),('framework_code','sgl-project/sglang'))
  self.assertFalse(evidence.dual_code_roles(s,publisher_ships_code=True))
 def test_policy_set_matches_the_recorded_audit(self):
  # The policy root must not claim a publisher ships no code without the evidence saying so.
  root=Path(__file__).resolve().parents[2]
  audit=evidence.loads((root/'evidence/publisher-code/audit.json').read_text())
  config_only={r['repo'] for r in audit['records'] if not r['python_files']}
  self.assertEqual(evidence.CONFIG_ONLY_PUBLISHERS,config_only)
  self.assertTrue(config_only,'audit recorded no config-only publisher')
 def test_every_authorized_publisher_has_an_audit_record(self):
  # A repo missing from the audit entirely contributes to neither side of
  # test_policy_set_matches_the_recorded_audit's comparison, so it is silently treated as
  # "ships code" by omission -- indistinguishable from a publisher actually verified to
  # ship code. An absent record must not read as evidence of anything; every repo in
  # REPOS['publisher_config'] must have been actually queried and recorded.
  root=Path(__file__).resolve().parents[2]
  audit=evidence.loads((root/'evidence/publisher-code/audit.json').read_text())
  audited={r['repo'] for r in audit['records']}
  missing=evidence.REPOS['publisher_config']-audited
  self.assertEqual(missing,set(),f'no audit record for: {sorted(missing)}')
