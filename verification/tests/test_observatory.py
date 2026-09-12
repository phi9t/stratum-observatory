import json,subprocess,unittest
from pathlib import Path
from verification.domains.contracts import validate
from verification.evidence import check_source,check_coverage,HEX40
ROOT=Path(__file__).resolve().parents[2]
class ObservatoryTests(unittest.TestCase):
 def dense(self):return {'layers':2,'layer_groups':{'gqa':[1,2]},'experts':0,'top_k':0,'shared':0,'d':4096,'latent':4096,'heads':32,'kv_heads':8,'head_dim':128,'index_base':1}
 def test_dense_model_is_not_forced_to_invent_experts(self):
  self.assertEqual([f for f in validate('model',self.dense()) if f.status!='PASS'],[])
 def test_dense_with_routed_topk_is_rejected(self):
  x=self.dense();x['top_k']=1
  self.assertIn('MODEL_ROUTING',[f.code for f in validate('model',x)])
 def qwen_reference(self,role):
  return {'id':'qwen-ref','role':role,'kind':'git','provider':'github','repo':'huggingface/transformers','path':'src/transformers/models/qwen3/modeling_qwen3.py','revision':'c93057d4835cd31752bb56f59989dd27696eb45b','url':'https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3/modeling_qwen3.py','cache':None,'sha256':None,'license':'No source redistributed.'}
 def test_registered_qwen_reference_is_code_not_a_config_substitute(self):
  # A real code reference is authorized evidence; only its bytes are missing.
  findings=check_source(self.qwen_reference('framework_code'),ROOT)
  self.assertNotIn('SOURCE_AUTHORITY',[f.code for f in findings]);self.assertIn('SOURCE_UNFETCHED',[f.code for f in findings])
 def test_qwen_has_no_publisher_implementation_to_register(self):
  # Qwen does not publish modeling code for these presets. transformers reimplements
  # it, so registering transformers as the publisher would let one third-party
  # implementation corroborate another and satisfy MODEL_DUAL_CODE.
  self.assertIn('SOURCE_AUTHORITY',[f.code for f in check_source(self.qwen_reference('publisher_code'),ROOT)])
 def test_live_inventory_covers_new_renderer_modules_and_all_presets(self):
  r=subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)
  inv=json.loads(r.stdout)
  self.assertEqual(len(inv['models']),9)
  self.assertEqual(inv['samples']['decoder_layers'],544)
  for path in ['src/models.mjs','src/diagram-model.mjs','src/diagram-systems.mjs','src/diagram-meta.mjs']:
   self.assertIn('file.'+path,inv['functions'])
  manifest=json.loads((ROOT/'verification/manifests/observatory.json').read_text())
  self.assertEqual(check_coverage(inv,manifest['claims']),[])
 def test_paper_sources_are_pinned_to_full_text(self):
  """An /abs/ landing page carries the abstract only; architecture facts are not on it.

  This is an arXiv-specific concern: arXiv's /abs/ page and /html/ rendering are two
  different documents at two different URLs for the same paper. A git-pinned source
  (kind='git') has no such split -- the URL already names the exact raw bytes at an
  immutable commit, so the /html/ requirement does not apply to it."""
  catalog=json.loads((ROOT/'verification/manifests/sources.json').read_text())
  papers=[s for s in catalog['sources'] if s['role']=='paper' and s.get('kind')=='document']
  self.assertTrue(papers,'catalog has no document-kind paper sources')
  for s in papers:
   with self.subTest(source=s['id']):
    self.assertIn('/html/',s['url'],f"{s['id']} is pinned to a landing page, not the paper")
  # The other branch of the role's partition: a paper pinned as kind='git' has no /abs/
  # vs /html/ split to guard against, but it must still be pinned to an exact immutable
  # commit rather than a movable ref -- that is what makes "no split" true in the first
  # place, so this is not a free pass, it is the actual guarantee being relied on.
  gitpapers=[s for s in catalog['sources'] if s['role']=='paper' and s.get('kind')=='git']
  self.assertTrue(gitpapers,'catalog has no git-kind paper sources')
  for s in gitpapers:
   with self.subTest(source=s['id']):
    self.assertTrue(HEX40.fullmatch(str(s.get('revision',''))),
                     f"{s['id']} is kind='git' but not pinned to a full 40-hex commit")
 def test_paper_source_versions_are_not_pending(self):
  """Bytes are locked, so a version reading 'pending' is a stale lie about the snapshot."""
  catalog=json.loads((ROOT/'verification/manifests/sources.json').read_text())
  for s in catalog['sources']:
   if s.get('kind')=='document' and s.get('sha256'):
    with self.subTest(source=s['id']):
     self.assertNotIn('pending',s['version'].lower())
 def test_scene_claims_carry_no_unanchored_copies_of_dependency_citations(self):
  """A derived contract's provenance is its dependencies: an unanchored citation whose
  source a dependency already cites is a copy and must be gone. An unanchored citation
  whose source no dependency cites is not a copy of anything -- it is the claim's own
  only evidence for that role (e.g. a gpu scene claim's vendor_document), and removing
  it would manufacture a false role gap, so it must stay."""
  target=json.loads((ROOT/'verification/manifests/observatory.json').read_text())
  by_id={c['id']:c for c in target['claims']}
  offenders=[]
  for c in target['claims']:
   if not c['id'].startswith('scene.'):continue
   dep_sources=set()
   for dep_id in c.get('dependencies',[]):
    dep=by_id.get(dep_id)
    if dep is None:continue
    for e in dep.get('evidence',[]):dep_sources.add(e['source'])
   for e in c['evidence']:
    if e['anchor'] is None and e['source'] in dep_sources:
     offenders.append(c['id']);break
  self.assertEqual(offenders[:5],[],f'{len(offenders)} scene claims still carry a copied (unanchored, dependency-cited) citation')
 def test_every_scene_claim_still_declares_dependencies(self):
  target=json.loads((ROOT/'verification/manifests/observatory.json').read_text())
  for c in target['claims']:
   if c['id'].startswith('scene.'):
    with self.subTest(claim=c['id']):
     self.assertTrue(c['dependencies'],'a derived claim with no inputs proves nothing')
