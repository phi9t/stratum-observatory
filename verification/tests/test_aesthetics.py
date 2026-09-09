import json,unittest
from pathlib import Path
from verification import aesthetics as a

ROOT=Path(__file__).resolve().parents[2]

class ColorMathTests(unittest.TestCase):
 """WCAG arithmetic, checked against values computed independently from src/styles.css."""
 def test_parses_the_css_colour_forms_the_renderer_emits(self):
  self.assertEqual(a.parse_color('rgb(154, 169, 171)'),(154,169,171,1.0))
  self.assertEqual(a.parse_color('rgba(154, 169, 171, 0.5)'),(154,169,171,0.5))
  self.assertEqual(a.parse_color('#9aa9ab'),(154,169,171,1.0))
  self.assertIsNone(a.parse_color('none'))
  self.assertIsNone(a.parse_color('url(#grad)'))
 def test_known_contrast_ratios(self):
  self.assertAlmostEqual(a.contrast_ratio((255,255,255),(0,0,0)),21.0,places=2)
  self.assertAlmostEqual(a.contrast_ratio((0,0,0),(0,0,0)),1.0,places=2)
 def test_light_theme_failures_are_rediscovered_from_the_stylesheet(self):
  # These are the real, currently-shipping violations. If the algorithm drifts in
  # either direction these numbers move and this test fails.
  tokens=a.theme_tokens(ROOT/'src/styles.css')['light']
  def ratio(fg,bg):return a.contrast_ratio(a.parse_color(tokens[fg])[:3],a.parse_color(tokens[bg])[:3])
  self.assertAlmostEqual(ratio('--subtle','--bg'),3.28,places=2)
  self.assertAlmostEqual(ratio('--subtle','--canvas'),3.58,places=2)
  self.assertAlmostEqual(ratio('--amber','--amber-bg'),4.15,places=2)
  self.assertAlmostEqual(ratio('--purple','--purple-bg'),4.30,places=2)
 def test_translucent_foreground_is_composited_before_measuring(self):
  # .dim sets opacity .22 on a group; measuring the raw fill would overstate legibility.
  solid=a.composite([( (255,255,255),1.0 )],(0,0,0))
  self.assertEqual(solid,(255,255,255))
  faded=a.composite([((255,255,255),0.22)],(0,0,0))
  self.assertLess(a.contrast_ratio(faded,(0,0,0)),a.contrast_ratio(solid,(0,0,0)))
 def test_theme_token_key_parity(self):
  tokens=a.theme_tokens(ROOT/'src/styles.css')
  self.assertEqual(set(tokens['dark']),set(tokens['light']))
  self.assertGreater(len(tokens['dark']),15)

class ThresholdTests(unittest.TestCase):
 def setUp(self):self.policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
 def test_large_text_uses_the_relaxed_threshold(self):
  c=self.policy['contrast']
  self.assertEqual(a.required_ratio(11.0,400,c),c['text_normal'])
  self.assertEqual(a.required_ratio(24.0,400,c),c['text_large'])
  self.assertEqual(a.required_ratio(19.0,700,c),c['text_large'])
  self.assertEqual(a.required_ratio(19.0,400,c),c['text_normal'])
 def test_indeterminate_backdrop_blocks_rather_than_passes(self):
  sample={'kind':'text','id':'t','fill':'rgb(0,0,0)','alpha':1.0,'px':12.0,'weight':400,'backdrop':None}
  finding=a.judge_contrast(sample,self.policy)
  self.assertEqual(finding.status,'BLOCKED')
  self.assertEqual(finding.code,'CONTRAST_INDETERMINATE')
 def test_failing_pair_is_reported_as_a_contradiction(self):
  sample={'kind':'text','id':'t','fill':'rgb(140,140,140)','alpha':1.0,'px':11.0,'weight':400,
          'backdrop':[{'color':'rgb(255,255,255)','alpha':1.0}]}
  self.assertEqual(a.judge_contrast(sample,self.policy).status,'FAIL')
 def test_passing_pair_is_reported_as_pass(self):
  sample={'kind':'text','id':'t','fill':'rgb(20,20,20)','alpha':1.0,'px':11.0,'weight':400,
          'backdrop':[{'color':'rgb(255,255,255)','alpha':1.0}]}
  self.assertEqual(a.judge_contrast(sample,self.policy).status,'PASS')

class LayoutRuleTests(unittest.TestCase):
 def setUp(self):self.policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
 def test_zero_clearance_is_reported_even_though_boxes_do_not_intersect(self):
  # app.mjs overlap() tests intersection, so touching labels pass the geometry gate.
  boxes=[{'id':'a','x':0,'y':0,'w':10,'h':10},{'id':'b','x':10.1,'y':0,'w':10,'h':10}]
  codes=[f.code for f in a.judge_clearance(boxes,self.policy)]
  self.assertIn('AESTHETIC_CLEARANCE',codes)
 def test_adequate_clearance_passes(self):
  boxes=[{'id':'a','x':0,'y':0,'w':10,'h':10},{'id':'b','x':20,'y':0,'w':10,'h':10}]
  self.assertEqual([f for f in a.judge_clearance(boxes,self.policy) if f.status!='PASS'],[])
 def test_crossing_edges_are_counted(self):
  cross=[{'id':'e1','points':[[0,5],[10,5]]},{'id':'e2','points':[[5,0],[5,10]]}]
  self.assertEqual(a.count_crossings(cross),1)
  parallel=[{'id':'e1','points':[[0,5],[10,5]]},{'id':'e2','points':[[0,8],[10,8]]}]
  self.assertEqual(a.count_crossings(parallel),0)

class CompositionBaselineTests(unittest.TestCase):
 def test_palette_retune_does_not_move_the_baseline(self):
  scene={'nodes':[{'id':'n','x':1.2,'y':2.8,'w':10,'h':10,'name':'A','sub':'','tone':'accent','kind':'op'}],
         'edges':[],'groups':[],'labels':[]}
  before=a.composition_signature(scene,{'--accent':'#4dd0e1'})
  after=a.composition_signature(scene,{'--accent':'#00bcd4'})
  self.assertEqual(before,after,'a colour tweak that preserves roles changed the baseline')
 def test_role_change_does_move_the_baseline(self):
  base={'nodes':[{'id':'n','x':1,'y':2,'w':10,'h':10,'name':'A','sub':'','tone':'accent','kind':'op'}],
        'edges':[],'groups':[],'labels':[]}
  other=json.loads(json.dumps(base));other['nodes'][0]['tone']='amber'
  self.assertNotEqual(a.composition_signature(base,{}),a.composition_signature(other,{}))
 def test_moved_lane_does_move_the_baseline(self):
  base={'nodes':[{'id':'n','x':1,'y':2,'w':10,'h':10,'name':'A','sub':'','tone':'accent','kind':'op'}],
        'edges':[],'groups':[],'labels':[]}
  other=json.loads(json.dumps(base));other['nodes'][0]['y']=40
  self.assertNotEqual(a.composition_signature(base,{}),a.composition_signature(other,{}))
 def test_subpixel_jitter_does_not_move_the_baseline(self):
  base={'nodes':[{'id':'n','x':1.0,'y':2.0,'w':10,'h':10,'name':'A','sub':'','tone':'accent','kind':'op'}],
        'edges':[],'groups':[],'labels':[]}
  other=json.loads(json.dumps(base));other['nodes'][0]['x']=1.4
  self.assertEqual(a.composition_signature(base,{}),a.composition_signature(other,{}))

class GateContractTests(unittest.TestCase):
 """The gate must be wired the way the project's other gates are."""
 def test_policy_thresholds_meet_wcag_aa(self):
  policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
  # A gate whose thresholds drift below the standard it cites is decorative.
  self.assertGreaterEqual(policy['contrast']['text_normal'],4.5)
  self.assertGreaterEqual(policy['contrast']['text_large'],3.0)
  self.assertGreaterEqual(policy['contrast']['graphics'],3.0)
 def test_policy_is_bound_to_the_receipt_but_baselines_are_not(self):
  from verification.engineering import toolchain_digest
  import tempfile,subprocess
  # Loosening a threshold must invalidate every receipt; re-recording a reviewed
  # layout change must not.
  before=toolchain_digest(ROOT)
  with tempfile.TemporaryDirectory() as tmp:
   copy=Path(tmp)/'repo'
   subprocess.run(['cp','-R',str(ROOT),str(copy)],check=True,capture_output=True)
   p=copy/'verification/baselines/aesthetics/policy.json'
   d=json.loads(p.read_text());d['contrast']['text_normal']=1.0;p.write_text(json.dumps(d))
   self.assertNotEqual(before,toolchain_digest(copy),'policy.json is outside toolchain_digest')
   # Restore the exact original bytes; re-serialising would move the digest by itself.
   p.write_bytes((ROOT/'verification/baselines/aesthetics/policy.json').read_bytes())
   b=copy/'verification/baselines/aesthetics/composition.json'
   b.write_text(json.dumps({'schema':'stratum-aesthetics-baseline/v1','composition':{}}))
   self.assertEqual(before,toolchain_digest(copy),'re-recording a baseline invalidated the receipt')
 def test_aesthetics_is_not_a_scientific_domain(self):
  from verification.domains.contracts import DOMAINS
  # DESIGN.md trust boundary 1: local inputs are never evidence for themselves.
  # Adding an aesthetics domain would also force a contract and a runtime obligation.
  self.assertNotIn('aesthetics',DOMAINS)
 def test_gate_exit_codes_match_the_other_gates(self):
  from verification.model import aggregate,exit_code
  from verification.model import Finding
  self.assertEqual(exit_code(aggregate([Finding('X','aesthetics','PASS','a','ok')])),0)
  self.assertEqual(exit_code(aggregate([Finding('X','aesthetics','FAIL','a','bad')])),1)
  self.assertEqual(exit_code(aggregate([Finding('X','aesthetics','BLOCKED','a','unknown')])),2)
 def test_recorded_baseline_covers_every_scenario_in_both_themes(self):
  b=json.loads((ROOT/'verification/baselines/aesthetics/composition.json').read_text())['composition']
  self.assertTrue(b)
  scenarios={}
  for label,sig in b.items():
   model,view,theme=label.split('/');scenarios.setdefault((model,view),{})[theme]=sig
  for key,per_theme in scenarios.items():
   self.assertEqual(set(per_theme),{'dark','light'},f'{key} is not baselined in both themes')
   self.assertEqual(len(set(per_theme.values())),1,f'{key} composition depends on the theme')
