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
 def test_token_pairs_are_measured_to_known_values(self):
  """Pin the exact ratios so the algorithm stays calibrated, not merely passing.

  These five pairs were the original defects (3.28 / 3.49 / 4.15 / 4.30 / 4.38 :1).
  Asserting only "clears 4.5" would still pass if the maths broke in the safe
  direction, so the measured value itself is pinned.
  """
  for theme,expected in (('light',{('--subtle','--bg'):4.75,('--subtle','--surface'):5.24,
                                   ('--amber','--amber-bg'):4.63,('--purple','--purple-bg'):4.63,
                                   ('--muted','--amber-bg'):4.63}),
                         ('dark',{('--subtle','--bg'):5.38,('--subtle','--surface'):4.89,
                                  ('--amber','--amber-bg'):6.96,('--purple','--purple-bg'):6.73})):
   tokens=a.theme_tokens(ROOT/'src/styles.css')[theme]
   for (fg,bg),want in expected.items():
    got=a.contrast_ratio(a.parse_color(tokens[fg])[:3],a.parse_color(tokens[bg])[:3])
    self.assertAlmostEqual(got,want,places=2,msg=f'{theme} {fg} on {bg}')
    self.assertGreaterEqual(got,4.5,f'{theme} {fg} on {bg} regressed below AA')
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
  self.assertEqual(a.required_ratio(11.0,400,c,'chrome'),c['text_normal'])
  self.assertEqual(a.required_ratio(24.0,400,c,'chrome'),c['text_large'])
  self.assertEqual(a.required_ratio(19.0,700,c,'chrome'),c['text_large'])
  # 19px at normal weight is reading size, so the ratchet holds it to AAA rather than AA.
  self.assertEqual(a.required_ratio(19.0,400,c,'chrome'),c['text_enhanced'])
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
   # Keys are model/view/stage/layer/theme; only the theme is split off.
   scenario,theme=label.rsplit('/',1);scenarios.setdefault(scenario,{})[theme]=sig
  for key,per_theme in scenarios.items():
   self.assertEqual(set(per_theme),{'dark','light'},f'{key} is not baselined in both themes')
   self.assertEqual(len(set(per_theme.values())),1,f'{key} composition depends on the theme')
  stages={k.split('/')[2] for k in scenarios}
  self.assertTrue({'inputs','residual','ple'}<=stages,'focal stages are not baselined')

class CompositionBaselineTests(unittest.TestCase):
 """The baseline is only drift protection over the scenarios it actually compares.

 The gate used to report `len(current)` scenarios as "matching the recorded baseline" while
 comparing only the labels the baseline already held, and `current - recorded` was covered by
 nothing -- so a new checkpoint's signatures were unprotected behind a PASS line, and the public
 record quoted the baselined count as the measured one.
 """
 def codes(self,findings,status):
  return {f.code for f in findings if f.status==status}
 def test_a_measured_scenario_with_no_recorded_signature_is_caught(self):
  f=a.judge_composition({'x/dark':'1','new/dark':'2'},{'x/dark':'1'})
  self.assertIn('COMPOSITION_NEW',self.codes(f,'FAIL'))
  self.assertEqual(self.codes(f,'PASS'),set())
 def test_the_stable_count_is_the_compared_count_and_not_the_measured_one(self):
  """The exact defect: two measured, one baselined, and a PASS line saying two matched."""
  f=a.judge_composition({'x/dark':'1','new/dark':'2'},{'x/dark':'1'})
  self.assertNotIn('COMPOSITION_STABLE',{x.code for x in f})
 def test_a_moved_signature_is_still_caught(self):
  f=a.judge_composition({'x/dark':'2'},{'x/dark':'1'})
  self.assertIn('COMPOSITION_DRIFT',self.codes(f,'FAIL'))
 def test_a_vanished_scenario_is_still_caught(self):
  f=a.judge_composition({},{'x/dark':'1'})
  self.assertIn('COMPOSITION_MISSING',self.codes(f,'FAIL'))
 def test_an_exact_match_passes_and_states_the_compared_number(self):
  f=a.judge_composition({'x/dark':'1','y/dark':'2'},{'x/dark':'1','y/dark':'2'})
  self.assertIn('COMPOSITION_STABLE',self.codes(f,'PASS'))
  self.assertIn('2 measured scenarios',f[0].message)
 def test_comparing_nothing_does_not_pass(self):
  """The same defect this gate exists to catch, found in this gate's own PASS line: an empty
  current set against an empty baseline used to return COMPOSITION_STABLE with '0 measured
  scenarios', reporting success having compared nothing at all."""
  f=a.judge_composition({},{})
  self.assertEqual(self.codes(f,'PASS'),set())
  self.assertIn('COMPOSITION_EMPTY',self.codes(f,'BLOCKED'))

class ChromeProbeCoverageTests(unittest.TestCase):
 """The probe's selector list is only evidence if something ties it to the page.

 The failure these close is the one this gate had already been caught making one layer down: a
 hardcoded selector string, no completeness guard, and a rename that removes an element from
 measurement while the report still says PASS.
 """
 def setUp(self):
  self.policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
  self.css=ROOT/'src/styles.css'
  self.markup=(ROOT/'src/app.mjs',ROOT/'src/worked-examples.mjs',ROOT/'src/workspace-ui.mjs',ROOT/'src/mechanism-visuals.mjs',ROOT/'index.html')
 def test_the_shipped_probe_list_is_live_and_complete(self):
  for judge in (a.judge_chrome_probe(self.css,self.markup),
                a.judge_chrome_coverage(self.css,self.policy)):
   bad=[f for f in judge if f.status!='PASS']
   self.assertEqual(bad,[],'\n'.join(f'{f.subject}: {f.message}' for f in bad))
 def test_the_probe_string_is_built_from_the_list(self):
  """Otherwise the list is documentation and the probe still carries its own copy."""
  self.assertNotIn('__CHROME_TEXT__',a.PROBE)
  for selector in a.CHROME_TEXT_SELECTORS:
   self.assertIn(selector,a.PROBE,f'{selector} is listed but not measured')
 def test_a_renamed_class_is_caught_rather_than_silently_unmeasured(self):
  """The exact reported failure: rename .innovation-note and measurement disappears."""
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   css=Path(tmp)/'x.css';css.write_text('.innovation-lede{font-size:12px;color:var(--text)}\n')
   markup=Path(tmp)/'x.mjs';markup.write_text('`<p class="innovation-lede">…</p>`\n')
   codes={f.code for f in a.judge_chrome_probe(css,(markup,)) if f.status!='PASS'}
   self.assertIn('CHROME_PROBE_STALE',codes)
 def test_a_class_kept_in_css_but_dropped_from_the_markup_is_caught(self):
  """A selector that only the stylesheet still knows about measures nothing either."""
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   css=Path(tmp)/'x.css';css.write_text('.innovation-note{font-size:12px}\n.chart text{}\n')
   markup=Path(tmp)/'x.mjs';markup.write_text('nothing renders these\n')
   bad=[f for f in a.judge_chrome_probe(css,(markup,)) if f.status!='PASS']
   self.assertTrue(bad)
   self.assertIn('markup',bad[0].message)
 def test_new_reading_size_prose_in_neither_list_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   css=Path(tmp)/'x.css';css.write_text('.brand-new-prose{font-size:12px;color:var(--muted)}\n')
   findings=a.judge_chrome_coverage(css,self.policy)
   self.assertEqual({f.code for f in findings if f.status!='PASS'},{'CHROME_UNMEASURED'})
 def test_the_font_shorthand_is_not_a_way_around_the_coverage_guard(self):
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   css=Path(tmp)/'x.css';css.write_text('.sneaky{font:12px ui-monospace,Consolas,monospace}\n')
   self.assertIn('CHROME_UNMEASURED',{f.code for f in a.judge_chrome_coverage(css,self.policy)})
 def test_canvas_and_large_text_are_out_of_scope_deliberately(self):
  """Canvas labels are held to AA on the semantic tints, and >= large_text_px is a heading under
  WCAG's own relaxation. Including either would make the guard fire on text the AAA rule does not
  reach, which is how a check gets turned off."""
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   css=Path(tmp)/'x.css'
   css.write_text('.schematic .node-title{font-size:12px}\n.huge{font-size:32px}\n'
                  '.tiny{font-size:10px}\n')
   self.assertEqual([f for f in a.judge_chrome_coverage(css,self.policy) if f.status!='PASS'],[])
 def test_the_recorded_debt_is_named_as_debt_and_not_as_coverage(self):
  """A reader of the PASS message must be told the measured set is not everything."""
  message=a.judge_chrome_coverage(self.css,self.policy)[0].message
  self.assertIn('recorded as unmeasured',message)
  self.assertTrue(set(a.UNMEASURED_CHROME).isdisjoint(a.CHROME_TEXT_SELECTORS),
                  'a selector cannot be both measured and recorded as unmeasured')

class StylesheetScaleTests(unittest.TestCase):
 """The scales must be enforced, or the ad-hoc values silently return."""
 def setUp(self):
  self.policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
  self.css=ROOT/'src/styles.css'
 def test_declarations_are_parsed_from_the_stylesheet(self):
  decls=a.stylesheet_declarations(self.css)
  self.assertGreater(len(decls),200,'the stylesheet parser found almost nothing')
  # Rules inside @media must be attributed to their own selector, not the media query.
  self.assertTrue(any(d.selector.startswith('.schematic') for d in decls))
  self.assertFalse(any(d.selector.lstrip().startswith('@') for d in decls))
 def test_schematic_rules_are_distinguished_from_chrome(self):
  decls=a.stylesheet_declarations(self.css)
  canvas=[d for d in decls if d.canvas and d.prop=='font-size']
  chrome=[d for d in decls if not d.canvas and d.prop=='font-size']
  self.assertTrue(canvas and chrome)
  self.assertTrue(all('.schematic' in d.selector for d in canvas))
 def test_every_font_size_belongs_to_its_scale(self):
  findings=a.judge_type_scale(self.css,self.policy)
  bad=[f for f in findings if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f'{f.subject}: {f.message}' for f in bad))
 def test_every_spacing_value_belongs_to_the_scale(self):
  findings=a.judge_spacing_scale(self.css,self.policy)
  bad=[f for f in findings if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f'{f.subject}: {f.message}' for f in bad))
 def test_the_font_shorthand_is_not_a_way_around_the_scale(self):
  import tempfile
  # .source-role and .lens-tab .idx set their size via `font:`, and that is where two of
  # the original defects lived. Checking only `font-size` would miss them.
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.css';p.write_text('.a{font:8px ui-monospace,Consolas,monospace}\n')
   bad=[f for f in a.judge_type_scale(p,self.policy) if f.status!='PASS']
   self.assertTrue(bad,'the font shorthand escaped the type scale')
   self.assertIn('8.0px',bad[0].message)
 def test_an_off_scale_size_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.css';p.write_text('.a{font-size:13.5px}\n.schematic .b{font-size:9.25px}\n')
   codes={f.code for f in a.judge_type_scale(p,self.policy) if f.status!='PASS'}
   self.assertIn('TYPE_OFF_SCALE',codes)
 def test_an_off_scale_gap_is_caught(self):
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.css';p.write_text('.a{padding:7px 13px}\n')
   codes={f.code for f in a.judge_spacing_scale(p,self.policy) if f.status!='PASS'}
   self.assertIn('SPACING_OFF_SCALE',codes)

class ContrastRatchetTests(unittest.TestCase):
 def setUp(self):self.policy=json.loads((ROOT/'verification/baselines/aesthetics/policy.json').read_text())
 def test_reading_size_chrome_prose_is_held_to_the_enhanced_ratio(self):
  # AA is the floor everywhere; reading-size chrome prose is held to the AAA target.
  c=self.policy['contrast']
  self.assertIn('text_enhanced',c)
  self.assertGreaterEqual(c['text_enhanced'],7.0)
  self.assertEqual(a.required_ratio(14.0,400,c,'chrome'),c['text_enhanced'])
  self.assertEqual(a.required_ratio(10.0,400,c,'chrome'),c['text_normal'])
 def test_canvas_labels_keep_the_aa_floor(self):
  # Canvas labels sit on semantic tint fills -- amber means retained state, violet means
  # communication (OBSERVATORY.md). Forcing 7:1 there would wash the grammar out of the
  # diagram, so the canvas is held to AA and its AAA clearance is reported instead.
  c=self.policy['contrast']
  self.assertEqual(a.required_ratio(14.0,400,c,'svg'),c['text_normal'])
  self.assertEqual(a.required_ratio(12.0,550,c,'svg'),c['text_normal'])
 def test_large_text_still_uses_the_relaxed_threshold(self):
  c=self.policy['contrast']
  self.assertEqual(a.required_ratio(24.0,400,c),c['text_large'])

class CoverageTests(unittest.TestCase):
 """The gate must measure what it claims to measure."""
 @classmethod
 def setUpClass(cls):
  cls.root=ROOT
  cls.models,cls.focal,cls.views=a.scenario_inventory(ROOT)
 def test_model_list_comes_from_the_application_not_a_literal(self):
  import subprocess
  inv=json.loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  self.assertEqual(set(self.models),set(inv['models']),'the gate measures a different model set than the application defines')
  # The floor that stood here was `assertGreaterEqual(len(self.models),8)`. It could not fire for
  # the reason it was written -- it would have to drop below eight, not fail to reach nine -- and
  # the assertEqual above is the real check. A bound that cannot detect the drift it names reads
  # as coverage and is not.
 def test_view_list_comes_from_the_application_not_a_literal(self):
  """The mirror of the model check, for the axis that had no check at all.

  `run()` held its own seven-element literal beside the derived model list, so a new lens was
  measured in no theme and baselined under no label -- and the gate's own PASS line would have
  reported a smaller compared count with nothing to compare it against."""
  import subprocess
  inv=json.loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  self.assertEqual(list(self.views),list(inv['fields']['VIEWS']),'the gate measures a different lens set than the application defines')
  measured={c['view'] for c in a.cases(self.models,self.views,['dark'],self.focal)}
  self.assertEqual(measured,set(inv['fields']['VIEWS']),'a lens the application ships is measured in no scenario')
  # Both assertions above read the helper, so neither would see a literal smuggled back into
  # `run()`. This one reads what `run()` actually swept, recorded in the report's own `axes`.
  report=ROOT/'verification/reports/aesthetics.json'
  if not report.exists():
   self.skipTest('no aesthetics report; run npm run verify:aesthetics')
  axes=json.loads(report.read_text()).get('axes')
  self.assertIsNotNone(axes,'the aesthetics report records no swept axes, so its coverage cannot be reconciled')
  self.assertEqual(list(axes['views']),list(inv['fields']['VIEWS']),
                   'the gate swept a different lens set than the application defines')
  self.assertEqual(set(axes['models']),set(inv['models']),
                   'the gate swept a different model set than the application defines')
 def test_focal_stages_are_measured(self):
  # cases() has always promised focal stages in its docstring and emitted none, so the
  # inputs / residual / ple scenes have never been contrast-checked.
  cases=a.cases(self.models,['model'],['dark'],self.focal)
  stages={c.get('stage') for c in cases}
  self.assertIn('inputs',stages)
  self.assertIn('residual',stages)
  self.assertIn('ple',stages)
 def test_the_focal_stage_map_is_the_applications_own_and_not_re_derived(self):
  """The sixth copy of the predicate lived here. `core.focalStages` is now the only one, so
  the gate cannot generate a page the application does not offer -- which is how a signature
  labelled `<mid>/model/ple` could be measured on the decoder block and baselined under that
  name."""
  import subprocess
  inv=json.loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
  self.assertEqual(self.focal,inv['samples']['focal_stages'])
  self.assertEqual(set(self.focal),set(self.models),'a model has no focal-stage entry at all')
  for mid,stages in self.focal.items():
   measured={c['stage'] for c in a.cases(self.models,['model'],['dark'],self.focal) if c['model']==mid}-{'block'}
   self.assertEqual(measured,set(stages),f'{mid} focal coverage')
 def test_focal_stages_only_where_the_model_has_them(self):
  cases=a.cases(self.models,['model'],['dark'],self.focal)
  ple={c['model'] for c in cases if c.get('stage')=='ple'}
  self.assertEqual(ple,{'q38flash'},'ple is a Flash-Next-only scene')
  inputs={c['model'] for c in cases if c.get('stage')=='inputs'}
  self.assertTrue(inputs and inputs<=set(self.models))
 def test_more_than_one_layer_is_sampled(self):
  layers={c.get('layer') for c in a.cases(self.models,['model'],['dark'],self.focal) if c.get('stage')=='block'}
  self.assertGreater(len(layers),1,'only layer 4 was ever measured')
 def test_clearance_result_is_reported_even_when_it_passes(self):
  report=json.loads((ROOT/'verification/reports/aesthetics.json').read_text())
  codes={f['code'] for f in report['findings']}
  self.assertTrue(codes&{'CLEARANCE_OK','AESTHETIC_CLEARANCE'},'a reader cannot tell clearance was checked')
 def test_edge_crossing_budget_is_live(self):
  report=json.loads((ROOT/'verification/reports/aesthetics.json').read_text())
  observed=report['edge_crossings']['max'];budget=report['edge_crossings']['budget']
  self.assertLessEqual(budget,max(observed*2,observed+4),
                       f'budget {budget} against an observed max of {observed} cannot fail')

class WorkspacePanelCoverageTests(unittest.TestCase):
 def test_workspace_inventory_derives_panel_support_from_live_operations(self):
  states=a.workspace_states(ROOT)
  self.assertTrue(states)
  self.assertTrue(all(s['panel']!='schematic' and s['module'] for s in states))
  self.assertEqual(len(states),len({json.dumps(s,sort_keys=True) for s in states}))
  import subprocess
  code="""import {sanitizeState} from './src/core.mjs';
import {buildScene} from './src/diagrams.mjs';
import {resolveWorkspace} from './src/workspace.mjs';
let text='';for await(const chunk of process.stdin)text+=chunk;
const cases=JSON.parse(text);const invalid=cases.filter(c=>{const s=sanitizeState(c),w=resolveWorkspace(s,buildScene(s));return w.panel!==c.panel||w.module!==c.module||(c.example&&w.exampleId!==c.example)});console.log(JSON.stringify(invalid));"""
  invalid=json.loads(subprocess.check_output(['node','--input-type=module','-e',code],cwd=ROOT,input=json.dumps(states),text=True))
  self.assertEqual(invalid,[])
 def test_every_live_model_has_its_supported_reading_panels_measured(self):
  states=a.workspace_states(ROOT)
  models=a.scenario_models(ROOT)
  self.assertEqual({s['model'] for s in states},set(models))
  for mid in models:
   panels={s['panel'] for s in states if s['model']==mid}
   self.assertTrue({'maths','sources'}<=panels,mid)
  self.assertTrue(any(s['example'] for s in states if s['panel']=='worked'))
 def test_workspace_preview_coverage_rejects_a_missing_panel(self):
  from verification.presentations import judge_workspace_preview_coverage
  states=[{'model':'m','view':'model','stage':'chapter','module':'op','panel':'maths','example':''}]
  self.assertTrue(any(f.status=='FAIL' for f in judge_workspace_preview_coverage([],states)))
  captures=[{'state':states[0]}]
  self.assertTrue(all(f.status=='PASS' for f in judge_workspace_preview_coverage(captures,states)))
