import json,subprocess,unittest
from pathlib import Path
from verification import schematics as s

ROOT=Path(__file__).resolve().parents[2]

def scene(**over):
 base={'width':100,'height':100,'caption':'','defaultSelected':None,
       'nodes':[{'id':'a','x':0,'y':0,'w':10,'h':10,'name':'A','sub':'','kind':'op','tone':'neutral','small':False,'info':None},
                {'id':'b','x':40,'y':0,'w':10,'h':10,'name':'B','sub':'','kind':'op','tone':'neutral','small':False,'info':None}],
       'edges':[{'id':'wire-0','source':'a','target':'b','kind':'flow','arrow':True,'delay':False,'points':[[10,5],[40,5]]}],
       'groups':[],'labels':[],'decorations':[]}
 base.update(over);return base

class IRContractTests(unittest.TestCase):
 """The IR has closed value sets and structuralReport checks none of them."""
 def test_clean_scene_passes(self):
  self.assertEqual([f for f in s.judge_ir(scene()) if f.status!='PASS'],[])
 def test_unknown_node_tone_is_rejected(self):
  g=scene();g['nodes'][0]['tone']='chartreuse'
  self.assertIn('IR_VALUE',{f.code for f in s.judge_ir(g) if f.status!='PASS'})
 def test_unknown_edge_kind_is_rejected(self):
  g=scene();g['edges'][0]['kind']='telepathy'
  self.assertIn('IR_VALUE',{f.code for f in s.judge_ir(g) if f.status!='PASS'})
 def test_group_outside_the_canvas_is_rejected(self):
  # structuralReport never inspects groups at all.
  g=scene(groups=[{'id':'g','x':-5,'y':0,'w':20,'h':20,'title':'T','subtitle':''}])
  self.assertIn('IR_GROUP',{f.code for f in s.judge_ir(g) if f.status!='PASS'})
 def test_duplicate_group_id_is_rejected(self):
  gr={'id':'g','x':0,'y':0,'w':20,'h':20,'title':'T','subtitle':''}
  self.assertIn('IR_GROUP',{f.code for f in s.judge_ir(scene(groups=[gr,dict(gr)])) if f.status!='PASS'})
 def test_single_point_wire_is_rejected(self):
  g=scene();g['edges'][0]['points']=[[10,5]]
  self.assertIn('IR_EDGE',{f.code for f in s.judge_ir(g) if f.status!='PASS'})
 def test_endpoints_are_semantic_not_geometric(self):
  # A residual rail declares where the stream comes from, not where this leg is drawn.
  # geometryReport uses source/target to exempt a wire from crossing its own endpoints;
  # the IR has never promised the drawn points touch those boxes.
  g=scene();g['edges'][0]['points']=[[80,80],[90,80]]
  self.assertEqual([f for f in s.judge_ir(g) if f.status!='PASS'],[])

class StateRailTests(unittest.TestCase):
 def test_delay_kind_and_delay_flag_must_agree(self):
  g=scene();g['edges'][0]['kind']='delay';g['edges'][0]['delay']=False
  self.assertIn('RAIL_DELAY',{f.code for f in s.judge_state_rails(g) if f.status!='PASS'})
  g2=scene();g2['edges'][0]['delay']=True
  self.assertIn('RAIL_DELAY',{f.code for f in s.judge_state_rails(g2) if f.status!='PASS'})
 def test_state_toned_annotation_card_is_not_a_defect(self):
  # Annotation cards are legitimately toned 'state' without a rail reaching them.
  g=scene();g['nodes'][1]['tone']='state'
  self.assertEqual([f for f in s.judge_state_rails(g) if f.status!='PASS'],[])

class OrphanMetricTests(unittest.TestCase):
 """Connectivity is recorded, not enforced: annotation cards are unwired on purpose."""
 def test_unwired_node_is_counted(self):
  g=scene();g['nodes'].append({'id':'note','x':70,'y':70,'w':10,'h':10,'name':'N','sub':'','kind':'op','tone':'neutral','small':False,'info':None})
  self.assertEqual(s.count_orphans(g),1)
 def test_fully_wired_scene_counts_zero(self):
  self.assertEqual(s.count_orphans(scene()),0)
 def test_panel_layout_without_wires_counts_zero(self):
  self.assertEqual(s.count_orphans(scene(edges=[])),0)

class LaneTests(unittest.TestCase):
 def test_collinear_overlapping_wires_are_reported(self):
  # count_crossings cannot see this: the segments never cross, they coincide.
  g=scene(edges=[{'id':'w0','source':'a','target':'b','kind':'flow','arrow':True,'delay':False,'points':[[0,5],[50,5]]},
                 {'id':'w1','source':'c','target':'d','kind':'flow','arrow':True,'delay':False,'points':[[20,5],[70,5]]}])
  self.assertIn('LANE_COLLINEAR',{f.code for f in s.judge_lanes(g) if f.status!='PASS'})
 def test_fan_out_from_one_node_may_share_a_trunk(self):
  g=scene(edges=[{'id':'w0','source':'a','target':'b','kind':'flow','arrow':True,'delay':False,'points':[[0,5],[50,5]]},
                 {'id':'w1','source':'a','target':'c','kind':'flow','arrow':True,'delay':False,'points':[[0,5],[30,5]]}])
  self.assertEqual([f for f in s.judge_lanes(g) if f.status!='PASS'],[])
 def test_parallel_wires_in_different_lanes_pass(self):
  g=scene(edges=[{'id':'w0','source':'a','target':'b','kind':'flow','arrow':True,'delay':False,'points':[[0,5],[50,5]]},
                 {'id':'w1','source':'a','target':'b','kind':'flow','arrow':True,'delay':False,'points':[[0,25],[50,25]]}])
  self.assertEqual([f for f in s.judge_lanes(g) if f.status!='PASS'],[])

class SceneVersusModelTests(unittest.TestCase):
 """The diagram must agree with core.mjs, not merely with itself."""
 def test_dense_model_must_not_draw_a_router(self):
  g=scene();g['nodes'][0]['id']='router'
  expect={'model':'q38','experts':0,'topK':0,'shared':0,'layers':36,'partition':{},'denseLayers':0}
  self.assertIn('SCENE_MODEL_ROUTER',{f.code for f in s.judge_scene_vs_model(g,expect,{'view':'model','stage':'block','layer':4}) if f.status!='PASS'})
 def test_shared_expert_absent_when_the_model_declares_none(self):
  g=scene();g['nodes'][0]['id']='shared-up'
  expect={'model':'q330','experts':128,'topK':8,'shared':0,'layers':48,'partition':{},'denseLayers':0}
  self.assertIn('SCENE_MODEL_SHARED',{f.code for f in s.judge_scene_vs_model(g,expect,{'view':'model','stage':'block','layer':4}) if f.status!='PASS'})
 def test_layer_tiles_must_match_the_partition(self):
  g=scene(decorations=[{'kind':'layer','layer':1,'id':'layer-1','x':0,'y':0,'w':23,'h':21,'attention':'mla','active':False,'dense':False}])
  expect={'model':'ds2','experts':160,'topK':6,'shared':2,'layers':2,'partition':{'mla':2},'denseLayers':1}
  bad={f.code for f in s.judge_scene_vs_model(g,expect,{'view':'model','stage':'block','layer':1}) if f.status!='PASS'}
  self.assertIn('SCENE_MODEL_TILES',bad)

class RealSceneTests(unittest.TestCase):
 """Every judge must hold over the actual 684-scene matrix."""
 @classmethod
 def setUpClass(cls):
  cls.data=s.export_scenes(ROOT)
 def test_the_matrix_covers_every_decoder_layer(self):
  layers=[x for x in self.data['scenes'] if x['key'].split('/')[1]=='model' and x['key'].split('/')[2]=='block']
  self.assertEqual(len(layers),504)
 def test_row_alignment_across_presets(self):
  bad=[f for f in s.judge_row_alignment(self.data,4.0) if f.status!='PASS']
  self.assertEqual(bad,[],'\n'.join(f'{f.subject}: {f.message}' for f in bad[:10]))
 def test_core_operations_are_pixel_identical_across_every_preset(self):
  # The principle's strongest form, asserted directly rather than inferred from the rule.
  from collections import defaultdict
  rows=defaultdict(dict)
  for e in self.data['scenes']:
   m,v,st,l=e['key'].split('/')
   if v!='model' or st!='block' or l!='4':continue
   for n in e['scene']['nodes']:rows[n['id']][m]=n['y']+n['h']/2
  for node_id in ('norm1','attention'):
   centres=set(rows[node_id].values())
   self.assertEqual(len(centres),1,f'{node_id} is not on one row across all presets: {rows[node_id]}')
 def test_a_third_row_is_rejected(self):
  # Two rows are permitted for a structural variant; three means no row at all.
  fake={'scenes':[{'key':f'm{i}/model/block/4','scene':{'nodes':[{'id':'op','x':0,'y':i*40,'w':10,'h':10}]}} for i in range(3)]}
  self.assertIn('ROW_DRIFT',{f.code for f in s.judge_row_alignment(fake,4.0) if f.status!='PASS'})
