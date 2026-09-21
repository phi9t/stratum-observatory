import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS,DEFAULT_STATE,sanitizeState} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {resolveWorkspace} from '../src/workspace.mjs';
import {SOURCES,DSPARK_PRO} from '../src/models.mjs';
import {MECHANISM_STAGES,mechanismChapter} from '../src/mechanism-chapters.mjs';
import {renderMechanismChapter,normalizeChapterExample} from '../src/mechanism-visuals.mjs';
import {WORKED_RESULTS} from '../src/worked-results.mjs';
import {INPUT_CASES} from '../src/input-cases.mjs';
const state=stage=>sanitizeState({...DEFAULT_STATE,model:'dsv41',view:'model',stage,panel:'chapter'});
test('every authored chapter section resolves to a real operation, CPU fixture and pinned source',()=>{
 for(const stage of MECHANISM_STAGES){
  const s=state(stage),scene=buildScene(s);
  for(const section of mechanismChapter(MODELS.dsv41,stage).sections){
   assert.equal(resolveWorkspace({...s,module:section.module},scene).module,section.module);
   assert.ok(WORKED_RESULTS[section.example]);
   for(const id of section.sources){assert.ok(SOURCES[id],id);assert.ok(!SOURCES[id].url.includes('/main/'),id);}
  }
  const html=renderMechanismChapter(s,MODELS.dsv41);
  assert.doesNotMatch(html,/undefined|NaN|>missing</);
  assert.ok(html.includes('<math ')&&html.includes('Plain-text formula'));
 }
});
test('chapter selections reject cross-mechanism, out-of-bounds and injected identities',()=>{
 for(const [stage,example] of [['csa2','chap--residual--1'],['inputs','chap--chart-question--99999'],['residual','chap--residual---1'],['engram','chap--engram--2'],['inputs','chap--<script>--0']])assert.equal(normalizeChapterExample({...state(stage),example}),'');
 assert.equal(resolveWorkspace({...state('csa2'),model:'kimi3'},buildScene({...state('csa2'),model:'kimi3'})).panel,'schematic');
});
test('query and normalization selections render different stored CPU intermediates',()=>{
 for(const [stage,a,b] of [['csa2',2,3],['residual',0,39],['engram',0,1]]){
  const htmlA=renderMechanismChapter({...state(stage),example:`chap--${stage}--${a}`},MODELS.dsv41);
  const htmlB=renderMechanismChapter({...state(stage),example:`chap--${stage}--${b}`},MODELS.dsv41);
  assert.notEqual(htmlA,htmlB);
 }
 assert.match(renderMechanismChapter(state('residual'),MODELS.dsv41),/Max \|row sum − 1\|/);
});
test('every mixed sequence position and contributing patch is selectable with its saved identity',()=>{
 const c=INPUT_CASES.cases.find(c=>c.image),position=c.positions.find(p=>p.type_name==='IMAGE');
 const html=renderMechanismChapter({...state('inputs'),example:`chap--${c.id}--${position.index}`},MODELS.dsv41);
 assert.equal((html.match(/class="sequence-position /g)||[]).length,c.positions.length);
 assert.equal((html.match(/image-patch is-selected/g)||[]).length,c.image.patches.filter(p=>p.token_index===position.index).length);
 assert.match(html,/Exact pinned formatter \+ tokenizer/);
});
test('prefill exposes per-query causality while decode keeps its selected position',()=>{
 const decode=renderMechanismChapter({...state('csa2'),phase:'decode'},MODELS.dsv41);
 const prefill=renderMechanismChapter({...state('csa2'),phase:'prefill'},MODELS.dsv41);
 assert.match(prefill,/CPU local membership · query cases/);
 assert.doesNotMatch(decode,/CPU local membership · query cases/);
});

test('dSpark labels preserve tap timing, one-based depth and token-only Markov correction',()=>{
 const c=mechanismChapter(MODELS.dsv41,'dspark'),network=c.sections[0];
 assert.ok(network.notes.some(n=>n.includes('after target blocks '+DSPARK_PRO.targetLayers.map(x=>x+1).join(', '))));
 assert.ok(network.notes.some(n=>n.includes('before target attention')));
 assert.equal(c.sections[1].equations[0].text,'logits[j] = parallel_logits[j] + Markov(sample[j−1])');
 assert.ok(mechanismChapter(MODELS.dsv41,'engram').sections[2].equations[1].text.includes('copysign'));
});
