import test from 'node:test';
import assert from 'node:assert/strict';
import { MODELS, SOURCES } from '../src/models.mjs';
import { implementationReferences } from '../src/implementation-sources.mjs';

test('each exact checkpoint starts with its pinned publisher configuration', () => {
  for (const model of Object.values(MODELS)) {
    const { baseline } = implementationReferences(model, 'model');
    assert.equal(baseline[0].scope, 'exact-checkpoint');
    assert.equal(baseline[0].status, 'reference-only');
    assert.ok(baseline[0].url.startsWith(`https://huggingface.co/${model.repo}/blob/`));
    assert.match(baseline[0].revision, /^[a-f0-9]{40}$/);
  }
});
test('Transformers stays distinct from publisher implementation', () => {
  for (const id of ['q38', 'q330', 'q3235', 'q3827', 'q38flash']) {
    const refs = implementationReferences(MODELS[id], 'model').baseline;
    assert.ok(refs.some(ref => ref.provider === 'Transformers'));
    assert.ok(refs.filter(ref => ref.kind === 'implementation').every(ref => ref.scope === 'architecture-family'));
    assert.ok(!refs.some(ref => ref.provider === 'Hugging Face publisher' && ref.kind === 'implementation'));
  }
});
test('generic training and partial serving sources cannot imply checkpoint support', () => {
  for (const model of Object.values(MODELS)) {
    const refs = implementationReferences(model, 'training');
    assert.ok(refs.training.length >= 2);
    for (const ref of [...refs.training, ...refs.serving]) {
      assert.notEqual(ref.scope, 'exact-checkpoint');
      assert.equal(ref.status, 'reference-only');
      assert.equal(ref.executed, false);
      assert.ok(ref.explanation.length);
    }
    assert.ok(refs.training.every(ref => ref.scope === 'generic-mechanism'));
  }
  assert.equal(implementationReferences(MODELS.q38flash, 'serving').serving[0].scope, 'component-only');
});
test('selection marks existing references without adding unrelated checkpoint sources', () => {
  const refs = implementationReferences(MODELS.dsv41, 'serving', { sources: ['dsv41-sglang-cache', 'dsv4-dspark-code'] });
  assert.ok(refs.serving.find(ref => ref.sourceId === 'dsv41-sglang-cache').selected);
  assert.ok(!Object.values(refs).flat().some(ref => ref.sourceId === 'dsv4-dspark-code'));
  assert.deepEqual(implementationReferences({ id: 'invented' }, 'model'), { baseline: [], serving: [], training: [] });
});
test('references resolve only catalog identities and returned arrays are independent', () => {
  for (const model of Object.values(MODELS)) {
    const refs = Object.values(implementationReferences(model, 'model')).flat();
    for (const ref of refs) {
      assert.equal(ref.url, SOURCES[ref.sourceId].url);
      assert.equal(ref.revision, SOURCES[ref.sourceId].revision);
      assert.ok(ref.symbols.length);
    }
  }
  const first = implementationReferences(MODELS.ds2, 'model');
  first.baseline[0].symbols.push('invented');
  assert.ok(!implementationReferences(MODELS.ds2, 'model').baseline[0].symbols.includes('invented'));
});

test('operation filtering excludes unrelated publisher files and serving components', () => {
 const residual = implementationReferences(MODELS.dsv41, 'model', {operatorId:'merge-attn', sources:MODELS.dsv41.sources});
 assert.ok(!residual.baseline.some(r=>r.sourceId==='dsv41-engram'));
 assert.deepEqual(residual.training, []);
 assert.ok(!residual.baseline.find(r=>r.sourceId==='dsv41-code').symbols.includes('Attention'));
 const ffn = implementationReferences(MODELS.q38flash,'model',{operatorId:'expert-up',sources:MODELS.q38flash.sources});
 assert.ok(!ffn.serving.some(r=>r.sourceId==='q4-runtime'));
 const attention = implementationReferences(MODELS.q38flash,'model',{operatorId:'attention',sources:MODELS.q38flash.sources});
 assert.ok(attention.serving.some(r=>r.sourceId==='q4-runtime'));
});
test('training operations map only to corresponding generic Megatron mechanisms', () => {
 const refs = id => implementationReferences(MODELS.dsv41,'training',{operatorId:id,sources:['fsdp']}).training;
 assert.deepEqual(refs('optimizer').map(r=>r.sourceId),['megatron-optimizer']);
 assert.deepEqual(refs('gradient-sync').map(r=>r.sourceId),['megatron-ddp']);
 assert.deepEqual(refs('data'),[]);
 assert.deepEqual(refs('microbatch').map(r=>r.sourceId),['megatron-schedules']);
 const parallel = implementationReferences(MODELS.q38,'model',{operatorId:'q-proj',sources:MODELS.q38.sources}).training;
 assert.deepEqual(parallel.map(r=>r.sourceId),['megatron-parallel-layers']);
 const dispatch = implementationReferences(MODELS.q330,'model',{operatorId:'dispatch',sources:MODELS.q330.sources}).training;
 assert.deepEqual(dispatch.map(r=>r.sourceId),['megatron-expert-dispatch']);
 assert.ok(dispatch.every(r=>r.scope==='generic-mechanism'&&r.executed===false));
});
test('serving cache selects cache sources without unrelated training or Engram code', () => {
 const refs=implementationReferences(MODELS.dsv41,'serving',{operatorId:'state-indexer',sources:[...MODELS.dsv41.sources,'dsv41-sglang-cache']});
 assert.deepEqual(refs.serving.map(r=>r.sourceId),['dsv41-sglang-cache']);
 assert.deepEqual(refs.training,[]);
 assert.ok(!refs.baseline.some(r=>r.sourceId==='dsv41-engram'));
});
test('selected FFN symbol does not point readers at Qwen attention', () => {
 const refs=implementationReferences(MODELS.q330,'model',{operatorId:'expert-up',sources:MODELS.q330.sources});
 assert.deepEqual(refs.baseline.find(r=>r.sourceId==='q3-moe-code').symbols,['Qwen3MoeSparseMoeBlock']);
});
test('unknown operations and absent cache mechanisms never inherit whole-model implementation coverage', () => {
 for(const operatorId of ['unmapped-operation','state-recurrent','state-conv']) {
  const refs=implementationReferences(MODELS.dsv41,'serving',{operatorId,sources:MODELS.dsv41.sources});
  assert.equal(refs.baseline.length,1);
  assert.deepEqual(refs.serving,[]);
 }
});
