import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS, sanitizeState, servingLedger, formatBytes} from '../src/core.mjs';
import {buildServing} from '../src/diagram-systems.mjs';
import {structuralReport} from '../src/scene.mjs';

test('CED serving displays the complete logical state breakdown with scoped formulas', () => {
  const state = sanitizeState({model:'dsv41', view:'serving'});
  const ledger = servingLedger(state.model, state);
  const scene = buildServing(state);
  const rows = Object.fromEntries(scene.nodes.filter(n=>n.id.startsWith('state-')).map(n=>[n.id,n]));
  assert.equal(rows['state-window'].sub, formatBytes(ledger.windowKV));
  assert.equal(rows['state-history'].sub, formatBytes(ledger.history));
  assert.equal(rows['state-indexer'].sub, formatBytes(ledger.indexer));
  assert.doesNotMatch(rows['state-history'].info.formula, /b_FP8|window/);
  assert.equal(rows['state-window'].info.formula, 'B × window × layers × headDim × b_FP8');
  assert.equal(ledger.total, ledger.history + ledger.recurrent + ledger.conv + ledger.indexer + ledger.windowKV);
  assert.match(rows['state-bundle'].sub, /2 SWA-only/);
  assert.deepEqual(structuralReport(scene), []);
});

test('CED serving phase selects inspection with the participating depth and active scope', () => {
  const m = MODELS.dsv41;
  for (const phase of ['prefill','decode']) {
    const scene = buildServing(sanitizeState({model:m.id, view:'serving', phase}));
    assert.equal(scene.defaultSelected, phase);
    const inspected = scene.nodes.find(n=>n.id===scene.defaultSelected);
    assert.ok(inspected.info.facts.some(([name,value])=>name==='Reported participating layers' && value===String(phase==='prefill'?m.encoderLayers:m.layers)));
    assert.ok(inspected.info.facts.some(([name])=>name==='Published active parameters'));
  }
});

test('serving geometry remains valid for all models, phases and ownership modes', () => {
  for (const model of Object.keys(MODELS)) for (const phase of ['prefill','decode']) for (const placement of ['split','colocated']) {
    assert.deepEqual(structuralReport(buildServing(sanitizeState({model,view:'serving',phase,placement}))), [], `${model}/${phase}/${placement}`);
  }
});

test('CED phase depth does not claim the pinned reference implements encoder-only prefill', () => {
  const scene = buildServing(sanitizeState({model:'dsv41',view:'serving',phase:'prefill'}));
  const prefill = scene.nodes.find(n=>n.id==='prefill');
  assert.match(prefill.info.body, /pinned reference.*all backbone layers/);
  assert.match(prefill.info.body, /reported CED/);
});

test('CED serving separates reported phase accounting from opt-in framework replay', () => {
  const scene = buildServing(sanitizeState({model:'dsv41',view:'serving',phase:'prefill'}));
  const prefill = scene.nodes.find(n=>n.id==='prefill');
  assert.match(prefill.sub, /reported/);
  assert.ok(prefill.info.facts.some(([name])=>name==='Reported participating layers'));
  assert.match(prefill.info.body, /SGLang/);
  assert.match(prefill.info.body, /opt-in/);
  assert.match(prefill.info.body, /approximat/);
  assert.ok(prefill.info.sources.includes('dsv41-sglang'));
  const indexer = scene.nodes.find(n=>n.id==='state-indexer');
  assert.match(indexer.info.body, /FP4/);
  assert.doesNotMatch(indexer.info.body, /precision is not established/);
  assert.match(indexer.info.body, /amortized logical payload/);
});

test('SGLang indexer packing reconciles complete-pair KV payload without claiming allocated memory', () => {
  const one = servingLedger('dsv41',{context:32768,batch:1});
  assert.equal(one.indexer / 32768, 170);
  assert.equal((one.mainKV + one.indexer) / 32768, MODELS.dsv41.kvBytesPerToken);
  const two = servingLedger('dsv41',{context:65536,batch:2});
  assert.equal(two.indexer, one.indexer * 4);
  assert.equal(two.windowKV, one.windowKV * 2);
  assert.ok(one.exclusions.some(x=>/incomplete.*pair/i.test(x)));
  const row = buildServing(sanitizeState({model:'dsv41',view:'serving'})).nodes.find(n=>n.id==='state-indexer');
  assert.equal(row.sub, formatBytes(one.indexer));
  assert.ok(row.info.sources.includes('dsv41-sglang-cache'));
});
