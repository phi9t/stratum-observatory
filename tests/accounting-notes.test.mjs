import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../src/core.mjs';

test('training assumptions derive the accumulation alternative without changing the ledger',()=>{
 const state={view:'training',model:'q38'};
 const before=core.trainingLedger(state.model,state);
 const note=core.accountingAssumptions(state.model,state)[0];
 assert.equal(note.values.baseline,before.total);
 assert.equal(note.values.accumulation,before.master);
 assert.equal(note.values.withAccumulation,before.total+before.master);
 assert.equal(note.values.bytesPerParameter,16);
 assert.equal(note.values.alternativeBytesPerParameter,20);
 assert.deepEqual(core.trainingLedger(state.model,state),before);
});

test('specialized caches are not certified by a generic GQA method',()=>{
 for(const m of Object.values(core.MODELS)){
  const note=core.accountingAssumptions(m.id,{view:'serving'})[0];
  assert.equal(note.applicability,m.architecture==='gqa'?'applicable':'specialized');
  if(m.id==='dsv41')assert.match(note.body,/read.*multiple|repeated.*read/);
 }
});

test('book notes resolve sources for every live model and lens without changing scenario state',()=>{
 for(const m of Object.values(core.MODELS))for(const view of core.VIEWS){
  const state=core.sanitizeState({model:m.id,view});const snapshot=structuredClone(state);
  const rows=core.accountingAssumptions(m.id,state);
  assert.ok(rows.length>0);
  for(const row of rows){assert.ok(row.method);assert.ok(row.scope);for(const id of row.sources)assert.ok(core.SOURCES[id],id);}
  assert.deepEqual(state,snapshot);
 }
});
