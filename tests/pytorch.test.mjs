import test from 'node:test';
import assert from 'node:assert/strict';
import {PYTORCH_EXAMPLES,chapterExamples,exampleProgram} from '../src/pytorch.mjs';
import {SOURCES} from '../src/models.mjs';
import {DEEP_DIVE_TITLES} from '../src/deepdives.mjs';
test('every V4.1 chapter provides executable, attributed module examples',()=>{
 assert.ok(PYTORCH_EXAMPLES.length);
 assert.equal(new Set(PYTORCH_EXAMPLES.map(x=>x.id)).size,PYTORCH_EXAMPLES.length);
 for(const stage of Object.keys(DEEP_DIVE_TITLES))assert.ok(chapterExamples(stage).length,stage);
 for(const x of PYTORCH_EXAMPLES){
  assert.ok(x.summary&&x.limitations&&x.symbols.length&&x.sources.length,x.id);
  for(const source of x.sources)assert.ok(SOURCES[source],source);
  assert.ok(x.code.includes('def '),x.id);
  assert.doesNotMatch(x.code,/NotImplementedError|^\s*\.\.\.\s*$/m);
  assert.ok(exampleProgram(x.id).includes(x.code),x.id);
  for(const dependency of x.requires)assert.ok(exampleProgram(x.id).includes(PYTORCH_EXAMPLES.find(e=>e.id===dependency).code));
 }
 assert.deepEqual(chapterExamples('block'),[]);
 assert.throws(()=>exampleProgram('missing'),/Unknown/);
});
