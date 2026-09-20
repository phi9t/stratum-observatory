import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {syncWorkedResults} from '../tools/sync-worked.mjs';

test('portable build rejects changed Python sources and changed trace results',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'stratum-worked-'));
 try{
  for(const item of ['examples','tools','src','uv.lock'])fs.cpSync(item,path.join(root,item),{recursive:true});
  syncWorkedResults(root);
  const source=path.join(root,'examples/deepseek_v41/csa2_attention.py'),original=fs.readFileSync(source);
  fs.appendFileSync(source,'\n# stale recipe\n');
  assert.throws(()=>syncWorkedResults(root),/stale.*CPU/i);
  fs.writeFileSync(source,original);
  const receipt=path.join(root,'examples/deepseek_v41/worked-results.json');
  const value=JSON.parse(fs.readFileSync(receipt));
  value.examples.query.steps[0].values=[999];fs.writeFileSync(receipt,JSON.stringify(value));
  assert.throws(()=>syncWorkedResults(root),/digest/);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
