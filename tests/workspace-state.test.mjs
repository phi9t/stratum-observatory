import test from 'node:test';
import assert from 'node:assert/strict';
import {sanitizeState} from '../src/core.mjs';
test('workspace URL fields retain safe identities and reject markup or unsupported panels',()=>{
 const s=sanitizeState({panel:'worked',module:'csa2-reindex',example:'selection--candidate_mask'});
 assert.equal(s.panel,'worked');assert.equal(s.module,'csa2-reindex');assert.equal(s.example,'selection--candidate_mask');
 for(const bad of ['<script>','x'.repeat(101),{},null]){const x=sanitizeState({module:bad,example:bad,panel:'missing'});assert.equal(x.module,'');assert.equal(x.example,'');assert.equal(x.panel,'schematic');}
});
