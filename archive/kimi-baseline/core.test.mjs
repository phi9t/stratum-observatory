import test from 'node:test';
import assert from 'node:assert/strict';
import * as C from '../src/core.mjs';
test('Kimi K3 is the source-backed default',()=>assert.equal(C.DEFAULT_STATE.model,'kimi3'));
test('93 decoder layers partition into 69 KDA and 24 MLA with a global final layer',()=>{
 assert.equal(typeof C.layerKind,'function');
 const k=Array.from({length:93},(_,i)=>C.layerKind(i+1));
 assert.equal(k.filter(x=>x==='kda').length,69);assert.equal(k.filter(x=>x==='mla').length,24);assert.equal(k[92],'mla');
 assert.throws(()=>C.layerKind(0));assert.throws(()=>C.layerKind(94));
});
test('K3 exact configuration distinguishes latent and full model width',()=>{
 const m=C.MODELS.kimi3;assert.ok(m);assert.equal(m.d,7168);assert.equal(m.latent,3584);assert.equal(m.experts,896);assert.equal(m.topK,16);assert.equal(m.shared,2);assert.equal(m.vocab,163840);
});
test('routed expert parameter count is derived, reported totals stay separately rounded',()=>{
 assert.equal(typeof C.weightBudget,'function');const p=C.parameterLedger(C.MODELS.kimi3);
 assert.equal(p.routed,92*896*3*3584*3072);assert.equal(p.total,2.78e12);assert.equal(p.active,104.2e9);
 assert.equal(C.weightBudget('bf16').total,2*2.78e12);
 const mixed=C.weightBudget('mixed');assert.equal(mixed.routed,p.routed*(.5+1/32));assert.equal(mixed.total,mixed.routed+mixed.other);
});
test('NoPE does not erase the 64-dimensional shared key from compressed MLA cache',()=>{
 assert.equal(typeof C.cacheBudget,'function');const c=C.cacheBudget(1024,1,2,4,'compressed');
 assert.equal(c.mla,24*1024*(512+64)*2);assert.equal(c.recurrent,69*96*128*128*4);assert.equal(c.conv,69*3*96*128*4*2);
 assert.equal(c.total,c.mla+c.recurrent+c.conv);
});
test('only MLA history grows with context; all active state grows with batch',()=>{
 const a=C.cacheBudget(1024,1),b=C.cacheBudget(2048,1),c=C.cacheBudget(1024,4);
 assert.equal(b.mla,2*a.mla);assert.equal(b.recurrent,a.recurrent);assert.equal(b.conv,a.conv);assert.equal(c.total,4*a.total);
});
test('expanded reference cache is distinguished from optimized compressed MLA',()=>{
 const c=C.cacheBudget(1024,1,2,4,'expanded');assert.equal(c.mla,24*1024*96*(192+128)*2);
});
test('latent dispatch cuts vector payload in half, not the number of destinations',()=>{
 assert.equal(typeof C.moeTraffic,'function');const c=C.moeTraffic(4096,8,2);
 assert.equal(c.logical,4096*16*3584*2);assert.equal(c.sent,c.logical*7/8);assert.equal(c.fullWidth,2*c.logical);assert.equal(C.moeTraffic(4096,1,2).sent,0);
});
test('KCP exchanges both transition and zero-state matrices; single rank has no exchange',()=>{
 assert.equal(typeof C.kcpTraffic,'function');const c=C.kcpTraffic(8,4);
 assert.equal(c.local,96*(128*128+128*128)*4);assert.equal(c.sent,7*c.local);assert.equal(C.kcpTraffic(1,4).sent,0);
});
test('prefix reuse requires a common persisted boundary, not MLA match alone',()=>{
 assert.equal(typeof C.prefixHit,'function');assert.equal(C.prefixHit(2800,[[512,1536,2560],[512,2560]]),2560);
 assert.equal(C.prefixHit(2800,[[512,1536],[512,2560]]),512);assert.equal(C.prefixHit(2800,[[],[2560]]),0);
});
test('invalid sizes and precision fail closed',()=>{
 assert.throws(()=>C.cacheBudget(-1,1));assert.throws(()=>C.cacheBudget(1,1.5));assert.throws(()=>C.moeTraffic(1,0));assert.throws(()=>C.cacheBudget(1,1,0));
 assert.throws(()=>C.cacheBudget(1,1,2,4,'unknown'));assert.throws(()=>C.weightBudget('unknown'));
});
test('URL state validates the layer and maximum supported context',()=>{
 const s=C.sanitizeState({layer:93,context:1048576,foo:'bad',batch:1,hardware:'h200'});
 assert.equal(s.layer,93);assert.equal(s.context,1048576);assert.equal(s.foo,undefined);assert.equal(C.sanitizeState({layer:94}).layer,C.DEFAULT_STATE.layer);
});
test('derived state is finite for every lens',()=>{for(const view of ['model','training','serving','hardware','network']){const d=C.derived({...C.DEFAULT_STATE,view});assert.ok(Number.isFinite(d.perGPU));assert.ok(Number.isFinite(d.networkMs));}});
test('dense roofline uses a dense ceiling with explicit byte units',()=>{
 assert.equal(C.roofline(C.HARDWARE.h100,1),3.35);assert.equal(C.roofline(C.HARDWARE.h200,4096),989.5);
});
