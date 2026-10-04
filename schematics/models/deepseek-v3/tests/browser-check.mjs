// Focused executable-page checks beyond the shared play gate's sampling.
// Run after build: node models/deepseek-v3/tests/browser-check.mjs
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { launchChrome, serveDirectory } from '../../../tools/lib/chrome.mjs';
const vite = await createServer({configFile:false,server:{middlewareMode:true,hmr:false,watch:null},optimizeDeps:{noDiscovery:true}});
const ref = await vite.ssrLoadModule('/tnsr/tensor.ts');
const model = await vite.ssrLoadModule('/models/deepseek-v3/reference/model.ts');
const f = await vite.ssrLoadModule('/models/deepseek-v3/facts.ts');
const server = await serveDirectory('dist');
const browser = await launchChrome('/usr/bin/google-chrome');
const {page}=browser;
try {
 await page.setView('desktop-light');
 await page.open(`${server.base}deepseek-v3.html`);
 const tables = () => page.evaluate(`[...document.querySelectorAll('#ms-run table')].map(t=>[...t.tBodies[0].rows].map(r=>[...r.cells].map(c=>c.textContent)))`);
 const expected = (seed,layer,head,token) => {
  const ctx=new ref.Ctx('value',null,seed,f.TOY.gain);model.forward(ctx,model.TOY,f.TOY.ids);
  const get=id=>ctx.trace.get(`h${layer}.${id}`).data;
  const vector=(id,start,w)=>`[${Array.from(get(id).slice(start,start+w)).map(x=>x.toFixed(f.DEMO.precision)).join(', ')}]`;
  const k=f.TOY.nope+f.TOY.rotary;
  return {
   vectors:[vector('kn',token*f.TOY.kvRank,f.TOY.kvRank),vector('kpe',token*f.TOY.rotary,f.TOY.rotary),vector('joink',(head*f.TOY.tokens+token)*k,k),vector('kvsplit',(head*f.TOY.tokens+token)*f.TOY.valueDim,f.TOY.valueDim)],
   routes:Array.from({length:f.TOY.tokens},(_,t)=>{
    const ids=Array.from(get('topk').slice(t*f.TOY.topK,(t+1)*f.TOY.topK));
    const weights=Array.from(get('routeweights').slice(t*f.TOY.topK,(t+1)*f.TOY.topK));
    return [ `${t} / ${f.TOY.ids[0][t]}`,Array.from(get('groups').slice(t*f.TOY.topGroups,(t+1)*f.TOY.topGroups)).join(', '),ids.map((e,i)=>`${e} → ${weights[i].toFixed(f.DEMO.precision)}`).join('; ') ];
   }),
  };
 };
 const compare=async(seed,layer,head,token)=>{
  const actual=await tables(), want=expected(seed,layer,head,token);
  assert.deepEqual(actual[0].map(r=>r[1]),want.vectors);
  assert.deepEqual(actual[1],want.routes);
 };
 await compare(f.TOY.seed,f.TOY.firstDense,0,0);
 const change=async(index,value)=>{
  await page.evaluate(`(()=>{ const input=document.querySelectorAll('#ms-run input[type=range]')[${index}];Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,${JSON.stringify(String(value))});input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await page.settle();
 };
 const initial=(await tables())[0].map(r=>r[1]);
 await change(1,1);
 await compare(f.TOY.seed,f.TOY.firstDense,1,0);
 const headChanged=(await tables())[0].map(r=>r[1]);
 assert.deepEqual(initial.slice(0,2),headChanged.slice(0,2));
 assert.notDeepEqual(initial.slice(2),headChanged.slice(2));
 await change(2,3);
 await compare(f.TOY.seed,f.TOY.firstDense,1,3);
 await change(0,f.TOY.layers-1);
 await compare(f.TOY.seed,f.TOY.layers-1,1,3);
 const beforeSeed=await tables();
 await page.evaluate(`[...document.querySelectorAll('#ms-run button')].find(b=>b.textContent==='Draw new weights').click()`);
 await page.settle();
 await compare(f.TOY.seed+1,f.TOY.layers-1,1,3);
 assert.notDeepEqual(await tables(),beforeSeed);
 const hash=await page.evaluate('location.hash'), beforeReload=await tables();
 for(const key of ['run-head=1','run-token=3',`run-layer=${f.TOY.layers-1}`,`run-seed=${f.TOY.seed+1}`]) assert(hash.includes(key));
 await page.reload();
 assert.deepEqual(await tables(),beforeReload);
 assert.equal(page.errors.length,0);
 console.log('PASS: all computed vectors and all token routes match the reference; head preserves latent/shared key; layer/head/token/seed controls and URL reload checked.');
} finally {
 await browser.close();await server.close();await vite.close();
}
