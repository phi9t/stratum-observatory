import fs from 'node:fs';
import path from 'node:path';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import * as presets from './src/models.mjs';
import {PAGES} from './src/pages.mjs';
import {landingHTML,dataHTML} from './src/landing.mjs';
import {syncWorkedResults} from './tools/sync-worked.mjs';
import {syncPythonSources} from './tools/sync-pytorch.mjs';
syncPythonSources();
syncWorkedResults();
const inputCases=JSON.parse(fs.readFileSync('examples/deepseek_v41/input-cases.json','utf8'));
for(const [file,sha] of Object.entries(inputCases.recipe_sha256))if(createHash('sha256').update(fs.readFileSync(file)).digest('hex')!==sha)throw new Error('Stale prompt fixtures: '+file);
fs.writeFileSync('src/input-cases.mjs','/** Generated publisher-derived input fixtures. */\nexport const INPUT_CASES='+JSON.stringify(inputCases)+';\n');
const files=['models','worked-results','worked-examples','pytorch-source','pytorch-inputs','pytorch-backbone','pytorch-memory','pytorch-dspark','pytorch','core','scene','diagram-model','diagram-systems','diagram-residual','diagram-inputs','diagram-dspark','diagram-meta','deepdives','python-highlight','diagram-deepdives','diagram-pytorch-backbone','diagram-pytorch-memory','diagram-pytorch-dspark','diagram-pytorch','diagrams','innovations','pages','input-cases','mechanism-chapters','mechanism-visuals','workspace','implementation-sources','workspace-ui','app'];
const modules=[];
for(const name of files){let code=fs.readFileSync(`src/${name}.mjs`,'utf8');const exports=new Set();
 if(['pytorch-inputs','pytorch-backbone','pytorch-memory','pytorch-dspark'].includes(name)){
  const symbol=({ 'pytorch-inputs':'INPUTS_EXAMPLES','pytorch-backbone':'BACKBONE_EXAMPLES','pytorch-memory':'MEMORY_EXAMPLES','pytorch-dspark':'DSPARK_EXAMPLES'})[name];
  const metadata=(await import(`./src/${name}.mjs`))[symbol].map(({code,...entry})=>entry);
  const compressed=gzipSync(Buffer.from(JSON.stringify(metadata)),{level:9}).toString('base64');
  modules.push(`__modules['${name}.mjs'] = {${symbol}:(JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())).map(entry=>({...entry,code:__modules['pytorch-source.mjs'].PYTHON_SOURCES[entry.sourceFile.split('/').pop()]}))};`);
  continue;
 }
 if(['diagram-pytorch-backbone','diagram-pytorch-memory','diagram-pytorch-dspark'].includes(name)){
  const type=name.split('-').at(-1),symbol=type.toUpperCase()+'_SCHEMATICS';
  const builders=(await import(`./src/${name}.mjs`))[symbol];
  const examples=(await import(`./src/pytorch-${type}.mjs`))[type.toUpperCase()+'_EXAMPLES'];
  const scenes=Object.fromEntries(Object.entries(builders).map(([id,build])=>[id,build(presets.MODELS.dsv41,examples.find(e=>e.id===id))]));
  const compressed=gzipSync(Buffer.from(JSON.stringify(scenes)),{level:9}).toString('base64');
  modules.push(`__modules['${name}.mjs'] = await (async()=>{const scenes=JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text());return {${symbol}:Object.fromEntries(Object.entries(scenes).map(([id,scene])=>[id,()=>structuredClone(scene)]))};})();`);
  continue;
 }
 if(['diagram-inputs','diagram-residual','diagram-dspark'].includes(name)){
  // These fixed-checkpoint context scenes have no scenario-dependent geometry.
  // Compile their pure builders into compressed scene/chapter data; retain the
  // same callable interface and clone on access so reader interaction is isolated.
  const input=name==='diagram-inputs',dspark=name==='diagram-dspark',api=await import(`./src/${name}.mjs`);
  const sceneName=input?'buildV41Inputs':dspark?'buildDSparkArchitecture':'buildMHCResidual',chapterName=input?'inputsChapter':dspark?'dsparkChapter':'residualChapter';
  const data={scene:api[sceneName]({model:'dsv41'}),chapter:api[chapterName](presets.MODELS.dsv41)};
  const compressed=gzipSync(Buffer.from(JSON.stringify(data)),{level:9}).toString('base64');
  modules.push(`__modules['${name}.mjs'] = await (async()=>{const data=JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text());return {${sceneName}:()=>structuredClone(data.scene),${chapterName}:()=>structuredClone(data.chapter)};})();`);
  continue;
 }
 if(name==='models'){
  const compressed=gzipSync(Buffer.from(JSON.stringify(presets)),{level:9}).toString('base64');
  modules.push(`__modules['models.mjs'] = JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text());`);
  continue;
 }
 if(name==='pytorch-source'||name==='worked-results'||name==='input-cases'){
  // Decode exact local data before the dependent modules run; no external assets.
  const symbol=name==='pytorch-source'?'PYTHON_SOURCES':name==='input-cases'?'INPUT_CASES':'WORKED_RESULTS';
  const match=code.match(new RegExp('export const '+symbol+'=(.*);\\n$'));
  if(!match)throw new Error('Malformed generated data module: '+name);
  const compressed=gzipSync(Buffer.from(match[1]),{level:9}).toString('base64');
  modules.push(`__modules['${name}.mjs'] = {${symbol}: JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())};`);
  continue;
 }

 code=code.replace(/^import\s*\{([^}]+)\}\s*from\s*['"]\.\/([^'"]+)['"];?\s*$/gm,(_,names,dep)=>`const {${names.replace(/\s+as\s+/g,':')}} = __modules[${JSON.stringify(dep)}];`);
 code=code.replace(/^export\s*\{([^}]+)\};?\s*$/gm,(_,names)=>{names.split(',').map(x=>x.trim()).filter(Boolean).forEach(x=>exports.add(x));return'';});
 code=code.replace(/^export\s+(const|let|function|class)\s+(\w+)/gm,(_,type,id)=>{exports.add(id);return`${type} ${id}`;});
 if(/^import\s|^export\s/m.test(code))throw new Error(`Unsupported module syntax in ${name}`);
 modules.push(`__modules[${JSON.stringify(name+'.mjs')}] = (() => {\n${code}\nreturn {${[...exports].join(',')}};\n})();`);
}
const stylesheet=fs.readFileSync('src/styles.css','utf8');
const packedStyles=gzipSync(Buffer.from(stylesheet),{level:9}).toString('base64');
const js=`(async()=>{'use strict';document.getElementById('stratum-styles').textContent=await new Response(new Blob([Uint8Array.from(atob('${packedStyles}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text();const __modules=Object.create(null);\n${modules.join('\n')}\n})();`;
let html=fs.readFileSync('index.html','utf8');html=html.replace('<link rel="stylesheet" href="src/styles.css">',()=>'<style id="stratum-styles"></style>');const packedApp=gzipSync(Buffer.from(js),{level:9}).toString('base64');html=html.replace('<script type="module" src="src/app.mjs"></script>',()=>`<script>(async()=>{const app=document.createElement('script');app.textContent=await new Response(new Blob([Uint8Array.from(atob('${packedApp}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text();document.body.append(app);})();</script>`);
fs.mkdirSync('dist',{recursive:true});fs.writeFileSync('dist/index.html',landingHTML(fs.readFileSync('src/styles.css','utf8')));
fs.writeFileSync('dist/data-representation.html',dataHTML(fs.readFileSync('src/styles.css','utf8')));
for(const page of PAGES){
 const dedicated=html.replace('<body>', `<body data-page-id="${page.id}" data-page-model="${page.defaultModel}">`).replace('<title>STRATUM — Model Systems Observatory</title>', `<title>${page.title} — STRATUM</title>`);
 fs.writeFileSync(`dist/${page.file}`,dedicated);
}
const presetCount=(fs.readFileSync('src/models.mjs','utf8').match(/^ [A-Za-z0-9]+:\{\.\.\.common,/gm)||[]).length;
console.log(`Built landing, data reference and ${PAGES.length} interactive pages; ${presetCount} models, no external runtime assets.`);
