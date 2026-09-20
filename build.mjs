import fs from 'node:fs';
import path from 'node:path';
import {gzipSync} from 'node:zlib';
import * as presets from './src/models.mjs';
import {PAGES} from './src/pages.mjs';
import {landingHTML,dataHTML} from './src/landing.mjs';
import {syncWorkedResults} from './tools/sync-worked.mjs';
import {syncPythonSources} from './tools/sync-pytorch.mjs';
syncPythonSources();
syncWorkedResults();
const files=['models','worked-results','worked-examples','pytorch-source','pytorch-inputs','pytorch-backbone','pytorch-memory','pytorch-dspark','pytorch','core','scene','diagram-model','diagram-systems','diagram-residual','diagram-inputs','diagram-meta','deepdives','python-highlight','diagram-deepdives','diagram-pytorch-backbone','diagram-pytorch-memory','diagram-pytorch-dspark','diagram-pytorch','diagrams','innovations','pages','app'];
const modules=[];
for(const name of files){let code=fs.readFileSync(`src/${name}.mjs`,'utf8');const exports=new Set();
 if(['pytorch-inputs','pytorch-backbone','pytorch-memory','pytorch-dspark'].includes(name)){
  const symbol=({ 'pytorch-inputs':'INPUTS_EXAMPLES','pytorch-backbone':'BACKBONE_EXAMPLES','pytorch-memory':'MEMORY_EXAMPLES','pytorch-dspark':'DSPARK_EXAMPLES'})[name];
  const metadata=(await import(`./src/${name}.mjs`))[symbol].map(({code,...entry})=>entry);
  const compressed=gzipSync(Buffer.from(JSON.stringify(metadata)),{level:9}).toString('base64');
  modules.push(`__modules['${name}.mjs'] = {${symbol}:(JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())).map(entry=>({...entry,code:__modules['pytorch-source.mjs'].PYTHON_SOURCES[entry.sourceFile.split('/').pop()]}))};`);
  continue;
 }
 if(name==='diagram-inputs'||name==='diagram-residual'){
  // These fixed-checkpoint context scenes have no scenario-dependent geometry.
  // Compile their pure builders into compressed scene/chapter data; retain the
  // same callable interface and clone on access so reader interaction is isolated.
  const input=name==='diagram-inputs',api=await import(`./src/${name}.mjs`);
  const sceneName=input?'buildV41Inputs':'buildMHCResidual',chapterName=input?'inputsChapter':'residualChapter';
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
 if(name==='pytorch-source'||name==='worked-results'){
  // Decode exact local data before the dependent modules run; no external assets.
  const symbol=name==='pytorch-source'?'PYTHON_SOURCES':'WORKED_RESULTS';
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
const js=`(async()=>{'use strict';const __modules=Object.create(null);\n${modules.join('\n')}\n})();`;
let html=fs.readFileSync('index.html','utf8');html=html.replace('<link rel="stylesheet" href="src/styles.css">',()=>`<style id="stratum-styles">${fs.readFileSync('src/styles.css','utf8')}</style>`);html=html.replace('<script type="module" src="src/app.mjs"></script>',()=>`<script>${js.replace(/<\/script/gi,'<\\/script')}</script>`);
fs.mkdirSync('dist',{recursive:true});fs.writeFileSync('dist/index.html',landingHTML(fs.readFileSync('src/styles.css','utf8')));
fs.writeFileSync('dist/data-representation.html',dataHTML(fs.readFileSync('src/styles.css','utf8')));
for(const page of PAGES){
 const dedicated=html.replace('<body>', `<body data-page-id="${page.id}" data-page-model="${page.defaultModel}">`).replace('<title>STRATUM — Model Systems Observatory</title>', `<title>${page.title} — STRATUM</title>`);
 fs.writeFileSync(`dist/${page.file}`,dedicated);
}
const presetCount=(fs.readFileSync('src/models.mjs','utf8').match(/^ [A-Za-z0-9]+:\{\.\.\.common,/gm)||[]).length;
console.log(`Built landing, data reference and ${PAGES.length} interactive pages; ${presetCount} models, no external runtime assets.`);
