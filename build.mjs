import fs from 'node:fs';
import path from 'node:path';
import {gzipSync} from 'node:zlib';
import {PAGES} from './src/pages.mjs';
import {landingHTML,dataHTML} from './src/landing.mjs';
import {syncWorkedResults} from './tools/sync-worked.mjs';
import {syncPythonSources} from './tools/sync-pytorch.mjs';
syncPythonSources();
syncWorkedResults();
const files=['models','worked-results','worked-examples','pytorch-source','pytorch-backbone','pytorch-memory','pytorch-dspark','pytorch','core','scene','diagram-model','diagram-systems','diagram-meta','deepdives','python-highlight','diagram-deepdives','diagram-pytorch-backbone','diagram-pytorch-memory','diagram-pytorch-dspark','diagram-pytorch','diagrams','innovations','pages','app'];
const modules=[];
for(const name of files){let code=fs.readFileSync(`src/${name}.mjs`,'utf8');const exports=new Set();
 if(name==='pytorch-source'){
  // Preserve exact readable Python while keeping the offline single-file size bounded.
  const sources=code.match(/export const PYTHON_SOURCES=(.*);\n$/)?.[1];
  if(!sources)throw new Error('Malformed generated Python source module');
  const compressed=gzipSync(Buffer.from(sources),{level:9}).toString('base64');
  modules.push(`__modules['pytorch-source.mjs'] = {PYTHON_SOURCES: JSON.parse(await new Response(new Blob([Uint8Array.from(atob('${compressed}'),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())};`);
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
const presets=(fs.readFileSync('src/models.mjs','utf8').match(/^ [A-Za-z0-9]+:\{\.\.\.common,/gm)||[]).length;
console.log(`Built landing, data reference and ${PAGES.length} interactive pages; ${presets} models, no external runtime assets.`);
