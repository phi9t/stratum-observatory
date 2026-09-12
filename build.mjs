import fs from 'node:fs';
import path from 'node:path';
const files=['models','core','scene','diagram-model','diagram-systems','diagram-meta','diagrams','innovations','app'];
const modules=[];
for(const name of files){let code=fs.readFileSync(`src/${name}.mjs`,'utf8');const exports=new Set();
 code=code.replace(/^import\s*\{([^}]+)\}\s*from\s*['"]\.\/([^'"]+)['"];?\s*$/gm,(_,names,dep)=>`const {${names.replace(/\s+as\s+/g,':')}} = __modules[${JSON.stringify(dep)}];`);
 code=code.replace(/^export\s*\{([^}]+)\};?\s*$/gm,(_,names)=>{names.split(',').map(x=>x.trim()).filter(Boolean).forEach(x=>exports.add(x));return'';});
 code=code.replace(/^export\s+(const|let|function|class)\s+(\w+)/gm,(_,type,id)=>{exports.add(id);return`${type} ${id}`;});
 if(/^import\s|^export\s/m.test(code))throw new Error(`Unsupported module syntax in ${name}`);
 modules.push(`__modules[${JSON.stringify(name+'.mjs')}] = (() => {\n${code}\nreturn {${[...exports].join(',')}};\n})();`);
}
const js=`(()=>{'use strict';const __modules=Object.create(null);\n${modules.join('\n')}\n})();`;
let html=fs.readFileSync('index.html','utf8');html=html.replace('<link rel="stylesheet" href="src/styles.css">',()=>`<style id="stratum-styles">${fs.readFileSync('src/styles.css','utf8')}</style>`);html=html.replace('<script type="module" src="src/app.mjs"></script>',()=>`<script>${js.replace(/<\/script/gi,'<\\/script')}</script>`);
fs.mkdirSync('dist',{recursive:true});fs.writeFileSync('dist/index.html',html);
const dedicated=html.replace('<body>', '<body data-page-model="dsv41">').replace('<title>STRATUM — Model Systems Observatory</title>', '<title>DeepSeek V4.1 Flash — STRATUM</title>');
fs.writeFileSync('dist/deepseek-v4.1.html',dedicated);
const presets=(fs.readFileSync('src/models.mjs','utf8').match(/^ [A-Za-z0-9]+:\{\.\.\.common,/gm)||[]).length;
console.log(`Built dist/index.html: ${Math.round(Buffer.byteLength(html)/1024)} KiB, ${presets} models, no external runtime assets.`);
