/** Check CPU provenance before embedding saved results; no browser-side model arithmetic. */
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
export function syncWorkedResults(root=process.cwd()){
 const directory=path.join(root,'examples/deepseek_v41');
 const receipt=JSON.parse(fs.readFileSync(path.join(directory,'worked-results.json'),'utf8'));
 const files=[...fs.readdirSync(directory).filter(f=>f.endsWith('.py')).map(f=>'examples/deepseek_v41/'+f),'tools/export-worked.py','uv.lock'].sort();
 const actual=Object.fromEntries(files.map(f=>[f,hash(fs.readFileSync(path.join(root,f)))]));
 if(receipt.schema!=='stratum-cpu-worked/v1'||receipt.device!=='cpu'||JSON.stringify(actual)!==JSON.stringify(receipt.source_sha256))throw new Error('Stale CPU traces; run npm run generate:worked.');
 if(hash(receipt.payload)!==receipt.payload_sha256||JSON.stringify(JSON.parse(receipt.payload))!==JSON.stringify(receipt.examples))throw new Error('CPU trace payload digest mismatch; run npm run generate:worked.');
 const text='/** Generated from checked CPU results. Run npm run generate:worked. */\nexport const WORKED_RESULTS='+JSON.stringify(receipt.examples)+';\n';
 const output=path.join(root,'src/worked-results.mjs');
 if(!fs.existsSync(output)||fs.readFileSync(output,'utf8')!==text)fs.writeFileSync(output,text);
}
