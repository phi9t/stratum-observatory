/** Presentation only: every numerical value below comes from CPU-generated tensors. */
import {WORKED_RESULTS} from './worked-results.mjs';
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function tensorText(value){
 if(Array.isArray(value))return '['+value.map(tensorText).join(', ')+']';
 if(value===null)return 'missing';
 if(typeof value==='number')return Number.isInteger(value)?String(value):String(Number(value.toPrecision(6)));
 return String(value);
}
export function workedIds(stage){return ({decoder:['replay'],csa2:['selection','query'],'module-csa2-selection':['selection'],'module-csa2-attention':['query']})[stage]||[];}
function tensorRows(rows){return rows.map(r=>`<tr data-worked-step="${esc(r.id)}"><th scope="row">${esc(r.title)}</th><td>${esc(r.operation)}</td><td>${esc(r.axes.join(" × "))}<br>${esc(tensorText(r.shape))}<br>${esc(r.dtype)}</td><td><code data-worked-values="${esc(r.id)}">${esc(tensorText(r.values))}</code></td></tr>`).join('');}
function replayDiagram(example){
 const lanes=[['Full history',example.full],['Truncated replay',example.truncated]];
 const columns=example.full[0].positions.length,depths=example.full.length;
 const laneWidth=180+columns*180,height=110+depths*120;
 let content='';
 for(const [lane,[title,rows]]of lanes.entries()){
  const origin=0;
  content+=`<g transform="translate(0,${lane*height})">`;
  content+=`<text x="${origin+24}" y="30">${esc(title)}</text>`;
  rows[0].positions.forEach((position,i)=>{content+=`<text x="${origin+140+i*180}" y="65">position ${esc(position)}</text>`;});
  for(const [depth,row]of rows.entries()){
   const y=90+depth*120;
   content+=`<text x="${origin+20}" y="${y+30}">depth ${esc(row.depth)}</text>`;
   row.values.forEach((value,i)=>{
    const x=origin+125+i*180,missing=value===null;
    if(depth>0&&!missing){
     for(const previous of [i-1,i].filter(j=>j>=0)){
      const px=origin+125+previous*180+70,py=y-120+76;
      const absent=rows[depth-1].values[previous]===null;
      content+=`<path class="worked-wire${absent?' worked-missing':''}" d="M${px},${py} L${x+70},${y}"/><path class="worked-wire${absent?' worked-missing':''}" d="M${x+65},${y-7} L${x+70},${y} L${x+75},${y-7}"/>`;
     }
    }
    content+=`<rect class="worked-cell${missing?' worked-missing':''}" x="${x}" y="${y}" width="150" height="76" rx="6"/><text x="${x+8}" y="${y+32}" data-replay-value="${lane}-${depth}-${i}">${(Array.isArray(value)?value:[value]).map((v,j)=>`<tspan x="${x+8}" dy="${j?24:0}">${esc(tensorText(v))}</tspan>`).join('')}</text>`;
   });
  }
  content+='</g>';
 }
 return `<div class="table-scroll"><svg class="worked-diagram" viewBox="0 0 ${laneWidth} ${height*2}" role="img" aria-label="CPU replay experiment: token positions across layer depth, full history and truncated replay"><title>Missing local dependencies change the replayed result</title>${content}</svg></div><p>Arrows show local dependencies. Dashed amber paths start from missing state. Both runs also read the same global dictionary and sink, shown in the inputs below; those inputs are omitted from these local dependency arrows.</p>`;
}
export function workedHTML(stage){return workedIds(stage).map(id=>{
 const example=WORKED_RESULTS[id];
 return `<section class="pytorch-section worked-example" data-worked="${id}" aria-label="${esc(example.title)}"><h4>${esc(example.title)}</h4><p>${esc(example.scope)}</p><p>Computed with the checked-in CPU PyTorch implementation. Floating-point values are displayed to six significant digits; the saved results retain full precision. These fixed teaching inputs are independent of the page’s context and hardware controls.</p>${id==='replay'?replayDiagram(example):''}<div class="table-scroll"><table class="comparison-table worked-table"><caption>Inputs → tensor operations → computed results</caption><thead><tr><th scope="col">Step</th><th scope="col">Operation</th><th scope="col">Shape</th><th scope="col">CPU values</th></tr></thead><tbody>${tensorRows([...example.inputs,...example.steps])}</tbody></table></div><p><strong>Fixture settings.</strong> ${Object.entries(example.settings).map(([key,value])=>esc(key)+' = '+esc(tensorText(value))).join(' · ')}</p>${id==='replay'?'':`<p><button class="button" data-module-example="csa2-${id==='query'?'attention':'selection'}">Inspect the ${id==='query'?'attention':'selection'} schematic and implementation →</button></p>`}</section>`;
 }).join('');}
