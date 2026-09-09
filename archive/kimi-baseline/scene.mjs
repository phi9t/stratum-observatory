/** Visual IR. Positions belong to semantic motif templates, never to model data. */
export function makeScene(width=1320,height=850){
 return {width,height,nodes:[],edges:[],groups:[],labels:[],decorations:[],caption:'',defaultSelected:null};
}
export function addNode(g,id,x,y,w,h,name,sub='',options={}){
 const n={id,x,y,w,h,name,sub,tone:'neutral',kind:'op',...options};g.nodes.push(n);return n;
}
export function addGroup(g,id,x,y,w,h,title='',subtitle='',options={}){
 const box={id,x,y,w,h,title,subtitle,...options};g.groups.push(box);return box;
}
export function addLabel(g,text,x,y,tone='muted',options={}){
 const l={id:`label-${g.labels.length}`,text,x,y,tone,...options};g.labels.push(l);return l;
}
export function addWire(g,points,source='',target='',options={}){
 const e={id:`wire-${g.edges.length}`,points,source,target,kind:'flow',arrow:true,...options};g.edges.push(e);return e;
}
export function addSum(g,id,x,y,options={}){
 return addNode(g,id,x-12,y-12,24,24,'+','',{kind:'sum',...options});
}
export function makeInfo(category,title,body,formula,facts=[],sources=[],links=[]){return {category,title,body,formula,facts,sources,links};}
