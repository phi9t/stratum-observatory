/** Dedicated entry points share the atlas renderer and preserve exact variant identities. */
export const MODEL_VIEWS=['model','training','serving','evidence'];
export const PAGES=[
 {id:'foundations',file:'foundations.html',title:'Modeling Foundations',models:['ds2','ds3','q38','q330','q3235'],defaultModel:'ds3',stage:'block'},
 {id:'qwen38',file:'qwen3.8.html',title:'Qwen3.8',models:['q3827','q38flash'],defaultModel:'q3827',stage:'block'},
 {id:'kimi3',file:'kimi-k3.html',title:'Kimi K3',models:['kimi3'],defaultModel:'kimi3',stage:'block'},
 {id:'dsv41',file:'deepseek-v4.1.html',title:'DeepSeek V4.1 Flash',models:['dsv41'],defaultModel:'dsv41',stage:'ced'},
 {id:'hardware',file:'hardware.html',title:'GPU Hardware',models:[],defaultModel:'ds3',stage:'block',view:'hardware'},
 {id:'network',file:'networking.html',title:'Networking',models:[],defaultModel:'ds3',stage:'block',view:'network'},
];
export function pageForModel(model){return PAGES.find(page=>page.models.includes(model));}
export function pageHref(model){const page=pageForModel(model);return page?`${page.file}${model===page.defaultModel?'':`#model=${encodeURIComponent(model)}`}`:'index.html';}

export function pageForView(view){return PAGES.find(page=>page.view&&page.view===view);}
export function sectionHref(view,state){const page=pageForView(view);return `${page?.file||pageForModel(state.model)?.file||'index.html'}#${new URLSearchParams({...state,view})}`;}

export const GUIDES=[
 {id:'home',file:'index.html',title:'Home'},
 {id:'data',file:'data-representation.html',title:'Data Representation'},
];
