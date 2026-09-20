/** Dedicated entry points share the atlas renderer and preserve exact variant identities. */
export const PAGES=[
 {id:'qwen38',file:'qwen3.8.html',title:'Qwen3.8',models:['q3827','q38flash'],defaultModel:'q3827',stage:'block'},
 {id:'kimi3',file:'kimi-k3.html',title:'Kimi K3',models:['kimi3'],defaultModel:'kimi3',stage:'block'},
 {id:'dsv41',file:'deepseek-v4.1.html',title:'DeepSeek V4.1 Flash',models:['dsv41'],defaultModel:'dsv41',stage:'ced'},
];
export function pageForModel(model){return PAGES.find(page=>page.models.includes(model));}
export function pageHref(model){const page=pageForModel(model);return page?`${page.file}${model===page.defaultModel?'':`#model=${encodeURIComponent(model)}`}`:'index.html';}
