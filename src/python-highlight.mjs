/** Lossless Python display lexer. No execution, HTML parsing or external grammar assets. */
const KEYWORDS=new Set('and as assert async await break case class continue def del elif else except finally for from global if import in is lambda match nonlocal not or pass raise return try while with yield'.split(' '));
const CONSTANTS=new Set(['True','False','None','NotImplemented','Ellipsis']);
const BUILTINS=new Set('bool dict enumerate float int isinstance len list max min range set slice str super tuple zip'.split(' '));
const STRING=/^(?:br|rb|fr|rf|r|u|b|f)?(?:"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\[\s\S]|[^"\\\n])*(?:"|$)|'(?:\\[\s\S]|[^'\\\n])*(?:'|$))/i;
const NUMBER=/^(?:0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[jJ]?)/;
export function pythonTokens(source){
 const tokens=[];let offset=0,definition=null;
 while(offset<source.length){
  const rest=source.slice(offset);let text,kind='plain';
  if(rest[0]==='#'){text=rest.match(/^[^\n]*/)[0];kind='comment';}
  else if((text=rest.match(STRING)?.[0]))kind='string';
  else if((text=rest.match(NUMBER)?.[0]))kind='number';
  else if((text=rest.match(/^[A-Za-z_]\w*/)?.[0])){
   kind=definition||(KEYWORDS.has(text)?'keyword':CONSTANTS.has(text)?'constant':BUILTINS.has(text)?'builtin':/^[A-Z]/.test(text)?'type':/^\s*\(/.test(rest.slice(text.length))?'function':'plain');
   definition=text==='def'?'function':text==='class'?'type':null;
  }else if((text=rest.match(/^\s+/)?.[0])){if(text.includes('\n'))definition=null;}
  else{text=rest[0];kind=/[+\-*/%=<>!&|^~:@]/.test(text)?'operator':'plain';}
  tokens.push({kind,text});offset+=text.length;
 }
 return tokens;
}
const escapeHTML=text=>text.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function highlightPython(source){
 const lines=[''];
 for(const {kind,text} of pythonTokens(source)){
  const pieces=text.split('\n');
  pieces.forEach((piece,index)=>{
   if(index)lines.push('');
   if(piece)lines[lines.length-1]+=kind==='plain'?escapeHTML(piece):`<span class="python-${kind}">${escapeHTML(piece)}</span>`;
  });
 }
 return lines.map(line=>`<span class="code-line">${line}</span>`).join('\n');
}
