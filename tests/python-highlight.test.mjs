import test from 'node:test';
import assert from 'node:assert/strict';
import {pythonTokens,highlightPython} from '../src/python-highlight.mjs';
import {PYTORCH_EXAMPLES} from '../src/pytorch.mjs';

test('Python highlighting preserves every byte, including multiline strings and escapes',()=>{
 const cases=[...PYTORCH_EXAMPLES.map(x=>x.code),'@torch.no_grad()\nclass RMSNorm(nn.Module):\n    """A # is text\nnot a comment."""\n    value = 1e-6 + 0xFF\n',"text = r'quoted \\' string' # actual comment\n",'html = "</code><script>alert(1)</script>"\n'];
 for(const code of cases)assert.equal(pythonTokens(code).map(x=>x.text).join(''),code);
 const tokens=pythonTokens('def normalize(x: Tensor) -> Tensor:\n    return x.float() * 1e-6 # note\n');
 assert.ok(tokens.some(x=>x.kind==='keyword'&&x.text==='return'));
 assert.ok(tokens.some(x=>x.kind==='function'&&x.text==='normalize'));
 assert.ok(tokens.some(x=>x.kind==='number'&&x.text==='1e-6'));
 assert.ok(tokens.some(x=>x.kind==='comment'&&x.text==='# note'));
 assert.ok(pythonTokens('class RMSNorm:').some(x=>x.kind==='type'&&x.text==='RMSNorm'));
});
test('rendered Python escapes markup instead of interpreting source as HTML',()=>{
 const html=highlightPython('message = "</code><script>bad()</script>" # <img src=x>\n');
 assert.doesNotMatch(html,/<script|<img|<\/code>/);
 assert.match(html,/&lt;script&gt;/);
 assert.match(html,/class="python-string"/);
 assert.match(html,/class="code-line"/);
});
