/** One canonical, executable teaching example per mechanism; no upstream runtime imports. */
import {PYTHON_SOURCES} from './pytorch-source.mjs';
import {SOURCES} from './models.mjs';
import {BACKBONE_EXAMPLES} from './pytorch-backbone.mjs';
import {MEMORY_EXAMPLES} from './pytorch-memory.mjs';
import {DSPARK_EXAMPLES} from './pytorch-dspark.mjs';
export const PYTORCH_LICENSE=`MIT License

Copyright (c) 2023 DeepSeek
Copyright (c) 2026 The DeepSpec Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`;
export const PYTORCH_EXAMPLES=[...BACKBONE_EXAMPLES,...MEMORY_EXAMPLES,...DSPARK_EXAMPLES];
export function chapterExamples(stage){return PYTORCH_EXAMPLES.filter(x=>x.stages.includes(stage));}
export const PYTORCH_CONFIG=PYTHON_SOURCES['config.py'];
/** Flatten only the known package import for portable, single-file downloads. */
export function standaloneCode(code){return code.replace(/^from \.config import V41Config\n/gm,'');}
export function exampleProgram(id){
 const ordered=[],visiting=new Set();
 function visit(key){
  const example=PYTORCH_EXAMPLES.find(x=>x.id===key);
  if(!example)throw new Error(`Unknown PyTorch example: ${key}`);
  if(visiting.has(key))throw new Error(`Cyclic PyTorch dependency: ${key}`);
  if(ordered.includes(example))return;
  visiting.add(key);for(const dependency of example.requires)visit(dependency);visiting.delete(key);ordered.push(example);
 }
 visit(id);
 return PYTORCH_LICENSE.split('\n').map(line=>'# '+line).join('\n')+'\n\n'+PYTORCH_CONFIG+'\n\n'+ordered.map(x=>`# ${x.title}\n# Sources: ${x.sources.map(id=>SOURCES[id].url).join(' ')}\n# Source symbols: ${x.symbols.join(', ')}\n# Scope: ${x.limitations}\n${standaloneCode(x.code)}`).join('\n\n')+'\n';
}

/** Module views use the example inventory, so code and navigation cannot diverge. */
export function moduleStage(id){
 if(!PYTORCH_EXAMPLES.some(example=>example.id===id))throw new RangeError(`Unknown PyTorch example: ${id}`);
 return `module-${id}`;
}
export function moduleExample(stage){return PYTORCH_EXAMPLES.find(example=>`module-${example.id}`===stage)||null;}
