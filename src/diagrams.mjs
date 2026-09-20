import {sanitizeState,getModel} from './core.mjs';
import {buildModel} from './diagram-model.mjs';
import {buildTraining,buildServing,buildHardware,buildNetwork} from './diagram-systems.mjs';
import {buildCompare,buildEvidence,buildInputs,buildResidual,buildPLE,buildCED} from './diagram-meta.mjs';
import {buildEncoder,buildDecoder,buildEngram,buildCSA2,buildDSpark} from './diagram-deepdives.mjs';
import {moduleExample} from './pytorch.mjs';
import {buildModuleSchematic} from './diagram-pytorch.mjs';
/** One builder per focal stage. Its keys are the values `focalStages` can return, asserted by
 * tests rather than assumed, so wiring a stage into one without the other cannot compile away. */
const FOCAL={inputs:buildInputs,residual:buildResidual,ple:buildPLE,ced:buildCED,encoder:buildEncoder,decoder:buildDecoder,engram:buildEngram,csa2:buildCSA2,dspark:buildDSpark};
export function buildScene(input){
 const s=sanitizeState(input),m=getModel(s.model);
 if(s.view!=='model')return({compare:buildCompare,training:buildTraining,serving:buildServing,hardware:buildHardware,network:buildNetwork,evidence:buildEvidence})[s.view](s);
 const example=moduleExample(s.stage);
 if(example)return buildModuleSchematic(m,example);
 if(s.stage==='block')return buildModel(s);
 // Exhaustive or throw, the shape ruling T2g gave the layer-kind dispatch. The old form ended in
 // `return buildModel(s)`, so a stage this checkpoint does not offer silently drew the decoder
 // block under the other page's name. `sanitizeState` clamps an unreachable stage to 'block', so
 // nothing a reader can type reaches this throw; a caller constructing state by hand gets a
 // loud error instead of a false diagram. The guard used to also check
 // `!focalStages(m).includes(s.stage)`, but `s` is already sanitized above, and `sanitizeState`
 // clamps any non-'block' stage to one `focalStages(s.model)` offers -- so that half could never
 // be true and is gone. `!build` still throws for the developer error that matters: a stage
 // `focalStages` is extended to offer with no matching key in `FOCAL`.
 const build=FOCAL[s.stage];
 if(!build)throw new RangeError(`${m.id} offers no ${s.stage} focal page.`);
 return build(s);
}
