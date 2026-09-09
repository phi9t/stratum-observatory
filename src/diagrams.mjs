import {sanitizeState,getModel} from './core.mjs';
import {buildModel} from './diagram-model.mjs';
import {buildTraining,buildServing,buildHardware,buildNetwork} from './diagram-systems.mjs';
import {buildCompare,buildEvidence,buildInputs,buildResidual,buildPLE} from './diagram-meta.mjs';
export function buildScene(input){const s=sanitizeState(input);const m=getModel(s.model);if(s.view==='model'){if(s.stage==='inputs'&&m.vision)return buildInputs(s);if(s.stage==='residual'&&m.residual!=='standard')return buildResidual(s);if(s.stage==='ple'&&m.id==='q38flash')return buildPLE(s);return buildModel(s);}return({compare:buildCompare,training:buildTraining,serving:buildServing,hardware:buildHardware,network:buildNetwork,evidence:buildEvidence})[s.view](s);}
