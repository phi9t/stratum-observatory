/** Detailed operation graphs share the standard scene renderer and inspector. */
import {BACKBONE_SCHEMATICS} from './diagram-pytorch-backbone.mjs';
import {MEMORY_SCHEMATICS} from './diagram-pytorch-memory.mjs';
import {DSPARK_SCHEMATICS} from './diagram-pytorch-dspark.mjs';
const BUILDERS={...BACKBONE_SCHEMATICS,...MEMORY_SCHEMATICS,...DSPARK_SCHEMATICS};
export function buildModuleSchematic(model,example){
 const build=BUILDERS[example.id];
 if(!build)throw new RangeError(`Missing module schematic: ${example.id}`);
 return build(model,example);
}
