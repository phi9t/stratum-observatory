/** What is new in the Causal Encoder-Decoder family, and where each claim comes from.
 *
 *  Prose that asserts an architectural fact needs the same evidence chain as a number in a
 *  diagram, so three rules hold here. `effect` is a **function** of the model rather than a
 *  string: a literal cannot drift from the data because a literal cannot be written. `sources` is
 *  a list of ids that must resolve in `SOURCES`, so every factual statement stays one click from
 *  the bytes it came from.
 *
 *  And `replaces`/`mechanism` -- the only authored sentences here -- carry **no quantity at all**,
 *  not even spelled out as a word. A sentence saying "the bottom half of the stack" asserts
 *  `encoderLayers === layers / 2` in prose, where nothing can check it and where it survives a
 *  data change; it once sat beside a derived figure that mutation testing rendered as "10 of 40".
 *  These fields say what changed; `effect` says how much. `tests/core.test.mjs` enforces it.
 */
import {cacheLedger,servingLedger,layerPartition,parameterLedger,formatNum,formatParams,formatBytes} from './core.mjs';

// The short end of the bounded-state comparison. A chosen scenario floor, not a model fact --
// the long end is the checkpoint's own `maxContext`. The point of the row is that the figure does
// not move between the two, so both are rendered rather than asserted.
export const SHORT_CONTEXT=1024;

export const INNOVATIONS=[
 {id:'ced',title:'Causal Encoder-Decoder',
  replaces:'Every layer deriving global keys and values from its own hidden state',
  mechanism:'The stack is split at a fixed depth: the layers below the boundary are a causal encoder, those above it a causal decoder. Decoder layers do not project global KV from their own hidden states; it is projected from the final encoder hidden state with layer-dependent weights. Prefill therefore computes the encoder only, while the decoder is entered per generated token, which is why the published active-parameter figure differs between prefill and decode.',
  effect:m=>`prefill computes ${servingLedger(m,{phase:'prefill'}).phaseLayers} of ${m.layers} layers · ${formatParams(servingLedger(m,{phase:'prefill'}).phaseActive)} active at prefill against ${formatParams(servingLedger(m,{phase:'decode'}).phaseActive)} at decode`,
  sources:['dsv41-code','dsv41-report']},

 {id:'csa2',title:'Compressed Sparse Attention 2',
  replaces:'Every attention layer keeping its own KV and computing its own selection',
  mechanism:'Each layer is assigned a single static mode. Full computes main KV and indexer K and publishes a selection; Reindex reuses those caches but re-scores them for a fresh selection; Reuse takes the previous selection unchanged. Ownership is a property of the layer, fixed by configuration, not a runtime decision — so how many layers pay for a cache is decoupled from the depth.',
  effect:m=>{const p=layerPartition(m);return `${p['csa2-full']} Full · ${p['csa2-reindex']} Reindex · ${p['csa2-reuse']} Reuse · ${p.swa} window-only, over ${m.layers} layers`;},
  sources:['dsv41-config','dsv41-code','dsv41-report']},

 {id:'hier-indexer',title:'Hierarchical Sparse Indexer',
  replaces:'Scoring every causally visible position at every layer that selects',
  mechanism:'Selection is split into a block stage and a position stage. The candidate-source layer scores fixed-size blocks of compressed positions and pins a candidate pool; every later selecting layer re-scores only inside that pool. Scoring a block is not reading it, and the layers above the source never see the unmasked range.',
  effect:m=>`${formatNum(m.candidateBlocks)} blocks × ${m.candidateBlockSize} = ${formatNum(m.candidateBlocks*m.candidateBlockSize)} compressed candidate positions, narrowed to a ${formatNum(m.indexerBudget)}-position budget`,
  sources:['dsv41-config','dsv41-code','dsv41-kernel']},

 {id:'mhc',title:'Single-Pass mHC',
  replaces:'A single residual stream re-entered through an identity add',
  mechanism:'The residual is widened into parallel streams, and each block mixes them with learned depth and width coefficients rather than adding back into a single rail. A Sinkhorn projection normalises the mixing matrix before it is applied, and the expansion, mix and contraction run as a single pass rather than per stream. A diagram that draws this as a plain plus sign is drawing a different model.',
  effect:m=>`${m.streams} residual streams · ${m.sinkhornIters} Sinkhorn iterations per mix`,
  sources:['dsv41-config','dsv41-code','dsv41-kernel']},

 {id:'engram',title:'Engram conditional memory',
  replaces:'Holding all learned lookup in dense weights that every token pays for',
  mechanism:'Hashed n-gram identities index large embedding tables injected at fixed depths. The tables are resident but conditional: a token reads only the rows its n-grams hash to, so the memory adds parameters to the checkpoint without adding dense per-token work. They are carried as a separate residency term for exactly that reason.',
  effect:m=>{const n=m.engramLayers.length;return `${formatParams(parameterLedger(m).ngram)} of table parameters at ${n} injection point${n===1?'':'s'}, separate from ${formatParams(parameterLedger(m).main)} of backbone`;},
  sources:['dsv41-config','dsv41-engram','dsv41-report']},

 {id:'dspark',title:'DSpark speculative decoding',
  replaces:'A single accepted token per target forward pass',
  mechanism:'A draft head re-runs the last decoder layers over a fixed block of future positions, routed through its own small expert set rather than the backbone\'s. The target then verifies the whole block in a single pass, so an accepted draft costs no additional target forward. Its parameter count is unpublished, so it is excluded from the calculated backbone rather than guessed at.',
  effect:m=>`${m.dsparkBlock}-token draft blocks · ${m.dsparkTopK} of ${m.dsparkExperts} draft experts per token · ${m.dsparkTargets.length} target layers`,
  sources:['dsv41-code','dsv41-report','dsv41-card']},

 {id:'fp4-kv',title:'FP4 main KV cache',
  replaces:'Retaining keys and values at the precision the layer computes in',
  mechanism:'Main KV is stored at a low precision with a shared exponent scale per channel group, and that precision is trained in rather than applied at serving time — so it is an architectural property, not a deployment option, and a KV-element setting does not apply to it. Only the Full-mode layers store main KV; sliding-window KV is retained at its own, coarser-quantised format.',
  effect:m=>{const derived=cacheLedger(m,{context:1,batch:1}).mainKV;const reported=m.kvBytesPerToken;
   return `${formatNum(derived)} B/token of ${m.kvFormat} main KV derived over ${m.kvSourceLayers.length} Full-mode layers against ${formatNum(reported)} B/token reported — a ${formatNum(reported-derived)} B difference that is indexer K at a precision the publisher never states; window KV is retained as ${m.swaKvFormat}`;},
  sources:['dsv41-report','dsv41-card','dsv41-kernel']},

 {id:'swa-replay',title:'SWA Bounded Replay',
  replaces:'A cache whose every term grows with the sequence',
  mechanism:'Every layer also attends over a short sliding window of raw KV, and the window-only layers attend over nothing else — no compressed cache, no indexer, no candidate pool. A window is a fixed span, so the state it retains is a function of window length, depth and head dimension; what leaves the window is not retained at all, and the bounded term of the ledger does not move with context.',
  effect:m=>{const near=cacheLedger(m,{context:SHORT_CONTEXT,batch:1}),far=cacheLedger(m,{context:m.maxContext,batch:1});
   return `${formatBytes(near.bounded)} bounded at ${formatNum(SHORT_CONTEXT)} positions and ${formatBytes(far.bounded)} at ${formatNum(m.maxContext)}, while context-linear state goes from ${formatBytes(near.contextLinear)} to ${formatBytes(far.contextLinear)}`;},
  sources:['dsv41-code','dsv41-report']},
];

/** One rendered row per record, for one model. The only place `effect` is called. */
export function innovationRows(m){return INNOVATIONS.map(r=>({id:r.id,title:r.title,replaces:r.replaces,mechanism:r.mechanism,effect:r.effect(m),sources:r.sources}));}
