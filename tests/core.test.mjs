import test from 'node:test';
import assert from 'node:assert/strict';
let c={}; try { c=await import('../src/core.mjs'); } catch {}
test('multi-model API exists',()=>{for(const k of ['getModel','layerKind','cacheLedger','trainingLedger','networkLedger','sanitizeState','parameterLedger'])assert.equal(typeof c[k],'function',k);});
test('all nine exact variants are represented',()=>{assert.equal(Object.keys(c.MODELS).length,9);assert.equal(c.MODELS.q38flash.repo,'Qwen/Qwen3.8-Flash-Next');});
test('Kimi 93 layers include the last global layer',()=>{let t=Array.from({length:93},(_,i)=>c.layerKind('kimi3',i+1));assert.equal(t.filter(x=>x==='kda').length,69);assert.equal(t[92],'mla');});
test('Qwen3.8 is not ordinary Qwen3',()=>{assert.equal(c.layerKind('q3827',1),'delta');assert.equal(c.layerKind('q3827',4),'gqa');assert.equal(c.layerKind('q38flash',4),'qsa');assert.equal(c.MODELS.q38flash.residual,'hyper');assert.equal(c.MODELS.q3827.experts,0);});
test('Qwen3 heads are not forced to divide residual width',()=>{assert.equal(c.MODELS.q3235.heads*c.MODELS.q3235.headDim,8192);assert.equal(c.MODELS.q3235.d,4096);});
test('GQA counts K and V heads, not Q',()=>{const l=c.cacheLedger('q38',{...c.DEFAULT_STATE,context:1024,batch:1,kvBytes:2});assert.equal(l.history,36*1024*2*8*128*2);});
test('DeepSeek compressed MLA retains content latent and rotary key',()=>{const l=c.cacheLedger('ds2',{...c.DEFAULT_STATE,context:1,batch:1,kvBytes:2});assert.equal(l.history,60*576*2);assert.equal(l.recurrent,0);});
test('Kimi NoPE does not eliminate the shared 64 channels',()=>{const l=c.cacheLedger('kimi3',{...c.DEFAULT_STATE,context:1,batch:1,kvBytes:2,stateBytes:4});assert.equal(l.history,24*576*2);assert.equal(l.recurrent,69*96*128*128*4);});
test('hybrid recurrent state is independent of context',()=>{const a=c.cacheLedger('q3827',{...c.DEFAULT_STATE,context:2048});const b=c.cacheLedger('q3827',{...c.DEFAULT_STATE,context:4096});assert.equal(a.recurrent,b.recurrent);assert.equal(b.history,a.history*2);assert.equal(a.recurrent,48*48*128*128*4);});
test('QSA sparse reads do not bound the full history',()=>{const a=c.cacheLedger('q38flash',{...c.DEFAULT_STATE,context:8192});const b=c.cacheLedger('q38flash',{...c.DEFAULT_STATE,context:16384});assert.equal(b.history,2*a.history);assert.ok(a.exclusions.some(x=>x.includes('PLE')));assert.ok(a.indexer>0);});
test('uniform sharding changes per-rank state, not global state',()=>{const a=c.trainingLedger('ds3',{...c.DEFAULT_STATE,ranks:8});const b=c.trainingLedger('ds3',{...c.DEFAULT_STATE,ranks:16});assert.equal(a.total,b.total);assert.equal(a.perRank,2*b.perRank);});
test('dense models cannot produce expert dispatch',()=>{assert.equal(c.networkLedger('q38',{...c.DEFAULT_STATE,traffic:'expert'}).sent,0);});
test('single rank has zero off-rank collective or expert traffic',()=>{for(const traffic of ['expert','allreduce'])assert.equal(c.networkLedger('ds3',{...c.DEFAULT_STATE,traffic,ranks:1}).sent,0);});
test('invalid URL input is rejected or clamped, never injected',()=>{const x=c.sanitizeState({model:'<img>',view:'nope',batch:-2,context:Infinity,layer:9999,unknown:42});assert.equal(x.model,c.DEFAULT_STATE.model);assert.equal(x.batch,c.DEFAULT_STATE.batch);assert.ok(!('unknown' in x));assert.ok(x.layer<=c.MODELS[x.model].layers);});
test('hardware capacity and scope distinguish GPU from rack',()=>{assert.equal(c.HARDWARE.b200.hbmGB,180);assert.equal(c.HARDWARE.gb300.hbmGB,288);assert.equal(c.HARDWARE.gb300.gpus,72);assert.equal(c.HARDWARE.b200.switches,2);});
test('parameter ledgers declare rounded totals and extras',()=>{assert.equal(c.parameterLedger('q38flash').ngram,51e9);assert.equal(c.parameterLedger('q38flash').mtp,4e9);assert.equal(c.parameterLedger('q38flash').resident,176e9);assert.ok(c.parameterLedger('kimi3').approximate);});
test('absent mechanisms produce zero, never 0 times NaN',()=>{for(const id of Object.keys(c.MODELS)){const l=c.cacheLedger(id,c.DEFAULT_STATE);for(const k of ['history','recurrent','conv','indexer','total'])assert.ok(Number.isFinite(l[k]),`${id}.${k}`);
 assert.ok(Number.isFinite(c.servingLedger(id,c.DEFAULT_STATE).perRank));}});

test('dsv41 context-linear and bounded match an independent derivation from MODELS fields', () => {
  // Independent of cacheLedger's own history/indexer/recurrent/conv/windowKV decomposition:
  // this recomputes the FP4 main-KV and FP8 window byte costs straight from MODELS.dsv41,
  // so a wrong ratio, channel count, byte-per-channel cost or window arithmetic fails here.
  const m = c.MODELS.dsv41, B = 1, S = 4096;
  const cache = c.cacheLedger(m, { context: S, batch: B });
  const perToken = m.kvSourceLayers.reduce((a, l) =>
    a + m.headDim / (l < m.encoderLayers ? m.compressEncoder : m.compressDecoder), 0) * (0.5 + 1 / 16);
  assert.equal(cache.history, B * S * perToken, 'FP4 main KV over the four Full-mode layers');
  assert.equal(cache.contextLinear, cache.history + B * S * 170, 'packed indexer payload and scales');
  assert.equal(cache.bounded, B * m.window * m.layers * m.headDim, 'FP8 SWA window across all forty layers');
});

test('cacheLedger recurrent/conv match an independent derivation for a hybrid model', () => {
  // Independent of layerPartition: recomputes the delta-layer count from layers/globalEvery
  // and the byte costs from linearHeads/linearHeadDim/linearKeyHeads/convKernel directly.
  const m = c.MODELS.q3827, B = 1, S = 2048, stateBytes = 4;
  const cache = c.cacheLedger(m, { context: S, batch: B, stateBytes });
  const deltaLayers = m.layers - Math.floor(m.layers / m.globalEvery);
  const recurrent = deltaLayers * B * m.linearHeads * m.linearHeadDim * m.linearHeadDim * stateBytes;
  const conv = deltaLayers * B * (2 * m.linearKeyHeads + m.linearHeads) * m.linearHeadDim * m.convKernel * 2;
  assert.equal(cache.recurrent, recurrent, 'current recurrent-matrix state for every non-global layer');
  assert.equal(cache.conv, conv, 'streaming convolution buffers for every non-global layer');
});

test('dsv41 transcribes the published shapes and derives nothing by rounding', () => {
  const m = c.MODELS.dsv41;
  assert.equal(m.layers, 40);
  assert.equal(m.encoderLayers, 20);
  assert.equal(m.d, 5120);
  assert.equal(m.heads, 64);
  assert.equal(m.kvHeads, 1, 'one latent KV serves all 64 query heads');
  assert.equal(m.headDim, 512);
  assert.equal(m.nopeDim, m.headDim - m.ropeDim);
  assert.equal(m.experts, 384);
  assert.equal(m.topK, 6);
  assert.equal(m.shared, 1);
  assert.equal(m.denseLayers, 0, 'every Transformer block is MoE');
  assert.equal(m.architecture, 'ced-csa2');
  assert.equal(m.residual, 'mhc');
  assert.equal(m.scoring, 'sqrtsoftplus');
  assert.equal(m.active, 8e9);
  assert.equal(m.activeDecode, 16e9);
  // the exact figure is derivable, so the card's rounded "196B" is not good enough
  assert.equal(m.ngram, 196613849600);
  assert.equal(m.engramTables.reduce((a, b) => a + b, 0) * m.engramHeadDim, m.ngram);
});

// Derived from the class, not from a roster of it. A title that names a class over a body that
// names a set exempts the next checkpoint silently, which is this branch's recurring defect in
// miniature -- and the roster and the class coincide only for as long as nobody adds a model.
const withoutCED = () => Object.values(c.MODELS).filter(m => !m.encoderLayers).map(m => m.id);

test('every model without an encoder/decoder split keeps the common CED defaults', () => {
  assert.ok(withoutCED().length >= 8, 'the class must not be empty or accidentally narrowed');
  for (const id of withoutCED()) {
    const m = c.MODELS[id];
    assert.equal(m.encoderLayers, 0, `${id} has no encoder/decoder split`);
    assert.equal(m.activeDecode, 0, `${id} publishes one active figure`);
    assert.equal(m.window, 0, `${id} has no sliding-window branch`);
    assert.equal(m.oRank, 0);
    assert.equal(m.oGroups, 0);
  }
});

test('dsv41 layer kinds follow the published group structure', () => {
  const m = c.MODELS.dsv41;
  // encoder: SWA x2, then three groups of six led by a Full layer
  assert.equal(c.layerKind(m, 1), 'swa');
  assert.equal(c.layerKind(m, 2), 'swa');
  for (const l of [3, 9, 15]) assert.equal(c.layerKind(m, l), 'csa2-full', `layer ${l} owns its KV`);
  for (const l of [4, 8, 10, 14, 16, 20]) assert.equal(c.layerKind(m, l), 'csa2-reuse');
  // decoder: [Full, Reuse x3] then [Reindex, Reuse x3] x4
  assert.equal(c.layerKind(m, 21), 'csa2-full', 'the decoder projects its KV from H20 here');
  for (const l of [25, 29, 33, 37]) assert.equal(c.layerKind(m, l), 'csa2-reindex');
  for (const l of [22, 23, 24, 38, 39, 40]) assert.equal(c.layerKind(m, l), 'csa2-reuse');
});

test('dsv41 partition sums to its layer count and excludes the draft layers', () => {
  const p = c.layerPartition('dsv41');
  assert.deepEqual(p.swa, 2);
  assert.deepEqual(p['csa2-full'], 4);
  assert.deepEqual(p['csa2-reindex'], 4);
  assert.deepEqual(p['csa2-reuse'], 30);
  const sum = Object.values(p).reduce((a, b) => a + b, 0);
  assert.equal(sum, c.MODELS.dsv41.layers, 'DSpark draft layers are mtp, not backbone layers');
});

test('every model partitions to exactly its layer count', () => {
  for (const m of Object.values(c.MODELS)) {
    const sum = Object.values(c.layerPartition(m)).reduce((a, b) => a + b, 0);
    assert.equal(sum, m.layers, `${m.id} partition must sum to ${m.layers}`);
  }
});

test('layerKind still rejects layers outside the backbone', () => {
  assert.throws(() => c.layerKind('dsv41', 41), RangeError, 'draft layers are not addressable here');
  assert.throws(() => c.layerKind('dsv41', 0), RangeError);
});

test('every model splits its cache into context-linear and bounded halves', () => {
  for (const m of Object.values(c.MODELS)) {
    const cl = c.cacheLedger(m, { context: 8192 });
    assert.equal(cl.contextLinear + cl.bounded, cl.total, `${m.id} halves must sum to total`);
    assert.ok(cl.contextLinear >= 0 && cl.bounded >= 0, `${m.id} no negative cache`);
  }
});

test('bounded state does not move with context, for any model', () => {
  for (const m of Object.values(c.MODELS)) {
    const near = c.cacheLedger(m, { context: 4096 }).bounded;
    const far = c.cacheLedger(m, { context: 262144 }).bounded;
    assert.equal(near, far, `${m.id} bounded term must not scale with S`);
  }
});

test('models with recurrent state report it as bounded, not zero', () => {
  for (const id of ['q3827', 'q38flash', 'kimi3']) {
    assert.ok(c.cacheLedger(id, { context: 8192 }).bounded > 0, `${id} holds recurrent state`);
  }
});

test('dsv41 global KV is 720 bytes per token from the four Full-mode layers', () => {
  const one = c.cacheLedger('dsv41', { context: 1, batch: 1 });
  // 3 encoder Full layers at m=2 and 1 decoder Full layer at m=1, FP4 at 0.5625 B/channel
  assert.equal(one.mainKV, 720);
  assert.equal(one.contextLinear, 890);
  const ten = c.cacheLedger('dsv41', { context: 10, batch: 1 });
  assert.equal(ten.mainKV, 7200, 'main KV is linear in context');
});

test('dsv41 SWA KV is bounded by the window across all forty layers', () => {
  const cl = c.cacheLedger('dsv41', { context: 1048576, batch: 1 });
  assert.equal(cl.windowKV, 128 * 40 * 512, 'window x layers x headDim at FP8');
  assert.equal(cl.bounded, cl.windowKV);
});

test('dsv41 reports its global-history layers as the thirty-eight CSA2 layers', () => {
  assert.equal(c.cacheLedger('dsv41', {}).global, 38, 'the two SWA-only layers hold no global history');
  assert.equal(c.cacheLedger('dsv41', {}).linear, 0, 'no recurrent attention in this model');
});

test('dsv41 parameters are derived, not rounded', () => {
  const r = c.parameterLedger('dsv41');
  assert.equal(r.approximate, false, 'this branch calculates rather than quoting');
  assert.equal(r.ngram, 196613849600);
  assert.equal(r.resident, r.main + r.ngram);
});

test('the derived dsv41 backbone reconciles with the published 552B', () => {
  const r = c.parameterLedger('dsv41');
  const published = 552e9;
  const drift = Math.abs(r.main - published) / published;
  assert.ok(drift < 0.01, `derived ${r.main} drifts ${(drift * 100).toFixed(3)}% from 552B`);
  assert.ok(r.main < published * 1.01, 'the text backbone must not exceed the published figure');
});

test('dsv41 counts the Compressor and Indexer projections model.py actually instantiates', () => {
  const m = c.MODELS.dsv41;
  const r = c.parameterLedger('dsv41');
  // Compressor (model.py:437-457): wkv and norm (RMSNorm(head_dim)) on all 4 kvSourceLayers;
  // wgate only where compress_ratio > 1.
  const wgateLayers = m.kvSourceLayers.filter(l => (l < m.encoderLayers ? m.compressEncoder : m.compressDecoder) > 1).length;
  assert.equal(wgateLayers, 3, 'three of the four Full-mode layers compress; the decoder one passes through at ratio 1');
  const compressor = m.kvSourceLayers.length * (m.d * m.headDim + m.headDim) + wgateLayers * (m.d * m.headDim);
  // Indexer (model.py:496-526): wq_b + weights_proj on all 8 indexSourceLayers; wk and k_norm
  // (RMSNorm(index_head_dim)) only on the 4 kvSourceLayers, the same guard as wk.
  const indexer = m.indexSourceLayers.length * (m.qRank * m.indexerHeads * m.indexer + m.d * m.indexerHeads)
                 + m.kvSourceLayers.length * (m.headDim * m.indexer + m.indexer);
  assert.equal(compressor, 18352128, 'Compressor: wkv + norm on 4 layers, wgate on 3');
  assert.equal(indexer, 43516416, 'Indexer: wq_b + weights_proj on 8 layers, wk + k_norm on 4');
  assert.equal(compressor + indexer, 61868544, 'the previously omitted modules are ~61.9M parameters');
  // Pinned: Attention/Gate/mHC/Engram-only total was 551,818,966,640; counting the Compressor
  // and Indexer projections (including their RMSNorm submodules) must land on exactly that plus
  // the 61,868,544 above, not silently exclude any of them from `main` while still calling the
  // derivation exact.
  assert.equal(r.main, 551880835184, 'main must include the Compressor and Indexer projections, norms included');
  const publishedDriftWith = Math.abs(r.main - 552e9) / 552e9;
  const publishedDriftWithout = Math.abs((r.main - compressor - indexer) - 552e9) / 552e9;
  assert.ok(publishedDriftWith < publishedDriftWithout,
    'counting these instantiated modules must move drift closer to the published figure, not further');
});

test('dsv41 does not fabricate a DSpark parameter count', () => {
  const r = c.parameterLedger('dsv41');
  assert.equal(r.mtp, 0, 'no DSpark parameter count is published; mtp:3 is a layer count');
  assert.equal(r.hasMtp, true, 'the mechanism exists even though its size is unpublished');
  assert.match(r.scope, /DSpark/, 'scope must name what it excludes');
});

test('dsv41 prefill computes only the encoder half', () => {
  const pre = c.servingLedger('dsv41', { phase: 'prefill' });
  const dec = c.servingLedger('dsv41', { phase: 'decode' });
  assert.equal(pre.phaseLayers, 20);
  assert.equal(dec.phaseLayers, 40);
  assert.equal(pre.phaseActive, 8e9);
  assert.equal(dec.phaseActive, 16e9);
});

test('the prefill saving is the depth ratio the architecture claims', () => {
  const m = c.MODELS.dsv41;
  const depthRatio = m.encoderLayers / m.layers;
  const activeRatio = c.servingLedger(m, { phase: 'prefill' }).phaseActive /
                      c.servingLedger(m, { phase: 'decode' }).phaseActive;
  // published actives are rounded to whole billions, so this is a tolerance not an identity
  assert.ok(Math.abs(activeRatio - depthRatio) < 0.05,
    `active ratio ${activeRatio} should track depth ratio ${depthRatio}`);
});

test('phase is a no-op for models without an encoder/decoder split', () => {
  assert.ok(withoutCED().length >= 8, 'the class must not be empty or accidentally narrowed');
  for (const id of withoutCED()) {
    const pre = c.servingLedger(id, { phase: 'prefill' });
    const dec = c.servingLedger(id, { phase: 'decode' });
    assert.equal(pre.phaseLayers, c.MODELS[id].layers, `${id} computes every layer at prefill`);
    assert.equal(pre.phaseActive, dec.phaseActive, `${id} publishes one active figure`);
  }
});

test('an unknown phase falls back to the default rather than propagating', () => {
  assert.equal(c.servingLedger('dsv41', { phase: 'wishful' }).phase, 'decode');
});

// ---------------------------------------------------------------- innovation coverage
// Prose that asserts an architectural fact needs the same evidence chain as a number in a
// diagram. Two properties make that enforceable, and both are tested here rather than trusted:
// `effect` is a function of the model, so a literal cannot be written in the first place, and
// every record cites source ids that must resolve.
let inv = {}; try { inv = await import('../src/innovations.mjs'); } catch {}

test('every innovation record derives its effect rather than stating it', () => {
  assert.ok(Array.isArray(inv.INNOVATIONS) && inv.INNOVATIONS.length, 'INNOVATIONS must exist');
  for (const rec of inv.INNOVATIONS) {
    assert.equal(typeof rec.effect, 'function', `${rec.id} must derive its effect`);
    assert.ok(rec.sources.length > 0, `${rec.id} must cite a source`);
    for (const sid of rec.sources) assert.ok(c.SOURCES[sid], `${rec.id} cites unknown source ${sid}`);
    assert.ok(rec.title && rec.mechanism && rec.replaces, `${rec.id} needs all three prose fields`);
  }
});

test('the innovation coverage names every innovation the architecture actually has', () => {
  const ids = new Set(inv.INNOVATIONS.map(r => r.id));
  for (const id of ['ced', 'csa2', 'hier-indexer', 'mhc', 'engram', 'dspark', 'fp4-kv', 'swa-replay'])
    assert.ok(ids.has(id), `${id} is a published innovation and must be covered`);
});

test('rendered innovation effects contain no hardcoded architectural number', () => {
  const m = c.MODELS.dsv41;
  const rows = inv.innovationRows(m);
  assert.equal(rows.length, inv.INNOVATIONS.length);
  // the CED row must move when the model does, proving it is derived
  const shallow = inv.innovationRows({ ...m, encoderLayers: 10 });
  const cedBefore = rows.find(r => r.id === 'ced').effect;
  const cedAfter = shallow.find(r => r.id === 'ced').effect;
  assert.notEqual(cedBefore, cedAfter, 'the CED effect must follow encoderLayers');
});

// One targeted mutation per record. A record whose effect ignores the field its exposition is
// about would pass the "is a function" test while still being a literal in disguise; this is the
// assertion that catches that, and it covers all eight rather than only the CED row.
test('every innovation effect follows the field its exposition is about', () => {
  const m = c.MODELS.dsv41;
  const base = new Map(inv.innovationRows(m).map(r => [r.id, r.effect]));
  const mutations = {
    'ced': { encoderLayers: 10 },
    'csa2': { kvSourceLayers: [2, 8, 14, 20], indexSourceLayers: [2, 8, 14, 20, 24, 28, 32] },
    'hier-indexer': { candidateBlocks: 1024 },
    'mhc': { streams: 2 },
    'engram': { engramLayers: [1] },
    'dspark': { dsparkBlock: 4 },
    'fp4-kv': { kvBytesPerToken: 900 },
    'swa-replay': { window: 64 },
    // and the long end of its comparison is the checkpoint's own native context, not a literal

  };
  for (const [id, patch] of Object.entries(mutations)) {
    const moved = inv.innovationRows({ ...m, ...patch }).find(r => r.id === id).effect;
    assert.notEqual(moved, base.get(id), `${id} must derive its effect from ${Object.keys(patch)}`);
  }
  const shortContext = inv.innovationRows({ ...m, maxContext: 131072 }).find(r => r.id === 'swa-replay').effect;
  assert.notEqual(shortContext, base.get('swa-replay'), 'the bounded row must compare at the model\'s own native context');
});

// The "derived, never retyped" rule applies to `replaces` and `mechanism` too, not only to
// `effect`. Those sentences are exactly where a stale number survives a data change: "the bottom
// half of the stack" asserted `encoderLayers === layers / 2` in prose, and the mutation test above
// rendered "10 of 40" beside it. So authored prose carries no quantity at all — not a digit, and
// not a cardinal spelled as a word, which a digit scan alone would miss (the 16 in "one E4M3
// scale per sixteen channels" was exactly that shape).
const QUANTITY = /[0-9]|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|hundred|thousand|million|billion|half|halves|third|quarter|twice|double|triple|both)\b/i;

test('authored innovation prose carries no quantity, in digits or in words', () => {
  for (const rec of inv.INNOVATIONS) {
    // `title` is excluded: it is the mechanism's published name, and the 2 in "Compressed Sparse
    // Attention 2" is a generation, not a count of anything the model declares.
    for (const field of ['replaces', 'mechanism']) {
      const hit = rec[field].match(QUANTITY);
      assert.equal(hit, null,
        `${rec.id}.${field} states the quantity ${JSON.stringify(hit && hit[0])}; a quantity belongs in effect, where it is derived`);
    }
  }
});

test('the prose rule is enforced, not merely declared', () => {
  // The negative half: the exact sentence this rule was written for must be rejected. Without
  // this, a regex that matched nothing would pass the test above over any prose at all.
  for (const bad of ['The bottom half of the stack is a causal encoder.',
                     'one E4M3 scale per sixteen channels',
                     'Each layer takes one of three static modes.',
                     'prefill computes 20 of 40 layers']) {
    assert.ok(QUANTITY.test(bad), `the rule must reject ${JSON.stringify(bad)}`);
  }
  assert.ok(!QUANTITY.test('Ownership is a property of the layer, fixed by configuration.'),
    'the rule must not reject prose that states no quantity');
});

// The formats are model fields, so the FP4 row renders them rather than the prose asserting them.
test('the FP4 row names the cached precisions from the model table', () => {
  const m = c.MODELS.dsv41;
  const row = inv.innovationRows(m).find(r => r.id === 'fp4-kv');
  assert.match(row.effect, new RegExp(m.kvFormat.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(row.effect, new RegExp(m.swaKvFormat));
  const renamed = inv.innovationRows({ ...m, kvFormat: 'FP6 E3M2' }).find(r => r.id === 'fp4-kv');
  assert.notEqual(renamed.effect, row.effect, 'the precision must follow m.kvFormat');
});

test('the FP4 row states the derived figure alongside the reported one', () => {
  const m = c.MODELS.dsv41;
  const row = inv.innovationRows(m).find(r => r.id === 'fp4-kv');
  const derived = c.cacheLedger(m, { context: 1, batch: 1 }).mainKV;
  assert.notEqual(derived, m.kvBytesPerToken, 'the two figures differ; that is the point');
  assert.match(row.effect, new RegExp(`\\b${derived}\\b`), 'the derived figure must be shown');
  assert.match(row.effect, new RegExp(`\\b${m.kvBytesPerToken}\\b`), 'the reported figure must be shown');
});
