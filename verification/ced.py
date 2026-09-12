"""Architecture invariants for the causal encoder-decoder family.

Engineering-class, like the rest of `schematics.py`: every value is derived from our own
scene IR and our own model table, so per docs/verifiers/DESIGN.md trust boundary 1 it must
never contribute to a scientific PASS.

A manifold-hyper-connection residual drawn as an ordinary `+` is exactly the misrepresentation
this project forbids, and it is the kind of defect that survives every geometric check: the sum
node is contained, orthogonally wired and perfectly legible. So it is checked as a graph
property rather than trusted to a reviewer's eye.

Every check here is either a cross-check between two independent derivations or a structural
property that a mis-transcription breaks. A check that restates config.json proves nothing and
does not belong in this module.
"""
from __future__ import annotations
import json,re,subprocess
from pathlib import Path
import hashlib
from .evidence import pointer,safe_path
from .model import Finding

ARCH='ced-csa2'


def is_ced(m):
    """Whether a checkpoint is a Causal Encoder-Decoder one, by the predicate the rest of the tree uses.

    Every other consumer of this population selects it with `encoderLayers`: `core.focalStages`
    offers the `ced` page on it, `tools/export-verification.mjs` emits innovation rows for it,
    `tools/export-scenes.mjs` attaches the CED expectations to it, and `app.mjs` labels it. This
    module alone tested `architecture == 'ced-csa2'`. Two predicates for one population agree
    only by today's coincidence -- the exact defect `core.focalStages`'s docstring was rewritten
    to record. A tenth CED checkpoint tagged `ced-csa3` would get its focal page, its scenes, its
    innovation rows and its previews while all six judges below emitted nothing at all, and
    `aggregate()` over an empty finding list is PASS.

    So the selector is `encoderLayers`, and `judge_architecture_tag` holds the tag to it: a model
    with encoder layers whose architecture is not `ced-csa2` is a FAIL here, not a silence.
    """
    return bool(m.get('encoderLayers'))

#: The report's §4.2.1 group sizes, and the only numbers `modes_from_report_structure` is given
#: beyond the depths. Prose, not config: "the remaining 18 encoder layers ... are divided into
#: three identically configured groups of six layers"; "the 20 decoder layers ... are divided into
#: five groups of four layers".
ENCODER_GROUP,DECODER_GROUP=6,4

#: `Block.__init__` (model.py:938) sizes every hyper-connection coefficient tensor as
#: `mix_hc = (2 + hc_mult) * hc_mult`.
MHC_COEFFICIENTS=24

#: `EngramLayout.build` (engram.py:107-113) allocates one bucket range per (n-gram size, head)
#: pair over `max_ngram_size - 1` n-gram sizes.
ENGRAM_HASH_COLUMNS=24

#: Depth-ratio slack for CED_ACTIVE_RATIO. Wide on purpose: see the check.
ACTIVE_RATIO_SLACK=0.05


def _ok(code,subject,message):
    return Finding(code,'schematics','PASS',subject,message)


def _bad(code,subject,message,remedy):
    return Finding(code,'schematics','FAIL',subject,message,remedy)


def _evidence_blocked(code,subject,message,remedy):
    """The only BLOCKED constructor in this module, and it is evidence-domain by construction.

    There is deliberately no schematics-domain equivalent. `aggregate` turns any BLOCKED into
    exit 2, `schematics` is one of the four gates the Pages deploy requires, and a software gate
    that blocks stops the page shipping. So the judges wired into that gate can only ever return
    PASS or FAIL -- a property `test_no_schematics_judge_can_emit_blocked` exercises over every
    failure branch rather than asserting.
    """
    return Finding(code,'evidence','BLOCKED',subject,message,remedy)


def inventory(root:Path)->tuple[dict,dict]:
    """Read what the application actually exposes, never a literal copy of it.

    Returns (models, samples) from one exporter run, so a gate needs no second subprocess.
    """
    out=subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,
                       capture_output=True,check=True,timeout=300)
    doc=json.loads(out.stdout)
    return doc['models'],doc['samples']

#: The two nodes that close the residual path in `residuals()` (src/diagram-model.mjs).
RESIDUAL_MERGES = ('merge-attn', 'merge-ffn')
RAIL_PREFIX = 'residual stream · '


def _stream_bad(subject, message, remedy):
    return Finding('MHC_STREAM_BALANCE', 'schematics', 'FAIL', subject, message, remedy)


def judge_mhc_streams(scene, m):
    """An mHC residual must name its stream count and must not collapse to a summation.

    `m` is the per-scene `expect` record from `tools/export-scenes.mjs`, i.e. what core.mjs
    independently says is true, so the count is never read back out of the drawing.
    """
    rails = [l.get('text', '') for l in scene.get('labels', []) if l.get('text', '').startswith(RAIL_PREFIX)]
    if not rails:
        return [Finding('MHC_SKIP', 'schematics', 'PASS', 'scene', 'Not a decoder-block lens; no residual rail to judge.')]
    out = []
    streams = m.get('streams')
    if not streams:
        out.append(_stream_bad('streams', 'An mHC preset declares no stream count for the rail to state.',
                        'Record the stream count in models.mjs; the diagram must not invent one.'))
    else:
        want = f'{streams} streams'
        for text in rails:
            if want not in text:
                out.append(_stream_bad(text, f'Residual rail reads {text!r} and never states the {want} the model declares.',
                                'Label the rail with the declared stream count; a widened residual drawn as one stream is a false diagram.'))
    for n in scene.get('nodes', []):
        if n.get('id') in RESIDUAL_MERGES and n.get('kind') == 'sum':
            out.append(_stream_bad(n['id'], 'A bare summation node sits on an mHC residual path.',
                            'mHC collapses and re-expands through a learned doubly stochastic mix; a "+" asserts arithmetic the model does not perform.'))
    return out or [Finding('MHC_OK', 'schematics', 'PASS', 'scene',
                           f"Residual rails state {streams} streams and no summation node closes the residual path.")]


def _topology(subject, message, remedy):
    return Finding('CED_SCENE_TOPOLOGY', 'schematics', 'FAIL', subject, message, remedy)


def layer_bands(kinds, lo, hi):
    """Runs of consecutive one-indexed layers: one head layer plus the layers that reuse it.

    Deliberately a *second* implementation of the rule `buildCED` applies in JavaScript, fed
    the per-layer kinds `core.layerKind` reported. Two independent transcriptions of the same
    grouping rule disagreeing is exactly the defect this catches; reading the bands back out of
    the drawing would catch nothing.
    """
    out = []
    for layer in range(lo, hi + 1):
        kind = kinds[layer - 1]
        if out and (kind == 'csa2-reuse' or (kind == 'swa' and out[-1][0] == 'swa')):
            out[-1][2] = layer
            continue
        out.append([kind, layer, layer])
    return [tuple(band) for band in out]


def band_id(side, index, lo, hi):
    """`buildCED`'s node-naming contract.

    The decoder's first band keeps the stable id `dec-full`: it is the target of the single CED
    handoff edge and the source of the candidate pool, so those two relations are assertable by
    name rather than by position.
    """
    return 'dec-full' if side == 'dec' and index == 0 else f'{side}-band-{lo}-{hi}'


def judge_ced_topology(scene, m):
    """The encoder/decoder page must state a one-way handoff between two derived halves.

    `m` is the per-scene `expect` record from `tools/export-scenes.mjs` -- what core.mjs says is
    true, independent of how it was drawn. A back edge from a decoder node into an encoder node
    would make this an ordinary shared-KV Transformer, and that is checked as a graph property
    instead of being asserted in a caption no verifier reads.
    """
    groups = {g.get('id') for g in scene.get('groups', [])}
    if not {'encoder', 'decoder'} <= groups:
        return [Finding('CED_SKIP', 'schematics', 'PASS', 'scene',
                        'Not the encoder/decoder lens; there is no CED topology to judge.')]
    nodes = scene.get('nodes', [])
    edges = scene.get('edges', [])
    ids = {n.get('id') for n in nodes}
    encoder_layers, kinds, depth = m.get('encoderLayers'), m.get('layerKinds') or [], m.get('layers')
    if not encoder_layers or not kinds or len(kinds) != (depth or -1):
        return [_topology('bands', 'The preset declares no encoder split, or no per-layer kinds for the page to state.',
                          'Record encoderLayers in models.mjs and export the layer kinds; the diagram must not invent a split.')]
    bands = {'enc': layer_bands(kinds, 1, encoder_layers),
             'dec': layer_bands(kinds, encoder_layers + 1, depth)}
    out = []

    for side, declared in (('enc', encoder_layers), ('dec', depth - encoder_layers)):
        want = {band_id(side, i, b[1], b[2]) for i, b in enumerate(bands[side])}
        drawn = {i for i in ids if i.startswith(f'{side}-band-')} | ({'dec-full'} & ids if side == 'dec' else set())
        for missing in sorted(want - drawn):
            out.append(_topology(missing, f'The {side} half declares a band {missing} that the page does not draw.',
                                 'Draw every band layerKind produces; a missing band hides layers from the reader.'))
        for extra in sorted(drawn - want):
            out.append(_topology(extra, f'The page draws a band {extra} that the layer pattern does not produce.',
                                 'No band may be invented to fit the template.'))
        covered = sum(b[2] - b[1] + 1 for i, b in enumerate(bands[side])
                      if band_id(side, i, b[1], b[2]) in drawn)
        if covered != declared:
            out.append(_topology(f'{side}-band-coverage',
                                 f'The {side} band tiles cover {covered} layers against {declared} declared.',
                                 'The two halves must tile 1..layers exactly; expand the drawing, never the band.'))

    handoff = [e for e in edges if e.get('source') == 'enc-out' and e.get('target') == 'dec-full']
    if len(handoff) != 1:
        out.append(_topology('enc-out→dec-full', f'{len(handoff)} CED handoff edges, not exactly one.',
                             'The handoff is one residual stream into the decoder Full layer.'))
    for edge in edges:
        source, target = str(edge.get('source') or ''), str(edge.get('target') or '')
        if source.startswith('dec-') and target.startswith('enc-'):
            out.append(_topology(edge.get('id', '<edge>'),
                                 f'An edge runs from decoder node {source} back into encoder node {target}.',
                                 'CED is a one-way handoff. A back edge makes this an ordinary shared-KV Transformer.'))

    def holding(side, layers):
        """The bands that actually contain the declared one-indexed layers."""
        found = set()
        for layer in layers:
            for i, band in enumerate(bands[side]):
                if band[1] <= layer <= band[2]:
                    found.add(band_id(side, i, band[1], band[2]))
        return found

    reindex = [i for i, band in enumerate(bands['dec']) if band[0] == 'csa2-reindex']
    pool_edges = [e for e in edges if e.get('source') == 'candidate-pool']
    want_pool = {band_id('dec', i, bands['dec'][i][1], bands['dec'][i][2]) for i in reindex}
    if len(pool_edges) != len(reindex) or {e.get('target') for e in pool_edges} != want_pool:
        out.append(_topology('candidate-pool',
                             f"The pool feeds {sorted(e.get('target') for e in pool_edges)} against the "
                             f'{len(reindex)} decoder-band Reindex layers {sorted(want_pool)}.',
                             'The pool feeds exactly the Reindex bands; it is a decoder-only mechanism.'))
    sourced = [e for e in edges if e.get('target') == 'candidate-pool' and e.get('source') == 'dec-full']
    if len(sourced) != 1:
        out.append(_topology('candidate-pool',
                             f'{len(sourced)} edges make the decoder Full layer the candidate source, not one.',
                             'The pool is established by the candidate-source layer and by nothing else.'))

    engram = [l + 1 for l in (m.get('engramLayers') or [])]
    engram_edges = [e for e in edges if e.get('source') == 'engram']
    # One edge per declared site, which for this checkpoint is also one edge per band holding a
    # site. Two sites inside one band would trip this rather than be reported as one arrow.
    if len(engram_edges) != len(engram) or {e.get('target') for e in engram_edges} != holding('enc', engram) | holding('dec', engram):
        out.append(_topology('engram',
                             f"Engram fans out to {sorted(e.get('target') for e in engram_edges)} against "
                             f'{len(engram)} declared injection sites at layers {engram}.',
                             'One edge per declared Engram site, landing on the band that holds it.'))
    dspark = [l + 1 for l in (m.get('dsparkTargets') or [])]
    reads = {e.get('source') for e in edges if e.get('target') == 'dspark'}
    if dspark and reads != holding('dec', dspark) | holding('enc', dspark):
        out.append(_topology('dspark', f'DSpark reads {sorted(reads)} against the bands holding its declared '
                                       f'target layers {dspark}.',
                             'Draw the read from the bands that actually contain the target layers.'))
    for node_id, layers, what in (('dspark', dspark, 'target'), ('engram', engram, 'injection')):
        node = next((n for n in nodes if n.get('id') == node_id), None)
        if node is None:
            out.append(_topology(node_id, f'The page has no {node_id} node.', 'Draw the mechanism the model declares.'))
            continue
        text = f"{node.get('name') or ''} {node.get('sub') or ''}"
        for layer in layers:
            # Not \b: 'L38' has no word boundary between the prefix and the digits.
            if not re.search(rf'(?<!\d){layer}(?!\d)', text):
                out.append(_topology(node_id, f'{node_id} never states {what} layer {layer}: {text!r}.',
                                     'Name the declared layers; a fan-out a reader cannot resolve states nothing.'))
    ratio = judge_band_ratios(nodes, bands, m)
    if out:
        return out + ratio
    return [Finding('CED_OK', 'schematics', 'PASS', 'scene',
                    f"{len(bands['enc'])} encoder and {len(bands['dec'])} decoder bands tile "
                    f'{depth} layers, one handoff edge, no decoder-to-encoder edge.')] + ratio


def band_ratio(m, lo):
    """Which compression ratio the band starting at one-indexed layer `lo` owes.

    A fourth implementation of a rule that already exists three times -- `ratioOf` in
    diagram-meta.mjs and the same expression inside `parameterLedger` and `cacheLedger`. That is
    the point: the ratios come from `expect`, i.e. from core.mjs, and the band boundary is
    re-derived here, so `ratioOf` drifting from the model table is a FAIL rather than a silently
    mislabelled page.
    """
    return m['compressEncoder'] if lo - 1 < m['encoderLayers'] else m['compressDecoder']


def judge_band_ratios(nodes, bands, m):
    """Every Full band must state the ratio its half declares.

    The subtitle is the only place the page says how much each band compresses, and a wrong
    ratio there is invisible to every geometric and topological check: the box is contained,
    orthogonally wired and perfectly legible while asserting the wrong arithmetic.
    """
    if not all(isinstance(m.get(k), (int, float)) for k in ('compressEncoder', 'compressDecoder')):
        return [Finding('CED_BAND_RATIO_DRAWN', 'schematics', 'FAIL', 'bands',
                        'The preset declares no band compression ratios for the page to state.',
                        'Record compressEncoder and compressDecoder in models.mjs and export them; '
                        'the diagram must not invent a ratio.')]
    by_id = {n.get('id'): n for n in nodes}
    out = []
    judged = 0
    for side, declared in bands.items():
        for i, band in enumerate(declared):
            if band[0] != 'csa2-full':
                continue
            node = by_id.get(band_id(side, i, band[1], band[2]))
            if node is None:
                continue                      # a missing band is already reported as topology
            ratio = band_ratio(m, band[1])
            want = '1 latent / position' if ratio == 1 else f'÷{ratio} latents'
            judged += 1
            text = f"{node.get('name') or ''} {node.get('sub') or ''}"
            if want not in text:
                out.append(Finding('CED_BAND_RATIO_DRAWN', 'schematics', 'FAIL', node['id'],
                                   f'The band states {text!r} where its half compresses {ratio}:1, '
                                   f'so it never says {want!r}.',
                                   'State the ratio the band actually writes. The rule lives in '
                                   'ratioOf, parameterLedger and cacheLedger at once; a page '
                                   'disagreeing with the model table means one of the three drifted.'))
    return out or [Finding('CED_BAND_RATIO_DRAWN', 'schematics', 'PASS', 'scene',
                           f'{judged} Full bands each state the compression ratio their half '
                           f"declares ({m['compressEncoder']}:1 encoder, {m['compressDecoder']}:1 decoder).")]


def modes_from_report_structure(*,swa:int,encoder:int,total:int,encoder_group:int,decoder_group:int)->list[str]:
    """Generate the mode assignment from the report's prose, §4.2.1, and nothing else.

    "For the first two layers, we use pure sliding window attention. The remaining 18
    encoder layers ... are divided into three identically configured groups of six layers.
    In each group, the first layer operates in Full Mode, and the remaining five ... in
    Reuse Mode. The 20 decoder layers ... are divided into five groups of four layers. In
    the first group, the first layer operates in Full Mode ... The remaining four groups
    ... the first layer operates in Reindex Mode."

    Deliberately takes no layer-id arrays: reading them would make the comparison in
    judge_architecture a tautology. The two derivations share only the depths.
    """
    modes=['swa']*swa
    for _ in range(swa,encoder,encoder_group):
        modes.append('csa2-full');modes+=['csa2-reuse']*(encoder_group-1)
    for i,_ in enumerate(range(encoder,total,decoder_group)):
        modes.append('csa2-full' if i==0 else 'csa2-reindex');modes+=['csa2-reuse']*(decoder_group-1)
    return modes


def modes_from_config(m:dict)->list[str]:
    """The same assignment as the application derives it: from the pinned layer-id arrays.

    A second implementation of `core.layerKind`'s rule, in a different language. The exported
    per-layer kinds are compared against this in the test suite, so a drift in either is visible.
    """
    out=[]
    for l in range(m['layers']):
        if l<m['swaLayers']:out.append('swa')
        elif l in m['kvSourceLayers']:out.append('csa2-full')
        elif l in m['indexSourceLayers']:out.append('csa2-reindex')
        else:out.append('csa2-reuse')
    return out


def _one(out,code,subject,ok,passed,failed,remedy):
    """Exactly one finding per check per subject, PASS or FAIL and nothing else.

    A check that emits nothing when it passes is indistinguishable in the report from a check
    that never ran, which this repo has already been caught reporting as assurance.

    There is deliberately no `status` parameter. It used to default to FAIL, which left a
    keyword-sized route back to the hazard ruling T789b removed: any caller passing
    `status='BLOCKED'` would have reconstructed a schematics-domain BLOCKED and put exit 2 back on
    a gate the Pages deploy requires. Writing that now requires writing a new constructor.
    """
    out.append(_ok(code,subject,passed) if ok else _bad(code,subject,failed,remedy))
    return out


def judge_architecture(models:dict,samples:dict|None=None)->list[Finding]:
    """Structural invariants for every CED preset in the live model inventory.

    `samples` is the second half of `inventory()`; only CED_PARTITION_SUM reads it, and reports
    BLOCKED rather than silently skipping when a caller forgets to pass it.
    """
    out=list(judge_architecture_tag(models))
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        out+=_judge_preset(mid,m,samples)
    return out


def judge_architecture_tag(models:dict)->list[Finding]:
    """The two predicates for one population must not be able to disagree.

    `is_ced` selects on `encoderLayers`, as the rest of the tree does, and `ARCH` names the tag
    those models are expected to carry. Nothing previously asserted the implication in this
    direction -- only its converse, that non-CED models are skipped -- so a checkpoint with
    encoder layers and a different architecture string would simply have been judged by a module
    that no longer recognised it, or (before this) not judged at all.

    Reported rather than assumed: a model here is a contradiction between two files we own, which
    is a FAIL, not a BLOCKED.
    """
    out=[]
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        tag=m.get('architecture')
        if tag!=ARCH:
            out.append(Finding('CED_ARCH_TAG','model','FAIL',mid,
                               f'{mid} declares {m.get("encoderLayers")} encoder layers but carries '
                               f'architecture {tag!r}, not {ARCH!r}.',
                               'The CED judges select on encoderLayers and the CED renderers, focal '
                               'page and scene expectations follow the same predicate. Either give '
                               'the checkpoint the CED architecture tag or teach this module the '
                               'new one deliberately; do not let the two predicates disagree.'))
    covered=[mid for mid,m in sorted(models.items()) if is_ced(m)]
    # Only report a PASS when there is something to have passed. "0 checkpoints all carry the
    # tag" is the vacuous-coverage line this module exists to refuse; a caller handed no CED
    # model gets no finding, and the absence of the checkpoint itself is `judge_coverage`'s and
    # `judge_contract_model_coverage`'s business, not this judge's.
    if not out and covered:
        out.append(Finding('CED_ARCH_TAG','model','PASS','models',
                           f'{len(covered)} checkpoint(s) with encoder layers all carry '
                           f'architecture {ARCH!r}: {", ".join(covered)}.'))
    return out


def _judge_preset(mid,m,samples):
    out=[];enc,depth=m['encoderLayers'],m['layers']
    kv,idx=list(m['kvSourceLayers']),list(m['indexSourceLayers'])
    from_config=modes_from_config(m)
    from_report=modes_from_report_structure(swa=m['swaLayers'],encoder=enc,total=depth,
                                            encoder_group=ENCODER_GROUP,decoder_group=DECODER_GROUP)
    differ=[l for l in range(max(len(from_config),len(from_report)))
            if from_config[l:l+1]!=from_report[l:l+1]]
    _one(out,'CED_MODE_DISAGREEMENT',mid,not differ,
         f'{len(from_config)} layer modes derived twice -- from the pinned layer-id arrays and '
         f'from the report\'s {ENCODER_GROUP}/{DECODER_GROUP} group structure -- agree exactly.',
         f'The two derivations disagree at layers {differ[:12]}: config says '
         f'{[from_config[l:l+1] for l in differ[:6]]}, the report structure says '
         f'{[from_report[l:l+1] for l in differ[:6]]}.',
         'Re-derive the layer-id arrays from the publisher config; do not adjust the report '
         'structure to match a transcription.')

    missing=sorted(set(kv)-set(idx))
    _one(out,'CSA2_INDEX_SUBSET',mid,not missing,
         f'All {len(kv)} KV-source layers also own index keys.',
         f'Full-mode layers {missing} write a compressed cache but are absent from '
         f'index_source_layer_ids, so nothing publishes a selection over the bytes they wrote.',
         'Every Full layer derives its own index keys; kv_source_layer_ids must be a subset of '
         'index_source_layer_ids.')

    unreachable=[]
    for l,mode in enumerate(from_config):
        if mode not in ('csa2-reuse','csa2-reindex'):continue
        lo,hi=(0,enc) if l<enc else (enc,depth)
        if not any(lo<=k<l for k in kv):unreachable.append(l)
    _one(out,'CED_REUSE_UNREACHABLE',mid,not unreachable,
         f'Every reusing layer has an earlier same-band Full layer to read; {len(kv)} cache '
         f'writers serve {sum(1 for x in from_config if x!="swa")-len(kv)} readers.',
         f'Layers {unreachable[:12]} reuse or re-index a compressed cache that no earlier layer '
         f'in their own half ({enc} encoder / {depth-enc} decoder) writes.',
         'Layer-dimension cache sharing is only well-founded if the reader follows a writer in '
         'the same half. Fix the layer-id arrays, not the invariant.')

    ratios=(m['compressEncoder'],m['compressDecoder'])
    _one(out,'CED_BAND_RATIO',mid,all(r>=1 for r in ratios) and ratios[0]!=ratios[1],
         f'Encoder compresses {ratios[0]}:1 and decoder {ratios[1]}:1 -- two distinct bands.',
         f'Band compression ratios are {ratios}: a ratio below 1 stores more than it reads, and '
         f'two identical ratios leave no ratio change for the encoder boundary to be derived '
         f'from.',
         'Read both ratios from compress_ratios. Distinctness is load-bearing rather than '
         'aesthetic: _encoder_boundary derives encoderLayers as the index where the ratio '
         'changes, and with one ratio throughout that derivation is undefined and encoderLayers '
         'stops being verifiable against publisher bytes at all. A checkpoint that genuinely '
         'shared one ratio would need its boundary pinned some other way first.')

    decoder_sources=[l for l in kv if l>=enc]
    _one(out,'CED_KV_PROVENANCE',mid,decoder_sources==[enc],
         f'The decoder half projects its compressed KV at layer {enc} and nowhere else.',
         f'The decoder half owns KV sources {decoder_sources}, not exactly [{enc}].',
         'The whole decoder reads one cache written by the first decoder layer. A second decoder '
         'source makes this an ordinary shared-KV Transformer.')

    early=[l for l in idx if l>=enc and l<m['candidateSource']]
    _one(out,'CSA2_CANDIDATE_SOURCE',mid,m['candidateSource']==enc and not early,
         f'The candidate pool is established at layer {enc}, before every decoder index layer.',
         f'candidate_source_layer_id is {m["candidateSource"]} against an encoder depth of {enc}'
         +(f', and decoder index layers {early} precede it.' if early else '.'),
         'The pool must exist before any layer is masked to it; a consumer above its source reads '
         'a pool that does not yet exist.')

    derived=sum(m['engramTables'])*m['engramHeadDim']
    _one(out,'ENGRAM_TOTAL_DERIVATION',mid,derived==m['ngram'],
         f'{len(m["engramTables"])} tables x {m["engramHeadDim"]} = {derived:,} n-gram parameters, '
         f'which is the declared total exactly.',
         f'Declared n-gram residency {m["ngram"]:,} against {derived:,} derived from the table '
         f'sizes and the head dimension.',
         'The exact figure follows from engram_num_embeddings and engram_head_dim; do not carry '
         'the card\'s rounded total.')

    columns=(m['engramNgram']-1)*m['engramHeads']
    _one(out,'ENGRAM_HASH_COLUMNS',mid,columns==ENGRAM_HASH_COLUMNS,
         f'({m["engramNgram"]} - 1) n-gram sizes x {m["engramHeads"]} heads = {columns} hash '
         f'columns per position.',
         f'({m["engramNgram"]} - 1) x {m["engramHeads"]} = {columns} hash columns against the '
         f'{ENGRAM_HASH_COLUMNS} the layout allocates.',
         'engram.py builds one bucket range per (n-gram size, head) pair over max_ngram_size - 1 '
         'sizes; the two config fields must multiply out to that.')

    engram_ok=len(m['engramTables'])==len(m['engramLayers']) and all(0<=l<depth for l in m['engramLayers'])
    _one(out,'ENGRAM_LAYER_BOUNDS',mid,engram_ok,
         f'{len(m["engramLayers"])} injection sites at layers {m["engramLayers"]}, one table each, '
         f'all inside {depth} layers.',
         f'{len(m["engramTables"])} tables against {len(m["engramLayers"])} sites '
         f'{m["engramLayers"]} in a {depth}-layer backbone.',
         'One table per injection site, every site inside the backbone.')

    targets=list(m['dsparkTargets'])
    _one(out,'DSPARK_TARGET_COUNT',mid,len(targets)==m['mtp'] and all(0<=l<depth for l in targets),
         f'{m["mtp"]} draft layers read {len(targets)} distinct backbone layers {targets}.',
         f'{len(targets)} draft targets {targets} against mtp={m["mtp"]} in a {depth}-layer '
         f'backbone.',
         'One target hidden state per draft layer, each an existing backbone layer.')

    _one(out,'DSPARK_BLOCK_TRIVIAL',mid,m['dsparkBlock']>1,
         f'The draft head proposes {m["dsparkBlock"]} tokens per block.',
         f'dspark_block_size is {m["dsparkBlock"]}: a block of one is ordinary autoregression '
         f'wearing a speculative name.',
         'Read dspark_block_size from the config; a trivial block is not a draft mechanism.')

    _one(out,'DSPARK_MOE_DISTINCT',mid,(m['dsparkExperts'],m['dsparkTopK'])!=(m['experts'],m['topK']),
         f'The draft MoE is {m["dsparkExperts"]}/{m["dsparkTopK"]} against the backbone\'s '
         f'{m["experts"]}/{m["topK"]} -- a smaller router, not a copy.',
         f'The draft MoE and the backbone MoE are both {m["experts"]}/{m["topK"]}, so the draft '
         f'head is as expensive as the model it drafts for.',
         'The draft layers declare their own expert count and top-k; do not inherit the '
         'backbone\'s.')

    draft=list(range(depth,depth+m['mtp']))
    owning=[l for l in draft if l in kv or l in idx]
    _one(out,'CED_MTP_RATIO',mid,not owning,
         f'Draft layers {draft} own no compressed cache and no index keys.',
         f'Draft layers {owning} appear in kv_source_layer_ids or index_source_layer_ids.',
         'DSparkAttention.forward opens with `assert self.compress_ratio == 0`, and '
         'Attention.__init__ gates both roles on `layer_id < n_layers`. A draft layer owning a '
         'compressed cache contradicts the publisher\'s own assertion.')

    partition=(samples or {}).get('partitions',{}).get(mid)
    if partition is None:
        out.append(_bad('CED_PARTITION_SUM',mid,
                        'No exported layer partition to compare: judge_architecture was called '
                        'without the inventory samples, so this check could not run.',
                        'Pass the `samples` half of `inventory(root)`. Narrowing ruling P1: this '
                        'reports FAIL, not BLOCKED, because the inventory comes from our own '
                        'exporter in the same process -- its absence is a defect in the gate '
                        'wiring, and a software gate must never return BLOCKED. P1 required that '
                        'this not be skipped silently, which FAIL satisfies more loudly.'))
    else:
        total=sum(partition.values())
        _one(out,'CED_PARTITION_SUM',mid,total==depth,
             f'core.layerPartition tiles all {depth} layers: '
             f'{ {k:v for k,v in partition.items() if v} }.',
             f'The exported partition sums to {total} against {depth} declared layers: '
             f'{ {k:v for k,v in partition.items() if v} }.',
             'Every layer has exactly one kind. A partition that does not tile the depth means '
             'layerKind returns something layerPartition does not count.')

    _one(out,'CED_PHASE_DEPTH',mid,0<enc<depth,
         f'The phase split runs {enc} layers for the prompt and all {depth} after it.',
         f'encoderLayers is {enc} in a {depth}-layer model: a split covering every layer or none '
         f'is not a split.',
         'The encoder depth is strictly between zero and the total; otherwise drop the phase '
         'distinction rather than draw one.')

    return out


def judge_active_ratio(models:dict,domain:str='evidence')->list[Finding]:
    """The published prefill/decode actives should track the encoder/decoder depth split.

    BLOCKED, never FAIL, on a mismatch: the published actives are rounded to whole billions and
    the embedding and Engram terms are depth-independent, so an exact identity would be
    overfitting and a mismatch is weak evidence either way. That is also why it cannot live in
    `schematics` -- a BLOCKED there is exit 2 on a gate the Pages deploy requires (ruling T789b).
    It is evidence work regardless: the inputs are card figures, not scene IR.
    """
    out=[]
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        enc,depth,active,decode=m['encoderLayers'],m['layers'],m['active'],m['activeDecode']
        if not decode or not depth:
            out.append(_evidence_blocked('CED_ACTIVE_RATIO',mid,
                                         'No decode-phase active parameter count to compare '
                                         'against the depth ratio.',
                                         'Record activeDecode, or drop the phase split.'))
            continue
        ratio_a,ratio_d=active/decode,enc/depth
        ok=abs(ratio_a-ratio_d)<=ACTIVE_RATIO_SLACK
        out.append(_ok('CED_ACTIVE_RATIO',mid,
                       f'Active-parameter ratio {ratio_a:.3f} tracks the depth ratio '
                       f'{ratio_d:.3f} within {ACTIVE_RATIO_SLACK}.') if ok else
                   _evidence_blocked('CED_ACTIVE_RATIO',mid,
                                     f'Active-parameter ratio {ratio_a:.3f} against a depth ratio '
                                     f'of {ratio_d:.3f}, outside {ACTIVE_RATIO_SLACK}. Reported as '
                                     f'BLOCKED, not FAIL: the actives are published rounded and '
                                     f'carry depth-independent terms, so this is weak evidence '
                                     f'either way.',
                                     'Re-read the published prefill and decode actives. Do not '
                                     'adjust either to make the ratio land.'))
    return [Finding(f.code,domain,f.status,f.subject,f.message,f.remedy) for f in out]


MODEL_DEPENDENCY=re.compile(r'^MODELS\.([^.]+)\.')


def judge_claim_model_coverage(claims,models,domain:str='evidence')->list[Finding]:
    """A claim reconciling more than one model must reconcile every model.

    A checklist replaced by a check. Today 25 claims declare they reconcile every supported
    model; a manifest that names eight of nine asserts completeness it does not have, and the
    next person to add a model cannot forget them without this failing.

    Evidence-domain, and wired into `runner.audit` rather than `schematics.run`: it judges the
    claim manifest, not the scene IR, and its zero-comparison path is BLOCKED, which a gate the
    Pages deploy requires must never return (rulings P1 and T789b together).
    """
    supported=set(models);out=[];compared=0
    for claim in claims:
        named={MODEL_DEPENDENCY.match(d).group(1) for d in claim.get('dependencies') or []
               if MODEL_DEPENDENCY.match(d)}
        if len(named)<2:continue
        compared+=1
        missing,unknown=sorted(supported-named),sorted(named-supported)
        if missing or unknown:
            out.append(_bad('CLAIM_MODEL_COVERAGE',claim['id'],
                            (f'Names {len(named)} of {len(supported)} supported models'
                             +(f', missing {missing}' if missing else '')
                             +(f', and names {unknown} which no model table supplies' if unknown else '')+'.'),
                            'A multi-model claim reconciles every supported model. Add the '
                            'missing MODELS.<id>.d dependencies, or narrow the claim\'s statement '
                            'to the models it actually covers.'))
    def stamped(items):
        return [Finding(f.code,domain,f.status,f.subject,f.message,f.remedy) for f in items]
    if out:return stamped(out)
    if not compared:
        return stamped([_evidence_blocked('CLAIM_MODEL_COVERAGE','claims',
                         'No claim names two or more models, so nothing was compared.',
                         'Zero comparisons is not agreement. Check that the manifest still '
                         'carries the all-models claims before reading this as coverage.')])
    return stamped([_ok('CLAIM_MODEL_COVERAGE','claims',
                        f'{compared} multi-model claims each name all {len(supported)} supported '
                        f'models.')])


def _contract_models(rec:dict)->set[str]:
    """Which models in the live table a contract record speaks for, read off its own references.

    Never `f'model.{mid}'`. `model.kimi-k3` speaks for `MODELS.kimi3`, so a check keyed on a
    naming convention would report the one model whose contract id differs from its table key as
    uncovered and say nothing true about the other eight. Bindings and claims both name the
    application's own inputs, which is the only place the association is actually written down.
    """
    refs=list((rec.get('bindings') or {}).values())+list(rec.get('claims') or [])
    return {m.group(1) for m in (MODEL_DEPENDENCY.match(r) for r in refs) if m}


def judge_contract_model_coverage(models,contracts,obligations,domain:str='evidence')->list[Finding]:
    """Every model in the live inventory needs a `model.*` contract and a model-domain obligation.

    The tenth instance of this branch's recurring defect, and the one in the most load-bearing
    artifact on it: `observatory.json` held one `model.*` contract per model for eight models and
    stayed at eight while 482 claims for a ninth were authored. Nothing could report it.
    `RUNTIME_MATRIX` builds its required set from `{gpu, scale_up, scale_out, communication}` and
    excludes `model`; `RUNTIME_COVERAGE` asks only that a *domain* have some obligation, and
    `model` had eight; `DOMAIN_UNCOVERED` fires only on a domain with zero contracts. So the
    public record reported a 31/31 inventory over nine checkpoints, and the ninth -- the only one
    with four attention modes, two separately-quantized KV caches and a 4-stream residual -- had
    no contract to be internally consistent with and no obligation to be NOT_RUN against. Its
    absence was indistinguishable from having nothing to prove.

    FAIL, not BLOCKED. BLOCKED means we cannot know; a model our own exporter emits with no
    record beside it is a wiring defect we can know about (ruling P1). Evidence-domain and wired
    into `runner.audit` rather than a Pages-deploying gate, like its two neighbours, because its
    zero-comparison path is BLOCKED (rulings P1 and T789b together).
    """
    covered,obliged={},set()
    for rec in contracts:
        if rec.get('domain')!='model':continue
        for mid in _contract_models(rec):covered.setdefault(mid,set()).add(rec.get('id'))
    for o in obligations:
        if o.get('domain')=='model':obliged.add(o.get('contract'))
    out=[]
    for mid in sorted(models):
        ids=covered.get(mid,set())
        if not ids:
            out.append(_bad('CONTRACT_MODEL_COVERAGE',mid,
                            f'No model-domain contract binds or claims any MODELS.{mid} input, so '
                            f'this checkpoint has no contract record to be consistent with.',
                            'Author a model.<id> contract for it. The domain validator is '
                            'generic; do not fabricate a field that has no honest value.'))
            continue
        if not (ids & obliged):
            out.append(_bad('CONTRACT_MODEL_COVERAGE',mid,
                            f'{sorted(ids)} covers {mid} but no model-domain runtime obligation '
                            f'names it, so its absence of runtime evidence is never NOT_RUN.',
                            'Register a runtime obligation against the contract; a contract with '
                            'no obligation reports nothing when the experiment has not been run.'))
            continue
        out.append(_ok('CONTRACT_MODEL_COVERAGE',mid,
                       f'Covered by {sorted(ids)} with a model-domain runtime obligation.'))
    def stamped(items):
        return [Finding(f.code,domain,f.status,f.subject,f.message,f.remedy) for f in items]
    if not models:
        return stamped([_evidence_blocked('CONTRACT_MODEL_COVERAGE','models',
                         'The live model inventory is empty, so no model was checked for a '
                         'contract.',
                         'Zero comparisons is not coverage. Check that '
                         'tools/export-verification.mjs still emits the model table.')])
    return stamped(out)


#: FP4 E2M1 plus one E4M3 scale per 16 channels, §2.4. Trained in (QAT), so it is an
#: architectural fact rather than a display setting, which is why `s.kvBytes` does not apply.
FP4_BYTES_PER_CHANNEL=0.5+1/16
#: Any real KV precision lies in this range: FP4 without scales at the floor, FP8 at the ceiling.
#: Deliberately wide. The report states no indexer precision, so a tighter bound would be a guess
#: wearing a check's clothing.
FP4_FLOOR,FP8_CEILING=0.5,1.0


def derived_main_kv_bytes_per_token(m:dict)->float:
    """Independent re-derivation of the ledger's arithmetic.

    Only Full-mode layers store main KV, each at its own band's compression ratio. The ledger
    computes this in JavaScript; neither side is checked against itself.
    """
    total=0.0
    for l in m['kvSourceLayers']:
        ratio=m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder']
        total+=(m['headDim']/ratio)*FP4_BYTES_PER_CHANNEL
    return total


def indexer_channels_per_token(m:dict)->float:
    """Indexer K is stored by the same Full-mode layers, at index_head_dim.

    Channels, not bytes: the precision is exactly what the report does not state, and it is what
    KV_RESIDUAL_IMPLAUSIBLE bounds rather than assumes.
    """
    return sum(m['indexer']/(m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder'])
               for l in m['kvSourceLayers'])


def judge_kv_budget(models:dict,samples:dict)->list[Finding]:
    """The KV budget, cross-checked against a second implementation and bounded by physics.

    KV_BOUNDED_INVARIANT runs for every model, not only this architecture: `bounded` is a shared
    field, and a reclassification nothing measures is cosmetic.
    """
    out=[];near=(samples or {}).get('logical_state') or {};far=(samples or {}).get('logical_state_far') or {}
    for mid,m in sorted(models.items()):
        a,b=near.get(mid),far.get(mid)
        if a is None or b is None:
            out.append(_bad('KV_BOUNDED_INVARIANT',mid,
                            'No two-context ledger sample to compare bounded state against, so '
                            'this check could not run.',
                            'Export samples.logical_state and samples.logical_state_far from '
                            'tools/export-verification.mjs. FAIL rather than BLOCKED: the samples '
                            'are ours, so their absence is a wiring defect.'))
        elif a.get('S')==b.get('S'):
            out.append(_bad('KV_BOUNDED_INVARIANT',mid,
                            f'Both ledger samples are at S={a.get("S")}, so nothing was varied and '
                            f'this check could not run.',
                            'Sample the ledger at two different contexts; a state compared '
                            'against itself is not evidence that it is bounded.'))
        else:
            _one(out,'KV_BOUNDED_INVARIANT',mid,a['bounded']==b['bounded'],
                 f'Bounded state is {a["bounded"]:,} B at S={a["S"]:,} and at S={b["S"]:,} -- '
                 f'a {b["S"]//a["S"]}x context with no change.',
                 f'Bounded state moves from {a["bounded"]:,} B at S={a["S"]:,} to '
                 f'{b["bounded"]:,} B at S={b["S"]:,}.',
                 'Recurrent, convolution and sliding-window state are bounded by construction. '
                 'A term that grows with context belongs in contextLinear, not in bounded.')
        if not is_ced(m):continue
        try:
            derived=derived_main_kv_bytes_per_token(m)
            channels=indexer_channels_per_token(m)
        except (ZeroDivisionError,KeyError,TypeError) as exc:
            _one(out,'KV_BUDGET_DISAGREEMENT',mid,False,'',
                 f'The declared shape makes the main-KV derivation undefined: {exc!r}.',
                 'A compression ratio must be a positive number; CED_BAND_RATIO reports the same '
                 'shape. A judge must return a finding rather than raise.')
            _one(out,'KV_RESIDUAL_IMPLAUSIBLE',mid,False,'',
                 f'The declared shape makes the indexer-channel derivation undefined: {exc!r}.',
                 'A compression ratio must be a positive number.')
            continue
        if a is None:
            out.append(_bad('KV_BUDGET_DISAGREEMENT',mid,
                            'No exported cache ledger to compare the Python derivation against, '
                            'so this check could not run.',
                            'Export samples.logical_state from tools/export-verification.mjs.'))
        else:
            ledger=a['mainKV']/(a['B']*a['S'])
            _one(out,'KV_BUDGET_DISAGREEMENT',mid,abs(ledger-derived)<=1e-9,
                 f'cacheLedger and the Python re-derivation both put main KV at '
                 f'{derived:g} B/token over {len(m["kvSourceLayers"])} Full-mode layers.',
                 f'cacheLedger reports {ledger:g} B/token against {derived:g} B/token re-derived '
                 f'from the pinned ratios and head dimension.',
                 'One of the two implementations of the band-ratio rule drifted. Fix the '
                 'implementation, never the expectation.')
        residual=m['kvBytesPerToken']-derived
        per_channel=residual/channels if channels else float('inf')
        _one(out,'KV_RESIDUAL_IMPLAUSIBLE',mid,
             residual>=0 and FP4_FLOOR<=per_channel<=FP8_CEILING,
             f'Reported {m["kvBytesPerToken"]} B/token less {derived:g} B/token of main KV leaves '
             f'{residual:g} B over {channels:g} indexer channels = {per_channel:.3f} B/channel, '
             f'inside [{FP4_FLOOR}, {FP8_CEILING}].',
             f'Reported {m["kvBytesPerToken"]} B/token less {derived:g} B/token of main KV leaves '
             f'{residual:g} B over {channels:g} indexer channels = {per_channel:.3f} B/channel, '
             f'outside [{FP4_FLOOR}, {FP8_CEILING}].',
             'The report states no indexer precision, so the unexplained residual is held only to '
             'lying between FP4 and FP8. Re-read the published total; do not pick a precision that '
             'makes it come out exactly, and do not narrow the bound to fit.')
    return out


def judge_mhc(models:dict,samples:dict|None=None)->list[Finding]:
    """An mHC preset's stream count must produce the coefficient tensor the publisher allocates.

    Takes `samples` it does not read, so every member of SCHEMATICS_JUDGES has one signature and
    the gate and the property test can iterate the same list.
    """
    out=[]
    for mid,m in sorted(models.items()):
        if m.get('residual')!='mhc':continue
        streams=m.get('streams') or 0
        _one(out,'MHC_COEFF_COUNT',mid,streams>1 and (2+streams)*streams==MHC_COEFFICIENTS,
             f'{streams} streams give (2 + {streams}) x {streams} = {MHC_COEFFICIENTS} '
             f'hyper-connection coefficients per mix.',
             f'{streams} streams give (2 + {streams}) x {streams} = {(2+streams)*streams} '
             f'coefficients against the {MHC_COEFFICIENTS} allocated, and a single stream is an '
             f'ordinary residual wearing a widened name.',
             "Block.__init__ sizes every coefficient tensor as `mix_hc = (2 + hc_mult) * hc_mult` "
             '(model.py:938); the declared stream count must reproduce it.')
    return out


#: `Compressor.__init__` (model.py:437-457) allocates `wgate` only where the band's compression
#: ratio exceeds 1. It is the third implementation of the encoder/decoder ratio rule -- after
#: `ratioOf` in src/diagram-meta.mjs and the loop in `cacheLedger` -- and until now the only one
#: nothing compared, because no parameter *value* reached the inventory: `function.parameterLedger`
#: arrives as a source hash, and a hash says a function changed, never whether it is right.
def gated_compressor_layers(m:dict)->list[int]:
    """The Full-mode layers whose band ratio is greater than 1, and so carry a compressor gate."""
    return [l for l in m['kvSourceLayers']
            if (m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder'])>1]


def derived_compressor_gate_parameters(m:dict)->int:
    """The gate weights the ratio rule alone is responsible for.

    Re-derived here, not read back: the sampled ledger is asked the same question twice, once at
    the declared ratios and once with every ratio at 1. With unit ratios no band is gated, so the
    difference between the two samples is exactly this quantity and nothing else -- the encoder
    boundary, the ratio comparison and the gate's shape, isolated from the rest of the ledger.
    """
    return len(gated_compressor_layers(m))*m['d']*m['headDim']


def judge_parameter_bands(models:dict,samples:dict|None=None)->list[Finding]:
    """The parameter ledger's copy of the band-ratio rule, cross-checked in Python.

    A differential rather than a re-implementation of the whole ledger: re-deriving 551.9B of
    backbone in a second language would compare two transcriptions of the same arithmetic and
    fail on any unrelated term. This compares only the part the ratio rule owns.
    """
    out=[];real=(samples or {}).get('parameters') or {};unit=(samples or {}).get('parameters_unit_ratio') or {}
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        a,b=real.get(mid),unit.get(mid)
        if not a or not b:
            out.append(_bad('PARAM_BAND_RATIO',mid,
                            'No two-ratio parameter sample to compare the compressor gates '
                            'against, so this check could not run.',
                            'Export samples.parameters and samples.parameters_unit_ratio from '
                            'tools/export-verification.mjs. FAIL rather than BLOCKED: the samples '
                            'are ours, so their absence is a wiring defect.'))
            continue
        try:
            expected=derived_compressor_gate_parameters(m)
        except (KeyError,TypeError) as exc:
            _one(out,'PARAM_BAND_RATIO',mid,False,'',
                 f'The declared shape makes the compressor-gate derivation undefined: {exc!r}.',
                 'A band ratio must be a number and the Full-mode layers must be declared; '
                 'CED_BAND_RATIO reports the same shape. A judge must return a finding, not raise.')
            continue
        measured=a.get('main',0)-b.get('main',0)
        gated=len(gated_compressor_layers(m))
        _one(out,'PARAM_BAND_RATIO',mid,measured==expected and expected>0,
             f'parameterLedger allocates {measured:,} parameters of compressor gate, matching '
             f'{gated} of {len(m["kvSourceLayers"])} Full-mode layers above ratio 1 at '
             f'{m["d"]} x {m["headDim"]} re-derived independently.',
             f'parameterLedger allocates {measured:,} parameters of compressor gate against '
             f'{expected:,} re-derived from {gated} gated band(s) at {m["d"]} x {m["headDim"]}.',
             'Compressor allocates wgate only where compress_ratio > 1 (model.py:437-457). The '
             'ledger, ratioOf and cacheLedger are three copies of one rule; fix the copy that '
             'drifted, never the expectation. A zero difference means the rule is not there at all.')
    return out


#: Figures are read out of the rendered sentence rather than from a structured field beside it,
#: because the sentence is what a reader is asked to believe. Numbers, not wording: a check keyed
#: to the phrasing would fail when the prose improved and pass when the figure was dropped.
NUMBER=re.compile(r'\d[\d,]*(?:\.\d+)?')


def rendered_numbers(text:str)->set[float]:
    return {float(t.replace(',','')) for t in NUMBER.findall(text or '')}


#: Rows whose figures are raw configuration fields, and the fields each must render. These are
#: **not** compared against a second copy of one of our derivations: `FIELD_MAP` already pins every
#: one of them to a JSON pointer inside the publisher's own config document, and
#: `CED_PUBLISHER_AGREEMENT` re-derives the model table from those bytes. So requiring the sentence
#: to render the field ties the prose to publisher bytes through a chain that does not pass through
#: `core.mjs` at all. `judge_coverage` checks the FIELD_MAP membership rather than trusting this
#: comment, so a field that stops being pinned stops being evidence.
CONFIG_PINNED_FIGURES={
    'ced':(('encoder depth','encoderLayers'),('total depth','layers')),
    'hier-indexer':(('candidate blocks','candidateBlocks'),
                    ('positions per block','candidateBlockSize'),
                    ('selection budget','indexerBudget')),
    'mhc':(('residual streams','streams'),('Sinkhorn iterations','sinkhornIters')),
    'dspark':(('draft block size','dsparkBlock'),
              ('draft experts','dsparkExperts'),
              ('draft experts per token','dsparkTopK')),
}

#: The three rows that genuinely cannot be checked this way, and why. Recorded rather than left as
#: an unexplained gap: each figure is an output of one of our own ledgers, so a second Python
#: implementation would compare two transcriptions of the same rule instead of two derivations.
#: Each is named with the check that does hold the underlying quantity.
UNPINNED_COVERAGE_ROWS={
    'csa2':'layerPartition is our own rule; CED_MODE_DISAGREEMENT and CED_PARTITION_SUM check the '
           'mode assignment against the report structure and the exported partition.',
    'engram':'the table total is a parameterLedger output; ENGRAM_TOTAL_DERIVATION checks it '
             'against engram_num_embeddings and the hash columns.',
    'swa-replay':'bounded state is a cacheLedger output; KV_BOUNDED_INVARIANT checks that it does '
                 'not move between two contexts, which is the property the row asserts.',
}


def _coverage_expectations(m:dict)->tuple[dict,list[str]]:
    """(row id -> [(label, value)], unmapped field names).

    The FP4 row is the one entry that is not a pinned field: its derived half comes from
    `derived_main_kv_bytes_per_token`, the arithmetic `judge_kv_budget` holds `cacheLedger` to,
    and its reported half is the published total, which no config field supplies.
    """
    wanted={'fp4-kv':[('derived main KV bytes per token',derived_main_kv_bytes_per_token(m)),
                      ('the publisher-reported bytes per token',float(m['kvBytesPerToken']))]}
    unmapped=[]
    for rid,fields in CONFIG_PINNED_FIGURES.items():
        rows=[]
        for label,field in fields:
            if field not in FIELD_MAP:unmapped.append(f'{rid}:{field}')
            rows.append((f'{label} ({field})',float(m[field])))
        wanted[rid]=rows
    return wanted,unmapped


def judge_coverage(models:dict,samples:dict|None=None)->list[Finding]:
    """The written exposition, held to the same evidence chain as a number in a diagram.

    Two failure modes, both of which the prose form invites. A sentence can assert an
    architectural fact with nothing behind it -- COVERAGE_UNCITED. And a sentence can quote the
    publisher's headline figure while the ledger computes another -- COVERAGE_DERIVED, which
    re-uses `derived_main_kv_bytes_per_token`, the same independent arithmetic `judge_kv_budget`
    holds the cache ledger to, plus the `FIELD_MAP`-pinned configuration fields, whose provenance
    runs to publisher bytes without passing through `core.mjs`. So the exposition cannot become a
    second, unverified copy of the architecture: it is measured against the computation, not
    against itself. The rows that cannot be reached this way are named in
    UNPINNED_COVERAGE_ROWS and counted in the PASS message, so the scope is stated rather than
    implied.
    """
    out=[];rows_by_model=(samples or {}).get('innovations') or {}
    known=set((samples or {}).get('source_ids') or [])
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        rows=rows_by_model.get(mid)
        if not rows:
            for code in ('COVERAGE_UNCITED','COVERAGE_DERIVED'):
                out.append(_bad(code,mid,
                                'No exported innovation rows, so the written exposition was not '
                                'checked at all.',
                                'Export samples.innovations from tools/export-verification.mjs. '
                                'FAIL rather than BLOCKED: the rows are ours, so their absence is '
                                'a wiring defect, and an unchecked exposition must not read as a '
                                'checked one.'))
            continue
        uncited=[]
        for r in rows:
            declared=list(r.get('sources') or []);resolved=list(r.get('citations') or [])
            if not declared:
                uncited.append(f'{r.get("id")} cites nothing')
            elif len(resolved)!=len(declared):
                missing=[s for s in declared if s not in known]
                uncited.append(f'{r.get("id")} renders {len(resolved)} of {len(declared)} '
                               f'citations'+(f' ({missing} resolve to no source)' if missing else ''))
        _one(out,'COVERAGE_UNCITED',mid,not uncited,
             f'All {len(rows)} innovation records cite at least one source, and every declared '
             f'id resolves to a clickable reference.',
             'The exposition states architecture with no reachable provenance: '+'; '.join(uncited)+'.',
             'Every record carries source ids that must resolve in SOURCES. An id that does not '
             'resolve renders as nothing, which reads as an uncited assertion. Cite the bytes or '
             'delete the sentence.')
        try:
            expectations,unmapped=_coverage_expectations(m)
        except (KeyError,TypeError,ZeroDivisionError) as exc:
            _one(out,'COVERAGE_DERIVED',mid,False,'',
                 f'The declared shape makes the coverage derivation undefined: {exc!r}.',
                 'The model must declare the fields its exposition is about. A judge must return '
                 'a finding rather than raise.')
            continue
        by_id={r.get('id'):r for r in rows};absent=list(
            f'{name} is claimed to be pinned to publisher bytes but FIELD_MAP does not map it'
            for name in unmapped)
        for rid,wanted in sorted(expectations.items()):
            row=by_id.get(rid)
            if row is None:
                absent.append(f'the {rid} row is missing entirely')
                continue
            shown=rendered_numbers(row.get('effect'))
            for label,value in wanted:
                if not any(abs(value-x)<=1e-9 for x in shown):
                    absent.append(f'the {rid} row renders {sorted(shown)} and not {label} {value:g}')
        unchecked=sorted(set(r.get('id') for r in rows)-set(expectations))
        _one(out,'COVERAGE_DERIVED',mid,not absent,
             f'{len(expectations)} of {len(rows)} rows are checked against a figure this gate does '
             f'not take from the rendered page: '
             +', '.join(f'{rid} shows {[f"{v:g}" for _,v in w]}'
                        for rid,w in sorted(expectations.items()))
             +f'. The remaining {len(unchecked)} ({", ".join(unchecked)}) render outputs of our own '
             f'ledgers, each held by the check named in UNPINNED_COVERAGE_ROWS.',
             'A rendered row does not show the figure re-derived for it: '+'; '.join(absent)+'.',
             'The effect must be computed from the model, never quoted. Figures named in '
             'CONFIG_PINNED_FIGURES are pinned to the publisher config by FIELD_MAP, so the row '
             'must render them; the FP4 row must show the derived bytes per token beside the '
             'reported one, because showing only the published total hides a residual the '
             'publisher does not explain.')
    return out


def _text(c):
    return c['text_config']


def _ratios(c):
    """The per-layer compression ratios, backbone layers only.

    compress_ratios is 43 long against 40 layers: the three trailing zeros are the DSpark draft
    layers, which own no compressed cache.
    """
    return _text(c)['compress_ratios'][:_text(c)['num_hidden_layers']]


def _encoder_boundary(c):
    """The layer at which the nonzero compression ratio changes.

    This is *our* arithmetic, not a published field: config.json marks a boundary here
    (compress_ratios 2 -> 1, kv_source_layer_ids, candidate_source_layer_id) but nothing in it
    states the boundary is an encoder/decoder split. That is why MODELS.dsv41.encoderLayers
    carries `anchor:null` and must keep it -- a pointer would clear CLAIM_ANCHOR on an inference.
    A verifier expression is a different thing from an evidence anchor: this asserts the number
    follows from the bytes, it does not claim the publisher said so.
    """
    nonzero=[(i,r) for i,r in enumerate(_ratios(c)) if r]
    if not nonzero:raise ValueError('no compressed layer in compress_ratios')
    first=nonzero[0][1]
    for i,r in nonzero:
        if r!=first:return i
    # This is the dependency CED_BAND_RATIO's distinctness rule protects: with one ratio
    # throughout, the boundary is not derivable from these bytes and encoderLayers becomes
    # unverifiable. Raising makes that an unresolved field (BLOCKED), never a guessed one.
    raise ValueError('compress_ratios never changes, so no encoder boundary is derivable')


def _has_vision(c):
    """Whether the checkpoint ships a vision encoder: `vision_config` is present or it does not.

    Structural rather than keyed, so it needs an unresolved path of its own -- a rule that can
    only ever return an answer is the thing this module exists to prevent. A document with no
    `text_config` at all is a truncated or wrong file, not a text-only checkpoint, and must read
    as unresolved rather than as `vision: false`.
    """
    if not c.get('text_config'):raise KeyError('text_config')
    vision=c.get('vision_config')
    if vision is None:return False
    if not vision:raise ValueError('vision_config is present but empty')
    return True


#: Every transcribed dsv41 field and where it comes from in config.json. A field absent from this
#: map is a field nothing verifies, so adding one to MODELS must be a conscious decision here too
#: -- NOT_IN_CONFIG below is the other half of that decision, and CED_FIELD_UNMAPPED reports any
#: field in neither. Derived fields carry a callable over the config document instead of a pointer.
FIELD_MAP={
 'layers':'/text_config/num_hidden_layers',
 'd':'/text_config/hidden_size',
 'heads':'/text_config/num_attention_heads',
 'kvHeads':'/text_config/num_key_value_heads',
 'headDim':'/text_config/head_dim',
 'ropeDim':'/text_config/qk_rope_head_dim',
 'qRank':'/text_config/q_lora_rank',
 'oRank':'/text_config/o_lora_rank',
 'oGroups':'/text_config/o_groups',
 'experts':'/text_config/n_routed_experts',
 'shared':'/text_config/n_shared_experts',
 'topK':'/text_config/num_experts_per_tok',
 'expertFF':'/text_config/moe_intermediate_size',
 'vocab':'/text_config/vocab_size',
 'window':'/text_config/sliding_window',
 'scoring':'/text_config/scoring_func',
 'kvSourceLayers':'/text_config/kv_source_layer_ids',
 'indexSourceLayers':'/text_config/index_source_layer_ids',
 'indexer':'/text_config/index_head_dim',
 'indexerHeads':'/text_config/index_n_heads',
 'indexerBudget':'/text_config/index_topk',
 'candidateBlocks':'/text_config/candidate_topk_blocks',
 'candidateBlockSize':'/text_config/candidate_block_size',
 'candidateSource':'/text_config/candidate_source_layer_id',
 'streams':'/text_config/hc_mult',
 'sinkhornIters':'/text_config/hc_sinkhorn_iters',
 'engramLayers':'/text_config/engram_layer_ids',
 'engramTables':'/text_config/engram_num_embeddings',
 'engramNgram':'/text_config/engram_max_ngram_size',
 'engramHeads':'/text_config/engram_n_heads',
 'engramHeadDim':'/text_config/engram_head_dim',
 'mtp':'/text_config/num_nextn_predict_layers',
 'dsparkBlock':'/text_config/dspark_block_size',
 'dsparkExperts':'/text_config/dspark_n_routed_experts',
 'dsparkTopK':'/text_config/dspark_num_experts_per_tok',
 'dsparkTargets':'/text_config/dspark_target_layer_ids',
 'maxContext':'/text_config/max_position_embeddings',
 'configMax':'/text_config/max_position_embeddings',
 'routeScale':'/text_config/routed_scaling_factor',
 'swigluLimit':'/text_config/swiglu_limit',
 'visionLayers':'/vision_config/num_hidden_layers',
 'visionD':'/vision_config/hidden_size',
 'visionHeads':'/vision_config/num_attention_heads',
 'visionFF':'/vision_config/intermediate_size',
 'visionPatch':'/vision_config/patch_size',
 # Derived, not transcribed: checked against an expression over the same bytes.
 'nopeDim':lambda c:_text(c)['head_dim']-_text(c)['qk_rope_head_dim'],
 # One wkv per layer produces the head_dim vector serving as both K and V (model.py:643), so
 # there is no separate value width to publish.
 'vDim':lambda c:_text(c)['head_dim'],
 'encoderLayers':_encoder_boundary,
 'swaLayers':lambda c:sum(1 for r in _ratios(c) if r==0),
 'compressEncoder':lambda c:_ratios(c)[_encoder_boundary(c)-1],
 'compressDecoder':lambda c:_ratios(c)[_encoder_boundary(c)],
 'ngram':lambda c:sum(_text(c)['engram_num_embeddings'])*_text(c)['engram_head_dim'],
 'vision':_has_vision,
 # Keyed, not .get(): an absent hc_mult must read as unresolved rather than default
 # quietly to 'standard' and compare as a mismatch nobody's bytes support.
 'residual':lambda c:'mhc' if _text(c)['hc_mult']>1 else 'standard',
}

#: The other half of the decision FIELD_MAP forces: fields config.json does not carry, each with
#: the reason it cannot be compared against publisher bytes here. Recorded rather than omitted,
#: because a field quietly missing from both maps is a field nothing verifies and nothing explains.
NOT_IN_CONFIG={
 'id':'STRATUM identifier.',
 'name':'Display name.',
 'family':'Display grouping.',
 'repo':'Pinned in verification/manifests/sources.json, not in the config document.',
 'generation':'Display label.',
 'architecture':'STRATUM taxonomy, not a publisher field.',
 'summary':'Editorial one-line description.',
 'sources':'Citation list, checked by the evidence gate rather than here.',
 'ff':'Dense FFN width. denseLayers is 0 for this checkpoint and config.json declares no dense '
      'intermediate_size, so the field is inert and has no publisher counterpart.',
 'denseLayers':'config.json declares no first_k_dense_replace for this checkpoint. Comparing '
               'against a default of 0 would be a tautology with no publisher byte behind it.',
 'qkNorm':'config.json declares no query/key norm flag and Attention.__init__ applies none. The '
          'claim asserts an ABSENCE, so it carries no anchor at all: the `q_norm` in that class '
          'normalizes the query LATENT, not the per-head Q/K, and an absence read off a span that '
          'simply lacks the mechanism is an inference. See docs/verifiers/ANCHOR_RESIDUE.md.',
 'kvRank':'Shared default for an absent mechanism (no latent KV rank in this architecture).',
 'latent':'Shared default for an absent mechanism (no latent-width routing).',
 'linearHeads':'Shared default for an absent mechanism (no linear attention).',
 'linearKeyHeads':'Shared default for an absent mechanism (no linear attention).',
 'linearHeadDim':'Shared default for an absent mechanism (no linear attention).',
 'convKernel':'Shared default for an absent mechanism (no short convolution).',
 'gate':'Shared default for an absent mechanism (no output gate).',
 'pleLayers':'Shared default for an absent mechanism (no per-layer embeddings).',
 'total':'Published in the model card, not the config.',
 'active':'Published in the model card, not the config.',
 'activeDecode':'Published in the model card, not the config.',
 'kvBytesPerToken':'Published in the model card; held accountable by KV_RESIDUAL_IMPLAUSIBLE.',
 'kvFormat':'Stated in the technical report (§2.4), not the config.',
 'swaKvFormat':'Stated in the technical report (§2.4), not the config.',
}


def _resolve(config,rule):
    """The publisher value for one mapped field, or None when the rule does not resolve."""
    try:
        return rule(config) if callable(rule) else pointer(config,rule)
    except Exception:
        return None


def _normalise(value):
    return tuple(value) if isinstance(value,(list,tuple)) else value


def mid_hint(m):
    return m.get('id') or 'model'


def publisher_config(root,catalog,m):
    """The acquired config.json for this preset's publisher, or (None, why).

    Resolved through the source catalog by role and repo rather than by a hardcoded id, and read
    through `evidence.safe_path` so a crafted cache path cannot escape the repository.
    """
    records=[s for s in (catalog or {}).get('sources',[])
             if s.get('role')=='publisher_config' and s.get('repo')==m.get('repo')]
    if not records:
        return None,(f'No publisher_config source record for {m.get("repo")}, so there are no '
                     'publisher bytes this transcription can be compared against.',
                     'Add the record to verification/manifests/sources.json, then acquire it: '
                     'uv run --frozen python -m verification.sources acquire --candidate '
                     f'.verification-cache/candidates/{mid_hint(m)}-sources.json '
                     f'--only {mid_hint(m)}-config')
    record=records[0]
    acquire=('uv run --frozen python -m verification.sources acquire --candidate '
             f'.verification-cache/candidates/{record["id"]}.json --only {record["id"]}')
    if not record.get('cache'):
        return None,(f'Source {record["id"]} records no cached bytes, so there is nothing to '
                     'compare the transcription against.',f'Acquire them locally: {acquire}')
    try:
        path=safe_path(Path(root),record['cache'])
    except ValueError as exc:
        return None,(f'Source {record["id"]} names an unusable cache path: {exc}.',
                     'Repair the record; do not widen the path check.')
    if not path.exists():
        return None,(f'No publisher bytes on disk for {record["id"]} at {record["cache"]}. '
                     '`.verification-cache/` is git-ignored by design, so CI holds none and this '
                     'check cannot run there.',f'Acquire them locally: {acquire}')
    raw=path.read_bytes()
    if record.get('sha256') and hashlib.sha256(raw).hexdigest()!=record['sha256']:
        return None,(f'Cached bytes for {record["id"]} do not match the recorded sha256, so they '
                     'are not the reviewed snapshot.',
                     f'Never edit a recorded hash. Re-acquire and re-pin: {acquire}')
    try:
        return json.loads(raw.decode('utf-8')),None
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        return None,(f'Cached bytes for {record["id"]} are not decodable UTF-8 JSON: {exc}.',
                     f'Re-acquire: {acquire}')


def judge_publisher_agreement(models:dict,*,config=None,root=None,catalog=None,
                              domain:str='evidence')->list[Finding]:
    """Re-derive every transcribed field from the publisher's own bytes and compare.

    Not a comparison against committed values -- that would be self-consistency rather than
    verification. When there are no publisher bytes to compare against, this reports BLOCKED and
    names the acquire command; it never reports agreement having compared nothing.

    Findings default to the `evidence` domain because this check reads upstream bytes, which is
    audit work rather than a software gate: per ruling T789a it is wired into `runner.audit`'s
    publication profile and deliberately not into `schematics.run`, which deploys the page and
    must never BLOCK on evidence CI does not hold.
    """
    out=[]
    for mid,m in sorted(models.items()):
        if not is_ced(m):continue
        unmapped=sorted(f for f in m if f not in FIELD_MAP and f not in NOT_IN_CONFIG)
        _one(out,'CED_FIELD_UNMAPPED',mid,not unmapped,
             f'All {len(m)} transcribed fields are either mapped to config.json or recorded in '
             f'NOT_IN_CONFIG with a reason.',
             f'Fields {unmapped} appear in neither FIELD_MAP nor NOT_IN_CONFIG, so nothing '
             f'verifies them and nothing records why not.',
             'Map the field to a config pointer or expression, or record in NOT_IN_CONFIG why it '
             'has no publisher counterpart.')
        document,why=(config,None) if config is not None else publisher_config(root,catalog,m)
        if document is None:
            message,remedy=why or ('No publisher config was supplied to compare against.',
                                   'Pass `config`, or `root` and `catalog` so the acquired bytes '
                                   'can be resolved: uv run --frozen python -m '
                                   'verification.sources acquire --candidate '
                                   '.verification-cache/candidates/dsv41-sources.json '
                                   '--only dsv41-config')
            out.append(_evidence_blocked('CED_PUBLISHER_AGREEMENT',mid,message,remedy))
            continue
        compared=[];unresolved=[];bad=[]
        for field,rule in sorted(FIELD_MAP.items()):
            if field not in m:continue
            published=_resolve(document,rule)
            if published is None:
                unresolved.append(field);continue
            compared.append(field)
            if _normalise(published)!=_normalise(m[field]):
                bad.append(f'{field}: transcribed {m[field]!r}, publisher {published!r}')
        if bad:
            out.append(_bad('CED_PUBLISHER_AGREEMENT',mid,
                            f'{len(bad)} of {len(compared)} compared fields disagree with the '
                            f'publisher bytes -- '+'; '.join(bad),
                            'Re-transcribe from config.json. A derived field disagreeing means '
                            'the expression is wrong, not the bytes; never adjust the publisher '
                            'value.'))
        elif not compared:
            out.append(_evidence_blocked('CED_PUBLISHER_AGREEMENT',mid,
                                f'No mapped field resolved against the supplied config, so '
                                f'nothing was compared ({len(unresolved)} unresolved).',
                                'A comparison count of zero is not agreement. Acquire the real '
                                'config: uv run --frozen python -m verification.sources acquire '
                                '--candidate .verification-cache/candidates/dsv41-sources.json '
                                '--only dsv41-config'))
        elif unresolved:
            out.append(_evidence_blocked('CED_PUBLISHER_AGREEMENT',mid,
                                         f'{len(compared)} transcribed fields agree with the '
                                         f'publisher config, but {len(unresolved)} mapped fields '
                                         f'did not resolve against it: {unresolved}. A field that '
                                         f'does not resolve is a field nothing compared.',
                                         'Repoint or re-express the unresolved fields against the '
                                         'current config.json, or move them to NOT_IN_CONFIG with '
                                         'a reason. Partial agreement is not agreement.'))
        else:
            out.append(_ok('CED_PUBLISHER_AGREEMENT',mid,
                           f'{len(compared)} transcribed fields re-derived from the publisher '
                           f'config and equal, 0 unresolved.'))
    return [Finding(f.code,domain,f.status,f.subject,f.message,f.remedy) for f in out]


#: The model-level judges `schematics.run` wires in, as one list so that the gate and
#: `test_no_schematics_judge_can_emit_blocked` iterate the *same* object. Wiring a judge therefore
#: brings it under the property test automatically; the test asserts a property of whatever this
#: holds rather than mirroring a list of names. Every member takes `(models, samples)`.
SCHEMATICS_JUDGES=(judge_architecture,judge_kv_budget,judge_mhc,judge_parameter_bands,judge_coverage)

#: The per-scene judges, each paired with the condition `schematics.run` applies before calling
#: it. Same purpose: one list, iterated by the gate and by the property test.
SCENE_JUDGES=((lambda expect,state:expect.get('residual')=='mhc',judge_mhc_streams),
              (lambda expect,state:state.get('stage')=='ced',judge_ced_topology))
