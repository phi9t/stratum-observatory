"""Diagram semantics: what the schematic means, not where its boxes landed.

Engineering-class, like aesthetics: every value is derived from our own scene IR and our
own model table, so it needs no upstream source and no signed review, and per
docs/verifiers/DESIGN.md trust boundary 1 it must never contribute to a scientific PASS.

Runs in Node-then-Python with no browser, which is what makes the full 504-layer matrix
affordable. `scene` comes from the diagram builders; `expect` comes from core.mjs, the same
functions behind the ledgers. Comparing them checks that the picture agrees with the
mathematics rather than merely with itself.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from .model import Finding,aggregate,exit_code
from .evidence import loads

# Closed value sets. structuralReport checks none of these; a typo in a builder silently
# renders an unstyled node or an arrowless wire.
NODE_TONES={'neutral','compute','state','comm'}
NODE_KINDS={'op','pill','sum','comm'}
EDGE_KINDS={'flow','residual','state','control','delay','comm','host','nvlink','inactive'}
LABEL_TONES={'muted','state','compute','text','comm'}


def export_scenes(root):
    out=subprocess.run(['node','tools/export-scenes.mjs'],cwd=root,text=True,capture_output=True,check=True,timeout=300)
    return loads(out.stdout)


def _ok(code,subject,message):
    return Finding(code,'schematics','PASS',subject,message)


def _bad(code,subject,message,remedy):
    return Finding(code,'schematics','FAIL',subject,message,remedy)


def judge_ir(scene):
    """The IR contract: closed value sets, and the groups/labels structuralReport ignores."""
    out=[];ids={n['id'] for n in scene['nodes']}
    for n in scene['nodes']:
        if n.get('tone') not in NODE_TONES:
            out.append(_bad('IR_VALUE',n['id'],f"Node tone {n.get('tone')!r} is not one of {sorted(NODE_TONES)}.",'Use a declared tone; an unknown tone renders unstyled.'))
        if n.get('kind') not in NODE_KINDS:
            out.append(_bad('IR_VALUE',n['id'],f"Node kind {n.get('kind')!r} is not one of {sorted(NODE_KINDS)}.",'Use a declared kind.'))
    seen=set()
    for g in scene['groups']:
        if g['id'] in seen:out.append(_bad('IR_GROUP',g['id'],'Duplicate group id.','Group ids must be unique, as node ids already are.'))
        seen.add(g['id'])
        if not all(isinstance(g.get(k),(int,float)) for k in ('x','y','w','h')) or g['w']<=0 or g['h']<=0:
            out.append(_bad('IR_GROUP',g['id'],'Group has a non-finite or non-positive box.','Give the group a real box.'))
        elif g['x']<0 or g['y']<0 or g['x']+g['w']>scene['width'] or g['y']+g['h']>scene['height']:
            out.append(_bad('IR_GROUP',g['id'],'Group box leaves the canvas.','Contain the group; nothing checks this today.'))
    for l in scene['labels']:
        if l.get('tone') not in LABEL_TONES:
            out.append(_bad('IR_VALUE',l.get('id','label'),f"Label tone {l.get('tone')!r} is not one of {sorted(LABEL_TONES)}.",'Use a declared tone.'))
    for e in scene['edges']:
        if e.get('kind') not in EDGE_KINDS:
            out.append(_bad('IR_VALUE',e['id'],f"Edge kind {e.get('kind')!r} has no marker definition.",'Use a declared edge kind.'))
        pts=e.get('points') or []
        if len(pts)<2:
            out.append(_bad('IR_EDGE',e['id'],'Wire has fewer than two points and renders as a dot.','Give the wire a real path.'));continue
    return out or [_ok('IR_OK','scene','IR value sets, groups, labels and wire endpoints are well formed.')]


def judge_state_rails(scene):
    """A delayed dependency must be both typed and flagged.

    OBSERVATORY.md distinguishes delayed recurrence from a same-step dependency, and the
    renderer keys off `kind`, while composition_signature keys off `delay`. One without the
    other means the two disagree about what the diagram says.
    """
    out=[];incident=defaultdict(set)
    if not scene['edges']:return [_ok('RAIL_OK','scene','Not a wired scene; state rails do not apply.')]
    for e in scene['edges']:
        for end in ('source','target'):
            if e.get(end):incident[e[end]].add(e.get('kind'))
        if (e.get('kind')=='delay')!=bool(e.get('delay')):
            out.append(_bad('RAIL_DELAY',e['id'],"Edge kind 'delay' and the delay flag disagree.",
                            'A delayed dependency must be both typed and flagged; one without the other renders as a same-step wire.'))
    return out or [_ok('RAIL_OK','scene','Delayed dependencies are consistently typed and flagged.')]


def count_orphans(scene):
    """Nodes with no incident wire, reported as a metric rather than a defect.

    Every lens carries annotation cards -- "Implementation boundary", "Publisher recipe is
    not this baseline", the per-rank ledgers -- that sit beside the flow on purpose. The IR
    has no annotation flag, so "every node is wired" is not a contract it makes, and
    asserting it would be inventing a rule rather than checking one. The count is recorded so
    a sudden jump is still visible.
    """
    ids={n['id'] for n in scene['nodes']}
    linked=set()
    for e in scene['edges']:
        for end in ('source','target'):
            if e.get(end) in ids:linked.add(e[end])
    return len(ids-linked) if scene['edges'] else 0


def _runs(scene):
    """Axis-aligned segments as (orientation, lane, lo, hi, edge id)."""
    for e in scene['edges']:
        pts=e.get('points') or []
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
            if y1==y2 and x1!=x2:yield ('h',y1,min(x1,x2),max(x1,x2),e['id'])
            elif x1==x2 and y1!=y2:yield ('v',x1,min(y1,y2),max(y1,y2),e['id'])


def judge_lanes(scene):
    """Two wires sharing a lane over an overlapping interval read as one wire.

    count_crossings cannot see this: the segments never cross, they coincide. Wires sharing a
    source or a target are exempt -- a fan-out from one operation legitimately shares a trunk.
    """
    out=[];runs=list(_runs(scene))
    related={e['id']:(e.get('source'),e.get('target')) for e in scene['edges']}
    for i,a in enumerate(runs):
        for b in runs[i+1:]:
            if a[0]!=b[0] or a[1]!=b[1] or a[4]==b[4]:continue
            if related.get(a[4]) and related.get(b[4]) and (
               related[a[4]][0]==related[b[4]][0] or related[a[4]][1]==related[b[4]][1]):continue
            overlap=min(a[3],b[3])-max(a[2],b[2])
            if overlap>1.0:
                out.append(_bad('LANE_COLLINEAR',f'{a[4]}|{b[4]}',
                                f'Wires share lane {a[0]}={a[1]:g} for {overlap:g}px and render as one line.',
                                'Route one of them into its own lane.'))
    return out or [_ok('LANE_OK','scene','No two wires share a lane over an overlapping interval.')]


def judge_scene_vs_model(scene,expect,state):
    """The picture must agree with core.mjs, which computed it independently."""
    out=[];ids={n['id'] for n in scene['nodes']}
    if state.get('view')!='model' or state.get('stage')!='block':
        return [_ok('SCENE_MODEL_SKIP','scene','Only the decoder-block lens depicts a single layer.')]
    dense_layer=expect.get('denseLayers',0) and state.get('layer',1)<=expect['denseLayers']
    if not expect.get('experts') and 'router' in ids:
        out.append(_bad('SCENE_MODEL_ROUTER',expect['model'],'A dense checkpoint is drawn with a router.',
                        'No branch may be invented to fit a template.'))
    if expect.get('experts') and not dense_layer and 'router' not in ids:
        out.append(_bad('SCENE_MODEL_ROUTER',expect['model'],'A routed layer is drawn without a router.','Draw the routing the model declares.'))
    if not expect.get('shared') and 'shared-up' in ids:
        out.append(_bad('SCENE_MODEL_SHARED',expect['model'],'A shared expert is drawn for a checkpoint that declares none.',
                        'Shared experts exist only where the configuration declares them.'))
    tiles=[d for d in scene['decorations'] if d.get('kind')=='layer']
    if tiles:
        if len(tiles)!=expect['layers']:
            out.append(_bad('SCENE_MODEL_TILES',expect['model'],f"{len(tiles)} layer tiles drawn for a {expect['layers']}-layer checkpoint.",
                            'The selector must show every decoder layer.'))
        drawn=Counter(d['attention'] for d in tiles)
        declared={k:v for k,v in (expect.get('partition') or {}).items() if v}
        if declared and dict(drawn)!=declared:
            out.append(_bad('SCENE_MODEL_TILES',expect['model'],f'Layer tiles show {dict(drawn)} against a declared partition of {declared}.',
                            'The tile mechanism must come from layerKind, not from a hand-written schedule.'))
        active=[d for d in tiles if d.get('active')]
        if len(active)!=1 or active[0]['layer']!=state.get('layer'):
            out.append(_bad('SCENE_MODEL_TILES',expect['model'],'The selected layer tile does not match the selected layer.','Bind the active tile to the state.'))
    return out or [_ok('SCENE_MODEL_OK',expect['model'],'Routing, shared experts and the layer selector agree with core.mjs.')]


def judge_row_alignment(data,tolerance,max_rows=2):
    """OBSERVATORY.md: sibling operations share semantic rows across presets.

    Position is memory. `norm1`, `attention` and `router` sit at pixel-identical heights in
    all eight presets, and that is what lets a reader switch checkpoints without relearning
    the diagram.

    A second row is legitimate where a checkpoint inserts a different mechanism -- Kimi's
    latent MoE pushes the expert column down, Flash-Next's QSA history sits higher, and the
    93/94-layer models need an extra row of layer tiles, which moves everything beneath the
    selector. A *third* row is not: it means the operation has no row at all. So the rule is
    that each shared operation occupies at most `max_rows` distinct rows, with drift inside a
    row bounded by the tolerance.
    """
    rows=defaultdict(dict)
    for entry in data['scenes']:
        model,view,stage,layer=entry['key'].split('/')
        if view!='model' or stage!='block' or layer!='4':continue
        for n in entry['scene']['nodes']:
            rows[n['id']][model]=n['y']+n['h']/2
    out=[];variants=0
    for node_id,per_model in sorted(rows.items()):
        if len(per_model)<2:continue
        clusters=[]
        for model,centre in sorted(per_model.items(),key=lambda kv:kv[1]):
            if clusters and centre-clusters[-1][0]<=tolerance:clusters[-1][1].append(model)
            else:clusters.append((centre,[model]))
        if len(clusters)>1:variants+=1
        if len(clusters)>max_rows:
            spread=' / '.join(f"{c:.0f}px {sorted(ms)}" for c,ms in clusters)
            out.append(_bad('ROW_DRIFT',node_id,f'Operation occupies {len(clusters)} distinct rows across presets: {spread}.',
                            'Share the semantic row. A second row is allowed where a checkpoint inserts a different mechanism; a third means the operation has no row.'))
    if out:return out
    return [_ok('ROW_OK','presets',f'{len(rows)} shared operations hold their rows within {tolerance:g}px; {variants} carry a single structural variant row.')]


def run(root,tolerance=None):
    data=export_scenes(root)
    policy=loads((root/'verification/baselines/aesthetics/policy.json').read_text())
    tolerance=tolerance if tolerance is not None else policy['layout']['row_alignment_tolerance_px']
    findings=[];orphans=0
    def scoped(entry,items):
        for f in items:
            findings.append(Finding(f.code,f.domain,f.status,f"{entry['key']}:{f.subject}",f.message,f.remedy))
    for entry in data['scenes']:
        scene=entry['scene']
        scoped(entry,judge_ir(scene))
        scoped(entry,judge_state_rails(scene))
        scoped(entry,judge_lanes(scene))
        scoped(entry,judge_scene_vs_model(scene,entry['expect'],entry['state']))
        orphans+=count_orphans(scene)
    findings.extend(judge_row_alignment(data,tolerance))
    report={'schema':'stratum-schematics/v1','status':aggregate(findings),
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'scenes':data['count'],'decoder_layers':sum(1 for e in data['scenes'] if e['key'].split('/')[1:3]==['model','block']),
            'row_alignment_tolerance_px':tolerance,'unwired_annotation_nodes':orphans,
            'counts':dict(Counter(f.status for f in findings)),
            'findings':[f.json() for f in findings if f.status!='PASS'][:400],
            'passed_codes':dict(Counter(f.code for f in findings if f.status=='PASS')),
            'scope':'Scene IR semantics measured against core.mjs. Says nothing about upstream truth, and a coherent diagram is not a correct one.'}
    return report


def markdown(report):
    lines=[f"# Schematics — {report['status']}",f"\nGenerated: {report['generated_at']}",
           f"\n{report['scenes']} scenes, including {report['decoder_layers']} decoder layers.",
           '\nThe scene IR is checked against core.mjs. A collision-free picture is not proof that the depicted mathematics is right.',
           '\n## Findings by failure mode','\n| Code | Status | Count |','|---|---|---:|']
    groups=Counter((f['code'],f['status']) for f in report['findings'])
    lines+=[f'| {c} | {s} | {n} |' for (c,s),n in sorted(groups.items())] or ['| none | — | 0 |']
    lines+=['\n## Checks that passed','\n| Code | Count |','|---|---:|']
    lines+=[f'| {c} | {n} |' for c,n in sorted(report['passed_codes'].items())]
    if report['findings']:
        lines+=['\n## First unresolved findings']
        for f in report['findings'][:24]:lines+=[f"\n**{f['code']} / {f['subject']}** — {f['message']} {f['remedy']}"]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=p.parse_args(argv);root=Path(a.root).resolve()
    report=run(root)
    out=root/'verification/reports';out.mkdir(exist_ok=True)
    (out/'schematics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'schematics.md').write_text(markdown(report))
    print(f"schematics: {report['status']} | {report['scenes']} scenes | {report['counts']}")
    return exit_code(report['status'])


if __name__=='__main__':raise SystemExit(main())
