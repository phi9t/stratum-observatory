"""What actually reaches a reader: the exported files, the delivered page, print and previews.

Engineering-class, like aesthetics and schematics. This is the first suite to open the
artifact from a real `file://` URL -- every other suite uses `page.set_content()`, so the
path a reader actually takes has never been exercised.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from .model import Finding,aggregate,exit_code
from .evidence import loads
from .schema import validate as shape_validate
from .aesthetics import parse_color,contrast_ratio

EXTERNAL=re.compile(r'https?://(?!www\.w3\.org/)[^\s"\'<>)]+')
# A URL the page LOADS breaks portability. A URL the reader may CLICK is the provenance
# feature -- README says plainly that external source links require connectivity -- so the
# 23 arXiv and vendor-doc citations in the artifact are not defects.
LOADING=re.compile(r'''(?:\bsrc\s*=\s*["']|@import[^;]{0,80}?["']|\burl\(\s*["']?|\bfetch\(\s*["']|\bimportScripts\(\s*["']|<link\b[^>]{0,200}?\bhref\s*=\s*["'])(https?://[^\s"'<>)]+)''',re.I)
POLICY_PATH='verification/baselines/aesthetics/policy.json'


def load_policy(root):
    policy=loads((Path(root)/POLICY_PATH).read_text())
    return policy.get('presentation',{'max_artifact_kib':400})


def _ok(code,subject,message):return Finding(code,'presentations','PASS',subject,message)
def _bad(code,subject,message,remedy):return Finding(code,'presentations','FAIL',subject,message,remedy)


def judge_artifact_source(path,policy):
    """The portable file must be portable in its bytes, not merely at runtime.

    The browser suite proves no request is *made*; it cannot see a URL that is only built on
    a path nobody exercised. Reading the source closes that gap.
    """
    path=Path(path);out=[];raw=path.read_bytes()
    try:
        text=raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        return [_bad('PORTABILITY_ENCODING',path.name,f'Artifact is not valid UTF-8: {exc}.',
                     'The document declares utf-8; write it as utf-8.')]
    for url in sorted(set(LOADING.findall(text)))[:10]:
        out.append(_bad('PORTABILITY_EXTERNAL',path.name,f'Artifact loads from an external origin: {url}.',
                        'Inline the asset. A URL that is only fetched on an unexercised path still ships.'))
    citations=len(set(EXTERNAL.findall(text)))
    kib=len(raw)/1024
    limit=policy.get('max_artifact_kib',400)
    if kib>limit:
        out.append(_bad('PORTABILITY_SIZE',path.name,f'Artifact is {kib:.0f} KiB against a {limit} KiB budget.',
                        'Keep the single-file artifact small enough to open anywhere.'))
    return out or [_ok('PORTABILITY_OK',path.name,
                       f'{kib:.0f} KiB, valid utf-8, nothing loaded from an external origin; {citations} source citations are clickable by design.')]


def judge_exported_svg(svg,policy):
    """An export must stand alone on someone else's page."""
    out=[]
    if 'xmlns' not in svg[:400]:
        out.append(_bad('EXPORT_MALFORMED','svg','Exported SVG carries no xmlns declaration and will not render standalone.',
                        'Emit the SVG namespace.'))
    for url in sorted(set(EXTERNAL.findall(svg)))[:5]:
        out.append(_bad('EXPORT_EXTERNAL','svg',f'Export references an external origin: {url}.',
                        'An export that fetches is not an offline artifact.'))
    if 'canvas-bg' not in svg and 'rect' not in svg:
        out.append(_bad('EXPORT_BACKDROP','svg','Export has no backdrop rect, so its colours are undefined on a host page.',
                        'Paint an opaque full-viewBox backdrop into the export.'))
    return out or [_ok('EXPORT_OK','svg','Export is namespaced, self-contained and carries a backdrop.')]


def judge_scenario(doc,root):
    """The exported JSON must match the schema the page claims and keep its status honest."""
    schema=loads((Path(root)/'verification/schemas/scenario.schema.json').read_text())
    errors=shape_validate(doc,schema)
    out=[_bad('SCENARIO_SHAPE','export',e,'Correct the export or the schema; a declared schema that nothing validates is decoration.') for e in errors[:10]]
    return out or [_ok('SCENARIO_OK','export','Exported scenario matches stratum-observatory-scenario/v1 and reports BLOCKED / NOT_RUN.')]


def judge_preview_freshness(root,recorded=None):
    """previews/manifest.json records an artifact hash that nothing has ever read."""
    root=Path(root);manifest=root/'previews/manifest.json'
    if not manifest.exists():
        return [Finding('PREVIEW_ABSENT','presentations','BLOCKED','previews','No preview manifest to compare against.','Run npm run capture.')]
    data=loads(manifest.read_text())
    actual=hashlib.sha256((root/'dist/index.html').read_bytes()).hexdigest()
    declared=recorded if recorded is not None else data.get('artifact_sha256')
    if declared!=actual:
        return [_bad('PREVIEW_STALE','previews',f'Previews were captured from artifact {str(declared)[:16]}… but the build is {actual[:16]}….',
                     'Re-run npm run capture; published previews must depict the artifact that ships.')]
    for capture in data.get('captures',[]):
        name=capture.get('artifact')
        if name:
            if name not in ('index.html','deepseek-v4.1.html'):
                return [_bad('PREVIEW_STALE',str(name),'Unknown preview artifact.','Capture a delivered entry point.')]
            artifact=root/'dist'/name
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=capture.get('artifact_sha256'):
                return [_bad('PREVIEW_STALE',name,'Preview belongs to different page bytes.','Run npm run capture.')]
    return [_ok('PREVIEW_FRESH','previews',f"{len(data.get('captures',[]))} captures match the built artifacts.")]


def judge_preview_artifacts(root):
    root=Path(root);out=[]
    captures=loads((root/'previews/manifest.json').read_text()).get('captures',[])
    for capture in captures:
        expected='deepseek-v4.1.html' if capture.get('state',{}).get('model')=='dsv41' else 'index.html'
        page=root/'dist'/expected
        if capture.get('artifact')!=expected or not page.is_file() or capture.get('artifact_sha256')!=hashlib.sha256(page.read_bytes()).hexdigest():
            out.append(_bad('PREVIEW_ARTIFACT',capture.get('name','preview'),
                            'Preview does not bind the delivered page for its model.','Regenerate previews with npm run capture.'))
    return out or [_ok('PREVIEW_ARTIFACT','previews','Every capture binds its delivered entry point.')]


def judge_preview_coverage(root,models,focal):
    """Fresh is not complete, and this suite only ever asked the first question.

    `judge_preview_freshness` compares one artifact digest, so a manifest depicting eight of nine
    checkpoints is exactly as fresh as one depicting all nine. `tests/capture.py` built its cases
    from a hardcoded eight-model literal and therefore published no preview of the ninth
    checkpoint at all, and nothing reported it.

    Closing that with an `encoderLayers`-only requirement left the same defect one layer in: this
    judge asked for a decoder block plus, where applicable, a `ced` page, and `focalStages` offers
    four. So `dsv41/model/residual` -- the mHC 4-stream residual this checkpoint exists to draw --
    was in no published preview and reported PASS, as were `dsv41/model/inputs`,
    `q38flash/model/inputs` and `q3827/model/inputs`. The requirement is now every focal page a
    model actually offers, read from `core.focalStages` rather than from a property test here.
    """
    root=Path(root);manifest=root/'previews/manifest.json'
    if not manifest.exists():
        return [Finding('PREVIEW_ABSENT','presentations','BLOCKED','previews','No preview manifest to compare against.','Run npm run capture.')]
    captures=loads(manifest.read_text()).get('captures') or []
    stages={}
    for capture in captures:
        state=capture.get('state') or {}
        stages.setdefault(state.get('model'),set()).add(state.get('stage'))
    out=[];pages=0
    for mid in sorted(models):
        if mid not in stages:
            out.append(_bad('PREVIEW_COVERAGE',mid,f'No published preview depicts {mid} in any state.',
                            'tests/capture.py must derive its cases from the live model table; a literal list '
                            'reports success over the models it happens to name.'))
            continue
        wanted=set(focal.get(mid,()));pages+=len(wanted)
        missing=sorted(wanted-stages[mid])
        if missing:
            out.append(_bad('PREVIEW_COVERAGE',mid,
                            f'No published preview depicts the {mid} focal page(s) {missing}; '
                            f'this checkpoint offers {sorted(wanted)}.',
                            'A focal page carries a diagram and prose that exist in no other state. '
                            'tests/capture.py must capture every stage the control strip offers, not a '
                            'literal subset of them.'))
    return out or [_ok('PREVIEW_COVERAGE','previews',
                       f'All {len(models)} checkpoints appear in the {len(captures)} published previews, '
                       f'including every one of the {pages} focal pages they offer.')]


def judge_preview_integrity(root):
    """The 29 recorded per-file digests that nothing has ever read.

    `tests/capture.py` writes an `svg_sha256` for every capture. Across every `.py` and `.mjs` in
    this repository there was exactly one reference to that key: the line that writes it.
    `judge_preview_freshness` compares a single artifact digest and `judge_preview_coverage`
    counts models, so a `previews/*.svg` that had been hand-edited, truncated, or left behind by
    an older build passed both -- the manifest recorded exactly what was wrong with it and no one
    asked. A recorded digest that nothing verifies is not provenance; it is decoration that reads
    like provenance, which is worse.

    Missing files are reported separately from mismatching ones: one is an incomplete capture,
    the other is a published preview that does not depict what the manifest says it depicts.
    """
    root=Path(root);manifest=root/'previews/manifest.json'
    if not manifest.exists():
        return [Finding('PREVIEW_ABSENT','presentations','BLOCKED','previews','No preview manifest to compare against.','Run npm run capture.')]
    captures=loads(manifest.read_text()).get('captures') or []
    out=[];checked=0
    for capture in captures:
        name=capture.get('name');recorded=capture.get('svg_sha256')
        if not recorded:
            continue
        path=root/'previews'/f'{name}.svg'
        if not path.exists():
            out.append(_bad('PREVIEW_INTEGRITY',name,
                            f'The manifest records a digest for {name}.svg but the file is absent.',
                            'Re-run npm run capture; a recorded capture with no file is an incomplete publication.'))
            continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        checked+=1
        if actual!=recorded:
            out.append(_bad('PREVIEW_INTEGRITY',name,
                            f'{name}.svg hashes to {actual[:16]}… but the manifest records {str(recorded)[:16]}….',
                            'The published preview is not the one the manifest attests. Re-run npm run '
                            'capture rather than editing the recorded digest.'))
    if out:
        return out
    if not checked:
        return [Finding('PREVIEW_INTEGRITY','presentations','BLOCKED','previews',
                        'No capture carries a recorded SVG digest, so nothing here was verified.',
                        'A comparison count of zero is not integrity. Re-run npm run capture.')]
    return [_ok('PREVIEW_INTEGRITY','previews',
                f'{checked} published SVG previews each hash to the digest the manifest records.')]


def judge_preview_inventory(root):
    """Coverage in the direction nobody was checking: files on disk that the manifest never names.

    `judge_preview_coverage` walks the manifest and asks whether each model is in it. Nothing
    walked `previews/` and asked whether each published file is in the manifest, so bytes could
    sit in the published directory attested by nothing and depicting anything -- including an
    artifact several releases old. That is the same "absent coverage reads as passing coverage"
    shape as the manifest-side gap, with the arrow reversed.
    """
    root=Path(root);manifest=root/'previews/manifest.json'
    if not manifest.exists():
        return [Finding('PREVIEW_ABSENT','presentations','BLOCKED','previews','No preview manifest to compare against.','Run npm run capture.')]
    captures=loads(manifest.read_text()).get('captures') or []
    named={c.get('name') for c in captures}
    on_disk={path.stem for path in (root/'previews').iterdir()
             if path.suffix in ('.png','.svg')}
    orphans=sorted(on_disk-named)
    if orphans:
        return [_bad('PREVIEW_INVENTORY','previews',
                     f'{len(orphans)} published preview file(s) are named by no capture record: '
                     f'{", ".join(orphans)}.',
                     'Every file in previews/ must be produced by npm run capture and recorded in the '
                     'manifest, or it is bytes attested by nothing. Capture it properly or move it out '
                     'of the published directory; do not leave it unrecorded.')]
    return [_ok('PREVIEW_INVENTORY','previews',
                f'Every one of the {len(on_disk)} published preview files is named by a capture record.')]


PROBE_A11Y=r"""
() => {
  const out = {issues: [], counts: {}};
  // A <label for> association is the most standard naming mechanism and browsers expose it
  // as el.labels; checking only aria-* and text content misses correctly labelled inputs.
  const named = el => !!(el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') ||
                         (el.labels && el.labels.length && [...el.labels].some(l => (l.textContent || '').trim())) ||
                         el.closest('label') || (el.textContent || '').trim() || el.getAttribute('title'));
  const controls = Array.from(document.querySelectorAll('button, a[href], select, input, [role="button"][tabindex]'));
  out.counts.controls = controls.length;
  for (const el of controls) if (!named(el))
    out.issues.push({kind: 'unnamed-control', detail: (el.id || el.className || el.tagName).toString().slice(0, 60)});
  const levels = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).map(h => +h.tagName[1]);
  out.counts.headings = levels.length;
  for (let i = 1; i < levels.length; i++) if (levels[i] - levels[i - 1] > 1)
    out.issues.push({kind: 'heading-skip', detail: `h${levels[i - 1]} to h${levels[i]}`});
  if (!document.querySelector('h1')) out.issues.push({kind: 'no-h1', detail: 'document has no h1'});
  const live = document.querySelectorAll('[aria-live], [role="status"]');
  out.counts.live = live.length;
  if (!live.length) out.issues.push({kind: 'no-live-region', detail: 'state changes are announced to nobody'});
  const svg = document.querySelector('#scene-mount svg');
  if (svg && !named(svg)) out.issues.push({kind: 'unnamed-diagram', detail: 'the schematic has no accessible name'});
  out.title = document.title;
  out.lang = document.documentElement.lang;
  return out;
}
"""


PROBE_PROSE=r"""
() => {
  // Prose and data tables, measured on the page a reader is actually on. `PROBE_A11Y` covers
  // headings and control names; this covers the part of a data table a screen reader needs and
  // nothing else checks: a real header row, a row header per body row, and links that say where
  // they go. A five-column table of <td>s is perfectly legible and completely unnavigable.
  const out = {issues: [], counts: {}};
  const tables = Array.from(document.querySelectorAll('table'));
  out.counts.tables = tables.length;
  out.counts.rows = 0;
  for (const t of tables) {
    const id = t.className || t.tagName;
    const head = t.querySelector('thead');
    if (!head) { out.issues.push({kind: 'table-no-thead', detail: id}); continue; }
    const cols = Array.from(head.querySelectorAll('th'));
    if (!cols.length) out.issues.push({kind: 'table-no-column-headers', detail: id});
    for (const th of cols) if (th.getAttribute('scope') !== 'col')
      out.issues.push({kind: 'table-unscoped-column', detail: `${id}: ${(th.textContent||'').trim().slice(0,24)}`});
    const bodyRows = Array.from(t.querySelectorAll('tbody tr'));
    out.counts.rows += bodyRows.length;
    for (const tr of bodyRows) {
      const first = tr.firstElementChild;
      if (!first) continue;
      if (first.tagName !== 'TH') { out.issues.push({kind: 'table-no-row-header', detail: id}); break; }
      if (first.getAttribute('scope') !== 'row') { out.issues.push({kind: 'table-unscoped-row', detail: id}); break; }
    }
  }
  const links = Array.from(document.querySelectorAll('a[href^="http"]'));
  out.counts.links = links.length;
  for (const el of links) {
    if (!(el.textContent || '').trim() && !el.getAttribute('aria-label'))
      out.issues.push({kind: 'unnamed-citation', detail: el.getAttribute('href').slice(0, 60)});
    if (!/noopener/.test(el.getAttribute('rel') || ''))
      out.issues.push({kind: 'citation-without-noopener', detail: el.getAttribute('href').slice(0, 60)});
  }
  return out;
}
"""


PROBE_VISUALIZATION=r"""
() => {
  // getBBox() returns user units, and the SVG carries viewBox="0 0 w h", so text bounds are
  // already in scene coordinates -- the same space geometryReport compares against.
  const scene = STRATUM.getScene();
  const svg = document.querySelector('#scene-mount svg');
  const nodes = [];
  for (const n of scene.nodes) {
    const host = svg.querySelector(`[data-node="${CSS.escape(n.id)}"]`);
    if (!host) continue;
    const texts = [...host.querySelectorAll('text')].map(t => t.getBBox());
    if (!texts.length) continue;
    const x0 = Math.min(...texts.map(b => b.x)), y0 = Math.min(...texts.map(b => b.y));
    const x1 = Math.max(...texts.map(b => b.x + b.width)), y1 = Math.max(...texts.map(b => b.y + b.height));
    nodes.push({id: n.id, bounds: [n.x, n.y, n.w, n.h], text_bounds: [x0, y0, x1 - x0, y1 - y0]});
  }
  return {nodes, edges: scene.edges.map(e => ({source: e.source, target: e.target, points: e.points})),
          canvas: [scene.width, scene.height], state: STRATUM.getState()};
}
"""


def visualization_record(probe,target):
    """A visualization contract record measured from the rendered diagram.

    The record the manifest has carried is `{"id": "visualization.pending"}`, so the scene
    validator in domains/contracts.py -- which has been fully implemented all along -- has
    never run on a real diagram. Node claims are the scene claims the inventory already
    registers, so provenance is checked against the actual claim set rather than asserted.
    """
    st=probe['state']
    prefix=f"scene.{st['model']}.{st['view']}.{st['stage']}."
    claims={}
    for c in target.get('claims',[]):
        consumer=c.get('consumer') or ''
        if consumer.startswith(prefix):
            node_id=consumer[len(prefix):].rsplit('.',1)[0]
            claims.setdefault(node_id,[]).append(c['id'])
    ids={n['id'] for n in probe['nodes']}
    return {'id':'visualization.observatory','claim_ids':sorted({cid for v in claims.values() for cid in v}),
            'canvas':probe['canvas'],
            'nodes':[{'id':n['id'],'claims':claims.get(n['id'],[]),'bounds':n['bounds'],'text_bounds':n['text_bounds']}
                     for n in probe['nodes']],
            'edges':[e for e in probe['edges'] if e['source'] in ids and e['target'] in ids]}



def focal_states(root,models=None,focal=None):
    """The states a reader can reach that this suite must also reach.

    Derived from the live model table, not from a literal: `tests/browser.test.py` shipped a
    hardcoded eight-checkpoint list and went on sweeping it after a ninth arrived, which
    `VERIFICATION.md` records. A focal stage carries prose and a data table that exist in no other
    state, so measuring only the default state measures a page nobody is complaining about.

    The model list was read live and then narrowed by a literal `'ced'`, which is the same defect
    one layer in: `focalStages` offers four stages and this emitted one, so the `inputs`, `ple`
    and mHC `residual` pages reached a reader measured in no state while `PROSE_SCOPE` correctly
    reported having measured two. Every stage a model offers is now a state here.

    `models`/`focal` let a caller that has already paid for one exporter run reuse it; omitted,
    the table is read here, so the derivation is the same either way.
    """
    if models is None or focal is None:
        from .ced import inventory
        models,samples=inventory(root)
        focal=samples['focal_stages']
    return [{'model':mid,'view':'model','stage':stage,'layer':4}
            for mid in sorted(models) for stage in focal.get(mid,())]


def run(root):
    root=Path(root)
    sys.path.insert(0,str(root/'tests'))
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    policy=load_policy(root)
    artifact=root/'dist/index.html'
    from .ced import inventory
    models,samples=inventory(root)
    focal=samples['focal_stages']
    findings=list(judge_artifact_source(artifact,policy))
    dedicated=root/'dist/deepseek-v4.1.html'
    findings.extend(judge_artifact_source(dedicated,policy))
    findings.extend(judge_preview_freshness(root))
    findings.extend(judge_preview_artifacts(root))
    findings.extend(judge_preview_coverage(root,models,focal))
    findings.extend(judge_preview_integrity(root))
    findings.extend(judge_preview_inventory(root))
    checks={'models':sorted(models)}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True,args=['--no-sandbox','--allow-file-access-from-files'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        # The path a reader actually takes. Every other suite injects with set_content().
        page.goto(artifact.resolve().as_uri(),wait_until='load')
        page.wait_for_function('window.STRATUM?.version === "3.0.0"')
        page.evaluate('document.fonts.ready')
        checks['file_url']=page.url.startswith('file://')
        if not checks['file_url']:
            findings.append(_bad('DELIVERY_ORIGIN','file://','Artifact did not load from a file:// origin.','The delivered path must be the tested path.'))
        else:
            findings.append(_ok('DELIVERY_ORIGIN','file://','Artifact loads and initialises from a real file:// origin.'))
        if errors:
            findings.append(_bad('DELIVERY_RUNTIME','file://',f'{len(errors)} page errors on the file:// path: {errors[0][:160]}',
                                 'localStorage, history and downloads behave differently on an opaque origin.'))
        else:
            findings.append(_ok('DELIVERY_RUNTIME','file://','No page errors on the file:// path.'))

        findings.extend(judge_exported_svg(page.evaluate('STRATUM.exportSVG()'),policy))
        scenario=json.loads(page.evaluate('STRATUM.exportConfig()'))
        findings.extend(judge_scenario(scenario,root))
        page_version=page.evaluate('STRATUM.version')
        pkg=loads((root/'package.json').read_text())['version']
        if not (scenario['version']==page_version==pkg):
            findings.append(_bad('VERSION_DRIFT','version',f"export {scenario['version']}, page {page_version}, package.json {pkg} disagree.",
                                 'Three independent literals must not drift apart.'))
        else:
            findings.append(_ok('VERSION_OK','version',f'Export, page and package.json all report {pkg}.'))

        a11y=page.evaluate(PROBE_A11Y)
        for issue in a11y['issues']:
            findings.append(_bad('A11Y_'+issue['kind'].replace('-','_').upper(),issue['kind'],
                                 f"{issue['kind']}: {issue['detail']}",'The delivered page must be usable by someone not using a mouse or a screen.'))
        if not a11y['issues']:
            findings.append(_ok('A11Y_OK','page',f"{a11y['counts']['controls']} controls named, {a11y['counts']['headings']} headings ordered, {a11y['counts']['live']} live regions."))
        if not a11y.get('lang'):
            findings.append(_bad('A11Y_LANG','page','Document declares no language.','Set lang on the html element.'))

        # The focal pages. `page.goto` lands on the default state only, so until now the CED
        # page's headings, control names, table semantics and citations were checked by nothing --
        # and a report that says "9 headings ordered" was describing a different page.
        states=[{'label':'default','patch':None}]+[{'label':f"{c['model']}/{c['stage']}",'patch':c}
                                                   for c in focal_states(root,models,focal)]
        checks['states']=[st['label'] for st in states]
        for st in states:
            if st['patch']:
                target=dedicated if st['patch']['model']=='dsv41' else artifact
                page.goto(target.resolve().as_uri(),wait_until='load')
                page.wait_for_function('window.STRATUM')
                page.evaluate('(p)=>STRATUM.setState(p)',st['patch'])
            where=st['label']
            # The default state's headings and control names are already reported above as
            # A11Y_OK; re-running the same probe on it would double every finding it produces.
            local=page.evaluate(PROBE_A11Y) if st['patch'] else a11y
            if st['patch']:
                for issue in local['issues']:
                    findings.append(_bad('A11Y_'+issue['kind'].replace('-','_').upper(),where,
                                         f"{issue['kind']}: {issue['detail']}",
                                         'The delivered page must be usable by someone not using a mouse or a screen.'))
            prose=page.evaluate(PROBE_PROSE)
            for issue in prose['issues']:
                findings.append(_bad('A11Y_'+issue['kind'].replace('-','_').upper(),where,
                                     f"{issue['kind']}: {issue['detail']}",
                                     'A data table needs a real header row, a row header per row and named links; '
                                     'legible is not navigable.'))
            if not local['issues'] and not prose['issues']:
                findings.append(_ok('PROSE_OK',where,
                                    f"{local['counts']['headings']} headings ordered, "
                                    f"{local['counts']['controls']} controls named, "
                                    f"{prose['counts']['tables']} table(s) with scoped headers over "
                                    f"{prose['counts']['rows']} rows, {prose['counts']['links']} citations named."))
        if len(states)<2:
            findings.append(_bad('PROSE_SCOPE','states',
                                 'No focal state was measured, so this suite only saw the page it opened.',
                                 'focal_states reads the live model table; a checkpoint with an encoder/decoder '
                                 'split must appear there or its page reaches a reader unmeasured.'))
        else:
            findings.append(_ok('PROSE_SCOPE','states',
                                f"Measured {len(states)} states from a real file:// origin: {', '.join(checks['states'])}."))
        page.goto(artifact.resolve().as_uri(),wait_until='load')
        page.wait_for_function('window.STRATUM')
        page.evaluate('(p)=>STRATUM.setState(p)',{'model':'ds3','view':'model','stage':'block'})

        # Focus must be visible to someone navigating by keyboard.
        page.keyboard.press('Tab')
        focus=page.evaluate("""() => {const el=document.activeElement; if(!el||el===document.body) return null;
            const s=getComputedStyle(el); return {tag:el.tagName, outline:s.outlineStyle, width:s.outlineWidth, shadow:s.boxShadow};}""")
        if not focus:
            findings.append(_bad('A11Y_FOCUS','page','Tab from the top of the document reaches nothing focusable.','Give the page a keyboard entry point.'))
        elif focus['outline']=='none' and focus['shadow']=='none':
            findings.append(_bad('A11Y_FOCUS',focus['tag'],'First focused control shows no visible focus indicator.','Keyboard users must be able to see where they are.'))
        else:
            findings.append(_ok('A11Y_FOCUS',focus['tag'],'The first focusable control shows a visible focus indicator.'))

        # Print, reduced motion and OS colour preference: none of these has ever been emulated.
        page.emulate_media(media='print')
        printed=page.evaluate("""() => {const b=getComputedStyle(document.body);
            return {bg:b.backgroundColor, fg:b.color, hidden:!!document.querySelector('footer') && getComputedStyle(document.querySelector('footer')).display};}""")
        fg,bg=parse_color(printed['fg']),parse_color(printed['bg'])
        if fg and bg:
            ratio=contrast_ratio(fg[:3],bg[:3])
            if ratio<4.5:
                findings.append(_bad('PRINT_CONTRAST','print',f'Print body text measures {ratio:.2f}:1.','The print stylesheet forces its own colours; they must still be readable.'))
            else:
                findings.append(_ok('PRINT_CONTRAST','print',f'Print body text measures {ratio:.2f}:1.'))
        else:
            findings.append(Finding('PRINT_CONTRAST','presentations','BLOCKED','print','Print colours are not resolvable.','Set explicit print colours.'))
        page.emulate_media(media='screen',reduced_motion='reduce')
        motion=page.evaluate("""() => {const el=document.querySelector('.model-card')||document.body;
            const s=getComputedStyle(el); return {duration:s.transitionDuration, animation:s.animationDuration};}""")
        lively=[v for v in motion.values() if v and v not in ('0s','0s, 0s')]
        if lively:
            findings.append(_bad('MOTION_REDUCED','page',f'Transitions remain active under prefers-reduced-motion: {lively}.','Honour the preference the user has already expressed.'))
        else:
            findings.append(_ok('MOTION_REDUCED','page','Motion is suppressed under prefers-reduced-motion.'))
        page.emulate_media(media='screen',reduced_motion='no-preference')
        target=loads((root/'verification/manifests/observatory.json').read_text())
        record=visualization_record(page.evaluate(PROBE_VISUALIZATION),target)
        (root/'verification/reports/visualization-record.json').write_text(json.dumps(record,indent=2)+'\n')
        from .domains.contracts import validate as validate_domain
        problems=[f for f in validate_domain('visualization',record) if f.status!='PASS']
        for f in problems[:6]:
            findings.append(_bad('VISUAL_RECORD',f.code,f.message,f.remedy))
        if not problems:
            findings.append(_ok('VISUAL_RECORD','scene',
                                f"Measured visualization record over {len(record['nodes'])} nodes satisfies the scene contract."))
        page.goto(dedicated.resolve().as_uri(),wait_until='load')
        page.wait_for_function('window.STRATUM')
        findings.extend(judge_exported_svg(page.evaluate('STRATUM.exportSVG()'),policy))
        findings.extend(judge_scenario(json.loads(page.evaluate('STRATUM.exportConfig()')),root))
        page.keyboard.press('Tab')
        focus=page.evaluate("() => {const e=document.activeElement,s=getComputedStyle(e);return e!==document.body&&(s.outlineStyle!=='none'||s.boxShadow!=='none')}")
        findings.append((_ok if focus else _bad)('DEDICATED_FOCUS','dedicated','Dedicated page keyboard entry has visible focus.',*(() if focus else ('Restore visible keyboard focus.',))))
        page.emulate_media(media='print')
        colors=page.evaluate("() => {const s=getComputedStyle(document.body);return [s.color,s.backgroundColor]}")
        fg,bg=map(parse_color,colors)
        readable=bool(fg and bg and contrast_ratio(fg[:3],bg[:3])>=4.5)
        findings.append((_ok if readable else _bad)('DEDICATED_PRINT','dedicated','Dedicated page print contrast meets AA.',*(() if readable else ('Set readable print colors.',))))
        page.emulate_media(media='screen',reduced_motion='reduce')
        duration=page.locator('#atlas-link').evaluate("e=>getComputedStyle(e).transitionDuration")
        quiet=duration in ('0s','0s, 0s')
        findings.append((_ok if quiet else _bad)('DEDICATED_MOTION','dedicated','Dedicated navigation honors reduced motion.',*(() if quiet else ('Honor reduced motion.',))))
        # Exercise native disclosure and actual Chromium printing on every delivered page.
        from accounting_panel import exercise_page as exercise_accounting_page
        from .engineering import delivery_artifacts
        from playwright.sync_api import Error as BrowserError
        checks['accounting']={}
        for name in delivery_artifacts(root):
            accounting_page=browser.new_page(viewport={'width':1440,'height':1000})
            accounting_page.on('pageerror',lambda e:errors.append(str(e)))
            try:
                result=exercise_accounting_page(accounting_page,root/'dist'/name)
                checks['accounting'][name]=result
                findings.append(_ok('ACCOUNTING_DELIVERY',name,
                                    'Native keyboard disclosure, dynamic controls, 390px layout, offline delivery and actual print restoration passed.'))
            except (AssertionError,BrowserError) as exc:
                findings.append(_bad('ACCOUNTING_DELIVERY',name,str(exc),
                                     'Restore accessible disclosure, responsive notes, live accounting and printable assumptions on this entry point.'))
            finally:
                accounting_page.close()
        if errors:findings.append(_bad('DELIVERY_RUNTIME','all pages',str(errors),'Fix all delivered page errors.'))
        browser.close()

    report={'schema':'stratum-presentations/v1','status':aggregate(findings),
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'delivery':checks,'counts':dict(Counter(f.status for f in findings)),
            'findings':[f.json() for f in findings],
            'scope':'The delivered artifact as a reader receives it. Says nothing about upstream truth, and a legible page is not a correct one.'}
    return report


def markdown(report):
    lines=[f"# Presentations — {report['status']}",f"\nGenerated: {report['generated_at']}",
           f"\nArtifact `{report['artifact_sha256'][:16]}…`",
           '\nExports, delivered-page accessibility, print, portability and preview freshness.',
           '\n## Results','\n| Code | Status | Subject | Detail |','|---|---|---|---|']
    for f in report['findings']:
        lines.append(f"| {f['code']} | {f['status']} | {f['subject'][:28]} | {f['message'][:96]} |")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=p.parse_args(argv);root=Path(a.root).resolve()
    report=run(root)
    out=root/'verification/reports';out.mkdir(exist_ok=True)
    (out/'presentations.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'presentations.md').write_text(markdown(report))
    print(f"presentations: {report['status']} | {report['counts']}")
    return exit_code(report['status'])


if __name__=='__main__':raise SystemExit(main())
