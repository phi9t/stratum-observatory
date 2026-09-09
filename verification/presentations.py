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
    return [_ok('PREVIEW_FRESH','previews',f"{len(data.get('captures',[]))} captures match the built artifact.")]


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


def run(root):
    root=Path(root)
    sys.path.insert(0,str(root/'tests'))
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    policy=load_policy(root)
    artifact=root/'dist/index.html'
    findings=list(judge_artifact_source(artifact,policy))
    findings.extend(judge_preview_freshness(root))
    checks={}
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
