"""Measured legibility, layout composition and visual-drift gate for the built artifact.

Engineering-class evidence, deliberately separate from the science track: every value here
is measured on our own bytes and our own renderer, so it needs no upstream source and no
signed review. DESIGN.md trust boundary 1 says local inputs are never evidence for
themselves -- an aesthetics result must never contribute to a scientific PASS.

Ratios follow WCAG 2.2 and are computed from specified colors. That is an upper bound on
legibility, not a guarantee of it: anti-aliasing and stem weight are not modelled.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,re,subprocess,sys
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from .model import Finding,aggregate,exit_code
from .evidence import digest,loads

POLICY_PATH='verification/baselines/aesthetics/policy.json'
BASELINE_PATH='verification/baselines/aesthetics/composition.json'
HEX=re.compile(r'^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$')
RGB=re.compile(r'^rgba?\(([^)]*)\)$')


# ---------------------------------------------------------------- colour arithmetic

def parse_color(value):
    """Return (r,g,b,alpha) for the CSS forms this renderer emits, or None if unpaintable."""
    if not isinstance(value,str):return None
    text=value.strip()
    if not text or text in ('none','transparent','currentColor') or text.startswith('url('):return None
    m=HEX.match(text)
    if m:
        h=m.group(1)
        if len(h)==3:h=''.join(c*2 for c in h)
        return (int(h[0:2],16),int(h[2:4],16),int(h[4:6],16),1.0)
    m=RGB.match(text)
    if m:
        parts=[p.strip() for p in m.group(1).replace('/',' ').split(',')] if ',' in m.group(1) else m.group(1).split()
        try:
            nums=[float(p.rstrip('%')) for p in parts[:4]]
        except ValueError:return None
        if len(nums)<3:return None
        alpha=nums[3] if len(nums)>3 else 1.0
        return (int(round(nums[0])),int(round(nums[1])),int(round(nums[2])),alpha)
    return None


def _linear(channel):
    c=channel/255
    return c/12.92 if c<=0.03928 else ((c+0.055)/1.055)**2.4


def relative_luminance(rgb):
    r,g,b=(_linear(c) for c in rgb[:3])
    return 0.2126*r+0.7152*g+0.0722*b


def contrast_ratio(a,b):
    la,lb=relative_luminance(a),relative_luminance(b)
    hi,lo=max(la,lb),min(la,lb)
    return (hi+0.05)/(lo+0.05)


def composite(layers,base):
    """Source-over composite of (rgb, alpha) layers onto an opaque base, back to front."""
    out=tuple(base[:3])
    for rgb,alpha in layers:
        alpha=max(0.0,min(1.0,float(alpha)))
        out=tuple(int(round(rgb[i]*alpha+out[i]*(1-alpha))) for i in range(3))
    return out


def theme_tokens(stylesheet):
    """Custom properties per theme, read from :root and :root[data-theme=light]."""
    css=Path(stylesheet).read_text()
    def block(pattern):
        m=re.search(pattern+r'\{([^}]*)\}',css,re.S)
        return dict(re.findall(r'(--[\w-]+):\s*([^;]+)',m.group(1))) if m else {}
    return {'dark':{k:v.strip() for k,v in block(r':root').items()},
            'light':{k:v.strip() for k,v in block(r':root\[data-theme=light\]').items()}}


# ---------------------------------------------------------------- judgement

def required_ratio(px,weight,contrast_policy,surface='chrome'):
    """WCAG thresholds applied to the rendered size rather than the declared one.

    Large text gets the documented relaxation. Reading-size *chrome* prose is held to the
    enhanced (AAA) ratio, because that is the text people read for comprehension.

    Canvas labels stay at AA deliberately. They sit on semantic tint fills -- amber for
    retained state, violet for communication (docs/design/OBSERVATORY.md) -- and forcing 7:1
    there would wash that grammar out of the diagram. Their AAA clearance is reported as a
    metric instead of enforced as a gate.
    """
    large=px>=contrast_policy['large_text_px'] or (px>=contrast_policy['large_bold_px'] and weight>=contrast_policy['bold_weight'])
    if large:return contrast_policy['text_large']
    if surface!='svg' and px>=contrast_policy.get('reading_px',1e9):return contrast_policy['text_enhanced']
    return contrast_policy['text_normal']


def judge_contrast(sample,policy):
    """One measured element becomes one finding. An unresolvable backdrop blocks, never passes."""
    contrast=policy['contrast']
    subject=sample.get('id','<unknown>')
    fg=parse_color(sample.get('fill'))
    if fg is None:
        return Finding('CONTRAST_INDETERMINATE','aesthetics','BLOCKED',subject,
                       'Foreground paint is not a resolvable colour.','Give the element an explicit colour or exclude it from the gate.')
    stack=sample.get('backdrop')
    if not stack:
        return Finding('CONTRAST_INDETERMINATE','aesthetics','BLOCKED',subject,
                       'No opaque backdrop fully contains this element, so its contrast cannot be resolved.',
                       'Paint an opaque backdrop behind it, or record why the ratio is indeterminate.')
    layers=[]
    for layer in stack:
        colour=parse_color(layer.get('color'))
        if colour is None:
            return Finding('CONTRAST_INDETERMINATE','aesthetics','BLOCKED',subject,
                           'A backdrop layer is not a resolvable colour.','Resolve the backdrop or exclude the element.')
        layers.append((colour[:3],colour[3]*float(layer.get('alpha',1.0))))
    backdrop=composite(layers[1:],layers[0][0]) if layers else None
    foreground=composite([(fg[:3],fg[3]*float(sample.get('alpha',1.0)))],backdrop)
    ratio=contrast_ratio(foreground,backdrop)
    px=float(sample.get('px',0));weight=int(sample.get('weight',400) or 400)
    needed=contrast['graphics'] if sample.get('kind')=='graphics' else required_ratio(px,weight,contrast,sample.get('surface','chrome'))
    detail=f'{ratio:.2f}:1 against {needed:.1f}:1 at {px:.1f}px weight {weight}'
    if ratio+1e-9 < needed:
        return Finding('CONTRAST_BELOW_POLICY','aesthetics','FAIL',subject,f'Contrast {detail}.',
                       'Adjust the token lightness or enlarge the text; do not lower the policy threshold.')
    return Finding('CONTRAST_OK','aesthetics','PASS',subject,f'Contrast {detail}.')


def judge_typography(sample,policy):
    floor=policy['typography']['min_svg_px'] if sample.get('surface')=='svg' else policy['typography']['min_chrome_px']
    px=float(sample.get('px',0))
    subject=sample.get('id','<unknown>')
    if px+1e-9 < floor:
        return Finding('TYPOGRAPHY_BELOW_FLOOR','aesthetics','FAIL',subject,
                       f'Rendered text is {px:.2f}px against a {floor:.1f}px floor.',
                       'Expand the lane or box. Shrinking type to fit is the failure this floor exists to catch.')
    return Finding('TYPOGRAPHY_OK','aesthetics','PASS',subject,f'{px:.2f}px at or above the {floor:.1f}px floor.')


def _gap(a,b):
    """Separation between two boxes; negative when they intersect."""
    dx=max(a['x'],b['x'])-min(a['x']+a['w'],b['x']+b['w'])
    dy=max(a['y'],b['y'])-min(a['y']+a['h'],b['y']+b['h'])
    return max(dx,dy)


def judge_clearance(boxes,policy):
    """Require separation between independent labels, not merely absence of intersection.

    Two lines of one label are deliberately tight -- that is leading, not a collision -- so
    only text belonging to different owners is compared.
    """
    floor=policy['layout']['min_text_clearance_px'];out=[]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if a.get('owner') is not None and a.get('owner')==b.get('owner'):continue
            gap=_gap(a,b)
            if gap+1e-9 < floor:
                out.append(Finding('AESTHETIC_CLEARANCE','aesthetics','FAIL',f"{a['id']}|{b['id']}",
                                   f'Only {gap:.2f}px separates these labels; the floor is {floor:.1f}px.',
                                   'Move the labels apart or widen the lane.'))
    return out or [Finding('CLEARANCE_OK','aesthetics','PASS','labels',f'All label pairs clear {floor:.1f}px.')]


def _segments(edge):
    pts=edge.get('points',[])
    return [(pts[i],pts[i+1]) for i in range(len(pts)-1)]


def _crosses(p,q):
    """True when two axis-aligned segments properly cross."""
    (ax1,ay1),(ax2,ay2)=p;(bx1,by1),(bx2,by2)=q
    a_horizontal=ay1==ay2;b_horizontal=by1==by2
    if a_horizontal==b_horizontal:return False
    # Normalise so a is the vertical segment and b the horizontal one.
    if a_horizontal:(ax1,ay1),(ax2,ay2),(bx1,by1),(bx2,by2)=(bx1,by1),(bx2,by2),(ax1,ay1),(ax2,ay2)
    lo_y,hi_y=sorted((ay1,ay2));lo_x,hi_x=sorted((bx1,bx2))
    return lo_x<ax1<hi_x and lo_y<by1<hi_y


def count_crossings(edges):
    total=0
    for i,a in enumerate(edges):
        for b in edges[i+1:]:
            for sa in _segments(a):
                for sb in _segments(b):
                    if _crosses(sa,sb):total+=1
    return total


# ---------------------------------------------------------------- composition baseline

def composition_signature(scene,tokens):
    """A drift signature that tracks structure and semantic role, not exact pixels or hexes.

    Coordinates are rounded so sub-pixel jitter is invisible, and colours are named by the
    token that produced them, so retuning the palette does not churn the baseline while a
    label silently changing role does.
    """
    reverse={str(v).lower():k for k,v in (tokens or {}).items()}
    def role(value):
        return reverse.get(str(value).lower(),value)
    def box(n):return [round(float(n.get(k,0))) for k in ('x','y','w','h')]
    nodes=sorted([[n.get('id'),box(n),role(n.get('tone')),n.get('kind'),n.get('name'),n.get('sub')] for n in scene.get('nodes',[])],key=lambda r:str(r[0]))
    groups=sorted([[g.get('id'),box(g),g.get('title'),g.get('subtitle')] for g in scene.get('groups',[])],key=lambda r:str(r[0]))
    edges=sorted([[e.get('source'),e.get('target'),e.get('kind'),bool(e.get('delay')),
                   [[round(float(x)),round(float(y))] for x,y in e.get('points',[])]] for e in scene.get('edges',[])],key=lambda r:json.dumps(r,sort_keys=True))
    labels=sorted([[l.get('text'),round(float(l.get('x',0))),round(float(l.get('y',0))),role(l.get('tone'))] for l in scene.get('labels',[])],key=lambda r:json.dumps(r,sort_keys=True))
    return digest({'nodes':nodes,'groups':groups,'edges':edges,'labels':labels})


# ---------------------------------------------------------------- stylesheet scales

@dataclass(frozen=True)
class Declaration:
    selector: str
    prop: str
    value: str
    canvas: bool


SPACING_PROPS=('margin','padding','gap','row-gap','column-gap')


def stylesheet_declarations(path):
    """Every declaration in the stylesheet, tagged as canvas (inside the SVG scene) or chrome.

    The innermost-rule pattern cannot cross a brace, so rules nested in @media are attributed
    to their own selector rather than to the media query.
    """
    css=re.sub(r'/\*.*?\*/','',Path(path).read_text(),flags=re.S)
    out=[]
    for selector,body in re.findall(r'([^{}]+)\{([^{}]*)\}',css):
        selector=' '.join(selector.split())
        if selector.startswith('@'):continue
        canvas='.schematic' in selector
        for prop,value in re.findall(r'([a-z-]+)\s*:\s*([^;]+)',body):
            out.append(Declaration(selector,prop.strip(),value.strip(),canvas))
    return out


def _px(token):
    m=re.fullmatch(r'(-?\d*\.?\d+)px',token)
    return float(m.group(1)) if m else None


def judge_type_scale(stylesheet,policy):
    """Every font-size must belong to a declared scale.

    Chrome and canvas have separate scales: canvas sizes are geometry-locked to authored node
    boxes across 544 layers, so they are frozen and enumerated rather than designed.
    """
    typography=policy['typography']
    chrome=set(typography['chrome_scale']);canvas=set(typography['canvas_scale']);out=[]
    for d in stylesheet_declarations(stylesheet):
        # `font:` shorthand counts too; two of the original defects were declared that way.
        if d.prop=='font-size':size=_px(d.value)
        elif d.prop=='font':
            head=re.match(r'(-?\d*\.?\d+px)',d.value.strip())
            size=_px(head.group(1)) if head else None
        else:continue
        if size is None:continue  # 0, inherit and keyword fonts are not scale members
        allowed=canvas if d.canvas else chrome
        if size not in allowed:
            surface='canvas' if d.canvas else 'chrome'
            out.append(Finding('TYPE_OFF_SCALE','aesthetics','FAIL',f'{d.selector} {{font-size}}',
                               f'{size}px is not in the {surface} type scale {sorted(allowed)}.',
                               'Use a scale step. If the canvas genuinely needs a new size, re-author the node boxes and re-verify the geometry suite.'))
    return out or [Finding('TYPE_SCALE_OK','aesthetics','PASS','stylesheet','Every font-size belongs to its declared scale.')]


def judge_spacing_scale(stylesheet,policy):
    """Margins, padding and gaps come from one spacing scale."""
    allowed=set(policy['spacing']['scale']);out=[]
    for d in stylesheet_declarations(stylesheet):
        if not (d.prop in SPACING_PROPS or d.prop.startswith('margin-') or d.prop.startswith('padding-')):continue
        for token in d.value.split():
            size=0.0 if token=='0' else _px(token)
            if size is None:continue  # auto, percentages and calc() are not scale members
            if abs(size) not in allowed:
                out.append(Finding('SPACING_OFF_SCALE','aesthetics','FAIL',f'{d.selector} {{{d.prop}}}',
                                   f'{token} is not in the spacing scale {sorted(allowed)}.',
                                   'Use a scale step so spacing reads as rhythm rather than drift.'))
    return out or [Finding('SPACING_SCALE_OK','aesthetics','PASS','stylesheet','Every margin, padding and gap belongs to the spacing scale.')]


# ---------------------------------------------------------------- in-page measurement

#: The chrome text the probe measures. A list, not a string in the middle of the probe, because
#: two independent checks now bind it: `judge_chrome_probe` requires every class token here to
#: still exist in the stylesheet *and* in the markup that emits it, and `judge_chrome_coverage`
#: requires every reading-size chrome rule in the stylesheet to be either listed here or recorded
#: in `UNMEASURED_CHROME`. Without those, renaming a class silently removes it from measurement
#: while the gate still reports PASS -- which is the failure this list was added to fix, one layer
#: up from the prose it was added for.
CHROME_TEXT_SELECTORS=('.worked-example .worked-lead','.worked-caption','.worked-number','.worked-path-title','.worked-block-label','.worked-query-label','.worked-pair-label','.worked-token-label','.worked-result span','.worked-block-score','.worked-dictionary-row span','.worked-audit summary','.chapter-notes>summary','.worked-table th','.worked-table td','.worked-table code','.worked-table caption','.chart text','.metric-note','.chart-caption','.source-role','.status-pill',
                       '.fact dt','.lens-tab .idx','.innovation-note','.innovation-table th',
                       '.innovation-table td','.innovation-table a','.accounting-assumptions summary',
                       '.accounting-assumptions h4','.accounting-assumptions p','.accounting-citations a','.pytorch-section h4','.pytorch-section p','.pytorch-example summary','.pytorch-code','.worked-code summary','.pytorch-license summary','.pytorch-code span','.code-language')

#: Reading-size chrome rules that the probe does **not** measure. This is recorded debt, not an
#: exemption on principle: each entry is chrome text at or above `reading_px` whose contrast is
#: therefore unverified by this gate. Eight of these measure between 4.89:1 and 6.87:1 when probed
#: by hand, against the 7:1 the policy sets for reading-size prose -- so the list is a statement
#: that the gate's coverage is incomplete, and shrinking it is real work rather than bookkeeping.
#: It is inside `toolchain_digest`, so adding a line here invalidates every engineering receipt.
UNMEASURED_CHROME=(
    '#audit-output','#toast','#search-input',
    '.analysis-copy p','.analysis-copy pre','.analysis-copy h4','.analysis-header h3',
    '.button','.button.compact','.evidence-footer p','.formula','.identity-sub',
    '.identity-metric strong','.inspector p','.inspector h3','.lens-tab','.model-card .variant',
    '.principle p','.search-head','.search-head h2','.search-result span','.section-identity h2',
    '.verifier-card h4','.brand','.icon-button','.metric-value',
    '.command-button span:after','.workspace-actions #focus-button:before',
    '.workspace-actions #export-svg:before','.workspace-actions #trace-button:before',
    ':root','h1','h2',
)

CLASS_TOKEN=re.compile(r'\.([A-Za-z][\w-]*)')


def judge_chrome_probe(stylesheet,markup,selectors=CHROME_TEXT_SELECTORS):
    """Every class the probe measures by must still exist in the stylesheet and in the markup.

    The failure this closes: rename `.innovation-note` in `src/styles.css` and `src/app.mjs`, and
    `querySelectorAll` quietly matches nothing. Every measurement for that element disappears and
    the gate reports PASS over a page it no longer looks at. A selector list is only evidence if
    something ties it to the names the page actually uses.
    """
    css=Path(stylesheet).read_text();rendered='\n'.join(Path(m).read_text() for m in markup)
    out=[]
    for selector in selectors:
        for token in CLASS_TOKEN.findall(selector):
            where=[name for name,text in (('src/styles.css',css),('markup',rendered))
                   if not re.search(rf'[.\'"\s]{re.escape(token)}\b',text)]
            if where:
                out.append(Finding('CHROME_PROBE_STALE','aesthetics','FAIL',selector,
                                   f'The probe measures by .{token}, which no longer appears in '
                                   f'{" and ".join(where)}.',
                                   'A renamed class makes querySelectorAll match nothing and the '
                                   'measurement silently vanish. Update CHROME_TEXT_SELECTORS with '
                                   'the rename, or remove the entry deliberately.'))
    return out or [Finding('CHROME_PROBE_LIVE','aesthetics','PASS','stylesheet',
                           f'All {len(selectors)} measured chrome selectors still name classes the '
                           f'stylesheet defines and the markup emits.')]


def judge_chrome_coverage(stylesheet,policy,selectors=CHROME_TEXT_SELECTORS,
                          recorded=UNMEASURED_CHROME):
    """Reading-size chrome text is measured, or it is recorded as unmeasured. Never neither.

    Scoped to the band where the AAA rule actually bites: `reading_px` up to `large_text_px`,
    above which WCAG's large-text relaxation applies and the text is a heading rather than prose.
    Static, so it cannot introduce a BLOCKED outcome on a gate the Pages deploy requires.

    Known limitation, stated rather than hidden: text that reaches reading size by *inheritance*
    declares no font-size of its own, so this attributes it to the ancestor rule that sets the
    size. It catches a new rule and a renamed rule, which is what it is for.
    """
    lo,hi=policy['contrast']['reading_px'],policy['contrast']['large_text_px']
    probed=set(selectors);known=set(recorded);out=[];checked=0
    for d in stylesheet_declarations(stylesheet):
        if d.canvas:continue
        if d.prop=='font-size':size=_px(d.value)
        elif d.prop=='font':
            head=re.match(r'(-?\d*\.?\d+px)',d.value.strip());size=_px(head.group(1)) if head else None
        else:continue
        if size is None or not (lo<=size<hi):continue
        for part in (x.strip() for x in d.selector.split(',')):
            if not part:continue
            checked+=1
            if part in probed or part in known:continue
            if any(part.startswith(p+' ') or part.startswith(p+':') for p in probed):continue
            out.append(Finding('CHROME_UNMEASURED','aesthetics','FAIL',f'{part} {{{d.prop}}}',
                               f'{size:g}px chrome text is neither measured by the probe nor '
                               f'recorded as unmeasured, so its contrast is unverified and nothing '
                               f'says so.',
                               'Add the selector to CHROME_TEXT_SELECTORS so it is measured, or to '
                               'UNMEASURED_CHROME to record that it is not. Do not leave '
                               'reading-size prose in neither list.'))
    return out or [Finding('CHROME_COVERAGE_OK','aesthetics','PASS','stylesheet',
                           f'All {checked} reading-size chrome rules between {lo:g}px and {hi:g}px '
                           f'are either measured ({len(probed)} selectors) or recorded as '
                           f'unmeasured ({len(known)} selectors).')]


PROBE=r"""
() => {
  const CHROME_TEXT = __CHROME_TEXT__;
  const rgba = el => getComputedStyle(el);
  const chain = el => { let a = 1; for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const o = parseFloat(getComputedStyle(n).opacity); if (!Number.isNaN(o)) a *= o; } return a; };
  const svg = document.querySelector('#scene-mount svg');
  const out = { texts: [], boxes: [], tokens: {}, theme: document.documentElement.dataset.theme || 'dark' };
  const rootStyle = getComputedStyle(document.documentElement);
  for (const name of Array.from(document.styleSheets).flatMap(s => { try { return Array.from(s.cssRules); } catch { return []; } })
        .flatMap(r => r.style ? Array.from(r.style) : []).filter(p => p.startsWith('--')))
    out.tokens[name] = rootStyle.getPropertyValue(name).trim();

  // An SVG text node's backdrop is every painted rect earlier in document order that
  // fully contains it. Partial coverage is deliberately left unresolved.
  const painted = [];
  if (svg) {
    const ctm = svg.getScreenCTM();
    const toScreen = el => { const b = el.getBBox(); const m = el.getScreenCTM();
      const pt = (x, y) => { const p = svg.createSVGPoint(); p.x = x; p.y = y; return p.matrixTransform(m); };
      const a = pt(b.x, b.y), c = pt(b.x + b.width, b.y + b.height);
      return { x: Math.min(a.x, c.x), y: Math.min(a.y, c.y), w: Math.abs(c.x - a.x), h: Math.abs(c.y - a.y) }; };
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    const all = Array.from(svg.querySelectorAll('*'));
    for (const el of all) {
      if (!(el instanceof SVGGraphicsElement)) continue;
      const st = rgba(el);
      if (el.tagName === 'text') {
        let box; try { box = toScreen(el); } catch { continue; }
        const stack = [{ color: bodyBg, alpha: 1 }];
        for (const p of painted) if (p.box.x <= box.x + 0.5 && p.box.y <= box.y + 0.5 &&
              p.box.x + p.box.w >= box.x + box.w - 0.5 && p.box.y + p.box.h >= box.y + box.h - 0.5)
          stack.push({ color: p.fill, alpha: p.alpha });
        const owner = el.closest('[data-node]');
        out.texts.push({ surface: 'svg', id: (owner && owner.dataset.node) || el.textContent.slice(0, 40),
          fill: st.fill, alpha: chain(el), px: parseFloat(st.fontSize) * (ctm ? ctm.a : 1),
          weight: parseInt(st.fontWeight, 10) || 400, backdrop: stack, text: el.textContent.slice(0, 60) });
        out.boxes.push({ id: el.textContent.slice(0, 24) || 'text', owner: (owner && owner.dataset.node) || null, ...box });
      } else if (el.tagName === 'rect' || el.tagName === 'path') {
        const fill = st.fill;
        if (fill && fill !== 'none' && !fill.startsWith('url(')) {
          let box; try { box = toScreen(el); } catch { continue; }
          painted.push({ box, fill, alpha: chain(el) });
        }
      }
    }
  }
  // Chrome text lives outside #schematic and is invisible to geometryReport().
  // The innovation coverage is the one place this page runs reading-size prose, so its cells and
  // citations are measured here rather than left to the eye; at >= reading_px they are held to AAA.
  for (const el of document.querySelectorAll(CHROME_TEXT)) {
    const st = rgba(el); const box = el.getBoundingClientRect();
    if (!box.width || !box.height) continue;
    let bg = null;
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const c = getComputedStyle(n).backgroundColor;
      if (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') { bg = c; break; }
    }
    out.texts.push({ surface: 'chrome', accounting: !!el.closest('.accounting-assumptions'), id: el.className && el.className.baseVal !== undefined ? el.className.baseVal : (el.className || el.tagName),
      fill: st.color, alpha: chain(el),
      px: parseFloat(st.fontSize), weight: parseInt(st.fontWeight, 10) || 400,
      backdrop: bg ? [{ color: bg, alpha: 1 }] : null, text: (el.textContent || '').slice(0, 60) });
  }
  // "Core labels cannot ellipsize" -- OBSERVATORY.md.
  out.truncated = Array.from(document.querySelectorAll('.model-card .variant, .model-card .motif, .comparison-table td, .comparison-table th, .search-result small'))
    .filter(el => el.scrollWidth > el.clientWidth + 1)
    .map(el => ({ id: el.className || el.tagName, text: (el.textContent || '').slice(0, 60) }));
  return out;
}
"""


PROBE=PROBE.replace('__CHROME_TEXT__',json.dumps(', '.join(CHROME_TEXT_SELECTORS)))

LAYER_SAMPLE=(1,2,4)


def scenario_inventory(root):
    """The application's own model table, so the sweep cannot drift from what ships.

    Read from the same Node inventory the science gate uses rather than a literal; the
    previous code probed a `window.__models` global that has never existed, so it always
    fell through to a hardcoded list and would not have noticed a ninth checkpoint.

    The lens axis returns here too. `run()` held its own seven-element view literal while calling
    this function for the models -- and `cases()` already takes `views` as a parameter precisely
    so it need not. An eighth lens would have been contrast-checked in no state, baselined under
    no label, and `COMPOSITION_STABLE` would have reported a compared count that simply got
    smaller relative to the page.
    """
    inventory=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,
                                   capture_output=True,check=True,timeout=60).stdout)
    return inventory['models'],inventory['samples']['focal_stages'],inventory['fields']['VIEWS']


def scenario_models(root):
    """The model table alone, for a caller that does not need the focal-stage or lens lists."""
    return scenario_inventory(root)[0]


def cases(models,views,themes,focal):
    """Every model in every lens in both themes, plus the focal scenes each model actually has.

    Layer 4 alone left the first layer (often dense, or the one that establishes the residual
    stream) and the recurrent layers unmeasured.

    `focal` is `core.focalStages` as the exporter publishes it, not re-derived here. This file
    used to re-implement the predicate from `vision`/`residual`/`pleLayers`/`encoderLayers`, which
    was the fifth copy of it; the copies in `app.mjs` and `diagrams.mjs` gated the N-gram page on
    a model id instead, so the two forms agreed only by today's coincidence.
    """
    out=[]
    for model,meta in models.items():
        for view in views:
            layers=LAYER_SAMPLE if view=='model' else (4,)
            for theme in themes:
                for layer in layers:
                    out.append({'model':model,'view':view,'theme':theme,'stage':'block','layer':layer,'phase':'decode'})
        if meta.get('encoderLayers'):
            for theme in themes:
                out.append({'model':model,'view':'serving','theme':theme,'stage':'block','layer':4,'phase':'prefill'})
        for theme in themes:
            for stage in focal.get(model,()):
                out.append({'model':model,'view':'model','theme':theme,'stage':stage,'layer':4})
    return out


def sample_key(t):
    """Collapse identical measurements so one real defect is one finding, not thousands."""
    backdrop=tuple((b.get('color'),round(float(b.get('alpha',1)),2)) for b in (t.get('backdrop') or []))
    return (t.get('surface'),t.get('fill'),round(float(t.get('alpha',1)),2),round(float(t.get('px',0)),1),
            int(t.get('weight',400) or 400),backdrop)


def judge_composition(current,recorded):
    """Compare the measured composition against the baseline, and report what was actually compared.

    The previous form asked `if label in recorded` for drift and then
    `all(recorded.get(l,s)==s for l,s in current.items())` for stability -- vacuously true for
    any label the baseline has never seen. So `COMPOSITION_STABLE` reported `len(current)`
    scenarios as "matching the recorded baseline" whether or not they had been compared, and
    `current - recorded` was covered by nothing at all. That is how a ninth checkpoint's 22
    signatures reached a PASS line with no drift protection, and how `VERIFICATION.md` came to
    quote the baselined count as though it were the measured one. Fixing the baseline left the
    mechanism intact; this function is the mechanism.

    A measured scenario with no recorded signature is FAIL -- the mirror of
    `COMPOSITION_MISSING`, and for the same reason: the baseline is stale against our own bytes,
    which is something we can know rather than something we cannot (ruling P1). The remedy is a
    deliberate `--record` and promotion, never a silent pass. And `COMPOSITION_STABLE` counts the
    compared set, so its figure cannot again be larger than the comparison behind it.
    """
    out=[];compared=sorted(recorded.keys()&current.keys())
    for label in compared:
        if recorded[label]!=current[label]:
            out.append(Finding('COMPOSITION_DRIFT','aesthetics','FAIL',label,
                               'Layout composition changed against the recorded baseline.',
                               'Review the change, then re-record deliberately with --record.'))
    for label in sorted(recorded.keys()-current.keys()):
        out.append(Finding('COMPOSITION_MISSING','aesthetics','FAIL',label,
                           'A baselined scenario no longer renders.',
                           'Restore the scenario or re-record the baseline.'))
    for label in sorted(current.keys()-recorded.keys()):
        out.append(Finding('COMPOSITION_NEW','aesthetics','FAIL',label,
                           'A measured scenario has no recorded signature, so nothing compares it '
                           'and no drift in it can be detected.',
                           'Re-record with --record and promote it only after checking both halves '
                           'of a promotion: that every new signature belongs to a genuinely new '
                           'scenario, and that no existing signature moved. Unbaselined is '
                           'unprotected, not stable.'))
    if out:return out
    if not compared:
        return [Finding('COMPOSITION_EMPTY','aesthetics','BLOCKED','baseline',
                        'Zero scenarios were compared against the recorded baseline, so there is '
                        'nothing here for a PASS to mean.',
                        'A comparison count of zero is not stability. Confirm the scenario '
                        'inventory and the recorded baseline are both non-empty before trusting '
                        'this gate.')]
    return [Finding('COMPOSITION_STABLE','aesthetics','PASS','baseline',
                    f'{len(compared)} measured scenarios were each compared against a recorded '
                    f'signature and all match; the measured, compared and baselined counts are '
                    f'the same number.')]


def run(root,record=False):
    sys.path.insert(0,str(root/'tests'))
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    policy=loads((root/POLICY_PATH).read_text())
    models,focal,views=scenario_inventory(root)
    artifact=root/'dist/foundations.html'
    findings=[];signatures={};truncations=[];crossing_report={}
    unique={}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chromium_path(p),headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1840,'height':1400},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(artifact.read_text(),wait_until='load')
        page.wait_for_function('window.STRATUM?.version === "3.0.0"')
        page.evaluate('document.fonts.ready')
        for case in cases(models,views,policy['themes'],focal):
            from .delivery import page_for_model,artifact_name
            dedicated=page_for_model(root,case['model'],case['view'])
            if page.evaluate('document.body.dataset.pageId || null')!=(dedicated['id'] if dedicated else None):
                page.evaluate('delete window.STRATUM')
                page.set_content((root/'dist'/artifact_name(root,case['model'],case['view'])).read_text(),wait_until='load')
                page.wait_for_function('window.STRATUM')
            page.evaluate('(p)=>STRATUM.setState(p)',case)
            # Closed native details hide the prose from rendered measurement. Measure the
            # expanded reader state for every model, lens and theme in the live matrix.
            accounting_expected=page.evaluate("""() => {
                const notes=document.querySelector('details.accounting-assumptions');
                if(!notes)return 0;
                notes.open=true;
                return notes.querySelectorAll('summary,h4,p,.accounting-citations a').length;
            }""")
            page.evaluate("() => document.querySelectorAll('.pytorch-example,.pytorch-license,.worked-code,.worked-audit,.chapter-notes').forEach(el=>el.open=true)")
            data=page.evaluate(PROBE)
            measured=sum(1 for text in data['texts'] if text.get('accounting'))
            if not accounting_expected or measured!=accounting_expected:
                findings.append(Finding('ACCOUNTING_TEXT_MEASURED','aesthetics','FAIL',str(case),
                                        f'{measured} accounting text elements measured against {accounting_expected} rendered elements.',
                                        'Expand the disclosure and measure every heading, paragraph, summary and citation.'))
            scenario=f"{case['model']}/{case['view']}/{case['stage']}/{case['layer']}"
            if case.get("phase")=="prefill":scenario+="/prefill"
            label=f"{scenario}/{case['theme']}"
            for t in data['texts']:
                key=sample_key(t)
                entry=unique.setdefault(key,{'sample':t,'count':0,'cases':set()})
                entry['count']+=1;entry['cases'].add(label)
            boxes=data['boxes']
            for f in judge_clearance(boxes,policy):
                findings.append(Finding(f.code,f.domain,f.status,f'{label}:{f.subject}',f.message,f.remedy))
            scene=page.evaluate('STRATUM.getScene()')
            crossings=count_crossings(scene.get('edges',[]))
            crossing_report[label]=crossings
            if crossings>policy['layout']['max_edge_crossings']:
                findings.append(Finding('AESTHETIC_CROSSINGS','aesthetics','FAIL',label,
                                        f'{crossings} edge crossings exceed the budget of {policy["layout"]["max_edge_crossings"]}.',
                                        'Reroute through reserved lanes rather than accepting a denser tangle.'))
            signatures[label]=composition_signature(scene,data['tokens'])
            for item in data.get('truncated',[]):truncations.append({'case':label,**item})
        browser.close()
    findings.extend(judge_type_scale(root/'src/styles.css',policy))
    findings.extend(judge_spacing_scale(root/'src/styles.css',policy))
    findings.extend(judge_chrome_probe(root/'src/styles.css',(root/'src/app.mjs',root/'src/worked-examples.mjs',root/'index.html')))
    findings.extend(judge_chrome_coverage(root/'src/styles.css',policy))
    if errors:
        findings.append(Finding('AESTHETIC_RUNTIME','aesthetics','FAIL','page',f'{len(errors)} runtime errors while measuring: {errors[0][:200]}','Fix the page error; measurements taken during an exception are not trustworthy.'))
    for key,entry in sorted(unique.items(),key=lambda kv:str(kv[0])):
        sample=dict(entry['sample']);sample['id']=f"{sample.get('id')} ({entry['count']} occurrences)"
        findings.append(judge_contrast(sample,policy))
        findings.append(judge_typography(sample,policy))
    for item in truncations:
        findings.append(Finding('AESTHETIC_TRUNCATION','aesthetics','FAIL',f"{item['case']}:{item['id']}",
                                f"Label is clipped: {item['text']!r}.",'Core labels cannot ellipsize; widen the column or shorten the source string.'))
    # Position must not depend on theme.
    by_scenario={}
    for label,sig in signatures.items():
        scenario,theme=label.rsplit('/',1);by_scenario.setdefault(scenario,{})[theme]=sig
    for scenario,per_theme in sorted(by_scenario.items()):
        if len(set(per_theme.values()))>1:
            findings.append(Finding('THEME_COMPOSITION_DRIFT','aesthetics','FAIL',scenario,
                                    'Composition differs between themes; a theme may change colour, not position.',
                                    'Move the theme-dependent value into a colour token.'))
    baseline_file=root/BASELINE_PATH
    current={k:v for k,v in sorted(signatures.items())}
    if record:
        candidate=baseline_file.with_name('composition.candidate.json')
        candidate.write_text(json.dumps({'schema':'stratum-aesthetics-baseline/v1','composition':current},indent=2)+'\n')
        findings.append(Finding('BASELINE_RECORDED','aesthetics','BLOCKED','baseline',
                                f'Wrote {candidate.relative_to(root)} for review.',
                                'Promote it deliberately; a new signature is not an approval.'))
    elif baseline_file.exists():
        findings.extend(judge_composition(current,loads(baseline_file.read_text()).get('composition',{})))
    else:
        findings.append(Finding('BASELINE_ABSENT','aesthetics','BLOCKED','baseline',
                                'No recorded composition baseline to compare against.','Record one with --record and promote it deliberately.'))
    report={'schema':'stratum-aesthetics/v1','status':aggregate(findings),
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'policy_sha256':hashlib.sha256((root/POLICY_PATH).read_bytes()).hexdigest(),
            'scenarios':len(signatures),'distinct_measurements':len(unique),
            # The axes this run actually swept, so a reader -- and
            # `test_aesthetics.CoverageTests` -- can reconcile the gate's coverage against the
            # application instead of inferring it from a scenario count. A count of 180 looks
            # identical whether it covers seven lenses or six, which is how a literal view list
            # sat beside a derived model list here without anything noticing.
            'axes':{'models':sorted(models),'views':list(views),
                    'stages':sorted({c['stage'] for c in cases(models,views,policy['themes'],focal)}),
                    'themes':list(policy['themes'])},
            'edge_crossings':{'max':max(crossing_report.values()) if crossing_report else 0,'budget':policy['layout']['max_edge_crossings']},
            'counts':{},'findings':[f.json() for f in findings],
            'scope':'Measured on this artifact in one Chromium build. WCAG ratios use specified colours and are an upper bound on legibility; they are not a substitute for a person reading the page.'}
    from collections import Counter
    report['counts']=dict(Counter(f.status for f in findings))
    return report


def markdown(report):
    lines=[f"# Aesthetics — {report['status']}",f"\nGenerated: {report['generated_at']}",
           f"\nArtifact `{report['artifact_sha256'][:16]}…`, policy `{report['policy_sha256'][:16]}…`",
           f"\n{report['scenarios']} scenarios, {report['distinct_measurements']} distinct measurements.",
           '\nMeasured legibility and composition only. A passing gate is not a claim that the page is well designed.',
           '\n## Findings by failure mode','\n| Code | Status | Count |','|---|---|---:|']
    from collections import Counter
    groups=Counter((f['code'],f['status']) for f in report['findings'])
    lines+=[f'| {c} | {s} | {n} |' for (c,s),n in sorted(groups.items())]
    lines+=['\n## First unresolved findings']
    for f in [x for x in report['findings'] if x['status'] not in ('PASS',)][:24]:
        lines+=[f"\n**{f['code']} / {f['subject']}** — {f['message']} {f['remedy']}"]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',action='store_true',help='write a candidate composition baseline for deliberate promotion')
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=p.parse_args(argv);root=Path(a.root).resolve()
    report=run(root,record=a.record)
    out=root/'verification/reports';out.mkdir(exist_ok=True)
    (out/'aesthetics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'aesthetics.md').write_text(markdown(report))
    print(f"aesthetics: {report['status']} | {report['scenarios']} scenarios | {report['counts']}")
    return exit_code(report['status'])


if __name__=='__main__':raise SystemExit(main())
