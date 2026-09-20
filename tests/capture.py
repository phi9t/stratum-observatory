"""Capture current portable application and export its own SVGs; no model execution."""
from pathlib import Path
import hashlib,json,os,sys
from playwright.sync_api import sync_playwright
from browser_env import chromium_path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification.delivery import artifact_name
OUT=ROOT/'previews';OUT.mkdir(exist_ok=True)
BASE={'model':'ds3','view':'model','stage':'block','layer':4,'theme':'dark','hardware':'h200','context':32768,'batch':1,'kvBytes':2,'stateBytes':4,'weightBytes':2,'ranks':8,'placement':'split','trainMode':'fsdp','traffic':'expert','fabric':'ib','library':'nccl','utilization':70,'oversub':1,'intensity':128,'compare':'kimi3','detail':'shapes','zoom':1,'trace':False}
#: Read out of the rendered page rather than written down here. The model strip, the decoder-layer
#: options and the focus segments are all emitted from `MODELS` -- the segments from
#: `core.focalStages` -- so this list is the application's own inventory. The literal it replaces
#: named eight checkpoints and therefore published no preview of the ninth in any state, while
#: `verification/presentations.py` compared one artifact digest and reported the previews that did
#: exist as fresh. Reading the segments is not enough on its own: they were read and then filtered
#: to `'ced'` below, which left three focal pages undepicted. `PREVIEW_COVERAGE` now requires every
#: focal page a model offers, so a tenth checkpoint cannot go undepicted in silence either.
MODEL_IDS="[...document.querySelectorAll('#model-strip [data-model]')].map(b=>b.dataset.model)"
LAYER_OPTIONS="[...document.querySelectorAll('#control-layer option')].map(o=>o.textContent)"
FOCUS_STAGES="[...document.querySelectorAll('[data-setting=\"stage\"]')].map(b=>b.dataset.value)"

def representative_layer(options):
 """The layer whose attention motif the stack is mostly made of.

 Each option reads `NN · KIND · dense|MoE`, so the kinds come from the application's own
 `layerKind` and not from a table here. Layer 4 ordinarily; where a periodic stack makes layer 4
 the minority kind -- a hybrid's global-attention layer, which the curated cases below capture by
 name -- the lowest layer above the first, layer 1 being a dense warm-up block in several
 checkpoints.
 """
 kinds=[o.split('·')[1].strip() for o in options]
 modal=max(set(kinds),key=kinds.count)
 if len(kinds)>=4 and kinds[3]==modal:return 4
 return next((i+1 for i,k in enumerate(kinds) if k==modal and i>0),1)

def per_model_cases(page):
 """One decoder-block preview per checkpoint, plus every focal page the checkpoint offers.

 The docstring was true of the intent and false of the code: the focus segments were read out of
 the page and then narrowed by `if stage=='ced'`, so the only focal pages ever published were the
 encoder/decoder ones. `dsv41/model/residual` -- the mHC 4-stream residual -- was in no preview,
 nor were `dsv41/model/inputs`, `q38flash/model/inputs` or `q3827/model/inputs`, while
 `PREVIEW_COVERAGE` reported PASS over a requirement written to the same literal. The 'block'
 segment is the decoder block captured above at its representative layer, not a focal page.
 """
 cases=[]
 for mid in page.evaluate(MODEL_IDS):
  page.evaluate('(m)=>STRATUM.setState({model:m,view:"model",stage:"block"})',mid)
  cases.append((mid,{'model':mid,'view':'model','stage':'block',
                     'layer':representative_layer(page.evaluate(LAYER_OPTIONS))}))
  for stage in page.evaluate(FOCUS_STAGES):
   if stage!='block':cases.append((f'{mid}-{stage}',{'model':mid,'view':'model','stage':stage}))
   if stage=='ced':
    for phase in ['prefill','decode']:cases.append((f'{mid}-serving-{phase}',{'model':mid,'view':'serving','phase':phase}))
 return cases

#: Curated states, each named for the motif it shows rather than for its checkpoint, and each
#: showing something `per_model_cases` does not. The four that duplicated a focal page
#: (`flash-ple`, `flash-residual`, `kimi-residual`, `kimi-vision`) are gone: now that every
#: offered focal stage is captured systematically, they were a second copy of the same state
#: under a nicer name, and a curated list beside a derived one is how a page ends up depicted
#: only where someone remembered to name it. The two that remain pin a specific attention layer
#: the representative-layer rule does not choose.
EXTRA_CASES=[
 ('flash-qsa',{'model':'q38flash','layer':4}),
 ('kimi-mla',{'model':'kimi3','layer':4}),
 ('training',{'model':'ds3','view':'training'}),
 ('agentic-rl',{'model':'kimi3','view':'training','trainMode':'rl'}),
 ('serving',{'model':'q38flash','view':'serving'}),
 ('hardware-b200',{'view':'hardware','hardware':'b200'}),
 ('hardware-gb300',{'view':'hardware','hardware':'gb300'}),
 ('network-roce',{'view':'network','fabric':'roce','library':'nixl','traffic':'handoff','model':'kimi3'}),
 ('evidence',{'model':'q38flash','view':'evidence'}),
 ('paper',{'model':'q38flash','layer':4,'theme':'light'}),
]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=chromium_path(p),headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1840,'height':1300},device_scale_factor=1.25)
 page.set_content((ROOT/'dist/index.html').read_text(),wait_until='load')
 page.wait_for_function('window.STRATUM?.version === "3.0.0"');page.evaluate('document.fonts.ready')
 manifest=[]
 for name,patch in per_model_cases(page)+EXTRA_CASES:
  state={**BASE,**patch}
  artifact=ROOT/'dist'/artifact_name(ROOT,state['model'],state['view'])
  page.goto(artifact.as_uri(),wait_until='load');page.wait_for_function('window.STRATUM')
  page.evaluate('(s)=>STRATUM.setState(s)',state)
  page.locator('.accounting-assumptions').evaluate('(el)=>el.open=true')
  page.evaluate('document.fonts.ready');page.wait_for_timeout(50)
  report=page.evaluate('STRATUM.geometryReport()')
  if report['geometry_defects']:raise AssertionError((name,report))
  page.screenshot(path=str(OUT/f'{name}.png'),full_page=True)
  svg=page.evaluate('STRATUM.exportSVG()');(OUT/f'{name}.svg').write_text(svg)
  manifest.append({'name':name,'accounting_expanded':True,'artifact':artifact.name,'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'state':page.evaluate('STRATUM.getState()'),'svg_sha256':hashlib.sha256(svg.encode()).hexdigest()})
  print(name,flush=True)
 artifact=ROOT/'dist'/artifact_name(ROOT,'q38flash','serving')
 page.goto(artifact.as_uri(),wait_until='load');page.wait_for_function('window.STRATUM')
 page.set_viewport_size({'width':390,'height':844})
 page.evaluate('(s)=>STRATUM.setState(s)',{**BASE,'model':'q38flash','view':'serving'})
 page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
 # Recorded like every other capture. This one is a screenshot with no SVG export, so it carries
 # a viewport and a state but no `svg_sha256` -- an absent digest, not a fabricated one. Without
 # a record it was a published file the manifest never named, which `PREVIEW_INVENTORY` now
 # reports: bytes in the published directory attested by nothing.
 manifest.append({'name':'mobile','artifact':artifact.name,'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'viewport':{'width':390,'height':844},
                  'state':page.evaluate('STRATUM.getState()')})
 browser.close()
(OUT/'manifest.json').write_text(json.dumps({'artifact_sha256':hashlib.sha256((ROOT/'dist/index.html').read_bytes()).hexdigest(),'captures':manifest},indent=2)+'\n')
