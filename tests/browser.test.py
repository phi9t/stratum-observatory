"""Exercise the delivered offline HTML in Chromium, including rendered geometry.

No synthetic claim of upstream-code/GPU correctness is made by this suite.
"""
from pathlib import Path
import json, os, subprocess, sys
from playwright.sync_api import sync_playwright
from browser_env import chromium_path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification.delivery import page_for_model,artifact_name
def app_models(root):
    """Layer counts and focal stages read from the application's own table, not a literal.

    A hardcoded list stops covering a checkpoint the moment one is added, silently: the ninth
    preset shipped with 40 layers while this matrix still swept the same eight it always had.
    The inventory comes from the same Node exporter the science gate uses -- and the focal-stage
    map now comes from `core.focalStages` through it, rather than from a fourth re-derivation of
    the same predicate here.

    The view and hardware axes now come through the same door. They were literals in a file whose
    own docstring argues against literals: `VIEWS` restated `models.mjs` and the roofline loop
    restated `HARDWARE`, so an eighth lens or a fifth SKU would have been swept in no state while
    `assertion(not failures...)` passed over the smaller matrix. Both are published by the
    exporter already -- `fields.VIEWS` and `hardware` -- which is how `tests/sweep.py` reads them.
    """
    out=subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,
                       capture_output=True,check=True,timeout=120).stdout
    doc=json.loads(out)
    return doc['models'],doc['samples']['focal_stages'],doc['fields']['VIEWS'],doc['hardware']

APP,FOCUS,VIEWS,HARDWARE=app_models(ROOT)
MODELS={k:m['layers'] for k,m in APP.items()}
errors=[];requests=[];cases=[];responsive=[];failures=[]

def check(page, patch, label='geometry'):
    current=page.evaluate('STRATUM.getState()')
    model=patch.get('model',current['model'])
    dedicated=page_for_model(ROOT,model)
    if page.evaluate('document.body.dataset.pageId || null')!=(dedicated['id'] if dedicated else None):
        page.set_content((ROOT/'dist'/artifact_name(ROOT,model)).read_text(),wait_until='load')
        page.wait_for_function('window.STRATUM')
        patch={**current,**patch}
    r=page.evaluate('(p)=>{STRATUM.setState(p);return STRATUM.geometryReport()}',patch)
    invalid=page.locator('svg').evaluate_all('''els=>els.flatMap(e=>[e,...e.querySelectorAll('*')]).filter(e=>[...e.attributes].some(a=>/(^|[^a-z])(?:NaN|Infinity)([^a-z]|$)/.test(a.value))).map(e=>e.outerHTML.slice(0,180))''')
    record={'state':page.evaluate('STRATUM.getState()'),'defects':r['geometry_defects'],'label':label}
    cases.append(record)
    if len(cases)%100==0: print('Geometry states:',len(cases),flush=True)
    if r['geometry_defects'] or invalid:
        failures.append({'state':record['state'],'defects':r['defects'],'invalid_svg':invalid})

def assertion(ok, message):
    if not ok: raise AssertionError(message)

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=chromium_path(p),headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1840,'height':1400},device_scale_factor=1,accept_downloads=True)
    page.on('pageerror',lambda err:errors.append(str(err)))
    page.on('request',lambda req:requests.append(req.url) if req.url.startswith(('http://','https://')) else None)
    page.set_content((ROOT/'dist/index.html').read_text(),wait_until='load')
    page.wait_for_function('window.STRATUM?.version === "3.0.0"')
    page.evaluate('document.fonts.ready')
    for model,L in MODELS.items():
        for detail in ['shapes','operators']:
            for layer in range(1,L+1):
                check(page,{'model':model,'layer':layer,'view':'model','stage':'block','theme':'dark','detail':detail},'every-layer')
        for view in VIEWS:
            for theme in ['light','dark']:
                check(page,{'model':model,'layer':min(4,L),'view':view,'stage':'block','theme':theme,'detail':'shapes'},'lens-theme')
    for m,stages in FOCUS.items():
        for stage in stages:
            for theme in ['dark','light']: check(page,{'model':m,'view':'model','stage':stage,'theme':theme},'focus-stage')
    for model,meta in APP.items():
        if meta.get('encoderLayers'):
            for phase in ['prefill','decode']:
                for theme in ['dark','light']:check(page,{'model':model,'view':'serving','stage':'block','phase':phase,'theme':theme},'serving-phase')
    for h in HARDWARE:
        for theme in ['dark','light']:
            for ai in [1,128,2048]: check(page,{'view':'hardware','hardware':h,'intensity':ai,'theme':theme},'hardware-roofline')
    for fabric in ['ib','roce']:
        for traffic in ['expert','allreduce','handoff']:
            for library in ['nccl','nvshmem','nixl']:
                for ranks in [1,1024]: check(page,{'model':'q38flash','view':'network','fabric':fabric,'traffic':traffic,'library':library,'ranks':ranks,'hardware':'gb300'},'network-protocol')
    for model in MODELS:
        for mode in ['fsdp','ddp','rl']: check(page,{'model':model,'view':'training','trainMode':mode,'ranks':1024},'training-mode')
        for placement in ['colocated','split']:
            for n in [1,1048576]: check(page,{'model':model,'view':'serving','placement':placement,'context':n,'batch':256,'weightBytes':1,'kvBytes':1,'ranks':1},'serving-extremes')
    for width in [390,768,1280,1840]:
        page.set_viewport_size({'width':width,'height':1100})
        for theme in ['light','dark']:
            for view in VIEWS:
                check(page,{'model':'q38flash','view':view,'theme':theme,'stage':'block','layer':4,'zoom':1,'context':32768,'batch':1,'kvBytes':2,'weightBytes':2,'ranks':8},'responsive')
                dims=page.evaluate('({body:document.documentElement.scrollWidth,viewport:innerWidth,scrollable:document.getElementById("scene-viewport").scrollWidth})')
                assertion(dims['body']<=width+1,f'Document overflow at {width}: {dims}')
                responsive.append({'width':width,'theme':theme,'view':view,**dims})
    # Font bounds depend on device-scale factors; do not infer coverage from DPR=1.
    for ratio in [1.25,2]:
        dense=browser.new_page(viewport={'width':1840,'height':1300},device_scale_factor=ratio)
        dense.on('pageerror',lambda err:errors.append(str(err)))
        dense.on('request',lambda req:requests.append(req.url) if req.url.startswith(('http://','https://')) else None)
        dense.set_content((ROOT/'dist/index.html').read_text(),wait_until='load')
        dense.evaluate('document.fonts.ready')
        for model in MODELS:
            for theme in ['light','dark']:
                check(dense,{'model':model,'view':'model','layer':2,'theme':theme,'detail':'shapes'},f'device-scale-{ratio}')
        dense.close()
    page.set_viewport_size({'width':1840,'height':1400})
    page.evaluate('STRATUM.setState({model:"ds3",view:"model",stage:"block",layer:4,theme:"dark",trace:false,zoom:1})')
    (ROOT/'tests/geometry-results.json').write_text(json.dumps({'geometry_states':len(cases),'failures':failures},indent=2))
    assertion(not failures,'Geometry failures: inspect tests/geometry-results.json')
    # Real UI changes, not API-only smoke tests.
    page.goto((ROOT/'dist/index.html').as_uri());page.wait_for_function('window.STRATUM')
    page.locator('a[data-model="q3827"]').click();page.wait_for_function('window.STRATUM')
    assertion(page.evaluate('STRATUM.getState().model')=='q3827','model card')
    page.select_option('#control-layer','4')
    assertion(page.evaluate('STRATUM.getScene().nodes.some(n=>n.id==="kv-cache")'),'global attention selector')
    page.locator('[data-layer="1"]').click()
    assertion(page.evaluate('STRATUM.getScene().nodes.some(n=>n.id==="state-prev")'),'recurrent layer tile')
    page.locator('[data-node="attention"]').focus();page.keyboard.press('Enter')
    assertion('Gated' in page.locator('#inspector').inner_text() or 'delta' in page.locator('#inspector').inner_text().lower(),'keyboard inspector')
    page.locator('#trace-button').click()
    assertion(page.evaluate('STRATUM.getState().trace'),'trace control')
    assertion(page.locator('.trace-edge').count()>0,'dependency highlighting')
    page.locator('#trace-button').click()
    page.locator('#theme-button').click()
    assertion(page.evaluate('document.documentElement.dataset.theme')=='light','theme toggle')
    page.locator('#focus-button').click();assertion(page.locator('body').evaluate('e=>e.classList.contains("focus-mode")'),'focus mode')
    page.keyboard.press('Escape');assertion(not page.locator('body').evaluate('e=>e.classList.contains("focus-mode")'),'focus Escape')
    page.locator('#search-button').click();page.fill('#search-input','Flash-Next');page.keyboard.press('Enter')
    assertion(page.evaluate('STRATUM.getState().model')=='q38flash','command search')
    page.locator('button[data-setting="stage"][data-value="ple"]').click()
    assertion(page.evaluate('STRATUM.getScene().nodes.some(n=>n.id==="ngram-lookup")'),'PLE focus action')
    page.locator('#lens-tabs button[data-view="serving"]').click();page.select_option('#control-context','131072')
    assertion(page.evaluate('STRATUM.getState().context')==131072,'context control')
    page.locator('button[data-setting="placement"][data-value="colocated"]').click()
    assertion(page.evaluate('STRATUM.getState().placement')=='colocated','placement control')
    page.locator('#lens-tabs button[data-view="network"]').click();page.select_option('#control-library','nvshmem');page.locator('button[data-setting="fabric"][data-value="roce"]').click()
    assertion(page.evaluate('STRATUM.getState().fabric')=='roce','fabric control')
    page.locator('#lens-tabs button[data-view="network"]').focus();page.keyboard.press('ArrowRight')
    assertion(page.evaluate('STRATUM.getState().view')=='evidence','keyboard lens navigation')
    page.locator('body').click(position={'x':3,'y':3});page.keyboard.press('1')
    assertion(page.evaluate('STRATUM.getState().view')=='model','numeric lens shortcut')
    # URL state is restored by the built artifact itself.
    expected=page.evaluate('STRATUM.getState()');saved_hash=page.evaluate('location.hash');page.goto('about:blank'+saved_hash);page.set_content((ROOT/'dist/index.html').read_text(),wait_until='load');page.wait_for_function('window.STRATUM')
    assertion(page.evaluate('STRATUM.getState()')==expected,'URL round trip')
    page.set_viewport_size({'width':390,'height':1000});page.locator('#zoom-in').click()
    assertion(page.evaluate('STRATUM.getState().zoom')>1,'mobile zoom')
    assertion(page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),'zoom must stay within scroll viewport')
    page.locator('#zoom-reset').click();page.set_viewport_size({'width':1840,'height':1400})
    page.evaluate('STRATUM.setState({model:"ds3",view:"model",layer:4,stage:"block",theme:"dark"})')
    with page.expect_download() as download:
        page.locator('#export-svg').click()
    svgp=ROOT/'tests/test-export.svg';download.value.save_as(str(svgp))
    assertion('<svg' in svgp.read_text(),'real SVG download')
    export_page=browser.new_page();export_page.on('pageerror',lambda e:errors.append(str(e)))
    export_page.set_content('<html><body></body></html>');parsed=export_page.evaluate('(text)=>{const d=new DOMParser().parseFromString(text,"image/svg+xml");if(d.querySelector("parsererror"))return false;document.body.appendChild(document.importNode(d.documentElement,true));return true}',svgp.read_text());assertion(parsed,'standalone SVG XML parses')
    assertion(export_page.locator('svg').count()==1,'independent SVG parses');assertion(export_page.locator('parsererror').count()==0,'no XML parser error');export_page.close()
    with page.expect_download() as download:
        page.locator('#export-config').click()
    jp=ROOT/'tests/test-scenario.json';download.value.save_as(str(jp));conf=json.loads(jp.read_text())
    assertion(conf['scientificStatus']=='BLOCKED' and conf['runtimeStatus']=='NOT_RUN','export cannot claim certification')
    assertion(conf['model']['id']=='ds3','config corresponds to selected model')
    page.locator('#audit-button').click();assertion(page.locator('#audit-output').is_visible(),'audit dialog');page.keyboard.press('Escape')
    # Invalid source/URL state is normalized, not injected.
    page.evaluate('STRATUM.setState({model:"<svg/onload=1>",layer:999999,batch:-2})')
    assertion(page.locator('script[src]').count()==0,'no external script dependency')
    assertion(page.evaluate('STRATUM.geometryReport().geometry_defects')==0,'sanitized state remains valid')
    from dedicated_navigation import exercise as exercise_navigation
    exercise_navigation(page,ROOT)
    from dedicated_page import exercise
    delivery=browser.new_page()
    delivery.on('pageerror',lambda e:errors.append(str(e)))
    delivery.on('request',lambda req:requests.append(req.url) if req.url.startswith(('http://','https://')) else None)
    exercise(delivery,ROOT)
    from accounting_panel import exercise as exercise_accounting
    exercise_accounting(delivery,ROOT)
    delivery.close()
    browser.close()
report={'schema':'stratum-browser-tests/v2','browser':'Chromium','geometry_states':len(cases),'responsive_states':len(responsive),'device_scale_states':sum(1 for c in cases if c['label'].startswith('device-scale-')),'geometry_defects':sum(len(f['defects'])+len(f['invalid_svg']) for f in failures),'runtime_errors':len(errors),'external_runtime_requests':len(requests),'interactive_checks':'passed','exports':'independent SVG rendering and JSON passed','models':MODELS,'failures':failures,'errors':errors,'requests':requests}
(ROOT/'tests/browser-results.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'tests/browser-cases.json').write_text(json.dumps({'states':cases,'responsive':responsive},indent=2)+'\n')
print(json.dumps(report,indent=2))
assertion(not failures and not errors and not requests,'See browser-results.json for release-blocking findings')
