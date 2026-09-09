"""Real-browser release verifier. No server or external runtime resources are required."""
import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]

def run():
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1900,'height':1250},device_scale_factor=1)
  errors=[];requests=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('request',lambda r:requests.append(r.url))
  page.set_content((ROOT/'dist/index.html').read_text(),wait_until='load')
  page.wait_for_function('!!window.STRATUM')
  cases=[]
  for layer in range(1,94):
   for detail in ['shapes','operators']:
    cases.append({'view':'model','layer':layer,'detail':detail,'theme':'light' if layer%2==0 else 'dark'})
  for theme in ['dark','light']:
   for layer in [1,2,4,13,92,93]:
    cases.append({'view':'model','layer':layer,'detail':'shapes','theme':theme})
   for mode in ['pretrain','agentic']:cases.append({'view':'training','trainingMode':mode,'theme':theme})
  for context in [1024,32768,131072,1048576]:
   for batch in [1,128]:
    for cache in ['compressed','expanded']:
     for mode in ['disaggregated','colocated']:
      for precision in [1,2]:
       cases.append({'view':'serving','context':context,'batch':batch,'cacheMode':cache,'serveMode':mode,'kvPrecision':precision,'statePrecision':4 if precision==1 else 2,'weightMode':'bf16' if batch==128 else 'mixed','theme':'light' if precision==2 else 'dark','match':6144 if batch==128 else 2800})
  for mode in ['moe','kcp','cache']:
   for ranks in [1,2,8,64]:
    for theme in ['light','dark']:
     cases.append({'view':'network','trafficMode':mode,'nodes':ranks,'theme':theme,'tokens':65536,'context':1048576,'oversub':4})
  for hw in ['h100','h200']:
   for ai in [1,4,64,256,1024,4096]:cases.append({'view':'hardware','hardware':hw,'intensity':ai})
  bad=[]
  for case in cases:
   page.evaluate('(c)=>STRATUM.setState(c)',case)
   report=page.evaluate('STRATUM.geometryReport()')
   if any(report.values()):bad.append({'state':case,'defects':report})
  print(f'Geometry states: {len(cases)}; defective states: {len(bad)}')
  if bad:
   (ROOT/'tests/geometry-failures.json').write_text(json.dumps(bad,indent=2));print(json.dumps(bad[:5],indent=2))
   raise AssertionError('Rendered geometry failed')
  assert not errors,errors
  responsive=0
  for width in [390,768,1280,1920]:
   page.set_viewport_size({'width':width,'height':1000})
   for view in ['model','training','serving','hardware','network']:
    for theme in ['dark','light']:
     page.evaluate('(s)=>STRATUM.setState(s)',{'view':view,'theme':theme})
     assert page.evaluate('document.documentElement.scrollWidth')<=width+1,(width,view)
     assert not any(page.evaluate('STRATUM.geometryReport()').values()),(width,view,theme)
     responsive+=1
  page.set_viewport_size({'width':1900,'height':1250})
  page.evaluate("STRATUM.setState({view:'model',layer:2,detail:'shapes',theme:'dark'})")
  assert page.locator('h1').inner_text().startswith('Kimi K3.')
  page.locator('[data-key="layer"]').select_option('1')
  assert page.locator('[data-node="dense-up"]').count()==1
  page.locator('[data-key="layer"]').select_option('4')
  assert page.locator('[data-node="mla-core"]').count()==1
  page.locator('[data-node="kv-down"]').click()
  assert '64-dimensional' in page.locator('#inspector').inner_text()
  page.locator('[data-node="layer-93"]').click()
  assert page.evaluate('STRATUM.getState().layer')==93
  page.locator('#tab-training').click()
  page.locator('[data-key="trainingMode"][data-value="agentic"]').click()
  page.locator('[data-node="external-pool"]').click()
  assert 'write-back' in page.locator('#inspector').inner_text().lower() or 'evicted' in page.locator('#inspector').inner_text().lower()
  page.locator('#tab-serving').click()
  page.locator('[data-key="context"]').select_option('1024');small=page.locator('#metrics').inner_text()
  page.locator('[data-key="context"]').select_option('1048576');large=page.locator('#metrics').inner_text();assert small!=large
  page.locator('[data-key="match"]').select_option('2800')
  assert '2,560' in page.locator('[data-node="prefix-match"]').text_content()
  page.locator('#theme-button').click();assert page.evaluate('STRATUM.getState().theme')=='light'
  page.locator('#sources-button').click();assert page.locator('#sources-dialog').evaluate('(e)=>e.open')
  assert page.locator('#source-list .source-row').count()==12
  page.keyboard.press('Escape')
  page.locator('#search-button').click();page.locator('#search-input').fill('Hardware');page.locator('.search-result').first.click()
  assert page.evaluate('STRATUM.getState().view')=='hardware'
  page.locator('#tab-hardware').focus();page.keyboard.press('ArrowRight');assert page.evaluate('STRATUM.getState().view')=='network'
  page.locator('[data-key="trafficMode"][data-value="kcp"]').click();page.locator('[data-key="nodes"]').select_option('1')
  assert '0 MiB' in page.locator('[data-node="traffic-formula"]').text_content()
  page.locator('#zoom-in').click();assert page.evaluate('STRATUM.getState().zoom')>1
  page.locator('#zoom-fit').click();assert page.evaluate('STRATUM.getState().zoom')==1
  page.locator('#focus-button').click();assert page.locator('body').evaluate('(e)=>e.classList.contains("focus-mode")');page.locator('#focus-button').click()
  with page.expect_download() as event:page.locator('#export-button').click()
  event.value.save_as(str(ROOT/'previews/test-export.svg'))
  with page.expect_download() as event:page.locator('#export-config').click()
  event.value.save_as(str(ROOT/'previews/test-configuration.json'))
  config=json.loads((ROOT/'previews/test-configuration.json').read_text())
  assert config['schema']=='stratum-kimi-k3/v1';assert config['model']['layers']==93;assert len(config['sources'])==12
  svgp=browser.new_page(viewport={'width':1400,'height':900});svgp.set_content((ROOT/'previews/test-export.svg').read_text())
  assert svgp.locator('svg .node').count()>=10
  assert svgp.locator('.node-title').first.evaluate('(e)=>getComputedStyle(e).fill')!='rgb(0, 0, 0)'
  page.set_viewport_size({'width':390,'height':844});page.locator('#zoom-fit').click();before=page.locator('#diagram-host').bounding_box()['width'];page.locator('#zoom-in').click();assert page.locator('#diagram-host').bounding_box()['width']>before*1.2
  assert page.evaluate('location.hash').startswith('#view=')
  page.evaluate('location.hash="view=model&layer=93&hardware=h200&theme=light"')
  page.wait_for_function('STRATUM.getState().layer===93 && STRATUM.getState().view==="model"')
  assert page.locator('[data-node="mla-core"]').count()==1
  assert not errors,errors;assert not requests,requests
  result={'geometry_states':len(cases),'geometry_defects':0,'responsive_states':responsive,'runtime_errors':len(errors),'external_runtime_requests':len(requests),'interactive_checks':'passed','exports':'SVG and JSON passed','browser':'Chromium','coverage':'All 93 layers; representative view, mode, precision, extreme-size and viewport states. Not exhaustive all-combinations certification.'}
  (ROOT/'tests/browser-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));browser.close()
if __name__=='__main__':run()
