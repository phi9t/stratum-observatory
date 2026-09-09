from playwright.sync_api import sync_playwright
from browser_env import chromium_path
from pathlib import Path
import json,os
R=Path(__file__).resolve().parents[1]
models={'ds2':60,'ds3':61,'q38':36,'q330':48,'q3235':94,'q3827':64,'q38flash':48,'kimi3':93}
cases=[]
for m,L in models.items():
 for l in sorted({1,2,4,L}): cases.append({'model':m,'view':'model','stage':'block','layer':l})
 for v in ['compare','training','serving','hardware','network','evidence']: cases.append({'model':m,'view':v,'stage':'block'})
for m in ['q3827','q38flash','kimi3']:
 cases.append({'model':m,'view':'model','stage':'inputs'})
for m in ['q38flash','kimi3']:cases.append({'model':m,'view':'model','stage':'residual'})
cases.append({'model':'q38flash','view':'model','stage':'ple'})
for h in ['h100','h200','b200','gb300']:cases.append({'view':'hardware','hardware':h})
for traffic in ['expert','allreduce','handoff']:
 for lib in ['nccl','nvshmem','nixl']:cases.append({'view':'network','traffic':traffic,'library':lib})
for train in ['fsdp','ddp','rl']:cases.append({'view':'training','trainMode':train})
errs=[]; report=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=chromium_path(p),headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1840,'height':1400},device_scale_factor=1)
 page.on('pageerror',lambda e:errs.append(str(e)))
 page.set_content((R/'dist/index.html').read_text(),wait_until='load')
 page.wait_for_function('window.STRATUM')
 for i,c in enumerate(cases):
  data=page.evaluate('(s)=>{STRATUM.setState(s);return STRATUM.geometryReport()}',c)
  invalid=page.locator('svg').evaluate_all('(els)=>els.map(x=>x.outerHTML).filter(x=>/NaN|Infinity/.test(x)).length')
  if data['geometry_defects'] or invalid:
   report.append({'case':c,'defects':data['defects'],'invalid':invalid})
   print(c,':',data['geometry_defects'],'invalid',invalid,flush=True)
  if c.get('stage') in ['inputs','residual','ple'] or (c.get('model')=='q38flash' and c.get('view')!='model') or c.get('hardware')=='gb300' or (c.get('model')=='kimi3' and c.get('layer') in [2,4]):
   name='-'.join(str(v) for v in c.values());page.screenshot(path=str(R/'previews'/f'{name}.png'),full_page=True)
 b.close()
(R/'tests/sweep-report.json').write_text(json.dumps({'cases':len(cases),'bad':report,'errors':errs},indent=2))
print('TOTAL',len(cases),'BAD',len(report),'JS',errs)
