"""Scratch geometry sweep. Not a gate: no workflow and no package.json script runs this.

The matrix is derived from the application, like every other sweep in this repo. It used to be an
eight-model literal with literal stage lists beside it -- byte-for-byte the pattern that swept
eight checkpoints after a ninth arrived in `tests/browser.test.py`, and the one that published no
preview of the ninth in `tests/capture.py`. Nothing reads this script's report, so no published
figure was ever wrong because of it; it is derived anyway, because a scratch script that *looks*
like a coverage sweep is the next maintainer's trap.
"""
from playwright.sync_api import sync_playwright
from browser_env import chromium_path
from pathlib import Path
import json,os,subprocess
R=Path(__file__).resolve().parents[1]
# Scratch shots go beside this script's report, never into `previews/`. That directory is
# published output: every file in it must be produced by `npm run capture` and named by a capture
# record, which `PREVIEW_INVENTORY` now enforces. A scratch sweep dropping 16 unattested PNGs
# there is precisely the trap this file's docstring warns the next maintainer about.
SHOTS=R/'tests/sweep-shots'
SHOTS.mkdir(exist_ok=True)
INV=json.loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=R,text=True,
                              capture_output=True,check=True,timeout=120).stdout)
models={k:m['layers'] for k,m in INV['models'].items()}
focal=INV['samples']['focal_stages']
# The last literal axis in this file. Its own docstring argues against exactly this.
views=[v for v in INV['fields']['VIEWS'] if v!='model']
cases=[]
for m,L in models.items():
 for l in sorted({1,2,4,L}): cases.append({'model':m,'view':'model','stage':'block','layer':l})
 for v in views: cases.append({'model':m,'view':v,'stage':'block'})
 for stage in focal[m]: cases.append({'model':m,'view':'model','stage':stage})
for h in INV['hardware']:cases.append({'view':'hardware','hardware':h})
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
   name='-'.join(str(v) for v in c.values());page.screenshot(path=str(SHOTS/f'{name}.png'),full_page=True)
 b.close()
(R/'tests/sweep-report.json').write_text(json.dumps({'cases':len(cases),'bad':report,'errors':errs},indent=2))
print('TOTAL',len(cases),'BAD',len(report),'JS',errs)
