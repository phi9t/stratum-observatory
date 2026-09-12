"""Scratch geometry probe. Not a gate: no workflow and no package.json script runs this.

It used to screenshot into `previews/`, the directory the presentations gate treats as published
output -- so a scratch run left a file there that no capture record named and no judge covered,
and `previews/observatory-initial.png` sat in the repository for three releases depicting an
artifact nobody could identify. Scratch output goes under `tests/`, like the rest of this file's
output; `npm run capture` is the only thing that writes a preview.
"""
from playwright.sync_api import sync_playwright
from browser_env import chromium_path
from pathlib import Path
import json,os
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=chromium_path(p),headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1840,'height':1400},device_scale_factor=1)
 page.on('pageerror',lambda e: print('JS ERROR',e))
 page.set_content((root/'dist/index.html').read_text(),wait_until='load')
 page.wait_for_timeout(400)
 print('loaded', page.evaluate('typeof STRATUM'))
 if page.evaluate('typeof STRATUM')!='undefined':
  data=page.evaluate('STRATUM.geometryReport()'); (root/'tests/smoke-report.json').write_text(json.dumps(data,indent=2)); print('Geometry',data['geometry_defects']);print(json.dumps(data['defects'][:25],indent=2))
 page.screenshot(path=str(root/'tests/smoke-initial.png'),full_page=True)
 b.close()
