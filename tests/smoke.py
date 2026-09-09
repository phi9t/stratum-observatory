from playwright.sync_api import sync_playwright
from pathlib import Path
import json,os
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=os.getenv('CHROMIUM_PATH','/usr/bin/chromium'),headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1840,'height':1400},device_scale_factor=1)
 page.on('pageerror',lambda e: print('JS ERROR',e))
 page.set_content((root/'dist/index.html').read_text(),wait_until='load')
 page.wait_for_timeout(400)
 print('loaded', page.evaluate('typeof STRATUM'))
 if page.evaluate('typeof STRATUM')!='undefined':
  data=page.evaluate('STRATUM.geometryReport()'); (root/'tests/smoke-report.json').write_text(json.dumps(data,indent=2)); print('Geometry',data['geometry_defects']);print(json.dumps(data['defects'][:25],indent=2))
 (root/'previews').mkdir(exist_ok=True)
 page.screenshot(path=str(root/'previews/observatory-initial.png'),full_page=True)
 b.close()
