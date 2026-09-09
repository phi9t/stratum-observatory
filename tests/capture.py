"""Capture current portable application and export its own SVGs; no model execution."""
from pathlib import Path
import hashlib,json,os
from playwright.sync_api import sync_playwright
from browser_env import chromium_path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'previews';OUT.mkdir(exist_ok=True)
BASE={'model':'ds3','view':'model','stage':'block','layer':4,'theme':'dark','hardware':'h200','context':32768,'batch':1,'kvBytes':2,'stateBytes':4,'weightBytes':2,'ranks':8,'placement':'split','trainMode':'fsdp','traffic':'expert','fabric':'ib','library':'nccl','utilization':70,'oversub':1,'intensity':128,'compare':'kimi3','detail':'shapes','zoom':1,'trace':False}
CASES=[(m,{'model':m,'layer':4 if m not in ['q3827','q38flash','kimi3'] else 2}) for m in ['ds2','ds3','q38','q330','q3235','q3827','q38flash','kimi3']]
CASES += [
 ('flash-qsa',{'model':'q38flash','layer':4}),
 ('flash-ple',{'model':'q38flash','stage':'ple'}),
 ('flash-residual',{'model':'q38flash','stage':'residual'}),
 ('kimi-mla',{'model':'kimi3','layer':4}),
 ('kimi-residual',{'model':'kimi3','stage':'residual'}),
 ('kimi-vision',{'model':'kimi3','stage':'inputs'}),
 ('comparison',{'model':'q38flash','view':'compare','compare':'ds3'}),
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
 for name,patch in CASES:
  state={**BASE,**patch};page.evaluate('(s)=>STRATUM.setState(s)',state)
  page.evaluate('document.fonts.ready');page.wait_for_timeout(50)
  report=page.evaluate('STRATUM.geometryReport()')
  if report['geometry_defects']:raise AssertionError((name,report))
  page.screenshot(path=str(OUT/f'{name}.png'),full_page=True)
  svg=page.evaluate('STRATUM.exportSVG()');(OUT/f'{name}.svg').write_text(svg)
  manifest.append({'name':name,'state':page.evaluate('STRATUM.getState()'),'svg_sha256':hashlib.sha256(svg.encode()).hexdigest()})
  print(name,flush=True)
 page.set_viewport_size({'width':390,'height':844})
 page.evaluate('(s)=>STRATUM.setState(s)',{**BASE,'model':'q38flash','view':'serving'})
 page.screenshot(path=str(OUT/'mobile.png'),full_page=True)
 browser.close()
(OUT/'manifest.json').write_text(json.dumps({'artifact_sha256':hashlib.sha256((ROOT/'dist/index.html').read_bytes()).hexdigest(),'captures':manifest},indent=2)+'\n')
