"""Site sections have one canonical home; model tabs stay model-specific."""
from pathlib import Path


def exercise(page,root):
    dist=Path(root)/'dist'
    page.goto((dist/'deepseek-v4.1.html').as_uri());page.wait_for_function('window.STRATUM')
    assert page.locator('#lens-tabs [data-view]').evaluate_all('xs=>xs.map(x=>x.dataset.view)')==['model','training','serving','evidence']
    page.locator('#lens-tabs [data-view="model"]').focus()
    for expected in ['training','serving']:
        page.keyboard.press('ArrowRight')
        assert page.evaluate('STRATUM.getState().view')==expected
        assert page.evaluate('document.activeElement.dataset.view')==expected
    page.locator('#site-nav a[data-section="network"]').click()
    page.wait_for_function('window.STRATUM && STRATUM.getState().view==="network"')
    assert page.url.split('#')[0]==(dist/'networking.html').as_uri()
    assert page.locator('h1').inner_text()=='Networking'
    assert page.locator('#control-model').input_value()=='dsv41'
    assert page.locator('#model-strip [data-model]').count()==0
    page.select_option('#control-model','kimi3')
    assert page.evaluate('STRATUM.getState().model')=='kimi3'
    page.reload();page.wait_for_function('window.STRATUM')
    assert page.locator('#control-model').input_value()=='kimi3'
    page.locator('#site-nav a[data-section="hardware"]').click()
    page.wait_for_function('window.STRATUM && STRATUM.getState().view==="hardware"')
    assert page.url.split('#')[0]==(dist/'hardware.html').as_uri()
    assert page.locator('h1').inner_text()=='GPU Hardware'
    assert page.locator('#control-model').count()==0
    assert page.locator('#identity-metrics').inner_text()==''
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    for link in page.locator('#site-nav a').all():assert link.is_visible()
    page.locator('#site-nav a[data-section="models"]').click()
    assert page.url.split('#')[0]==(dist/'index.html').as_uri()
    for filename in ['index.html','qwen3.8.html','kimi-k3.html','deepseek-v4.1.html']:
        page.goto((dist/filename).as_uri()+'#view=network&model=kimi3&hardware=b200')
        page.wait_for_url('**/networking.html#*')
        page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().hardware')=='b200'
    page.goto((dist/'index.html').as_uri()+'#view=compare')
    page.wait_for_function('window.STRATUM')
    assert page.evaluate('STRATUM.getState().view')=='model'
    page.set_viewport_size({'width':1840,'height':1400})

if __name__=='__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True)
        exercise(browser.new_page(),Path(__file__).resolve().parents[1]);browser.close()
    print('Site navigation passed')
