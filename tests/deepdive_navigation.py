"""Real chapter journeys through the delivered portable V4.1 file."""
from pathlib import Path
import json


def exercise(page,root):
    artifact=Path(root)/'dist/deepseek-v4.1.html'
    page.goto(artifact.as_uri());page.wait_for_function('window.STRATUM')
    expected=['encoder','decoder','engram','csa2','dspark']
    for stage in expected:
        page.locator(f'button[data-setting="stage"][data-value="{stage}"]').click()
        assert page.evaluate('STRATUM.getState().stage')==stage
        assert f'stage={stage}' in page.url
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().stage')==stage
        assert page.locator('#analysis-panel').get_by_text('Worked trace',exact=True).count()==1
        assert page.locator('#analysis-panel tbody tr').count()>=4
        assert page.evaluate('STRATUM.geometryReport().geometry_defects')==0
        assert json.loads(page.evaluate('STRATUM.exportConfig()'))['state']['stage']==stage
        links=page.locator('#analysis-panel > .analysis-split a[href^="https:"]').evaluate_all('xs=>xs.map(a=>a.href)')
        assert links and all('/main/' not in href for href in links)
        assert all('huggingface.co/deepseek-ai/' in href or 'github.com/sgl-project/sglang/' in href or 'github.com/deepseek-ai/DeepSpec/' in href or 'arxiv.org/html/2607.05147v1' in href for href in links)
        page.locator('#trace-button').click()
        assert page.evaluate('STRATUM.getState().trace')
        page.locator('#trace-button').click()
        page.set_viewport_size({'width':390,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.set_viewport_size({'width':1840,'height':1400})
    page.locator('button[data-setting="stage"][data-value="csa2"]').click()
    assert page.locator('button[data-setting="detail"]').count()==0
    for layer in ['3','25','26']:
        page.select_option('#control-layer',layer)
        assert page.evaluate('STRATUM.getState().layer')==int(layer)
        assert f'Layer {layer} is' in page.locator('#analysis-panel').inner_text()
    before=page.locator('#analysis-panel tbody').inner_text()
    page.select_option('#control-context','131072')
    assert page.locator('#analysis-panel tbody').inner_text()!=before
    page.locator('button[data-setting="stage"][data-value="ced"]').click()
    assert page.locator('.innovation-table').count()==1

if __name__=='__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True)
        exercise(browser.new_page(),Path(__file__).resolve().parents[1]);browser.close()
    print('V4.1 chapter navigation passed')
