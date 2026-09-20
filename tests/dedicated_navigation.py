"""Every dedicated page is discoverable and keeps exact checkpoint identities."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verification.delivery import page_inventory


def exercise(page, root):
    root = Path(root)
    routes = [(model,entry['file']) for entry in page_inventory(root) for model in entry['models']]
    assert dict(routes)=={'q3827':'qwen3.8.html','q38flash':'qwen3.8.html',
                          'kimi3':'kimi-k3.html','dsv41':'deepseek-v4.1.html'}
    for model, filename in routes:
        page.goto((root/'dist/index.html').as_uri())
        page.wait_for_function('window.STRATUM')
        page.locator(f'#model-strip a[data-model="{model}"]').click(timeout=2000)
        page.wait_for_function('(model)=>window.STRATUM && STRATUM.getState().model===model', arg=model)
        assert page.url.split('#')[0] == (root/'dist'/filename).as_uri()
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().model') == model
        assert page.evaluate('STRATUM.geometryReport().geometry_defects') == 0
        page.evaluate('STRATUM.setState({model:"ds2"})')
        assert page.evaluate('STRATUM.getState().model') in (["q3827", "q38flash"] if filename=='qwen3.8.html' else [model])
        page.locator('#atlas-link').click()
        page.wait_for_function('window.STRATUM')
        assert page.url.split('#')[0] == (root/'dist/index.html').as_uri()
    page.goto((root/'dist/qwen3.8.html').as_uri())
    page.wait_for_function('window.STRATUM')
    for model in ['q38flash','q3827']:
        page.locator(f'button[data-model="{model}"]').click()
        assert page.evaluate('STRATUM.getState().model') == model
        assert page.url.split('#')[0] == (root/'dist/qwen3.8.html').as_uri()
        assert page.evaluate('JSON.parse(STRATUM.exportConfig()).model.id') == model


if __name__ == '__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True)
        exercise(browser.new_page(),Path(__file__).resolve().parents[1])
        browser.close()
    print('Dedicated navigation passed')
