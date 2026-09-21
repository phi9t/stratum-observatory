"""Every dedicated page is discoverable and keeps exact checkpoint identities."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verification.delivery import page_inventory


def exercise(page, root):
    root = Path(root)
    routes = [(model,entry['file']) for entry in page_inventory(root) for model in entry['models']]
    assert len(dict(routes)) == len(routes), 'Each model has one canonical entry'
    for model, filename in routes:
        entry=next(p for p in page_inventory(root) if p['file']==filename)
        page.goto((root/'dist/index.html').as_uri())
        page.locator(f'main a[href="{filename}"]').click()
        page.wait_for_function('window.STRATUM')
        if page.evaluate('STRATUM.getState().model')!=model:
            page.get_by_role('combobox',name='Exact checkpoint').select_option(model)
        assert page.url.split('#')[0] == (root/'dist'/filename).as_uri()
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().model') == model
        assert page.evaluate('STRATUM.geometryReport().geometry_defects') == 0
        page.evaluate('STRATUM.setState({model:"invalid-checkpoint"})')
        assert page.evaluate('STRATUM.getState().model') in entry['models']
        page.locator('#chapter-nav .sidebar-sections').get_by_role('link',name='Home',exact=True).click()
        assert page.url.split('#')[0] == (root/'dist/index.html').as_uri()
    page.goto((root/'dist/qwen3.8.html').as_uri())
    page.wait_for_function('window.STRATUM')
    for model in ['q38flash','q3827']:
        page.get_by_role('combobox',name='Exact checkpoint').select_option(model)
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
