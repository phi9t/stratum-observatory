"""Reader journeys through the dedicated checkpoint page, using the portable files."""
import json
from pathlib import Path


def exercise(page, root):
    artifact = Path(root)/'dist/deepseek-v4.1.html'
    assert artifact.is_file(), 'Build must deliver the dedicated V4.1 page'
    page.goto((Path(root)/'dist/foundations.html').as_uri())
    page.wait_for_function('window.STRATUM')
    page.locator('[data-view="serving"]').first.click()
    page.locator('#model-strip [data-model="q38"]').click()
    assert page.evaluate('STRATUM.getState().view') == 'serving'
    page.locator('#atlas-link').click()
    page.locator('a[data-destination="dsv41"]').click()
    page.wait_for_function('window.STRATUM && STRATUM.getState().stage === "ced"')
    assert page.url.split('#')[0] == artifact.as_uri()
    assert page.evaluate('STRATUM.getState().model') == 'dsv41'
    assert page.locator('.innovation-table').count() == 1
    assert page.locator('#analysis-panel .chart').count() == 0
    assert 'Mechanism → state growth' not in page.locator('#analysis-panel').inner_text()
    assert page.locator('#schematic').is_visible()
    assert page.locator('#inspector .formula').count() > 0
    assert page.locator('#model-strip [data-model]').count() == 0
    page.locator('[data-view="serving"]').first.click()
    page.locator('[data-setting="phase"][data-value="prefill"]').click()
    assert page.evaluate('STRATUM.getState().phase') == 'prefill'
    page.reload()
    page.wait_for_function('window.STRATUM')
    assert page.evaluate('STRATUM.getState().phase') == 'prefill'
    exported = json.loads(page.evaluate('STRATUM.exportConfig()'))
    assert exported['model']['id'] == 'dsv41'
    assert exported['state']['view'] == 'serving'
    assert exported['scientificStatus'] == 'BLOCKED'
    page.goto(artifact.as_uri()+'#model=ds3&stage=ple&context=8192')
    page.wait_for_function('window.STRATUM')
    assert page.evaluate('STRATUM.getState().model') == 'dsv41'
    assert page.evaluate('STRATUM.getState().stage') == 'encoder'
    assert page.evaluate('STRATUM.getState().context') == 8192
    for width in [390,768,1840]:
        page.set_viewport_size({'width':width,'height':1000})
        for theme in ['dark','light']:
            page.evaluate('(theme)=>STRATUM.setState({theme,view:"model",stage:"ced"})',theme)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert page.evaluate('STRATUM.geometryReport().geometry_defects') == 0
    page.locator('#atlas-link').click()
    assert page.url.split('#')[0] == (Path(root)/'dist/index.html').as_uri()


if __name__ == '__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium_path(pw), headless=True)
        exercise(browser.new_page(), Path(__file__).resolve().parents[1])
        browser.close()
    print('Dedicated page reader journey passed')
