"""Integrated V4.1 chapter reader journeys on the portable artifact."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
from browser_env import chromium_path

ROOT = Path(__file__).resolve().parents[1]

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path(p), headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        url = (ROOT/'dist/deepseek-v4.1.html').as_uri()
        for chapter in ['csa2', 'engram', 'inputs', 'residual', 'dspark']:
            page.goto(url + '#stage=' + chapter + '&panel=chapter')
            page.wait_for_function('!!window.STRATUM')
            assert page.locator('.mechanism-chapter').is_visible(), chapter
            assert page.locator('.mechanism-section').count() >= 3, chapter
            assert page.locator('.mechanism-section math').count() >= 3, chapter
            assert page.locator('.tensor-matrix').count() == page.locator('.tensor-matrix thead').count(), chapter
            assert page.locator('.tensor-matrix thead th:not([scope=col])').count() == 0
            assert not page.locator('#inspector').is_visible(), chapter
            button = page.locator('.mechanism-section [data-chapter-node]').first
            button.click()
            assert page.locator('#inspector').is_visible(), chapter
            selected = page.evaluate('STRATUM.getState().module')
            if chapter == 'csa2':
                page.evaluate("location.hash='stage=engram&panel=chapter'")
                page.wait_for_function("STRATUM.getState().stage==='engram'")
                assert not page.locator('#inspector').is_visible(), 'Hash navigation must close the prior chapter inspector'
                page.go_back()
                page.wait_for_function("STRATUM.getState().stage==='csa2'")
            page.reload();page.wait_for_function('!!window.STRATUM')
            assert page.evaluate('STRATUM.getState().module') == selected
            page.locator('#workspace-tabs [data-panel="schematic"]').click()
            assert page.locator('#schematic').is_visible()
            page.go_back()
            assert page.locator('.mechanism-chapter').is_visible()
        page.goto(url+'#stage=residual&panel=chapter')
        page.wait_for_function('!!window.STRATUM')
        page.locator('#sinkhorn-step').select_option('chap--residual--39')
        selected = page.evaluate('STRATUM.getState()')
        assert selected['module'] == 'attention-mixer'
        page.locator('#workspace-tabs [data-panel="pytorch"]').click()
        assert page.locator('.code-entry').count() > 0
        page.locator('#workspace-tabs [data-panel="chapter"]').click()
        assert page.locator('#sinkhorn-step').input_value() == selected['example']
        page.reload();page.wait_for_function('!!window.STRATUM')
        assert page.locator('#sinkhorn-step').input_value() == selected['example']
        page.goto(url+'#stage=inputs&panel=chapter')
        page.wait_for_function('!!window.STRATUM')
        page.locator('[data-input-case="chart-question"]').click()
        page.locator('.sequence-position[data-position-type="IMAGE"]').first.click()
        assert 'IMAGE' in page.locator('.position-detail').inner_text()
        assert page.locator('.image-patch.is-selected').count() > 0
        for theme in ['light', 'dark']:
            page.evaluate('(theme)=>STRATUM.setState({theme})',theme)
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        assert not errors, errors
        print(json.dumps({'browser':browser.version,'chapters':5,'journeys':'PASS','errors':errors}))
        browser.close()

if __name__ == '__main__':
    main()
