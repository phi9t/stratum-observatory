"""Reader journeys through the delivered connected workspace, not upstream runtime proof."""
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
        page.on('pageerror', lambda e: errors.append(str(e)))
        url = (ROOT / 'dist/deepseek-v4.1.html').as_uri()
        page.goto(url + '#model=dsv41&stage=csa2&layer=25')
        page.wait_for_function('!!window.STRATUM')
        assert page.locator('#workspace-tabs').count() == 1, 'Missing connected workspace tabs'
        nodes = page.evaluate('STRATUM.getScene().nodes.map(n=>({id:n.id,name:n.name}))')
        node = next(n for n in nodes if 'index' in n['name'].lower())
        page.evaluate('(id)=>STRATUM.selectNode(id)', node['id'])
        original = page.evaluate('STRATUM.getState()')
        for panel in ['maths', 'pytorch', 'worked', 'sources', 'schematic']:
            page.locator(f'#workspace-tabs [data-panel="{panel}"]').click()
            state = page.evaluate('STRATUM.getState()')
            assert state['panel'] == panel, state
            assert state['module'] == original['module'] and state['layer'] == 25
        page.locator('#workspace-tabs [data-panel="pytorch"]').click()
        assert page.locator('#module-panel .code-line').count() > 0
        restored = page.url
        page.reload()
        page.wait_for_function('!!window.STRATUM')
        assert page.evaluate('STRATUM.getState().panel') == 'pytorch'
        assert page.evaluate('STRATUM.getState().module') == original['module']
        page.locator('#workspace-tabs [data-panel="sources"]').click()
        page.go_back()
        assert page.evaluate('STRATUM.getState().panel') == 'pytorch'
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#navigation-toggle').click()
        assert page.locator('#navigation-toggle').get_attribute('aria-expanded') == 'true'
        page.keyboard.press('Escape')
        assert page.locator('#navigation-toggle').get_attribute('aria-expanded') == 'false'
        # Keyboard panel activation must preserve focus across the rerender.
        page.set_viewport_size({'width': 1600, 'height': 1000})
        maths = page.locator('#workspace-tabs [data-panel="maths"]')
        maths.focus();page.keyboard.press('Enter')
        assert page.evaluate('document.activeElement.dataset.panel') == 'maths'
        page.keyboard.press('Tab')
        assert page.evaluate('document.activeElement.dataset.panel') == 'pytorch'
        page.locator('#workspace-tabs [data-panel="schematic"]').click()
        next_operator = page.get_by_role('button', name='Next operator', exact=True)
        next_operator.focus();page.keyboard.press('Enter')
        assert page.evaluate('document.activeElement.getAttribute("aria-label")') == 'Next operator'
        assert page.evaluate('STRATUM.getState().module') != original['module']
        # Shared systems and evidence must expose their actual operation contracts.
        for filename, fragment in [('hardware.html', ''), ('networking.html', ''),
                                   ('deepseek-v4.1.html', '#view=evidence')]:
            page.goto((ROOT / 'dist' / filename).as_uri() + fragment)
            page.wait_for_function('!!window.STRATUM')
            operation = page.evaluate("""() => {const s=STRATUM.getState();return STRATUM.getScene().nodes.find(n=>n.id===s.module);}""")
            page.locator('#inspector [data-panel="sources"]').click()
            assert page.locator('#module-panel').is_visible()
            assert not page.locator('#analysis-panel').is_visible()
            assert not page.locator('.canvas-frame').is_visible()
            chain = page.locator('#module-panel .complete-references')
            chain.locator('summary').click()
            assert chain.locator('a.source-card').count() == len(set(operation['info']['sources']))
            assert chain.locator('a.source-card').first.is_visible()
            if filename in ('hardware.html', 'networking.html'):
                assert page.locator('#module-panel .implementation-reference').count() == 0
                assert 'Publisher baseline & Transformers' not in page.locator('#module-panel').inner_text()
            page.locator('#workspace-tabs [data-panel="maths"]').click()
            assert page.locator('#module-panel .module-mathematics').is_visible()
            assert operation['info']['formula'] in page.locator('#module-panel').inner_text()
            page.locator('#workspace-tabs [data-panel="schematic"]').click()
            assert page.locator('.canvas-frame').is_visible()
            assert page.locator('#analysis-panel').is_visible()
        for target in ['evidence', 'model']:
            page.goto((ROOT / 'dist/networking.html').as_uri() + '#model=kimi3')
            page.wait_for_function('!!window.STRATUM')
            page.locator(f'#chapter-nav [data-view="{target}"]').click()
            page.wait_for_url('**/kimi-k3.html#*')
            page.wait_for_function('!!window.STRATUM')
            assert page.evaluate('STRATUM.getState().model') == 'kimi3'
            assert page.evaluate('STRATUM.getState().view') == target
        page.evaluate("Object.defineProperty(navigator,'clipboard',{value:undefined,configurable:true})")
        page.locator('#share-view').click()
        assert 'Clipboard unavailable' in page.locator('#toast').inner_text()
        page.evaluate('delete navigator.clipboard')
        assert not errors, errors
        print(json.dumps({'browser': browser.version, 'journey': 'PASS', 'url_restore': restored.split('#')[1], 'errors': errors}))
        browser.close()

if __name__ == '__main__':
    main()
