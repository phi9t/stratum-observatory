"""Native disclosure, dynamic accounting, print and offline delivery journeys."""
from pathlib import Path


def exercise_page(page, artifact):
    requests = []
    def requested(request):
        if request.url.startswith(('http://', 'https://')):
            requests.append(request.url)
    page.on('request', requested)
    page.context.set_offline(True)
    try:
        page.goto(Path(artifact).resolve().as_uri())
        page.wait_for_function('window.STRATUM')
        assert page.url.startswith('file://')
        panel = page.locator('details.accounting-assumptions')
        summary = panel.locator('summary')
        assert panel.count() == 1
        assert summary.inner_text().strip() == 'Accounting assumptions'
        assert 'Accounting assumptions' in panel.aria_snapshot(), 'Disclosure needs an accessible name'
        assert summary.evaluate('e=>e.tabIndex') == 0, 'Native summary must be keyboard reachable'
        assert not panel.evaluate('e=>e.open'), 'Notes must initially be collapsed'
        assert not panel.locator('section').first.is_visible()
        summary.focus()
        page.keyboard.press('Enter')
        assert panel.evaluate('e=>e.open'), 'Enter must use native details activation'
        assert panel.locator('section').first.is_visible()
        assert summary.evaluate("e=>{const s=getComputedStyle(e);return e===document.activeElement && (s.outlineStyle!=='none' && parseFloat(s.outlineWidth)>0 || s.boxShadow!=='none')}")
        page.keyboard.press('Space')
        assert not panel.evaluate('e=>e.open'), 'Space must close the native disclosure'
        page.keyboard.press('Enter')
        links = panel.locator('.accounting-citations a')
        assert links.count() > 0
        for link in links.all():
            assert link.inner_text().strip(), 'Citation needs an accessible name'
            assert link.get_attribute('href').startswith('https://')
        page.locator('[data-view="training"]').first.click()
        assert panel.evaluate('e=>e.open'), 'Disclosure state must survive a lens change'
        training = panel.locator('[data-accounting-method="book-training-state"]')
        assert training.count() == 1
        assert '16 bytes per parameter' in training.inner_text()
        assert '20 bytes per parameter' in training.inner_text()
        before = training.inner_text()
        ranks = page.locator('#control-ranks')
        options = ranks.locator('option').evaluate_all('xs=>xs.map(x=>x.value)')
        alternate = next(x for x in options if x != ranks.input_value())
        ranks.select_option(alternate)
        assert page.evaluate('STRATUM.getState().ranks') == int(alternate)
        assert panel.evaluate('e=>e.open')
        assert training.inner_text() == before, 'Global persistent totals must not change with rank count'
        if not page.evaluate('document.body.dataset.pageModel'):
            current = page.evaluate('STRATUM.getState().model')
            cards = page.locator('#model-strip button[data-model]').all()
            next(card for card in cards if card.get_attribute('data-model') != current).click()
            assert page.evaluate('STRATUM.getState().view') == 'training'
            assert training.inner_text() != before, 'Training totals must follow the selected model'
            assert panel.evaluate('e=>e.open')
        # A visible hardware control must update the derived prose, on both entry points.
        page.locator('[data-view="hardware"]').first.click()
        hardware = panel.locator('[data-accounting-method="book-roofline"]')
        before = hardware.inner_text()
        intensity = page.locator('#control-intensity')
        options = intensity.locator('option').evaluate_all('xs=>xs.map(x=>x.value)')
        intensity.select_option(next(x for x in options if x != intensity.input_value()))
        assert hardware.inner_text() != before, 'Derived notes must follow scenario controls'
        page.set_viewport_size({'width': 390, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), 'Disclosure causes mobile page overflow'
        assert panel.evaluate('e=>e.scrollWidth<=e.clientWidth+1'), 'Expanded notes overflow their container'
        summary.click()
        assert not panel.evaluate('e=>e.open')
        page.evaluate("""() => {
            window.accountingPrintProbe=[];
            addEventListener('beforeprint',()=>{
                const el=document.querySelector('details.accounting-assumptions');
                const nodes=[...el.querySelectorAll('section p')];
                accountingPrintProbe.push({event:'before',open:el.open,
                    visible:nodes.length>0 && nodes.every(n=>n.getClientRects().length>0 && getComputedStyle(n).visibility!=='hidden')});
            });
            addEventListener('afterprint',()=>accountingPrintProbe.push({event:'after',open:document.querySelector('details.accounting-assumptions').open}));
        }""")
        page.emulate_media(media='print')
        assert panel.locator('summary,h4,p,a').evaluate_all(
            "xs=>xs.length>0 && xs.every(e=>getComputedStyle(e).color==='rgb(34, 34, 34)')"
        ), 'Printed accounting prose must use dark ink, not screen-theme colors'
        pdf = page.pdf()
        page.emulate_media(media='screen')
        assert pdf.startswith(b'%PDF'), 'Real Chromium print must produce a PDF'
        events = page.evaluate('accountingPrintProbe')
        assert any(e['event']=='before' and e['open'] and e['visible'] for e in events), 'Print must expose closed notes'
        assert events[-1] == {'event': 'after', 'open': False}, 'Printing must restore collapsed notes'
        assert not panel.evaluate('e=>e.open')
        assert not requests, f'Offline notes loaded external runtime content: {requests}'
        return {'origin': 'file://', 'keyboard': True, 'dynamic_controls': True,
                'mobile_width': 390, 'print_restored': True, 'external_requests': len(requests)}
    finally:
        page.remove_listener('request', requested)
        page.context.set_offline(False)
        page.set_viewport_size({'width': 1440, 'height': 1000})


def exercise(page, root):
    artifacts = sorted((Path(root)/'dist').glob('*.html'))
    assert artifacts, 'No portable pages were built'
    return {artifact.name: exercise_page(page, artifact) for artifact in artifacts}


if __name__ == '__main__':
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium_path(pw), headless=True)
        print(exercise(browser.new_page(), root))
        browser.close()
