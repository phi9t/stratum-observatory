"""The landing directory exposes learning paths without opening a model workbench."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verification.delivery import guide_inventory
from verification.aesthetics import parse_color,contrast_ratio
from verification.presentations import PROBE_A11Y


def exercise(page,root):
    root=Path(root);dist=root/'dist'
    page.goto((dist/'index.html').as_uri())
    assert page.locator('h1').inner_text()=='Models, systems, and the data between them.'
    assert page.locator('#scene-mount').count()==0
    page.goto((dist/'index.html').as_uri()+'#directory')
    assert page.locator('#directory').is_visible()
    assert page.url.split('#')[0]==(dist/'index.html').as_uri()
    expected={'foundations.html','qwen3.8.html','deepseek-v4.1.html','kimi-k3.html','hardware.html','networking.html'}
    links=set(page.locator('main a[data-destination]').evaluate_all('xs=>xs.map(a=>a.getAttribute("href"))'))
    assert expected<=links
    page.locator('a[data-destination="foundations"]').click();page.wait_for_function('window.STRATUM')
    assert page.url.split('#')[0]==(dist/'foundations.html').as_uri()
    assert set(page.locator('#model-strip [data-model]').evaluate_all('xs=>xs.map(x=>x.dataset.model)'))=={'ds2','ds3','q38','q330','q3235'}
    page.locator('#atlas-link').click()
    assert page.url.split('#')[0]==(dist/'index.html').as_uri()
    for width in [390,768,1440]:
        page.set_viewport_size({'width':width,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    for name in expected:
        page.goto((dist/'index.html').as_uri());page.locator(f'main a[href="{name}"]').click();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.geometryReport().geometry_defects')==0
    page.goto((dist/'index.html').as_uri())
    page.locator('a[data-destination="data"]').click()
    assert page.locator('h1').inner_text()=='From text to training inputs.'
    assert set(page.locator('.guide section[id]').evaluate_all('xs=>xs.map(x=>x.id)'))=={'tokenizer','packing','masking','storage'}
    for link in page.locator('nav a').all():
        link.click()
        assert page.locator(page.evaluate('location.hash')).is_visible()
    page.locator('.directory-note a[href="index.html"]').click()
    assert page.url.split('#')[0]==(dist/'index.html').as_uri()
    for guide in guide_inventory(root):
        page.context.set_offline(True)
        try:
            page.goto((dist/guide['file']).as_uri())
            # Static documents have no changing application state to announce.
            issues=[i for i in page.evaluate(PROBE_A11Y)['issues'] if i['kind']!='no-live-region']
            assert not issues,issues
            page.keyboard.press('Tab')
            assert page.evaluate("() => {const e=document.activeElement,s=getComputedStyle(e);return e!==document.body&&(s.outlineStyle!=='none'||s.boxShadow!=='none')}")
            for width in [390,768,1440]:
                page.set_viewport_size({'width':width,'height':1000})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            for media in ['screen','print']:
                page.emulate_media(media=media,reduced_motion='reduce')
                samples=page.locator('main p,main h1,main h2,main h3,main a').evaluate_all("""xs=>xs.map(e=>{let b=e,bg;while(b){bg=getComputedStyle(b).backgroundColor;if(bg!=='rgba(0, 0, 0, 0)'&&bg!=='transparent')break;b=b.parentElement}const s=getComputedStyle(e);return {text:e.textContent,color:s.color,bg,size:parseFloat(s.fontSize),duration:s.transitionDuration}})""")
                for sample in samples:
                    fg,bg=map(parse_color,[sample['color'],sample['bg']])
                    assert fg and bg and contrast_ratio(fg[:3],bg[:3])>=(7 if media=='screen' else 4.5),sample
                    assert sample['size']>=12,sample
                    assert sample['duration'] in ('0s','0s, 0s'),sample
            assert page.pdf().startswith(b'%PDF')
        finally:
            page.context.set_offline(False)
            page.emulate_media(media='screen',reduced_motion='no-preference')
    for model,file in [('ds2','foundations.html'),('q3827','qwen3.8.html'),('dsv41','deepseek-v4.1.html')]:
        page.goto((dist/'index.html').as_uri()+f'#model={model}&view=serving&context=8192')
        page.wait_for_url(f'**/{file}#*');page.wait_for_function('window.STRATUM')
        state=page.evaluate('STRATUM.getState()')
        assert state['model']==model and state['view']=='serving' and state['context']==8192
    page.set_viewport_size({'width':1840,'height':1400})

if __name__=='__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True)
        exercise(browser.new_page(),Path(__file__).resolve().parents[1]);browser.close()
    print('Landing directory passed')
