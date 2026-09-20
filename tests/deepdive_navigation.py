"""Real chapter journeys through the delivered portable V4.1 file."""
from pathlib import Path
import json
import ast
import subprocess


def exercise(page,root):
    artifact=Path(root)/'dist/deepseek-v4.1.html'
    page.goto(artifact.as_uri());page.wait_for_function('window.STRATUM')
    examples=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {PYTORCH_EXAMPLES,exampleProgram} from './src/pytorch.mjs'; console.log(JSON.stringify(PYTORCH_EXAMPLES.map(x=>({...x,program:exampleProgram(x.id)}))));"],cwd=root,text=True))
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
        expected_examples=[x for x in examples if stage in x['stages']]
        assert page.locator('.pytorch-example').count()==len(expected_examples)>0
        for example in expected_examples:
            panel=page.locator(f'details[data-example="{example["id"]}"]')
            summary=panel.locator('summary')
            summary.focus();summary.press('Enter')
            assert panel.evaluate('el=>el.open')
            assert panel.locator('code').text_content()==example['code']
            with page.expect_download() as pending:
                panel.get_by_role('button',name='Download Python').click()
            downloaded=Path(pending.value.path()).read_text()
            assert downloaded==example['program']
            ast.parse(downloaded)
            assert 'MIT License' in downloaded and '# Sources: https://' in downloaded
            page.set_viewport_size({'width':390,'height':1000})
            code=panel.locator('.pytorch-code')
            panel.get_by_role('button',name='Download Python').focus()
            page.keyboard.press('Tab')
            assert code.evaluate('el=>el===document.activeElement')
            assert code.evaluate('el=>getComputedStyle(el).outlineStyle')!='none'
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.set_viewport_size({'width':1840,'height':1400})
            summary.press('Enter')
            assert not panel.evaluate('el=>el.open')
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        assert page.locator('.pytorch-example:not([open]),.pytorch-license:not([open])').count()==0
        assert page.locator('.pytorch-license').inner_text().find('Permission is hereby granted')>=0
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        assert page.locator('.pytorch-example[open],.pytorch-license[open]').count()==0
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
