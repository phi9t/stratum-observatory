"""Real chapter journeys through the delivered portable V4.1 file."""
from pathlib import Path
import json
import ast
import subprocess


def exercise(page,root):
    page.context.grant_permissions(['clipboard-read','clipboard-write'])
    artifact=Path(root)/'dist/deepseek-v4.1.html'
    page.goto(artifact.as_uri());page.wait_for_function('window.STRATUM')
    examples=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {PYTORCH_EXAMPLES,exampleProgram} from './src/pytorch.mjs'; console.log(JSON.stringify(PYTORCH_EXAMPLES.map(x=>({...x,program:exampleProgram(x.id)}))));"],cwd=root,text=True))
    config_source=(Path(root)/'examples/deepseek_v41/config.py').read_text()
    model=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {MODELS} from './src/core.mjs'; console.log(JSON.stringify(MODELS.dsv41));"],cwd=root,text=True))
    page.goto(artifact.as_uri()+'#model=dsv41&view=model&stage=block&layer=3')
    page.wait_for_function('window.STRATUM')
    assert page.evaluate('STRATUM.getState().stage')=='encoder'
    assert page.evaluate('STRATUM.getState().layer')==3
    assert page.locator('[data-setting="stage"][data-value="block"]').count()==0
    assert page.locator('#identity-metrics .identity-metric').nth(1).locator('strong').inner_text()==f"{model['encoderLayers']} + {model['layers']-model['encoderLayers']}"
    assert page.locator('#identity-metrics').get_by_text('encoder + decoder layers',exact=True).count()==1
    expected=['encoder','decoder','engram','csa2','dspark']
    for stage in expected:
        page.locator(f'button[data-setting="stage"][data-value="{stage}"]').click()
        assert page.evaluate('STRATUM.getState().stage')==stage
        credit=page.locator('#analysis-panel > .innovation-note a[href="https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html"]')
        assert credit.count()==int(stage in ('decoder','csa2','module-csa2-attention'))
        assert f'stage={stage}' in page.url
        if stage in ('encoder','decoder'):
            first,last=(1,model['encoderLayers']) if stage=='encoder' else (model['encoderLayers']+1,model['layers'])
            expected_layers=list(range(first,last+1))
            assert page.locator('[data-layer]').evaluate_all('xs=>xs.map(x=>Number(x.dataset.layer))')==expected_layers
            assert page.locator('#control-layer option').evaluate_all('xs=>xs.map(x=>Number(x.value))')==expected_layers
            for node in ('norm1','attention','router','out-proj','merge-ffn'):
                assert page.locator(f'[data-node="{node}"]').count()==1
            tile=page.locator(f'[data-layer="{last}"]')
            tile.focus();tile.press('Enter')
            assert page.evaluate('STRATUM.getState().stage')==stage
            assert page.evaluate('STRATUM.getState().layer')==last
            page.select_option('#control-layer',str(first))
            assert page.evaluate('STRATUM.getState().layer')==first
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().stage')==stage
        assert page.locator('#analysis-panel').get_by_text('Execution order',exact=True).count()==1
        notes=page.locator('.chapter-notes')
        if notes.count():
            assert not notes.evaluate('el=>el.open')
            notes.locator('summary').focus();notes.locator('summary').press('Enter')
        assert page.locator('#analysis-panel tbody tr').count()>=4
        assert page.evaluate('STRATUM.geometryReport().geometry_defects')==0
        assert json.loads(page.evaluate('STRATUM.exportConfig()'))['state']['stage']==stage
        links=page.locator('#analysis-panel .analysis-split a[href^="https:"]').evaluate_all('xs=>xs.map(a=>a.href)')
        assert links and all('/main/' not in href for href in links)
        assert all('huggingface.co/deepseek-ai/' in href or 'github.com/sgl-project/sglang/' in href or 'github.com/deepseek-ai/DeepSpec/' in href or 'arxiv.org/html/2607.05147v1' in href for href in links)
        assert page.locator('details[data-example="config"] code').text_content()==config_source
        expected_examples=[x for x in examples if stage in x['stages']]
        assert page.locator('.pytorch-example:not([data-example="config"])').count()==len(expected_examples)>0
        for example in expected_examples:
            panel=page.locator(f'details[data-example="{example["id"]}"]')
            summary=panel.locator('summary')
            summary.focus();summary.press('Enter')
            assert panel.evaluate('el=>el.open')
            assert panel.locator('code').text_content()==example['code']
            assert panel.locator('.python-keyword').count()>0
            assert panel.locator('.python-string').count()>0
            assert panel.locator('.code-line').count()==len(example['code'].split('\n'))
            copy=panel.get_by_role('button',name=f'Copy {example["title"]} Python')
            copy.click()
            assert page.evaluate('navigator.clipboard.readText()')==example['program']
            assert 'Python copied' in page.locator('#toast').inner_text()
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
        # Exercise the denied-clipboard boundary: the existing download remains available.
        first_panel=page.locator('.pytorch-example:not([data-example="config"])').first
        first_panel.locator('summary').click()
        page.evaluate("() => {window.originalCopy=navigator.clipboard.writeText.bind(navigator.clipboard);navigator.clipboard.writeText=async()=>{throw new Error('denied')};}")
        first_panel.locator('[data-copy-example]').click()
        assert 'Clipboard unavailable' in page.locator('#toast').inner_text()
        page.evaluate('() => {navigator.clipboard.writeText=window.originalCopy;}')
        first_panel.locator('summary').click()
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
    # Every code record opens its own complete scene through the real reader controls.
    for example in examples:
        chapter=example['stages'][0]
        stage='module-'+example['id']
        page.locator(f'button[data-setting="stage"][data-value="{chapter}"]').first.click()
        panel=page.locator(f'details[data-example="{example["id"]}"]')
        panel.locator('summary').click()
        panel.get_by_role('button',name='Inspect module schematic').click()
        assert page.evaluate('STRATUM.getState().stage')==stage
        assert f'stage={stage}' in page.url
        credit=page.locator('#analysis-panel > .innovation-note a[href="https://zartbot.github.io/blog/model_arch/dsv41flash_arch/en.html"]')
        assert credit.count()==int(example['id']=='csa2-attention')
        assert page.locator('#control-stage').input_value()==stage
        assert page.locator('.pytorch-example:not([data-example="config"])').count()==1
        assert page.locator('.pytorch-example:not([data-example="config"])').evaluate('el=>el.open')
        assert page.locator(f'details[data-example="{example["id"]}"] .pytorch-code code').text_content()==example['code']
        assert page.evaluate('STRATUM.geometryReport().geometry_defects')==0
        scene=page.evaluate('STRATUM.getScene()')
        assert len(scene['nodes'])>=5
        for node in scene['nodes']:
            page.locator(f'[data-node="{node["id"]}"]').focus()
            page.keyboard.press('Enter')
            assert page.locator('#inspector h3').inner_text()==node['info']['title']
            assert page.locator('#inspector a.source-card').count()>0
        page.locator('#trace-button').click()
        assert page.locator('.trace-edge').count()>0 or not any(e['target']==scene['nodes'][-1]['id'] for e in scene['edges'])
        page.locator('#trace-button').click()
        assert json.loads(page.evaluate('STRATUM.exportConfig()'))['state']['stage']==stage
        assert example['title'] in page.locator('#schematic > title').text_content()
        for view,title in [('training','Training'),('serving','Serving'),('evidence','Evidence')]:
            page.locator(f'#lens-tabs [data-view="{view}"]').click()
            assert title in page.locator('#schematic > title').text_content()
            assert example['title'] not in page.locator('#schematic > title').text_content()
            assert title in page.evaluate('STRATUM.exportSVG()')
        page.locator('#lens-tabs [data-view="model"]').click()
        assert page.evaluate('STRATUM.getState().stage')==stage
        assert example['title'] in page.locator('#schematic > title').text_content()
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.evaluate('STRATUM.getState().stage')==stage
        page.set_viewport_size({'width':390,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.set_viewport_size({'width':1840,'height':1400})
        page.select_option('#control-stage','ced')
        assert page.evaluate('STRATUM.getState().stage')=='ced'
    page.locator('button[data-setting="stage"][data-value="csa2"]').click()
    assert page.locator('button[data-setting="detail"]').count()==0
    for layer in ['3','25','26']:
        page.select_option('#control-layer',layer)
        assert page.evaluate('STRATUM.getState().layer')==int(layer)
        assert f'Layer {layer} is' in page.locator('#analysis-panel').text_content()
    fixed_cpu=page.locator('[data-worked-values]').all_text_contents()
    before=page.locator('#analysis-panel .chapter-notes > .table-scroll > table > tbody').text_content()
    page.select_option('#control-context','131072')
    assert page.locator('#analysis-panel .chapter-notes > .table-scroll > table > tbody').text_content()!=before
    assert page.locator('[data-worked-values]').all_text_contents()==fixed_cpu
    page.locator('button[data-setting="stage"][data-value="ced"]').click()
    assert page.locator('.innovation-table').count()==1

if __name__=='__main__':
    from playwright.sync_api import sync_playwright
    from browser_env import chromium_path
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=chromium_path(pw),headless=True)
        exercise(browser.new_page(),Path(__file__).resolve().parents[1]);browser.close()
    print('V4.1 chapter navigation passed')
