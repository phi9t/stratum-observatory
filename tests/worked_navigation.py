"""Reader-visible values must be the saved CPU values on the delivered artifact."""
import json
from pathlib import Path


def text(value):
    if isinstance(value, list):
        return '[' + ', '.join(text(x) for x in value) + ']'
    if value is None:
        return 'missing'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, '.6g')
    return str(value)


def exercise(page, root):
    examples = json.loads((Path(root)/'examples/deepseek_v41/worked-results.json').read_text())['examples']
    url = (Path(root)/'dist/deepseek-v4.1.html').as_uri()
    stages = {'decoder':['replay'], 'csa2':['selection','query'],
              'module-csa2-selection':['selection'], 'module-csa2-attention':['query']}
    for stage, ids in stages.items():
        page.goto(url + '#stage=' + stage)
        page.wait_for_function('window.STRATUM')
        for identifier in ids:
            panel = page.locator(f'[data-worked="{identifier}"]')
            assert panel.is_visible()
            example = examples[identifier]
            for step in example['inputs'] + example['steps']:
                assert panel.locator(f'[data-worked-values="{step["id"]}"]').inner_text() == text(step['values'])
                row_text = panel.locator(f'[data-worked-step="{step["id"]}"]').inner_text()
                assert step['dtype'] in row_text
                assert ' × '.join(step['axes']) in row_text
            if identifier == 'replay':
                assert panel.locator('svg[role="img"]').count() == 1
                assert panel.locator('.worked-wire.worked-missing').count() > 0
                for lane, key in enumerate(['full', 'truncated']):
                    for depth, row in enumerate(example[key]):
                        for position, value in enumerate(row['values']):
                            label = panel.locator(f'[data-replay-value="{lane}-{depth}-{position}"]')
                            expected = [text(v) for v in value] if isinstance(value, list) else [text(value)]
                            assert label.locator('tspan').all_text_contents() == expected
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        page.emulate_media(media='print')
        assert page.locator('.worked-code[open]').count() == 1
        page.emulate_media(media='screen')
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        assert page.locator('.worked-code[open]').count() == 0
        page.set_viewport_size({'width':390,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        page.set_viewport_size({'width':1440,'height':1000})
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.locator('[data-worked]').count() == len(ids)
