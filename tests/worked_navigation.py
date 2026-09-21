"""Reader-visible values must be the saved CPU values on the delivered artifact."""
import json
from decimal import Context, ROUND_HALF_UP
from pathlib import Path
from deepdive_navigation import open_chapter_reference


def text(value):
    if isinstance(value, list):
        return '[' + ', '.join(text(x) for x in value) + ']'
    if value is None:
        return 'missing'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        # JavaScript toPrecision rounds exact ties away from zero.
        if value.is_integer():
            return str(int(value))
        decimal = Context(prec=6, rounding=ROUND_HALF_UP).create_decimal_from_float(value)
        rounded = float(decimal)
        # Number.toString uses fixed notation here; Python's .6g switches earlier.
        if 1e-6 <= abs(rounded) < 1e21:
            fixed = format(decimal, 'f')
            return fixed.rstrip('0').rstrip('.') if '.' in fixed else fixed
        scientific = format(rounded, '.6g')
        if 'e' in scientific:
            mantissa, exponent = scientific.split('e')
            return f'{mantissa}e{int(exponent):+d}'
        return scientific
    return str(value)


def exercise(page, root):
    examples = json.loads((Path(root)/'examples/deepseek_v41/worked-results.json').read_text())['examples']
    url = (Path(root)/'dist/deepseek-v4.1.html').as_uri()
    stages = {'dspark':['dspark'], 'module-dspark-attention':['dspark'], 'module-dspark-heads':['dspark'], 'inputs':['inputs'], 'module-vision-inputs':['inputs'], 'residual':['residual'], 'decoder':['replay'], 'csa2':['selection','query'],
              'module-csa2-selection':['selection'], 'module-csa2-attention':['query']}
    for stage, ids in stages.items():
        page.goto(url + '#stage=' + stage + '&panel=schematic')
        page.wait_for_function('window.STRATUM')
        open_chapter_reference(page)
        for identifier in ids:
            panel = page.locator(f'[data-worked="{identifier}"]')
            assert panel.is_visible()
            example = examples[identifier]
            audit = panel.locator('.worked-audit')
            assert not audit.evaluate('el=>el.open')
            audit.locator('summary').focus();audit.locator('summary').press('Enter')
            assert audit.evaluate('el=>el.open')
            for step in example['inputs'] + example['steps']:
                assert panel.locator(f'[data-worked-values="{step["id"]}"]').inner_text() == text(step['values']), (stage, identifier, step['id'])
                row_text = panel.locator(f'[data-worked-step="{step["id"]}"]').inner_text()
                assert step['dtype'] in row_text
                assert ' × '.join(step['axes']) in row_text
            for cell in panel.locator('[data-worked-cell]').all():
                tensor_id, *indices = cell.get_attribute('data-worked-cell').split(':')
                value = next(row['values'] for row in example['inputs'] + example['steps'] if row['id'] == tensor_id)
                for index in indices:
                    value = value[int(index)]
                assert cell.inner_text() == text(value)
            if identifier == 'selection':
                mask = next(row['values'] for row in example['steps'] if row['id'] == 'candidate_mask')
                for index, keep in enumerate(mask):
                    assert panel.locator(f'[data-worked-mask="{index}"]').inner_text() == ('Keep' if keep else 'Exclude')
            if identifier == 'replay':
                for cell in panel.locator('[data-replay-value]').all():
                    lane, depth, position = map(int, cell.get_attribute('data-replay-value').split('-'))
                    value = example[['full','truncated'][lane]][depth]['values'][position]
                    assert cell.inner_text() == text(value)
                assert panel.locator('[data-replay-value]').count() == 6
            audit.locator('summary').press('Enter')
            assert not audit.evaluate('el=>el.open')
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        page.emulate_media(media='print')
        assert page.locator('.worked-code[open]').count() == 1
        assert page.locator('.worked-audit:not([open])').count() == 0
        page.emulate_media(media='screen')
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        assert page.locator('.worked-code[open]').count() == 0
        page.set_viewport_size({'width':390,'height':1000})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        page.set_viewport_size({'width':1440,'height':1000})
        page.reload();page.wait_for_function('window.STRATUM')
        assert page.locator('[data-worked]').count() == len(ids)
