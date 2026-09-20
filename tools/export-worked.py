"""Compute portable teaching traces on CPU; never claim checkpoint or GPU parity."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'examples/deepseek_v41/worked-results.json'


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    sources = sorted((root / 'examples/deepseek_v41').glob('*.py'))
    sources += [root / 'tools/export-worked.py', root / 'uv.lock']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)}


def compute() -> dict:
    import torch
    from examples.deepseek_v41.worked_examples import run_worked_examples
    torch.set_num_threads(1)
    with torch.inference_mode():
        examples = run_worked_examples()
    # A compact payload string also gives Node an unambiguous byte-level digest.
    payload = json.dumps(examples, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    return {'schema': 'stratum-cpu-worked/v1', 'device': 'cpu',
            'torch_version': torch.__version__, 'source_sha256': source_hashes(),
            'payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'payload': payload, 'examples': examples}


def compare(actual, expected, location='examples') -> None:
    """Float32 execution may vary in its final bits across CPU architectures."""
    if type(actual) is not type(expected):
        raise ValueError(f'{location}: type changed')
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'{location}: fields changed')
        for key in actual:
            compare(actual[key], expected[key], location + '.' + key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f'{location}: shape changed')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{location}[{i}]')
    elif isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f'{location}: CPU result differs: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'{location}: value changed')


def check(saved: dict, fresh: dict) -> None:
    for key in ('schema', 'device', 'source_sha256'):
        if saved.get(key) != fresh[key]:
            raise ValueError(f'Stale CPU traces: {key}; run npm run generate:worked')
    if hashlib.sha256(saved['payload'].encode()).hexdigest() != saved['payload_sha256']:
        raise ValueError('CPU trace payload digest mismatch')
    if json.loads(saved['payload']) != saved['examples']:
        raise ValueError('CPU trace payload differs from examples')
    compare(saved['examples'], fresh['examples'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    fresh = compute()
    if args.check:
        check(json.loads(OUTPUT.read_text()), fresh)
        print('CPU worked traces match current execution (rtol=atol=1e-6).')
    else:
        OUTPUT.write_text(json.dumps(fresh, indent=2, allow_nan=False) + '\n')
        print('Generated CPU worked traces: ' + ', '.join(fresh['examples']))


if __name__ == '__main__':
    main()
