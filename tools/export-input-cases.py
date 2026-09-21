#!/usr/bin/env python3
"""Reproduce derived input cases without distributing publisher assets.

Run with the locked examples environment. --check compares actual execution when
cache is available; missing cache/dependencies/sandbox exits 2 (BLOCKED).
"""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'examples/deepseek_v41/input-cases.json'
SOURCE_FILES = {'dsv41-encoding':'encoding.py','dsv41-tokenizer':'tokenizer.json',
    'dsv41-tokenizer-config':'tokenizer_config.json','dsv41-image-processor':'image_processor.py',
    'dsv41-config':'config.json'}
POLICY = '''(version 1)
(deny default)
(allow process-exec process-fork signal sysctl-read mach-lookup)
(allow file-read-metadata)
(allow file-read* (literal "/") (subpath "/private/var/db") (subpath "/private/etc") (subpath "/dev") (subpath "/System") (subpath "/usr") (subpath "/Library")
    (subpath (param "VENV_LINK")) (subpath (param "PYTHON")) (subpath (param "VENV")) (subpath (param "STAGE"))
    (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))
(allow file-write-data (literal "/dev/null"))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_fixture(fixture):
    """Check sequence ownership, including negative corruption of image mappings."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(fixture['schema'] == 'stratum-input-cases/v1', 'schema')
    for case in fixture['cases']:
        ids, types, positions = case['token_ids'],case['token_types'],case['positions']
        require(len(ids) == len(types) == len(positions) == len(case['token_pieces']), 'sequence lengths')
        for i, position in enumerate(positions):
            require(position['index'] == i and position['token_id'] == ids[i] and position['type'] == types[i]
                and position['piece'] == case['token_pieces'][i], 'position binding')
        image = case['image']
        if image is None:
            require(ids == case['serialized_token_ids'] and set(types) == {-1}, 'text sequence')
            continue
        start,end = image['span']
        image_id = image['preprocessing_args']['image_token_id']
        require(ids[:start]+[image_id]+ids[end:] == case['serialized_token_ids'], 'serialized image expansion')
        require(sha(base64.b64decode(image['chart']['png_base64'])) == image['chart']['png_sha256'], 'chart bytes')
        h,w = image['merged_grid']
        expected = [0] + ([1]*w+[2])*h+[3]
        require(types[start:end] == expected, 'image layout')
        require(all(t == -1 for t in types[:start]+types[end:]), 'image ownership')
        require(all(t == image['preprocessing_args']['image_token_id'] for t in ids[start:end]), 'image ids')
        require(len(image['merged']) == h*w, 'merged grid')
        vh,vw = image['patch_grid']
        require(len(image['patches']) == vh*vw, 'patch grid')
        ratio = image['preprocessing_args']['vision_downsample_ratio']
        for merged in image['merged']:
            idx = merged['index']
            r,c = divmod(idx,w)
            require(merged['row'] == r and merged['column'] == c, 'merge coordinates')
            require(merged['token_index'] == start+1+r*(w+1)+c, 'merge token ownership')
            require(merged['patch_indices'] == [p['index'] for p in image['patches'] if p['merged_index'] == idx], 'merge patch ownership')
        for patch in image['patches']:
            r,c = divmod(patch['index'],vw)
            mi = (r//ratio)*w+c//ratio
            require(patch['row'] == r and patch['column'] == c and patch['merged_index'] == mi, 'patch coordinates')
            require(patch['token_index'] == image['merged'][mi]['token_index'], 'patch token ownership')


def sandbox_command(stage):
    return ['/usr/bin/sandbox-exec','-D',f'PYTHON={Path(sys.base_prefix).resolve()}',
        '-D',f'VENV_LINK={sys.prefix}','-D',f'VENV={Path(sys.prefix).resolve()}','-D',f'STAGE={stage}', '-p',POLICY,
        str(Path(sys.executable).resolve()),'-I','-B']


def run_isolated(stage, args):
    env = {'PATH':'/usr/bin:/bin','TMPDIR':str(stage),
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    return subprocess.run(sandbox_command(stage)+args,cwd=stage,env=env,capture_output=True,text=True,timeout=90)


def prove_sandbox(stage):
    """Fake code must fail before any publisher module is executed."""
    target = stage / 'readonly'
    target.write_text('preserve')
    attempts = [f"open({str(path)!r}, 'w').write('bad')" for path in [
        stage/'extra',stage/'../escape',target,stage/'.git',stage/'.agents',stage/'.codex',stage/'.trae']]
    attempts += ["import socket; socket.socket().connect(('127.0.0.1',9))"]
    for code in attempts:
        result = run_isolated(stage,['-c',code])
        if result.returncode == 0 or not ('Operation not permitted' in result.stderr or 'PermissionError' in result.stderr):
            raise RuntimeError(f'Sandbox denial test failed ({result.returncode}, {code}): '+result.stderr)
    if target.read_text() != 'preserve' or (stage/'extra').exists():
        raise RuntimeError('Sandbox write boundary failed')


def generate():
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('BLOCKED: reproduction requires the audited macOS Seatbelt runner')
    for dependency in ('tokenizers','PIL','torch','numpy'):
        if importlib.util.find_spec(dependency) is None:
            raise RuntimeError(f'BLOCKED: locked examples dependency unavailable: {dependency}')
    catalog = {s['id']:s for s in json.loads((ROOT/'verification/manifests/sources.json').read_text())['sources']}
    sources = []
    with tempfile.TemporaryDirectory(prefix='stratum-input-cases-') as temporary:
        stage = Path(temporary).resolve()
        for id_, name in SOURCE_FILES.items():
            source = catalog[id_]
            path = ROOT/source['cache']
            if not path.exists():
                raise RuntimeError(f'BLOCKED: pinned cache unavailable: {id_}')
            raw = path.read_bytes()
            if sha(raw) != source['sha256']:
                raise ValueError(f'Pinned source drift: {id_}')
            (stage/name).write_bytes(raw)
            sources.append({key:source[key] for key in ('id','repo','path','revision','sha256','url')})
        shutil.copyfile(ROOT/'tools/input_cases_worker.py',stage/'worker.py')
        prove_sandbox(stage)
        site = str(Path(sys.prefix).resolve() / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
        code = f'import sys, runpy; sys.path.insert(0, {site!r}); sys.argv = [{str(stage/"worker.py")!r}, {str(stage)!r}]; runpy.run_path(sys.argv[0], run_name="__main__")'
        result = run_isolated(stage,['-c',code])
        if result.returncode:
            raise RuntimeError('BLOCKED: isolated publisher execution failed: '+result.stderr)
        fixture = json.loads(result.stdout)
        fixture.update({'schema':'stratum-input-cases/v1','model':'deepseek-ai/DeepSeek-V4.1-Flash',
            'sources':sources,'recipe_sha256':{path:sha((ROOT/path).read_bytes()) for path in
                ('tools/export-input-cases.py','tools/input_cases_worker.py')},
            'scope':'Exact pinned chat serialization, tokenization and image preprocessing; no model weights, vision inference or scientific acceptance.',
            'sandbox':{'executor':'/usr/bin/sandbox-exec','policy_sha256':sha(POLICY.encode()),
                'network':'denied','filesystem_writes':'denied','negative_boundary_checks':'PASS'}})
        validate_fixture(fixture)
        return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    try:
        fixture = generate()
        rendered = json.dumps(fixture,indent=2,ensure_ascii=False)+'\n'
        if args.check:
            if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
                raise ValueError('Input fixture differs from current pinned execution; regenerate')
            print('PASS: pinned input fixture reproduces exactly')
        else:
            OUTPUT.write_text(rendered)
            print(f'Wrote {OUTPUT.relative_to(ROOT)}')
    except RuntimeError as error:
        print(str(error),file=sys.stderr)
        return 2
    except (ValueError,KeyError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
