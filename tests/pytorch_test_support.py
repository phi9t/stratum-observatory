"""Inspect authored local modules and the portable programs delivered to readers."""
import importlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def records():
    output = subprocess.check_output(['node', '--input-type=module', '-e',
        "import {PYTORCH_EXAMPLES,exampleProgram} from './src/pytorch.mjs'; console.log(JSON.stringify(PYTORCH_EXAMPLES.map(x=>({...x,program:exampleProgram(x.id)}))));"], cwd=ROOT, text=True)
    return json.loads(output)


def local_module(record):
    return importlib.import_module(record['sourceFile'].removesuffix('.py').replace('/', '.'))
