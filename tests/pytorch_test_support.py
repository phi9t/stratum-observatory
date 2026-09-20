"""Execute only our authored teaching examples; never import publisher code or weights."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def records():
    output=subprocess.check_output(['node','--input-type=module','-e',
        "import {PYTORCH_EXAMPLES} from './src/pytorch.mjs'; console.log(JSON.stringify(PYTORCH_EXAMPLES));"],cwd=ROOT,text=True)
    return json.loads(output)

def load_examples():
    namespace={'__name__':'stratum_pytorch_examples'}
    for example in records():
        exec(compile(example['code'],f"<stratum:{example['id']}>",'exec'),namespace)
    return namespace
