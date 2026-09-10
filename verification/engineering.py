"""Run fresh software gates and bind the exact artifact to an evidence lock."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from .evidence import digest,loads

def toolchain_digest(root):
    files=[]
    # tests/ produces the browser evidence this receipt attests to, so it binds too.
    for folder,pattern in [('verification','*.py'),('tools','*.mjs'),('tools','*.py'),('tests','*.py'),('tests','*.mjs'),('.github/workflows','*.y*ml')]:
        files+=list((root/folder).rglob(pattern))
    files+=list((root/'verification/schemas').glob('*.json'))+list((root/'verification/trust').glob('*.json'))+list((root/'verification/trust').glob('*.pem'))
    # package.json defines the commands; the Python pin and lock define the interpreter and
    # the browser that produced the geometry result. A receipt must not survive either changing.
    # The aesthetics policy binds too: loosening a threshold is the only realistic attack on
    # that gate. Recorded baselines deliberately do not, so re-recording a reviewed layout
    # change does not invalidate every receipt.
    files+=[root/'package.json',root/'pyproject.toml',root/'uv.lock',root/'.python-version',
            root/'.github/CODEOWNERS',root/'verification/baselines/aesthetics/policy.json']
    return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()})

def scrape_count(text,pattern):
    """Read a test count out of a runner's summary line, or None if it cannot be read."""
    match=re.search(pattern,text)
    return int(match[1]) if match else None

def _openssl_identity():
    """Name the binary that would verify review signatures, or say none can."""
    from .evidence import ed25519_openssl
    found=ed25519_openssl()
    if not found:return 'none: no Ed25519-capable OpenSSL on this host'
    out=subprocess.run([found,'version'],capture_output=True,text=True,timeout=10)
    return out.stdout.strip() or found

def main():
    root=Path(__file__).resolve().parents[1];out=root/'verification/reports';out.mkdir(exist_ok=True)
    receipt=out/'engineering.json';receipt.unlink(missing_ok=True)
    started=datetime.now(timezone.utc).isoformat();commands={}
    # `uv run` can re-resolve the lock mid-run. If it does, the receipt would bind a
    # toolchain the tests never used, so refuse rather than attest to the wrong one.
    toolchain_before=toolchain_digest(root)
    steps=[('verifier-tests',[sys.executable,'-m','unittest','discover','-s','verification/tests','-v']),('node-tests',['npm','test']),('build',['npm','run','build'])]
    # 600s, not 180s: the verifier suite runs 383 tests taking 108-113s on an idle machine and
    # has already tripped 180s once under load. A timeout that fires on a slow machine reports a
    # failure that did not happen, which is the one thing this gate must never do.
    for name,cmd in steps:
        result=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        (out/f'{name}.log').write_text(result.stdout);commands[name]={'argv':cmd,'exit':result.returncode}
        print(result.stdout[-1800:])
        if result.returncode:return result.returncode
    artifact=root/'dist/index.html';a=hashlib.sha256(artifact.read_bytes()).hexdigest()
    result=subprocess.run(['npm','run','build'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (out/'repeat-build.log').write_text(result.stdout)
    if result.returncode:return result.returncode
    b=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if a!=b:print('FAIL: nondeterministic build');return 1
    old=root/'tests/browser-results.json';old.unlink(missing_ok=True)
    result=subprocess.run([sys.executable,'tests/browser.test.py'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    (out/'browser.log').write_text(result.stdout);commands['browser']={'argv':[sys.executable,'tests/browser.test.py'],'exit':result.returncode};print(result.stdout)
    if result.returncode:return result.returncode
    browser=loads(old.read_text())
    if any(browser.get(k)!=0 for k in ('geometry_defects','runtime_errors','external_runtime_requests')):return 1
    if toolchain_digest(root)!=toolchain_before:print('FAIL: verifier toolchain changed during the run');return 1
    inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,capture_output=True,check=True,timeout=60).stdout)
    target=loads((root/'verification/manifests/observatory.json').read_text());catalog=loads((root/'verification/manifests/sources.json').read_text())
    counts={'verifier':scrape_count((out/'verifier-tests.log').read_text(),r'Ran (\d+) tests'),
            'atlas':scrape_count((out/'node-tests.log').read_text(),r'tests (\d+)')}
    # A receipt that cannot count the tests it ran is not evidence that they ran.
    for name,value in counts.items():
        if not isinstance(value,int) or value<=0:print(f'FAIL: could not read the {name} test count; the runner output format changed');return 1
    data={'schema':'stratum-engineering/v1','status':'PASS','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),
        'consumer_sha256':digest(inv['functions']),'verifier_sha256':toolchain_digest(root),'lock_sha256':digest({'target':target,'catalog':catalog}),
        'artifact_sha256':a,'repeat_build_sha256':b,'deterministic_build':a==b,'commands':commands,'browser':browser,
        'test_counts':counts,
        'toolchain':{'python':sys.version.split()[0],'openssl':_openssl_identity(),'browser':browser.get('browser')},
        'scope':'Local software/geometry tests only. Does not attest to upstream source truth, GPU execution or physical fabric conformance.'}
    receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
