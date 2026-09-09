"""Run fresh software gates and bind the exact artifact to an evidence lock."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from .evidence import digest,loads

def toolchain_digest(root):
    files=[]
    for folder,pattern in [('verification','*.py'),('tools','*.mjs'),('tools','*.py'),('.github/workflows','*.yml')]:
        files+=list((root/folder).rglob(pattern))
    files+=list((root/'verification/schemas').glob('*.json'))+list((root/'verification/trust').glob('*.json'))+list((root/'verification/trust').glob('*.pem'))
    files+=[root/'package.json']
    return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()})

def main():
    root=Path(__file__).resolve().parents[1];out=root/'verification/reports';out.mkdir(exist_ok=True)
    receipt=out/'engineering.json';receipt.unlink(missing_ok=True)
    started=datetime.now(timezone.utc).isoformat();commands={}
    steps=[('verifier-tests',[sys.executable,'-m','unittest','discover','-s','verification/tests','-v']),('node-tests',['npm','test']),('build',['npm','run','build'])]
    for name,cmd in steps:
        result=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
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
    inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,capture_output=True,check=True,timeout=60).stdout)
    target=loads((root/'verification/manifests/observatory.json').read_text());catalog=loads((root/'verification/manifests/sources.json').read_text())
    def number(path,pat):
        m=re.search(pat,(out/path).read_text());return int(m[1]) if m else None
    data={'schema':'stratum-engineering/v1','status':'PASS','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),
        'consumer_sha256':digest(inv['functions']),'verifier_sha256':toolchain_digest(root),'lock_sha256':digest({'target':target,'catalog':catalog}),
        'artifact_sha256':a,'deterministic_build':True,'commands':commands,'browser':browser,
        'test_counts':{'verifier':number('verifier-tests.log',r'Ran (\d+) tests'),'atlas':number('node-tests.log',r'# tests (\d+)')},
        'scope':'Local software/geometry tests only. Does not attest to upstream source truth, GPU execution or physical fabric conformance.'}
    receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
