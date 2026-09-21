"""Run fresh software gates and bind the exact artifact to an evidence lock."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from .evidence import digest,loads
from .delivery import artifact_names

def toolchain_digest(root):
    files=[]
    # tests/ produces the browser evidence this receipt attests to, so it binds too.
    for folder,pattern in [('verification','*.py'),('tools','*.mjs'),('tools','*.py'),('tests','*.py'),('tests','*.mjs'),('examples','*.py'),('.github/workflows','*.y*ml')]:
        files+=list((root/folder).rglob(pattern))
    files+=list((root/'verification/schemas').glob('*.json'))+list((root/'verification/trust').glob('*.json'))+list((root/'verification/trust').glob('*.pem'))
    # package.json defines the commands; the Python pin and lock define the interpreter and
    # the browser that produced the geometry result. A receipt must not survive either changing.
    # The aesthetics policy binds too: loosening a threshold is the only realistic attack on
    # that gate. Recorded baselines deliberately do not, so re-recording a reviewed layout
    # change does not invalidate every receipt.
    files+=[root/'package.json',root/'pyproject.toml',root/'uv.lock',root/'.python-version',
            root/'.github/CODEOWNERS',root/'verification/baselines/aesthetics/policy.json',root/'verification/manifests/accounting-methods.json']
    return digest({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()})

def delivery_artifacts(root):
    """Bind every delivered entry point, rejecting a missing page."""
    directory=Path(root)/'dist'
    required=artifact_names(root)
    present={p.name for p in directory.glob('*.html')}
    if not required<=present:raise FileNotFoundError(f'Missing delivered pages: {sorted(required-present)}')
    return {name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in sorted(present)}


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

def public_argv(command):
    """Keep the invocation reproducible without exposing the local interpreter location."""
    return [Path(command[0]).name, *command[1:]]


def public_log(text, root):
    """Software logs are diagnostic output, not raw scientific evidence."""
    for path, label in sorted(((str(root), '<repo>'), (str(Path.home()), '<home>'),
                               (sys.executable, Path(sys.executable).name)), key=lambda x: -len(x[0])):
        text = text.replace(path, label)
    return text


def main():
    root=Path(__file__).resolve().parents[1];out=root/'verification/reports';out.mkdir(exist_ok=True)
    receipt=out/'engineering.json';receipt.unlink(missing_ok=True)
    started=datetime.now(timezone.utc).isoformat();commands={}
    # `uv run` can re-resolve the lock mid-run. If it does, the receipt would bind a
    # toolchain the tests never used, so refuse rather than attest to the wrong one.
    toolchain_before=toolchain_digest(root)
    steps=[('verifier-tests',[sys.executable,'-m','unittest','discover','-s','verification/tests','-v']),('node-tests',['npm','test']),('pytorch-validation',['uv','run','--frozen','--group','examples','python','-m','examples.deepseek_v41.validate']),('pytorch-examples',['uv','run','--frozen','--group','examples','python','-m','unittest','discover','-s','tests','-p','pytorch_*.py','-v']),('accounting',[sys.executable,'-m','verification.accounting']),('build',['npm','run','build'])]
    # The full verifier suite can exceed 600 seconds on a loaded host. Give that
    # step more time without changing its assertions or accepting partial results.
    for name,cmd in steps:
        result=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1200 if name=='verifier-tests' else 600)
        (out/f'{name}.log').write_text(public_log(result.stdout, root));commands[name]={'argv':public_argv(cmd),'exit':result.returncode}
        print(result.stdout[-1800:])
        if result.returncode:return result.returncode
    artifact=root/'dist/index.html';a=hashlib.sha256(artifact.read_bytes()).hexdigest();pages=delivery_artifacts(root)
    result=subprocess.run(['npm','run','build'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (out/'repeat-build.log').write_text(public_log(result.stdout, root))
    if result.returncode:return result.returncode
    b=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if a!=b or pages!=delivery_artifacts(root):print('FAIL: nondeterministic build');return 1
    old=root/'tests/browser-results.json';old.unlink(missing_ok=True)
    result=subprocess.run([sys.executable,'tests/browser.test.py'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    (out/'browser.log').write_text(public_log(result.stdout, root));commands['browser']={'argv':public_argv([sys.executable,'tests/browser.test.py']),'exit':result.returncode};print(result.stdout)
    if result.returncode:return result.returncode
    workspace_result=subprocess.run([sys.executable,'tests/workspace_browser.py'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
    (out/'workspace-browser.log').write_text(public_log(workspace_result.stdout,root));commands['workspace-browser']={'argv':['python','tests/workspace_browser.py'],'exit':workspace_result.returncode}
    print(workspace_result.stdout)
    if workspace_result.returncode:return workspace_result.returncode
    browser=loads(old.read_text())
    if any(browser.get(k)!=0 for k in ('geometry_defects','runtime_errors','external_runtime_requests')):return 1
    if toolchain_digest(root)!=toolchain_before:print('FAIL: verifier toolchain changed during the run');return 1
    inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=root,text=True,capture_output=True,check=True,timeout=60).stdout)
    target=loads((root/'verification/manifests/observatory.json').read_text());catalog=loads((root/'verification/manifests/sources.json').read_text())
    verifier_log=(out/'verifier-tests.log').read_text();node_log=(out/'node-tests.log').read_text()
    counts={'verifier':scrape_count(verifier_log,r'Ran (\d+) tests'),
            'atlas':scrape_count(node_log,r'tests (\d+)'),
            'pytorch':scrape_count((out/'pytorch-examples.log').read_text(),r'Ran (\d+) tests')}
    # A receipt that cannot count the tests it ran is not evidence that they ran.
    for name,value in counts.items():
        if not isinstance(value,int) or value<=0:print(f'FAIL: could not read the {name} test count; the runner output format changed');return 1
    # "Ran N tests" counts the tests unittest started, not the ones that asserted anything. Two
    # receipt-staleness guards cannot run inside this gate because it deletes the receipt first,
    # and in CI the seven publisher-cache guards skip as well because `.verification-cache/` is
    # absent by design (ruling T789a). A receipt that records only N invites quoting N as
    # "N tests ran", which is how two self-disabling tests became one of this branch's nine
    # instances. Absent skip line means zero skipped, which unittest states by omitting it.
    counts['verifier_skipped']=scrape_count(verifier_log,r'\(skipped=(\d+)\)') or 0
    counts['atlas_skipped']=scrape_count(node_log,r'skipped (\d+)') or 0
    data={'schema':'stratum-engineering/v1','status':'PASS','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),
        'consumer_sha256':digest(inv['functions']),'verifier_sha256':toolchain_digest(root),'lock_sha256':digest({'target':target,'catalog':catalog}),
        'artifact_sha256':a,'artifacts':pages,'repeat_build_sha256':b,'deterministic_build':a==b,'commands':commands,'browser':browser,
        'test_counts':counts,
        'toolchain':{'python':sys.version.split()[0],'openssl':_openssl_identity(),'browser':browser.get('browser')},
        'scope':'Local software/geometry tests only. Does not attest to upstream source truth, GPU execution or physical fabric conformance.'}
    receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
