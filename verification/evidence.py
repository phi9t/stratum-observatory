"""Offline integrity, code-anchor, claim coverage and review verification.

Never imports upstream Python or evaluates source expressions. Hashes establish
integrity, not truth; interpretive claims also need separately trusted review.
"""
from __future__ import annotations
import ast
import base64
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .model import Finding

HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
ROLES = {'publisher_config','publisher_code','framework_code','vendor_document','library_code','paper','tokenizer_code','checkpoint_metadata'}
# This is a policy root. Adding a publisher/framework requires code-owner review.
REPOS = {
    'publisher_config': {'Qwen/Qwen3.8-27B', 'moonshotai/Kimi-K3', 'Qwen/Qwen3-235B-A22B', 'deepseek-ai/DeepSeek-V2', 'Qwen/Qwen3-8B', 'deepseek-ai/DeepSeek-V3', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3.8-Flash-Next'},
    'publisher_code': {'moonshotai/Kimi-K3','huggingface/transformers','deepseek-ai/DeepSeek-V2','deepseek-ai/DeepSeek-V3'},
    'tokenizer_code': {'moonshotai/Kimi-K3'},
    'checkpoint_metadata': {'moonshotai/Kimi-K3'},
    'framework_code': {'sgl-project/sglang','vllm-project/vllm','NVIDIA/Megatron-LM'},
    'library_code': {'NVIDIA/nccl','NVIDIA/nvshmem','ai-dynamo/nixl','openucx/ucx','linux-rdma/rdma-core','NVIDIA/cutlass','MoonshotAI/FlashKDA'},
}
DOC_HOSTS = {'docs.nvidia.com','developer.nvidia.com','www.nvidia.com','arxiv.org','docs.sglang.io','docs.vllm.ai'}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def loads(text: str):
    def pairs(items):
        out = {}
        for k,v in items:
            if k in out: raise ValueError(f'Duplicate JSON key: {k}')
            out[k] = v
        return out
    def constant(v): raise ValueError(f'Nonfinite JSON value: {v}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
        raise ValueError('Expected a nonempty repository-relative path')
    if '..' in Path(relative).parts: raise ValueError('Parent path traversal rejected')
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()): raise ValueError('Path/symlink escapes repository')
    return p

def pointer(document, path: str):
    if path == '': return document
    if not isinstance(path,str) or not path.startswith('/'): raise ValueError('JSON pointer must begin with /')
    value = document
    try:
        for item in path[1:].split('/'):
            if re.search(r'~(?![01])',item): raise ValueError('Invalid pointer escape')
            key = item.replace('~1','/').replace('~0','~')
            if isinstance(value,list):
                if not re.fullmatch(r'0|[1-9][0-9]*',key): raise ValueError('Invalid array index')
                value = value[int(key)]
            else: value = value[key]
        return value
    except (KeyError,IndexError,TypeError) as exc:
        raise ValueError(f'Unresolved JSON pointer: {path}') from exc

def valid_url(source: dict) -> bool:
    try:
        u = urlsplit(source.get('url',''))
        if u.scheme!='https' or u.username or u.password or u.port not in (None,443) or u.query: return False
        if source.get('kind')=='git':
            rev,repo,path=source.get('revision',''),source.get('repo',''),source.get('path','')
            if not HEX40.fullmatch(rev): return False
            if not path or '..' in Path(path).parts or Path(path).is_absolute(): return False
            if source.get('provider')=='huggingface':
                return u.hostname=='huggingface.co' and unquote(u.path)==f'/{repo}/raw/{rev}/{path}'
            if source.get('provider')=='github':
                return u.hostname=='raw.githubusercontent.com' and unquote(u.path)==f'/{repo}/{rev}/{path}'
            return False
        return source.get('kind')=='document' and u.hostname in DOC_HOSTS
    except (ValueError,TypeError): return False

def check_source(source: dict, root: Path, *, allow_fixtures=False) -> list[Finding]:
    sid=source.get('id','<unnamed>'); out=[]
    def add(code,status,msg): out.append(Finding(code,'evidence',status,sid,msg,'Acquire exact pinned bytes; review the candidate lock.'))
    if source.get('fixture') and not allow_fixtures: add('SOURCE_FIXTURE','FAIL','Synthetic fixture cannot serve as production evidence.')
    role=source.get('role')
    if role not in ROLES: add('SOURCE_ROLE','FAIL','Unknown evidence role.')
    if role in REPOS and source.get('repo') not in REPOS[role]: add('SOURCE_AUTHORITY','FAIL','Repository is not authorized for this evidence role.')
    if source.get('kind')=='git' and not HEX40.fullmatch(str(source.get('revision',''))): add('SOURCE_PIN','FAIL','Git sources require a full immutable 40-hex commit.')
    if source.get('kind')=='document' and not source.get('version'): add('SOURCE_VERSION','BLOCKED','Document edition/snapshot identity is missing.')
    if not valid_url(source): add('SOURCE_URL','FAIL','URL does not match the pinned repository/path or trusted documentation host.')
    if not source.get('license'): add('SOURCE_LICENSE','BLOCKED','Evidence license/redistribution policy is unspecified.')
    sha=source.get('sha256')
    if not isinstance(sha,str) or not HEX64.fullmatch(sha): add('SOURCE_UNFETCHED','BLOCKED','No authenticated acquisition snapshot/checksum is locked.')
    try:
        if not source.get('cache'): raise FileNotFoundError()
        path=safe_path(root,source['cache']); raw=path.read_bytes()
        if len(raw)>8*1024*1024: add('SOURCE_SIZE','FAIL','Source exceeds 8 MiB inspection budget.')
        if isinstance(sha,str) and HEX64.fullmatch(sha) and hashlib.sha256(raw).hexdigest()!=sha:
            add('SOURCE_HASH','FAIL','Cached source bytes differ from the locked SHA-256.')
        raw.decode('utf-8','strict')
    except FileNotFoundError: add('SOURCE_CACHE','BLOCKED','Exact source bytes are absent from the offline cache.')
    except (ValueError,UnicodeError,OSError) as exc: add('SOURCE_PATH','FAIL',str(exc))
    return out

def select_anchor(raw: bytes, anchor: dict) -> tuple[str,list[int]]:
    text=raw.decode('utf-8','strict'); lines=text.splitlines(); kind=anchor.get('kind')
    if kind=='json_pointer':
        value=pointer(loads(text),anchor.get('pointer',''))
        transform=anchor.get('transform','identity')
        if transform=='length':
            if not isinstance(value,(list,dict,str)): raise ValueError('Length transform requires a collection')
            value=len(value)
        elif transform!='identity': raise ValueError('Unknown JSON transform')
        return canonical(value).decode(),[]
    if kind=='python_symbol':
        # Qualified AST lookup prevents a comment or string containing a class name from passing.
        try: tree=ast.parse(text)
        except SyntaxError as exc: raise ValueError('Source is not parsable Python') from exc
        matches=[]
        def visit(node,prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    name=prefix+child.name
                    if name==anchor.get('symbol'): matches.append(child)
                    visit(child,name+'.')
                else: visit(child,prefix)
        visit(tree)
        if len(matches)!=1: raise ValueError('Python symbol is missing or ambiguous')
        node=matches[0]; span=[node.lineno,node.end_lineno]; selected='\n'.join(lines[span[0]-1:span[1]])
    elif kind=='span':
        start,end=anchor.get('start'),anchor.get('end')
        if type(start)!=int or type(end)!=int or not 1<=start<=end<=len(lines): raise ValueError('Invalid line span')
        if not anchor.get('contains'): raise ValueError('Line spans require explicit semantic tokens')
        span=[start,end]; selected='\n'.join(lines[start-1:end])
    elif kind=='text_quote':
        # Quote is only an integrity anchor; meaning still requires reviewer approval.
        quote=anchor.get('quote','')
        if not quote or text.count(quote)!=1: raise ValueError('Quote missing or ambiguous')
        selected=quote;start=text[:text.index(quote)].count('\n')+1;span=[start,start+quote.count('\n')]
    else: raise ValueError('Unknown anchor kind')
    for token in anchor.get('contains',[]):
        if not isinstance(token,str) or not token or token not in selected: raise ValueError(f'Required token not in selected code: {token}')
    return selected,span

def check_anchor(source: dict, anchor: dict, root: Path, *, allow_fixtures=False) -> list[Finding]:
    issues=check_source(source,root,allow_fixtures=allow_fixtures)
    if issues: return issues
    sid=source['id']
    try:
        selected,span=select_anchor(safe_path(root,source['cache']).read_bytes(),anchor)
        actual=hashlib.sha256(selected.encode()).hexdigest()
        if not HEX64.fullmatch(str(anchor.get('sha256',''))):
            return [Finding('ANCHOR_UNLOCKED','evidence','BLOCKED',sid,'Anchor digest is not locked.','Resolve and review the exact source anchor.')]
        if actual!=anchor['sha256']:
            return [Finding('ANCHOR_HASH','evidence','FAIL',sid,'Anchor digest does not match the selected code/value.','Review upstream drift; never move a line anchor silently.')]
        return []
    except (ValueError,OSError,TypeError) as exc:
        return [Finding('ANCHOR_RESOLUTION','evidence','FAIL',sid,str(exc),'Correct the path/symbol/JSON pointer and re-review.')]

def dual_code_roles(sources: list[dict]) -> bool:
    pub={s.get('repo') for s in sources if s.get('role')=='publisher_code'}
    runtime={s.get('repo') for s in sources if s.get('role')=='framework_code'}
    return any(a and b and a!=b for a in pub for b in runtime)

def check_dependencies(claims: list[dict]) -> list[Finding]:
    byid={c.get('id'):c for c in claims};state={};out=[]
    def visit(cid):
        if state.get(cid)==1:
            out.append(Finding('CLAIM_CYCLE','evidence','FAIL',str(cid),'Evidence derivation is cyclic.','Use an acyclic dependency graph.'));return
        if state.get(cid)==2:return
        state[cid]=1
        for dep in byid[cid].get('dependencies',[]):
            if dep not in byid: out.append(Finding('CLAIM_DEPENDENCY','evidence','FAIL',str(cid),f'Missing dependency {dep}.','Register and verify the input claim.'))
            else:visit(dep)
        state[cid]=2
    for cid in byid: visit(cid)
    return out

def check_coverage(inventory: dict, claims: list[dict]) -> list[Finding]:
    values=dict(inventory.get('fields',{})); values.update({k:v['sha256'] for k,v in inventory.get('functions',{}).items()})
    values.update({k:v for k,v in inventory.get('scenes',{}).items()})
    out=[];seen_ids=set();bound={}
    def add(code,subject,msg):out.append(Finding(code,'evidence','FAIL',subject,msg,'Add/update an exact consumer binding with its evidence; do not exclude scientific inputs.'))
    if not values:add('COVERAGE_EMPTY','inventory','Empty input inventory cannot establish coverage.')
    for c in claims:
        cid=c.get('id');consumer=c.get('consumer')
        if cid in seen_ids:add('CLAIM_DUPLICATE',str(cid),'Duplicate claim ID.')
        seen_ids.add(cid)
        if not consumer:continue # Cross-cutting contract claims may lack a single local scalar.
        if consumer in bound:add('CONSUMER_DUPLICATE',str(consumer),'One input is bound by multiple claims.')
        bound[consumer]=cid
        if consumer not in values:add('CONSUMER_MISSING',str(cid),'Bound local consumer no longer exists.')
        elif 'expected' not in c or canonical(c['expected'])!=canonical(values[consumer]):add('CONSUMER_VALUE',str(cid),'Local value/function/scene changed since its evidence binding.')
    for name in values:
        if name not in bound:add('COVERAGE_MISSING',name,'Unregistered input/function/scene in the actual application.')
    return out

def check_review(claim: dict, subject_digest: str, reviews: dict, root: Path, trust: dict | None=None) -> Finding:
    cid=claim.get('id','<unknown>');review=reviews.get(cid)
    blocked=lambda msg: Finding('REVIEW_REQUIRED','evidence','BLOCKED',cid,msg,'Obtain an independent Ed25519-signed review of this exact claim/evidence/consumer digest.')
    if not review:return blocked('No signed semantic review. A reviewed boolean is not evidence.')
    if trust is None:
        try:trust=loads((root/'verification/trust/reviewers.json').read_text())
        except (OSError,ValueError):trust={}
    body=review.get('body',{});reviewer=body.get('reviewer');key=trust.get('reviewers',{}).get(reviewer)
    if not key:return blocked('Reviewer public key is not in the protected trust root.')
    if key.get('revoked'):return blocked('Reviewer key has been revoked in the protected trust root.')
    try:
        if body.get('subject_sha256')!=subject_digest:raise ValueError('Review does not bind the current evidence/consumer digest')
        if body.get('verdict')!='accept':raise ValueError('Review verdict is not acceptance')
        if not body.get('author') or body.get('author')==reviewer:raise ValueError('Independent reviewer is required')
        exp=datetime.fromisoformat(body['expires_at'].replace('Z','+00:00'))
        if exp.tzinfo is None or exp<=datetime.now(timezone.utc):raise ValueError('Review expired or lacks timezone')
        public=safe_path(root,key['public_key']);sig=base64.b64decode(review['signature'],validate=True)
        if len(sig)!=64:raise ValueError('Ed25519 signature must be 64 bytes')
        with tempfile.TemporaryDirectory() as tmp:
            payload=Path(tmp)/'body.json';payload.write_bytes(canonical(body))
            sf=Path(tmp)/'signature';sf.write_bytes(sig)
            run=subprocess.run(['openssl','pkeyutl','-verify','-pubin','-inkey',str(public),'-rawin','-in',str(payload),'-sigfile',str(sf)],capture_output=True,timeout=10)
            if run.returncode:raise ValueError('Review signature verification failed')
        return Finding('REVIEW_VALID','evidence','PASS',cid,'Independent signature binds the current claim and evidence.')
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
        return Finding('REVIEW_INVALID','evidence','FAIL',cid,str(exc),'Re-review; do not copy signatures across changed claims.')
