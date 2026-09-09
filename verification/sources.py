"""Explicit, read-only upstream acquisition. Writes candidates, never accepted locks.

Run only on a trusted, network-enabled development machine. Upstream files are
read as data; remote Python, pickle, installers and weights are never executed.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, ssl, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from .evidence import valid_url, safe_path, select_anchor, loads, ROLES, REPOS

MAX_BYTES=8*1024*1024
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError('Redirect refused: inspect the new origin and pin it explicitly.')

def fetch_source(source,root,*,opener=None):
    sid=source.get('id','unknown')
    if not valid_url(source):return {'id':sid,'status':'FAIL','reason':'URL must identify an allowed HTTPS origin and exact git commit/path.'}
    if source.get('role') not in ROLES or source.get('role') in REPOS and source.get('repo') not in REPOS[source['role']]:
        return {'id':sid,'status':'FAIL','reason':'Source authority is not allowed for its role.'}
    if not source.get('license'):return {'id':sid,'status':'BLOCKED','reason':'Record license/redistribution constraints before acquisition.'}
    if source.get('fixture'):return {'id':sid,'status':'FAIL','reason':'Synthetic fixtures cannot enter the production cache.'}
    try:
        request=urllib.request.Request(source['url'],headers={'User-Agent':'STRATUM-evidence/1.0','Accept':'text/plain,application/json,text/html'})
        open_url=opener or urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context())).open
        with open_url(request,timeout=15) as response:
            raw=response.read(MAX_BYTES+1)
            if len(raw)>MAX_BYTES:raise ValueError('Source exceeds 8 MiB budget')
            raw.decode('utf-8','strict')
            if response.getcode()!=200:raise ValueError('Expected HTTP 200')
            etag=response.headers.get('ETag')
        checksum=hashlib.sha256(raw).hexdigest();relative=f'.verification-cache/sha256/{checksum}.txt'
        path=safe_path(Path(root),relative);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Content-address collision')
        path.write_bytes(raw)
        return {'id':sid,'status':'PASS','sha256':checksum,'cache':relative,'retrieved_at':datetime.now(timezone.utc).isoformat(),'etag':etag,'bytes':len(raw)}
    except (ValueError,OSError,UnicodeError) as exc:
        return {'id':sid,'status':'BLOCKED','reason':f'Acquisition unavailable: {type(exc).__name__}: {exc}'}

def acquire(root,catalog,target,candidate,accepted,*,only=None):
    root=Path(root);candidate=Path(candidate);accepted=Path(accepted)
    if candidate.resolve()==accepted.resolve():raise ValueError('Never overwrite the accepted source lock')
    if candidate.exists():raise ValueError('Candidate already exists; use a fresh candidate path')
    out=copy.deepcopy(catalog);newtarget=copy.deepcopy(target);results=[];byid={}
    for s in out.get('sources',[]):
        if only and s['id'] not in only:continue
        r=fetch_source(s,root);results.append(r)
        if r['status']=='PASS':
            for key in ('sha256','cache','retrieved_at','etag'):s[key]=r[key]
            byid[s['id']]=s
    for claim in newtarget.get('claims',[]):
        for evidence in claim.get('evidence',[]):
            sid=evidence.get('source');a=evidence.get('anchor')
            if sid not in byid or not a:continue
            try:
                selected,span=select_anchor(safe_path(root,byid[sid]['cache']).read_bytes(),a)
                a['sha256']=hashlib.sha256(selected.encode()).hexdigest()
                if span:a['resolved_span']=span
            except (ValueError,OSError) as exc:
                results.append({'id':claim['id'],'status':'BLOCKED','reason':f'Anchor unresolved: {exc}'})
    bundle={'schema':'stratum-acquisition-candidate/v1','catalog':out,'target':newtarget,'results':results,
            'review_policy':'Candidate is NOT an accepted lock. Compare value/anchor changes; obtain signed independent reviews. Never copy old approvals.'}
    candidate.parent.mkdir(parents=True,exist_ok=True);candidate.write_text(json.dumps(bundle,indent=2)+'\n')
    return bundle

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['acquire']);p.add_argument('--candidate',required=True);p.add_argument('--only',action='append',default=[]);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args(argv);root=Path(a.root).resolve()
    try:
        cat=loads((root/'verification/manifests/sources.json').read_text());target=loads((root/'verification/manifests/observatory.json').read_text())
        ids={s['id'] for s in cat['sources']}
        if set(a.only)-ids:raise ValueError('Unknown source ID in --only')
        out=acquire(root,cat,target,safe_path(root,a.candidate),root/'verification/manifests/sources.json',only=set(a.only))
        print(json.dumps({'candidate':a.candidate,'results':out['results']},indent=2))
        return 0 if out['results'] and all(r['status']=='PASS' for r in out['results']) else 2
    except (OSError,ValueError,KeyError,TypeError) as exc:print(f'Acquisition error: {exc}');return 1
if __name__=='__main__':raise SystemExit(main())
