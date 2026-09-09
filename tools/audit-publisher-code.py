#!/usr/bin/env python3
"""Record, per publisher checkpoint repository, whether it ships modeling code.

A checkpoint whose publisher ships only configuration cannot satisfy a
publisher-implementation requirement, however much effort is spent. That is a fact about
upstream, so it is recorded as evidence with the repository revision it was observed at,
not asserted in a comment.

Read-only: queries the Hugging Face model API and writes a listing. Downloads no weights.
"""
import hashlib,json,sys,urllib.request
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/publisher-code/audit.json'


def listing(repo):
    url=f'https://huggingface.co/api/models/{repo}'
    raw=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'stratum-verifier'}),timeout=40).read()
    info=json.loads(raw)
    files=sorted(f['rfilename'] for f in info.get('siblings',[]))
    return {'repo':repo,'revision':info.get('sha'),'observed_at':datetime.now(timezone.utc).isoformat(),
            'api_response_sha256':hashlib.sha256(raw).hexdigest(),
            'file_count':len(files),'python_files':[f for f in files if f.endswith('.py')]}


def main():
    catalog=json.loads((ROOT/'verification/manifests/sources.json').read_text())
    repos=sorted({s['repo'] for s in catalog['sources'] if s.get('role')=='publisher_config'})
    records=[]
    for repo in repos:
        try:
            records.append(listing(repo))
        except Exception as exc:
            print(f'{repo}: lookup failed ({type(exc).__name__}); not recorded',file=sys.stderr);return 2
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({'schema':'stratum-publisher-code-audit/v1',
        'scope':'Whether each publisher checkpoint repository ships modeling code. Says nothing about the quality of any implementation.',
        'records':records},indent=2)+'\n')
    for r in records:
        state='ships modeling code' if r['python_files'] else 'CONFIG ONLY - no .py file'
        print(f"  {r['repo']:30} rev {str(r['revision'])[:12]}  {r['file_count']:>3} files  {state}")
    return 0


if __name__=='__main__':raise SystemExit(main())
