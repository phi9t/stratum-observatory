"""Export unsigned review requests; never creates an approval or private key."""
import argparse,json
from pathlib import Path
from .evidence import digest,loads
from .runner import source_subject

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--claim');p.add_argument('--out',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
 target=loads((root/'verification/manifests/observatory.json').read_text());sources={s['id']:s for s in loads((root/'verification/manifests/sources.json').read_text())['sources']}
 requests=[{'id':c['id'],'kind':'claim','subject_sha256':source_subject(c,sources),'statement':c['statement']} for c in target['claims'] if c.get('classification')!='presentation']
 requests += [{'id':c['id'],'kind':'contract','subject_sha256':digest(c),'statement':c['domain']+' contract and all constants/bindings'} for c in target['contracts']]
 if a.claim:requests=[r for r in requests if r['id']==a.claim]
 if not requests:raise SystemExit('No matching non-presentation claim/contract')
 out=Path(a.out)
 if out.exists():raise SystemExit('Refusing to overwrite a review request')
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'schema':'stratum-review-requests/v1','warning':'Unsigned requests, NOT approvals. Review actual bytes, branch guards, operand semantics and sources before signing.','requests':requests},indent=2)+'\n')
 print(f'Wrote {len(requests)} unsigned requests to {out}')
if __name__=='__main__':main()
