"""Bind book methods to publication consumers without manufacturing approvals.

Explicit local manifest migration; never fetches or alters accepted source bytes.
Existing claim assertions/evidence remain intact. Refreshing fingerprints alone
cannot satisfy the runner's source, role, contract or independent review gates.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification.evidence import check_source,check_anchor
from verification.runner import source_subject


def bind(root, *, refresh=False):
    root=Path(root)
    read=lambda p:json.loads((root/p).read_text())
    catalog=read('verification/manifests/sources.json')
    target=read('verification/manifests/observatory.json')
    original_target=copy.deepcopy(target);original_catalog=copy.deepcopy(catalog)
    reviews=read('verification/manifests/reviews.json')
    methods=read('verification/manifests/accounting-methods.json')
    candidate_path=root/'verification/candidates/books-20260912.json'
    candidate=read('verification/candidates/books-20260912.json') if candidate_path.exists() else {'catalog':{'sources':[]}}
    known={s['id']:s for s in catalog['sources']}
    for source in candidate['catalog']['sources']:
        if source['id'] in known:
            if known[source['id']]!=source:raise ValueError('Refusing to overwrite accepted source '+source['id'])
        else:
            issues=check_source(source,root)
            if issues:raise ValueError(str(issues))
            catalog['sources'].append(copy.deepcopy(source));known[source['id']]=source
    inv=json.loads(subprocess.check_output(['node','tools/export-verification.mjs'],cwd=root,text=True))
    values={**inv['fields'],**{k:v['sha256'] for k,v in inv['functions'].items()},**inv['scenes']}
    claims={c['id']:c for c in target['claims']}
    for kind,records in [('method',methods['methods']),('example',[e for e in methods['examples'] if e.get('sources')])]:
        for record in records:
            key='accounting.'+kind+'.'+record['id']
            refs=[dict(copy.deepcopy(e),relation='documents') for e in record['sources']]
            for ref in refs:
                if check_anchor(known[ref['source']],ref['anchor'],root):raise ValueError('Invalid anchor '+key)
            claim={'id':key,'domain':'numerics','classification':'derived' if kind=='example' else 'documented','consumer':key,
                   'expected':values[key], 'statement':record.get('scope', 'Analytical methodology with explicit assumptions and exclusions; no checkpoint, hardware or measured performance assertion.'),
                   'evidence':refs,'dependencies':['accounting.method.'+record['method']] if kind=='example' else []}
            if key in claims:
                if claims[key]!=claim and not refresh:raise ValueError('Changed method binding requires --refresh: '+key)
                claims[key].update(claim)
            else:target['claims'].append(claim);claims[key]=claim
    mapping={'book-gqa-parameters':'parameterLedger','book-gqa-cache':'cacheLedger','book-training-state':'trainingLedger','book-ring-allreduce':'networkLedger','book-roofline':'roofline'}
    for method in methods['methods']:
        c=claims['function.'+mapping[method['id']]]
        dependency='accounting.method.'+method['id']
        if dependency not in c['dependencies']:c['dependencies'].append(dependency)
        for e in method['sources']:
            ref=dict(copy.deepcopy(e),relation='documents')
            if ref not in c['evidence']:c['evidence'].append(ref)
    key='function.accountingAssumptions'
    new={'id':key,'domain':'numerics','classification':'derived','consumer':key,'expected':values[key],
         'statement':'Expose method identity, quantities, ownership, units and exclusions from the live ledgers; explanatory accounting metadata is not observed allocation or runtime evidence.',
         'evidence':[dict(copy.deepcopy(e),relation='documents') for m in methods['methods'] for e in m['sources']],
         'dependencies':['function.'+v for v in mapping.values()]+['accounting.method.'+m['id'] for m in methods['methods']]}
    if key in claims:
        if claims[key]!=new and not refresh:raise ValueError('Changed accounting metadata requires --refresh')
        claims[key].update(new)
    else:target['claims'].append(new)
    changed=[]
    if refresh:
        for c in target['claims']:
            key=c.get('consumer')
            if key and key not in values:raise ValueError('Missing consumer requires explicit migration: '+key)
            if key and c.get('expected')!=values[key]:
                allowed={'file.src/app.mjs','file.src/core.mjs','file.src/models.mjs','file.src/styles.css','function.accountingAssumptions'}
                if key not in allowed:raise ValueError('Consumer outside reviewed accounting scope requires explicit migration: '+key)
                changed.append(key);c['expected']=values[key]
    old_claims={c['id']:c for c in original_target['claims']}
    old_sources={s['id']:s for s in original_catalog['sources']}
    for claim in target['claims']:
        cid=claim['id']
        if reviews.get(cid) and cid in old_claims and source_subject(old_claims[cid],old_sources)!=source_subject(claim,known):
            raise ValueError('Reviewed subject would change: '+cid+'. Preserve the signed subject and obtain an explicit independently reviewed migration; no manifests written.')
    for path,value in [('verification/manifests/sources.json',catalog),('verification/manifests/observatory.json',target)]:
        (root/path).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'refreshed_consumers':changed,'book_sources':len(candidate['catalog']['sources']),'method_claims':len(methods['methods'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh',action='store_true');args=p.parse_args();bind(ROOT,refresh=args.refresh)
