"""Independent arithmetic cross-check; source acceptance and runtime are separate gates."""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
from pathlib import Path
from .evidence import loads

METHODS=('book-gqa-parameters','book-gqa-cache','book-training-state','book-ring-allreduce','book-roofline')
ROOT=Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def close(a,b):
    return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-6)

def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite number')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    if isinstance(value,list):
        for v in value:finite(v)

def gqa_parameters(m):
    """Enumerate tensor shapes; never call the JavaScript parameter ledger."""
    d=m['d']; layers=m['layers']; hd=m['headDim']
    embedding=m['vocab']*d*(1 if m.get('tieWordEmbeddings',False) else 2)
    attention=layers*(d*m['heads']*hd + d*m['kvHeads']*hd*2 + m['heads']*hd*d + (2*hd if m['qkNorm'] else 0))
    dense=min(layers,m['denseLayers']) if m['experts'] else layers
    routed=(layers-dense)*m['experts']*3*(m['latent'] or d)*m['expertFF']
    shared=(layers-dense)*m['shared']*3*d*m['expertFF']
    ffn=dense*3*d*m['ff']+routed+shared+(layers-dense)*d*m['experts']
    main=embedding+attention+ffn+(2*layers+1)*d
    return dict(embedding=embedding,attention=attention,routed=routed,shared=shared,main=main,resident=main+m['ngram'])

def audit_export(data):
    """Classify every required live scenario; malformed or incomplete inventory fails closed."""
    out=[]
    def finding(c, classification, scope, expected=None, actual=None, **extra):
        out.append(dict(id=c.get('id','inventory'),method=c.get('method'),model=c.get('model'),hardware=c.get('hardware'),settings=c.get('settings',{}),classification=classification,scope=scope,expected=expected,actual=actual,**extra))
    try:
        finite(data)
        if data['schema']!='stratum-accounting-export/v1':raise ValueError('Unknown export schema')
        hashes=[data['consumer_sha256'],*data['function_sha256'].values()]
        if set(data['function_sha256'])!={'sanitizeState','parameterLedger','cacheLedger','trainingLedger','networkLedger','roofline'} or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):raise ValueError('Missing or malformed consumer/function identity')
        models={m['id']:m for m in data['models']}; hardware={h['id']:h for h in data['hardware']}
        if not models or not hardware or len(models)!=len(data['models']) or len(hardware)!=len(data['hardware']):raise ValueError('Missing or duplicate live identities')
        axes=data['axes']; expected={}
        defaults=data['defaults']
        unique=lambda values:list(dict.fromkeys(values))
        ranks=unique([1,defaults['ranks']]+[h['gpus'] for h in hardware.values()])
        contexts=unique([1,defaults['context']]+[m['maxContext'] for m in models.values()])
        required_axes=dict(cache=[dict(context=s,batch=b,kvBytes=w) for s in contexts for b in unique([1,defaults['batch'],2]) for w in (1,2,4)],training=[dict(ranks=n,trainMode=mode) for n in ranks for mode in ('ddp','fsdp','rl')],network=[dict(ranks=n,traffic='allreduce') for n in ranks],intensity=unique([1,defaults['intensity'],2048]))
        if axes!=required_axes:raise ValueError('Scenario axes omit or add required live-model/hardware/settings cases')
        for s in axes['cache']:
            if any(type(s[k])!=int or s[k]<=0 for k in ('batch','context','kvBytes')):raise ValueError('Invalid cache settings')
        if any(type(n)!=int or n<1 or n>4096 for n in ranks):raise ValueError('Invalid rank axis')
        for m in models.values():
            if any(type(m[k])!=int or m[k]<=0 for k in ('layers','d','heads','headDim','vocab')):raise ValueError('Invalid model tensor dimension')
        for h in hardware.values():
            if any(not close(h[k],h[k]) or h[k]<=0 for k in ('peakTF','hbmTBps')):raise ValueError('Invalid hardware rate')
        for m in models:
            expected[f'book-gqa-parameters/{m}']=('book-gqa-parameters',m,{})
            for method,axis in zip(METHODS[1:4],('cache','training','network')):
                if not axes[axis]:raise ValueError('Empty scenario axis')
                for i,s in enumerate(axes[axis]):expected[f'{method}/{m}/{i}']=(method,m,s)
        if not axes['intensity']:raise ValueError('Empty intensity axis')
        for h in hardware:
            for i,intensity in enumerate(axes['intensity']):expected[f'book-roofline/{h}/{i}']=('book-roofline',h,{'intensity':intensity})
        cases={c['id']:c for c in data['cases']}
        if len(cases)!=len(data['cases']) or set(cases)!=set(expected):raise ValueError('Missing, extra or duplicate required cases')
        parameters={m:c['actual']['resident'] for m in models for c in [cases[f'book-gqa-parameters/{m}']]}
        if any(not close(p,p) or p<=0 for p in parameters.values()):raise ValueError('Invalid parameter-count input')
        for cid,(method,identity,settings) in expected.items():
            c=cases[cid]
            if c['method']!=method or c.get('model',c.get('hardware'))!=identity or c['settings']!=settings:raise ValueError('Case identity/settings mismatch: '+cid)
            a=c['actual']; e={}; meta={}; scope=''; classification='agreement'
            required={METHODS[0]:('main','resident','embedding','attention','routed','shared'),METHODS[1]:('gqa','mla','history','recurrent','conv','indexer','mainKV','windowKV','total'),METHODS[2]:('weights','gradients','master','moments','total','perRank','bytesPerParameter'),METHODS[3]:('sent','off','rate','seconds'),METHODS[4]:('memory','compute','attainable','ridge')}[method]
            if not isinstance(a,dict) or any(not close(a.get(k),a.get(k)) or a[k]<0 for k in required):raise ValueError('Missing or malformed actual numeric fields: '+cid)
            if method==METHODS[0]:
                m=models[identity]
                if m['architecture']!='gqa':
                    finding(c,'unsupported','Full parameter architecture: generic GQA tensor enumeration does not cover MLA, recurrence, CSA2, mHC or auxiliary projections.',units='parameters',denominator='base model',exclusions=['No independent parameter total established']);continue
                e=gqa_parameters(m);scope='GQA text parameters: vocabulary tying policy, Q/K/V/O projections, QK norms, FFN, router and residual/final norms.'
                meta=dict(units='parameters',denominator='base model',exclusions=['MTP','vision','quantization metadata'],method_applicability='Book projection formula; embedding, FFN and norm tensor policy require model-code provenance separately')
            elif method==METHODS[1]:
                m=models[identity]; arch=m['architecture']; b=settings['batch']; s=settings['context']; width=settings['kvBytes']
                scope='GQA K and V history, using KV heads rather than query heads.'
                meta=dict(units='bytes',denominator='batch × context across owned layers',exclusions=a['exclusions'])
                if arch in ('gqa','hybrid','hybrid-qsa'):
                    layers=m['layers'] if arch=='gqa' else m['layers']//m['globalEvery']
                    e={'gqa':2*b*s*layers*m['kvHeads']*m['headDim']*width}
                    if arch=='gqa':e.update(history=e['gqa'],total=e['gqa'])
                    else:finding(c,'unsupported','Full hybrid cache: generic GQA method checks global K/V only; recurrent, convolution and indexer components require specialized evidence.',**meta)
                elif arch=='ced-csa2':
                    owners=m['kvSourceLayers']
                    if len(set(owners))!=len(owners) or any(type(l)!=int or l<m['swaLayers'] or l>=m['layers'] for l in owners):raise ValueError('Duplicate or invalid CSA2 cache owner')
                    logical_positions=sum(b*s/(m['compressEncoder'] if l<m['encoderLayers'] else m['compressDecoder']) for l in set(owners))
                    e={'mainKV':logical_positions*(m['headDim']/2+m['headDim']/16),'indexer':logical_positions*(m['indexer']/2+m['indexer']/m['indexerScaleGroup']),'windowKV':b*m['window']*m['layers']*m['headDim']}
                    e['total']=sum(e.values());scope='Specialized logical packing cross-check: one allocation per unique Full-mode owner; FP4 payload/scales and FP8 window. This is not a generic book GQA application.'
                    meta['method_applicability']='specialized ownership arithmetic; model-code provenance required separately'
                    finding(c,'unsupported','Book GQA full architecture applicability: shared CSA2 keys/values and compressed ownership are outside the generic formula.',**meta)
                else:
                    finding(c,'unsupported','Full cache architecture is outside generic GQA: latent/recurrent or unknown representation.',**meta);continue
            elif method==METHODS[2]:
                p=parameters[identity]; n=settings['ranks'];mode=settings['trainMode']
                e=dict(weights=2*p,gradients=2*p,master=4*p,moments=8*p,total=16*p,bytesPerParameter=16,perRank=16*p/(1 if mode=='ddp' else n),ranks=n)
                scope='BF16 weights/gradients + FP32 master + Adam moments; conditional on exported resident parameter count.'
                meta=dict(inputs={'residentParameters':p},units='bytes (bytesPerParameter: bytes/parameter)',denominator='global persistent state; perRank divides only for ideal sharding',exclusions=['activations','temporary buffers','allocator overhead','independent parameter-count validation for unsupported architectures'])
                finding(c,'explained_difference','Separate FP32 gradient accumulation buffer adds four bytes per parameter to the declared sixteen-byte baseline.',expected={'baseline':16*p,'additionalFP32Accumulation':4*p,'alternativeTotal':20*p},actual={'baseline':a['total']},**meta)
            elif method==METHODS[3]:
                p=parameters[identity]; n=settings['ranks']; e={'sent':2*(n-1)/n*(2*p),'off':(n-1)/n}
                scope='Ring all-reduce full BF16 gradient buffer; conditional on resident parameters.'
                meta=dict(inputs={'residentParameters':p,'messageBytes':2*p,'ranks':n},units='bytes',denominator='per rank, one-way sent, full collective',exclusions=['received bytes','wire overhead','algorithm selection','measured time'])
            else:
                h=hardware[identity];i=settings['intensity']
                if h['peakFormat']!='Dense FP8':raise ValueError('Roofline precision/sparsity mismatch')
                memory=h['hbmTBps']*1e12*i/1e12; compute=h['peakTF']
                e=dict(memory=memory,compute=compute,attainable=min(memory,compute),ridge=compute*1e12/(h['hbmTBps']*1e12))
                scope='Dense FP8 roofline bound with decimal TB/s and TFLOP/s; supplied FLOP/byte intensity.'
                meta=dict(units='TFLOP/s (ridge: FLOP/byte)',denominator='per GPU',exclusions=['runtime throughput','model FLOP ledger','utilization measurement'])
            mismatch={k:{'expected':v,'actual':a.get(k)} for k,v in e.items() if not close(a.get(k),v)}
            finding(c,'contradiction' if mismatch else classification,scope,e,{k:a.get(k) for k in e},mismatches=mismatch,**meta)
    except (KeyError,TypeError,ValueError,ZeroDivisionError,OverflowError) as exc:
        finding({},'contradiction','Malformed or incomplete accounting export: '+str(exc),units='schema',denominator='export',exclusions=[])
    return out

def method_registry(manifest):
    if manifest['schema']!='stratum-accounting-methods/v1':raise ValueError('Unknown methods schema')
    records=manifest['methods']
    if not isinstance(records,list) or len(records)!=len(METHODS) or {m['id'] for m in records}!=set(METHODS):raise ValueError('Missing, duplicate or extra methods')
    for m in records:
        if not isinstance(m['sources'],list) or not m['sources']:raise ValueError('Missing method sources')
        for source in m['sources']:
            if not isinstance(source['source'],str) or not source['source'] or not isinstance(source['anchor'],dict) or not source['anchor']:raise ValueError('Malformed method source reference')
        for field in ('assumptions','exclusions'):
            if not isinstance(m[field],list) or not m[field] or any(not isinstance(v,str) or not v for v in m[field]):raise ValueError('Missing method '+field)
    return {m['id']:m for m in records}


def audit_examples(manifest):
    """Independent hand-computable fixtures; never claimed as book quotations."""
    findings=[]
    try:
        finite(manifest)
        method_registry(manifest)
        if manifest['schema']!='stratum-accounting-methods/v1':raise ValueError('Unknown methods schema')
        methods=manifest['methods']; examples=manifest['examples']
        if len(methods)!=len(METHODS) or {m['id'] for m in methods}!=set(METHODS):raise ValueError('Missing or duplicate methods')
        if not {'book-gqa-cache-llama13b-8192-v1',*(m+'-toy-v1' for m in METHODS)}=={e['id'] for e in examples} or {e['method'] for e in examples}!=set(METHODS) or len({e['id'] for e in examples})!=len(examples):raise ValueError('Missing, duplicate or unknown method examples')
        for e in examples:
            method=e['method']; x=e['inputs']
            if not isinstance(x,dict):raise ValueError('Malformed example inputs')
            for k,v in x.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool) and v<0:raise ValueError('Negative example dimension/rate')
            positive={METHODS[0]:('d','layers','headDim','vocab','heads','kvHeads','ff'),METHODS[1]:('batch','context','layers','kvHeads','headDim','kvBytes'),METHODS[2]:('parameters',),METHODS[3]:('ranks','messageBytes'),METHODS[4]:('peakTF','hbmTBps','intensity')}[method]
            if any(not close(x[k],x[k]) or x[k]<=0 for k in positive):raise ValueError('Nonpositive or malformed example dimension/rate')
            for field in ('id','units','denominator','scope','exclusions'):
                if not e[field]:raise ValueError('Missing example '+field)
            if method==METHODS[0]:expected=gqa_parameters(x)
            elif method==METHODS[1]:
                expected={'total':2*x['batch']*x['context']*x['layers']*x['kvHeads']*x['headDim']*x['kvBytes']}
                if e['id']=='book-gqa-cache-llama13b-8192-v1':
                    expected.update(decimalGB=expected['total']/10**9,binaryGiB=expected['total']/2**30)
                    if len(e['sources'])<2 or e['published']!={'proseGB':6.7,'tableGiB':6.7}:raise ValueError('Missing anchored book numerical example or published unit declarations')
            elif method==METHODS[2]:
                p=x['parameters'];expected=dict(weights=2*p,gradients=2*p,master=4*p,moments=8*p,total=16*p,additionalFP32Accumulation=4*p,alternativeTotal=20*p)
            elif method==METHODS[3]:
                n=x['ranks'];expected={'sent':2*(n-1)/n*x['messageBytes']}
            else:
                if x['peakFormat']!='Dense FP8':raise ValueError('Example precision mismatch')
                memory=x['hbmTBps']*x['intensity'];expected=dict(memory=memory,compute=x['peakTF'],attainable=min(memory,x['peakTF']),ridge=x['peakTF']/x['hbmTBps'])
            mismatch=set(e['expected'])!=set(expected) or any(not close(e['expected'].get(k),v) for k,v in expected.items())
            findings.append(dict(id=e['id'],method=method,classification='contradiction' if mismatch else 'agreement',scope=e['scope'],units=e['units'],denominator=e['denominator'],exclusions=e['exclusions'],expected=expected,actual=e['expected'],source_ids=[a['source'] for a in e.get('sources',[])]))
            if e.get('published'):
                findings.append(dict(id=e['id']+'/units',method=method,classification='explained_difference',scope='The equation rounds decimal GB to 6.7; the table labels that numeral GiB. Exact arithmetic gives 6.25 GiB. The table unit label is not used as an oracle.',units=e['units'],denominator=e['denominator'],exclusions=e['exclusions'],expected=expected,actual=e['published'],source_ids=[a['source'] for a in e['sources']]))
    except (KeyError,TypeError,ValueError,ZeroDivisionError,OverflowError) as exc:
        findings.append(dict(id='examples',classification='contradiction',scope='Malformed or incomplete examples: '+str(exc),units='schema',denominator='manifest',exclusions=[]))
    return findings


def main():
    raw=b'';data={};records={};path=ROOT/'verification/manifests/accounting-methods.json'
    try:
        raw=subprocess.check_output(['node','tools/export-accounting.mjs'],cwd=ROOT,timeout=120)
        data=loads(raw.decode('utf-8'));methods=loads(path.read_text())
        findings=audit_export(data)+audit_examples(methods)
        records=method_registry(methods)
    except (OSError,subprocess.SubprocessError,UnicodeError,ValueError,KeyError,TypeError) as exc:
        if not isinstance(data,dict):data={}
        findings=[dict(id='accounting-input',classification='contradiction',scope='Unable to read accounting inputs: '+str(exc),units='schema',denominator='audit',exclusions=[])]
    if not findings or any(f.get('classification') not in ('agreement','explained_difference','unsupported','contradiction') for f in findings):
        findings.append(dict(id='classification',classification='contradiction',scope='Missing or unrecognized result classification',units='schema',denominator='audit',exclusions=[]))
    for f in findings:
        record=records.get(f.get('method'),{})
        f['method_assumptions']=record.get('assumptions',[])
        f['source_ids']=list(dict.fromkeys(f.get('source_ids',[])+[s['source'] for s in record.get('sources',[])]))
        if f.get('model'):f['model_source_ids']=next((m['sources'] for m in data['models'] if m['id']==f['model']),[])
        if f.get('hardware'):f['hardware_source_ids']=next((h['sources'] for h in data['hardware'] if h['id']==f['hardware']),[])
    counts={name:sum(f['classification']==name for f in findings) for name in ('agreement','explained_difference','unsupported','contradiction')}
    report=dict(schema='stratum-accounting-report/v1',status='FAIL' if counts['contradiction'] else 'PASS',evidence_class='engineering arithmetic only',source_acceptance='SEPARATE_PUBLICATION_GATE',runtime='NOT_RUN',consumer_sha256=data.get('consumer_sha256'),function_sha256=data.get('function_sha256'),methods_sha256=sha(path.read_bytes()) if path.exists() else None,export_sha256=sha(raw),case_count=len(data.get('cases',[])),counts=counts,findings=findings)
    out=ROOT/'verification/reports';out.mkdir(exist_ok=True)
    (out/'accounting.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Arithmetic accounting audit','',f"Software arithmetic: {report['status']}; {report['case_count']} scenarios.",'','Source acceptance: separate publication gate. Runtime: NOT_RUN.', '', '| Classification | Findings |','|---|---:|']+[f'| {k} | {v} |' for k,v in counts.items()]
    lines+=['',f"Consumer: `{report['consumer_sha256']}`",f"Methods: `{report['methods_sha256']}`",'', 'Detailed raw units, denominators, scope, exclusions, expected/actual values and source IDs: `accounting.json`.','']
    (out/'accounting.md').write_text('\n'.join(lines));print(json.dumps({k:report[k] for k in ('status','case_count','counts')}))
    return int(bool(counts['contradiction']))

if __name__=='__main__':raise SystemExit(main())
