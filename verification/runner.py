"""Fail-closed offline publication/runtime gate. Never executes downloaded code."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .model import Finding, aggregate, exit_code
from .evidence import (CONFIG_ONLY_PUBLISHERS,canonical,digest,loads,safe_path,pointer,check_source,check_anchor,
                       select_anchor,dual_code_roles,check_dependencies,check_coverage,check_review,HEX64)
from .ced import judge_active_ratio,judge_claim_model_coverage,judge_contract_model_coverage,judge_publisher_agreement
from .domains.contracts import DOMAINS, validate
from .hardware import HARDWARE_DOMAINS, hardware_sha256 as attested_sha256, load as load_hardware, schema_sha256 as hardware_schema_sha256

PROFILES=('publication','runtime')
# Coverage and dependency checks run once over all claims, but these codes accuse one
# specific claim. They must reach that claim's status, or a claim whose bound value
# drifted still reports CLAIM_VERIFIED and still counts as verified science.
CLAIM_SCOPED_FINDINGS={'CLAIM_DUPLICATE','CONSUMER_MISSING','CONSUMER_VALUE','CLAIM_CYCLE','CLAIM_DEPENDENCY'}
PRESENTATION_SUFFIXES={'id','name','family','sources','blurb'}
PRESENTATION_FUNCTIONS={'function.fmt','function.compact','function.gb','function.sanitizeState'}

# Identity is what a reviewer actually read; the rest is refetch bookkeeping.
# Together these must cover every property in schemas/sources.schema.json, so that
# adding a field forces a conscious decision instead of silently escaping the subject.
SOURCE_IDENTITY={'id','kind','role','provider','repo','path','revision','url','sha256','license','version','fixture'}
SOURCE_VOLATILE={'cache','retrieved_at','etag'}

def source_identity(source):
    """Project a source onto the fields a review is actually about."""
    return {k:v for k,v in source.items() if k in SOURCE_IDENTITY}

def source_subject(claim,sources):
    """Any source pin, bytes, anchor, assertion or consumer change invalidates review.

    A no-op refetch does not: re-running acquire on byte-identical upstream rewrites
    retrieved_at/etag, and binding those would push operators toward hand-editing the
    accepted lock to keep signatures alive -- weakening the binding it protects.
    """
    return digest({'claim':claim,'sources':[source_identity(sources.get(e.get('source'),{})) for e in claim.get('evidence',[])]})

def classify_allowed(c):
    key=c.get('consumer',''); kind=c.get('classification')
    if kind not in {'documented','derived','assumed','presentation'}: return False
    if kind=='presentation':
        return key.startswith('DEFAULT_STATE.') or key.split('.')[-1] in PRESENTATION_SUFFIXES or key in PRESENTATION_FUNCTIONS or key in {'file.src/styles.css','file.index.html'}
    if kind=='assumed': return key.startswith('DEFAULT_STATE.') or not key
    return True

def required_roles(c,*,publisher_ships_code=True):
    if c.get('classification') in ('presentation','assumed'):return ()
    domain=c.get('domain')
    if domain=='model':
        # Where the publisher ships no code, the configuration it does ship is the anchor,
        # corroborated by independent implementations.
        return ('publisher_code','framework_code') if publisher_ships_code else ('publisher_config','framework_code')
    if domain in ('gpu','scale_up','scale_out'):return ('vendor_document',)
    if domain=='communication':return ('library_code','vendor_document')
    return ()

def release_record(receipt,*,consumer,lock):
    """The release contract, derived from the engineering receipt.

    Transcribing build hashes into the manifest would guarantee they are stale, and the
    validator only compares record fields to each other -- so a hand-written record with any
    equal pair passes. Deriving one side from the receipt and the other from what is true
    now makes each of those comparisons mean something: the source digest recomputed here
    against the one the receipt bound, and likewise the evidence lock.
    """
    commands=receipt.get('commands',{});browser=receipt.get('browser',{})
    return {'id':'release.observatory',
            'source_sha256':consumer,'built_from_sha256':receipt.get('consumer_sha256'),
            'source_lock_sha256':lock,'receipt_lock_sha256':receipt.get('lock_sha256'),
            'artifact_sha256':receipt.get('artifact_sha256'),'repeat_build_sha256':receipt.get('repeat_build_sha256'),
            'external_requests':browser.get('external_runtime_requests'),
            'geometry_defects':browser.get('geometry_defects'),
            'runtime_errors':browser.get('runtime_errors'),
            'commands':{'node':commands.get('node-tests',{}).get('exit'),'browser':commands.get('browser',{}).get('exit')}}


def apply_bindings(record,bindings,fields):
    """Replace declared record fields with the application's own values.

    Supports a dotted path so a nested constant can be bound too: GPU capacity lives at
    `memory.value`, and without this every capacity in the manifest stays hand-transcribed.
    Returns the consumers that do not exist, so the caller can report them.
    """
    missing=[]
    for field,consumer in bindings.items():
        if consumer not in fields:missing.append(consumer);continue
        target=record;parts=field.split('.')
        for part in parts[:-1]:
            nxt=target.get(part)
            if not isinstance(nxt,dict):nxt={};target[part]=nxt
            target=nxt
        target[parts[-1]]=fields[consumer]
    return missing


def contract_record(rec,fields):
    """The contract record as the gate judges it, plus the consumers its bindings did not find.

    A bound field (`memory.value` -> `HARDWARE.h100.hbmGB`) means the manifest text is not what
    the contract asserts; the application's own constant is. Anything binding a contract must
    bind this record, not the manifest text -- and must get it from here, so the record a
    receipt is digested against cannot drift from the record the domain validators judge.
    """
    record=json.loads(json.dumps(rec.get('record',{})))  # deep copy: bindings write into nested fields
    return record,apply_bindings(record,rec.get('bindings',{}),fields)


def contract_sha256(rec,fields):
    """The digest a hardware-attribute receipt carries to name the contract record it saw.

    Shared with the capture tooling so the receipt and the gate cannot compute it differently.
    """
    return digest(contract_record(rec,fields)[0])


def check_receipt(root, receipt, expected_lock, inventory):
    out=[]
    def add(code,status,msg):out.append(Finding(code,'release',status,'engineering',msg,'Run npm run verify:engineering on this exact checkout; never reuse an old report.'))
    if not receipt: add('ENGINEERING_NOT_RUN','NOT_RUN','No fresh engineering receipt for the current atlas.');return out
    if receipt.get('schema')!='stratum-engineering/v1':add('RECEIPT_SCHEMA','FAIL','Unrecognized engineering receipt.')
    from .engineering import toolchain_digest
    if receipt.get('verifier_sha256')!=toolchain_digest(root):add('VERIFIER_STALE','FAIL','Verification code/workflow changed since the receipt.')
    current=digest(inventory['functions'])
    if receipt.get('consumer_sha256')!=current or receipt.get('lock_sha256')!=expected_lock:add('RECEIPT_STALE','FAIL','Receipt binds different source files or evidence lock.')
    try:
        t=datetime.fromisoformat(receipt['completed_at'].replace('Z','+00:00'))
        if t.tzinfo is None or not timedelta(0)<=datetime.now(timezone.utc)-t<=timedelta(hours=24):add('RECEIPT_EXPIRED','FAIL','Engineering receipt is future-dated or older than 24 hours.')
        actual=hashlib.sha256((root/'dist/index.html').read_bytes()).hexdigest()
        if actual!=receipt.get('artifact_sha256'):add('ARTIFACT_STALE','FAIL','Portable HTML differs from the tested artifact.')
        for name in ('verifier-tests','node-tests','build','browser'):
            if receipt.get('commands',{}).get(name,{}).get('exit')!=0:add('ENGINEERING_COMMAND','FAIL',f'Missing/failing command: {name}.')
        b=receipt.get('browser',{})
        for key in ('geometry_defects','runtime_errors','external_runtime_requests'):
            if b.get(key)!=0:add('BROWSER_RESULT','FAIL',f'Browser result {key} is absent or nonzero.')
        if b.get('geometry_states',0)<=0 or b.get('responsive_states',0)<=0:add('BROWSER_EMPTY','FAIL','No browser state coverage.')
        if not receipt.get('deterministic_build') or receipt.get('repeat_build_sha256')!=receipt.get('artifact_sha256'):
            add('BUILD_NONDETERMINISTIC','FAIL','Two fresh builds did not produce identical bytes.')
        # The receipt copies the browser result into itself and is then re-read as truth.
        # Compare it with the file the suite actually wrote.
        try:
            written=loads((root/'tests/browser-results.json').read_text())
        except (OSError,ValueError):
            written=None;add('BROWSER_MISREPORTED','FAIL','No browser result file to corroborate the receipt.')
        if written is not None and any(written.get(k)!=b.get(k) for k in written):
            add('BROWSER_MISREPORTED','FAIL','Receipt browser result disagrees with tests/browser-results.json.')
    except (ValueError,KeyError,OSError,TypeError) as exc:add('RECEIPT_INVALID','FAIL',str(exc))
    # A locally produced receipt asserts every field it is judged on. That is a real limit,
    # so it is a finding rather than a sentence in a report nobody reads. CI supplies
    # `provenance` via attest-build-provenance; its absence blocks, it never passes.
    provenance=receipt.get('provenance') if receipt else None
    if not provenance:
        out.append(Finding('RECEIPT_UNATTESTED','release','BLOCKED','engineering',
                           'Engineering receipt is self-produced and carries no build attestation.',
                           'Run the gate in CI with attest-build-provenance, and protect the branch so the runner cannot be replaced.'))
    elif provenance.get('artifact_sha256')!=receipt.get('artifact_sha256'):
        out.append(Finding('RECEIPT_ATTESTATION','release','FAIL','engineering',
                           'Build attestation does not cover the artifact this receipt binds.',
                           'Attest the artifact that was actually tested.'))
    else:
        out.append(Finding('RECEIPT_ATTESTED','release','PASS','engineering',
                           f"Build attestation from {provenance.get('workflow','CI')} covers the tested artifact."))
    if not [f for f in out if f.status=='FAIL']:
        out.append(Finding('ENGINEERING_CURRENT','release','PASS','engineering','Fresh receipt binds tests, source, evidence lock and portable artifact, and its browser result agrees with the file the suite wrote.'))
    return out

def roles_from_dependencies(claim,claim_sources):
    """Source dicts contributed by a derived claim's declared dependencies.

    Provenance flows through the declared dependency edges: a derivation's evidence roles are
    the roles its inputs cite, so a dependency that cites no publisher implementation cannot
    lend one and CLAIM_ROLE still fires on the dependent.

    This deliberately does NOT consult the dependencies' own claim_status, and it is not what
    refuses an unverified input -- the dependency fixpoint is. A derived claim cannot reach PASS
    while any dependency is unverified, however its roles were computed, so conditioning role
    satisfaction on dependency status would add nothing to soundness while reporting a missing
    implementation where the real cause is a missing signature.
    """
    out=[]
    for dep in claim.get('dependencies',[]):
        out.extend(claim_sources.get(dep,[]))
    return out

def demoted_status(prior):
    """The status of a claim the role pass has just blocked.

    A passing claim is taken back to BLOCKED. A claim that was already FAIL keeps FAIL: a
    contradiction is not softened into missing evidence just because a role was also absent.
    Every current reader of claim_status only asks whether it is PASS, so this is inert today --
    it is here so that the first reader to distinguish the two is told the truth.
    """
    return 'BLOCKED' if prior=='PASS' else prior

def audit(root:Path,target:dict,inventory:dict,catalog:dict,*,profile='publication',reviews=None,receipt=None,runtime=None):
    if profile not in PROFILES:raise ValueError(f'Unknown verification profile: {profile}')
    root=Path(root); reviews=reviews or {}; runtime=runtime or {}; findings=[]
    def add(code,domain,status,subject,msg,remedy='Resolve the finding; do not weaken the gate.'):
        findings.append(Finding(code,domain,status,str(subject),msg,remedy))
    if target.get('schema')!='stratum-target/v1' or catalog.get('schema')!='stratum-sources/v1':
        add('MANIFEST_SCHEMA','evidence','FAIL','target','Missing or unsupported manifest version.')
    for name,allowed,obj in [('target',{'schema','id','description','claims','contracts','runtime_obligations'},target),('catalog',{'schema','sources'},catalog)]:
        for key in obj.keys()-allowed:add('MANIFEST_UNKNOWN','evidence','FAIL',name,f'Unknown manifest key {key}.')
    from .schema import validate as shape_validate
    for name,value in [('target',target),('sources',catalog),('runtime',runtime)]:
        path=root/f'verification/schemas/{name}.schema.json'
        if path.exists():
            for error in shape_validate(value,loads(path.read_text())):add('MANIFEST_SHAPE','evidence','FAIL',name,error)
    claims=target.get('claims',[]); contracts=target.get('contracts',[]); rawsources=catalog.get('sources',[])
    if not isinstance(claims,list) or not isinstance(contracts,list) or not isinstance(rawsources,list):
        add('MANIFEST_TYPE','evidence','FAIL','manifest','Claims, contracts and sources must be arrays.');claims=[];contracts=[];rawsources=[]
    ids=[s.get('id') for s in rawsources if isinstance(s,dict)]
    if len(ids)!=len(set(ids)) or len(ids)!=len(rawsources):add('SOURCE_DUPLICATE','evidence','FAIL','sources','Sources must have unique IDs and object values.')
    sources={s['id']:s for s in rawsources if isinstance(s,dict) and isinstance(s.get('id'),str)}
    lock=digest({'target':target,'catalog':catalog}); claim_status={}; claim_sources={}; source_status={}
    if not sources:add('SOURCES_EMPTY','evidence','BLOCKED','sources','No source catalog.')
    for sid,s in sources.items():
        issues=check_source(s,root);findings.extend(issues);source_status[sid]=aggregate(issues) if issues else 'PASS'
    crosscutting=check_coverage(inventory,claims)+check_dependencies(claims)
    findings.extend(crosscutting)
    by_claim={}
    for f in crosscutting:
        if f.code in CLAIM_SCOPED_FINDINGS:by_claim.setdefault(f.subject,[]).append(f)
    for c in claims:
        cid=c.get('id','<missing>');domain=c.get('domain');start=len(findings)
        if domain not in DOMAINS:add('CLAIM_DOMAIN','evidence','FAIL',cid,'Claim has no registered domain.')
        if not classify_allowed(c):add('CLAIM_CLASS','evidence','FAIL',cid,'Invalid classification or attempt to hide scientific input as presentation/assumption.')
        if not c.get('statement'):add('CLAIM_STATEMENT','evidence','BLOCKED',cid,'Claim needs an explicit assertion and scope.')
        evid=c.get('evidence',[]); referents=[]
        if not isinstance(evid,list):add('CLAIM_EVIDENCE','evidence','FAIL',cid,'Evidence must be an array.');evid=[]
        for e in evid:
            sid=e.get('source');s=sources.get(sid)
            if not s:add('CLAIM_SOURCE','evidence','FAIL',cid,f'Unknown source {sid}.');continue
            referents.append(s)
            if source_status[sid]!='PASS':
                add('CLAIM_SOURCE_BLOCKED',domain or 'evidence','BLOCKED',cid,f'Exact source snapshot for {sid} is not verified.');continue
            a=e.get('anchor')
            if not a or not a.get('kind'):
                add('CLAIM_ANCHOR','evidence','BLOCKED',cid,f'Missing concrete selector for {sid}; a repository URL is not a code reference.');continue
            issues=check_anchor(s,a,root);findings.extend(issues)
            if not issues and e.get('relation')=='equal':
                try:
                    selected,_=select_anchor(safe_path(root,s['cache']).read_bytes(),a)
                    if a['kind']!='json_pointer':raise ValueError('Exact value relation requires a typed JSON pointer; code text needs semantic review.')
                    if canonical(loads(selected))!=canonical(c.get('expected')):add('UPSTREAM_VALUE',domain,'FAIL',cid,'Local value contradicts the pinned upstream value.')
                except (ValueError,KeyError,TypeError) as exc:add('UPSTREAM_VALUE','evidence','FAIL',cid,str(exc))
            elif e.get('relation') not in ('equal','consumes','implements','documents','derives'):
                add('EVIDENCE_RELATION','evidence','FAIL',cid,'Unknown evidence relation.')
        claim_sources[cid]=referents
        if c.get('classification')=='documented' and not evid:
            add('DOCUMENTED_NO_SOURCE',domain or 'evidence','BLOCKED',cid,'A documented scientific claim requires a concrete source anchor, not only a review.')
        if c.get('classification')=='derived' and not c.get('dependencies'):
            add('DERIVATION_INPUTS',domain or 'evidence','BLOCKED',cid,'Derived claim has no explicit input-claim dependencies.')
        if c.get('classification')=='assumed' and (not c.get('scope') or not c.get('units')):
            add('ASSUMPTION_SCOPE',domain,'BLOCKED',cid,'Assumption needs units and a declared scenario scope.')
        # Every claim is reviewed, presentation included. Calling a user-visible value
        # presentational is itself a judgement, and it is the judgement most worth checking:
        # an unreviewed 'presentation' label is how a scientific number escapes the gate.
        findings.append(check_review(c,source_subject(c,sources),reviews,root))
        own=findings[start:]+by_claim.get(str(cid),[])
        claim_status[cid]=aggregate(own) if own else 'PASS'
        if claim_status[cid]=='PASS':add('CLAIM_VERIFIED',domain,'PASS',cid,'Registered consumer, source obligations and applicable review checks passed.')
    def unverify(cid):
        """A demoted claim must not keep a PASS finding asserting that it verified."""
        for verified in [f for f in findings if f.code=='CLAIM_VERIFIED' and f.subject==str(cid)]:findings.remove(verified)
    # Derivations cannot launder unresolved inputs into a verified total.
    def propagate():
        for _ in range(len(claims)):
            changed=False
            for c in claims:
                cid=c.get('id','<missing>')
                if claim_status.get(cid)=='PASS' and any(claim_status.get(d)!='PASS' for d in c.get('dependencies',[])):
                    claim_status[cid]='BLOCKED';add('DEPENDENCY_BLOCKED',c.get('domain'),'BLOCKED',cid,'A derivation input is unverified.');unverify(cid);changed=True
            if not changed:break
    propagate()  # settle the first pass's dependency demotions before any role is reported
    # Roles are resolved after the fixpoint, not inside the first pass, because the pass below
    # can itself demote a claim and so has to be followed by the fixpoint again.
    for c in claims:
        cid=c.get('id','<missing>');domain=c.get('domain');start=len(findings)
        # A derived claim's roles are its own citations together with those of its declared
        # inputs, so removing a citation a derivation only ever copied from its input does not
        # manufacture a role gap. Whether those inputs verified is settled by the fixpoint, not
        # here; see roles_from_dependencies.
        referents=list(claim_sources.get(cid,[]))
        if c.get('classification')=='derived':referents+=roles_from_dependencies(c,claim_sources)
        roles={s['role'] for s in referents}
        # A publisher that ships only configuration has no implementation to cite. Which path
        # applies is decided by the recorded audit, never by which sources happen to be present.
        publisher_repos={s.get('repo') for s in referents if s.get('role')=='publisher_config'}
        ships_code=not (publisher_repos and publisher_repos<=CONFIG_ONLY_PUBLISHERS)
        for role in required_roles(c,publisher_ships_code=ships_code):
            if role not in roles:add('CLAIM_ROLE',domain,'BLOCKED',cid,f'Required {role} anchor is missing.')
        if domain=='model' and c.get('classification') not in ('presentation','assumed') and not dual_code_roles(referents,publisher_ships_code=ships_code):
            add('MODEL_DUAL_CODE','model','BLOCKED',cid,
                'Requires publisher implementation AND independent framework implementation.' if ships_code else
                'Publisher ships configuration only, so this needs independent implementations from at least two organizations.')
        if findings[start:]:
            claim_status[cid]=demoted_status(claim_status.get(cid,'PASS'))
            unverify(cid)
    propagate()  # required: the role pass can demote a claim its dependents were cleared against
    bydomain={d:[] for d in DOMAINS}
    seen_contracts=set()
    for rec in contracts:
        if rec.get('domain') not in bydomain:add('CONTRACT_DOMAIN','evidence','FAIL',rec.get('id'),'Unknown contract domain.');continue
        if rec.get('id') in seen_contracts:add('CONTRACT_DUPLICATE','evidence','FAIL',rec.get('id'),'Duplicate contract ID.')
        seen_contracts.add(rec.get('id'));bydomain[rec['domain']].append(rec)
        d=rec['domain']
        # Core fields may be bound into the contract, not transcribed independently.
        record,unbound=contract_record(rec,inventory.get('fields',{}))
        if d=='release' and receipt:
            record=release_record(receipt,consumer=digest(inventory.get('functions',{})),lock=lock)
            unbound=apply_bindings(record,rec.get('bindings',{}),inventory.get('fields',{}))
        for consumer in unbound:
            add('CONTRACT_BINDING',d,'FAIL',rec['id'],f'No actual core input {consumer}.')
        findings.extend(validate(d,record))
        refs=rec.get('claims',[])
        if not refs or any(claim_status.get(cid)!='PASS' for cid in refs):add('CONTRACT_EVIDENCE',d,'BLOCKED',rec['id'],'Contract inputs/semantics do not yet have verified claims.')
        # Review pins every constant and binding, including hypothetical scenarios.
        findings.append(check_review({'id':rec['id']},digest(rec),reviews,root))
    for domain,rows in bydomain.items():
        if not rows:add('DOMAIN_UNCOVERED',domain,'BLOCKED',domain,'No source-bound contract for this required area. Register a complete contract. This release has no blanket domain-waiver switch.')
    findings.extend(check_receipt(root,receipt,lock,inventory))
    # Transcription fidelity: every field of a hand-transcribed architecture re-derived from the
    # publisher's own acquired bytes and compared. Evidence work, so it lives here rather than in
    # a software gate -- and BLOCKED where the git-ignored cache is absent, never a vacuous PASS.
    findings.extend(judge_publisher_agreement(inventory.get('models',{}),root=root,catalog=catalog))
    # Two more evidence checks whose honest outcome can be BLOCKED, and which therefore cannot
    # live in a Pages-deploying software gate (ruling T789b): the published active-parameter
    # ratio against the depth split, and the claim manifest's multi-model completeness.
    findings.extend(judge_active_ratio(inventory.get('models',{})))
    findings.extend(judge_claim_model_coverage(claims,inventory.get('models',{})))
    obligations=target.get('runtime_obligations',[])
    # The same enforcement one level up, and the reason this file could report 31/31 over nine
    # checkpoints: nothing above asks whether a *model* has a contract at all. RUNTIME_MATRIX
    # excludes the model domain, RUNTIME_COVERAGE asks only that the domain have some obligation,
    # and DOMAIN_UNCOVERED fires only on an empty domain.
    findings.extend(judge_contract_model_coverage(inventory.get('models',{}),contracts,obligations))
    if profile=='runtime':
        # The runtime matrix is about physical scenarios: a registered SKU, fabric or library has
        # to appear in it. The other domains are deliberately outside it -- a `numerics.*` or
        # `release.*` contract has no host to probe, and demanding a hardware obligation for one
        # would be a category error that inflated the BLOCKED count for a bad reason.
        #
        # What was wrong was not the membership but that it was a bare literal covering 4 of 12
        # domains, with the other 8 falling into neither set and nothing saying so. A thirteenth
        # domain would have joined them silently. The partition is now closed and checked below,
        # so adding a domain forces a decision instead of defaulting to "not in the matrix".
        matrix_domains={'gpu','scale_up','scale_out','communication'}
        off_matrix={'model','training','serving','numerics','storage','measurement',
                    'visualization','release'}
        unpartitioned=set(DOMAINS)-matrix_domains-off_matrix
        if unpartitioned:
            add('RUNTIME_MATRIX_PARTITION','evidence','FAIL','runtime',
                f'Contract domains in neither the runtime matrix nor its recorded exclusions: '
                f'{sorted(unpartitioned)}.',
                'RUNTIME_MATRIX only demands obligations for domains it names, so a domain in '
                'neither set is exempt by omission rather than by decision. Put it in the matrix '
                'or record why it is outside one.')
        required={r['id'] for r in contracts if r.get('domain') in matrix_domains}
        declared={o.get('contract') for o in obligations}
        for missing in required-declared:
            add('RUNTIME_MATRIX','evidence','BLOCKED',missing,'A registered SKU/fabric/library scenario is missing from the runtime matrix.')
        runtime_ids=[o.get('id') for o in obligations]
        if len(runtime_ids)!=len(set(runtime_ids)):add('RUNTIME_DUPLICATE','evidence','FAIL','runtime','Duplicate runtime obligation ID.')
        covered={o.get('domain') for o in obligations}
        for d in DOMAINS:
            if d not in covered:add('RUNTIME_COVERAGE',d,'BLOCKED',d,'Required domain has no runtime evidence obligation.')
        for o in obligations:
            rid=o['id'];r=runtime.get(rid)
            if not r:add('RUNTIME_NOT_RUN',o['domain'],'NOT_RUN',rid,o['description'],'Run the specified experiment on the declared stack; attach raw artifacts and a trusted signed receipt.');continue
            try:
                # The expiry window stays, hardware-attribute receipts included. Nothing in this
                # file can detect that someone re-capped a GPU's power limit, reflashed a NIC or
                # swapped a card between two probes of the same host: the digests would agree
                # because they only describe what the last probe saw. Expiry is the only
                # mechanism that forces someone to look again, so decoupling these receipts from
                # the manifest is not a reason to drop it.
                executed=datetime.fromisoformat(r['executed_at'].replace('Z','+00:00'))
                age=datetime.now(timezone.utc)-executed
                if executed.tzinfo is None or not timedelta(0)<=age<=timedelta(days=7 if o['domain']=='measurement' else 30):raise ValueError('Runtime receipt is expired or future-dated')
                if r.get('exit_code')!=0:raise ValueError('Experiment did not record a successful exit')
            except (KeyError,ValueError,TypeError) as exc:add('RUNTIME_EXECUTION',o['domain'],'FAIL',rid,str(exc))
            original=next((c for c in contracts if c['id']==o['contract']),None) if o.get('contract') else None
            # What a receipt is allowed to be retired by depends on what it attests.
            #
            # A hardware-attribute obligation (gpu/scale_up/scale_out) attests what a machine or
            # a fabric IS. How much HBM an H100 carries does not depend on our claim set or on
            # our source files, so binding those retires real fleet evidence for a relabelled
            # wire -- which is what happened, three times in one session. Such a receipt binds
            # instead the data model it was captured under, the contract record it vouches for,
            # and the attested hardware state it rests on. lock_sha256/consumer_sha256 stay in
            # the receipt as capture-time provenance; they are simply not compared here.
            #
            # Every other domain attests what our own code DOES, so it keeps binding the lock
            # and the consumer inventory exactly as before. The branch, not a per-receipt flag,
            # decides: a receipt cannot nominate itself for the relaxed rule.
            if o['domain'] in HARDWARE_DOMAINS:
                probed=r.get('probe_schema_sha256')
                if probed!=hardware_schema_sha256():
                    add('RUNTIME_SCHEMA',o['domain'],'BLOCKED',rid,
                        'Hardware record was captured under a superseded data model.' if probed else
                        'Receipt names no probe_schema_sha256, so the data model it was captured under is unknown.',
                        'Re-probe the host with tools/probe-system.py under the current verification/schemas/hardware.schema.json and rebuild the receipt.')
                # Re-derived, not merely carried. The receipt names the stratum-hardware/v1
                # capture it rests on; that capture is one of its own checksummed artifacts, and
                # the digest is recomputed from the attested half of those bytes. A digest typed
                # in by hand, or left behind when the capture was replaced, is therefore a
                # contradiction rather than an unchecked claim.
                carried=r.get('hardware_sha256');named=(r.get('hardware_artifact') or {}).get('path')
                checksummed={a.get('path') for a in r.get('artifacts',[]) if isinstance(a,dict)}
                if not carried:
                    add('RUNTIME_HARDWARE',o['domain'],'BLOCKED',rid,'Receipt names no hardware_sha256, so the captured hardware state it rests on is unknown.',
                        'Re-probe the host with tools/probe-system.py and rebuild the receipt with tools/build-runtime-receipts.py.')
                elif not named:
                    add('RUNTIME_HARDWARE',o['domain'],'BLOCKED',rid,'Receipt carries a hardware_sha256 but names no hardware_artifact, so nothing can re-derive it.',
                        'Rebuild the receipt with tools/build-runtime-receipts.py so it names the capture the digest was taken from.')
                elif named not in checksummed:
                    add('RUNTIME_HARDWARE',o['domain'],'BLOCKED',rid,'The hardware record this receipt names is not one of the artifacts it checksums, so its bytes are not pinned.',
                        'Rebuild the receipt so the hardware record is among its checksummed artifacts.')
                else:
                    try:
                        captured=load_hardware(safe_path(root,named))
                        if attested_sha256(captured)!=carried:
                            raise ValueError('Receipt names an attested hardware state that its own capture does not produce.')
                    except (OSError,ValueError,KeyError,TypeError) as exc:
                        add('RUNTIME_HARDWARE_DRIFT',o['domain'],'FAIL',rid,str(exc),
                            'Re-probe the host with tools/probe-system.py and rebuild the receipt; never edit a recorded digest in place.')
                if original is None:
                    if o.get('contract') is None:add('RUNTIME_HARDWARE',o['domain'],'BLOCKED',rid,'Hardware-attribute obligation names no contract for its receipt to bind.',
                                                    'Give the obligation the contract its probe attests, or record it as a behavioural obligation.')
                elif not r.get('contract_sha256'):
                    add('RUNTIME_HARDWARE',o['domain'],'BLOCKED',rid,'Receipt names no contract_sha256, so which contract record the probe saw is unknown.',
                        'Rebuild the receipt against the contract it attests; a receipt that names no contract record cannot vouch for one.')
                elif r['contract_sha256']!=contract_sha256(original,inventory.get('fields',{})):
                    add('RUNTIME_CONTRACT_DRIFT',o['domain'],'FAIL',rid,'Contract record changed since this receipt attested it; the receipt would vouch for a value the probe never saw.',
                        'Re-probe the hardware for the contract as it now reads, or restore the record the probe actually attested.')
            elif r.get('lock_sha256')!=lock or r.get('consumer_sha256')!=digest(inventory['functions']):
                add('RUNTIME_STALE',o['domain'],'FAIL',rid,'Runtime receipt belongs to a different atlas/lock.')
            if o.get('contract'):
                if original is None:add('RUNTIME_CONTRACT',o['domain'],'FAIL',rid,'Runtime obligation refers to an unknown contract.')
                else:
                    for key in ('sku','system','fabric','mode','library','tolerance_policy','accumulator'):
                        if key in original['record'] and r.get('record',{}).get(key)!=original['record'][key]:add('RUNTIME_TARGET',o['domain'],'FAIL',rid,f'Execution receipt is for a different {key}.')
            for key in o.get('environment',[]):
                fingerprint=r.get('environment',{}).get(key)
                if not isinstance(fingerprint,dict) or not fingerprint.get('value'):
                    add('RUNTIME_ENVIRONMENT',o['domain'],'BLOCKED',rid,f'Missing environment fingerprint {key}.')
                elif fingerprint.get('method')=='probed' and fingerprint.get('source_artifact') not in {a.get('path') for a in r.get('artifacts',[]) if isinstance(a,dict)}:
                    add('RUNTIME_ENVIRONMENT',o['domain'],'BLOCKED',rid,f'Fingerprint {key} claims to be probed but names no checksummed artifact.')
            try:
                artifacts=r.get('artifacts',[])
                if not artifacts:raise ValueError('No raw experiment artifacts')
                hashes={}
                for artifact in artifacts:
                    path=safe_path(root,artifact['path']);h=hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                    if not HEX64.fullmatch(str(artifact.get('sha256',''))) or h.hexdigest()!=artifact['sha256']:raise ValueError('Runtime artifact hash mismatch')
                    hashes[artifact['path']]=h.hexdigest()
                binding=r.get('record_artifact',{})
                if binding.get('path') not in hashes:raise ValueError('Normalized runtime record is not bound to a checksummed artifact')
                path=safe_path(root,binding['path'])
                if path.stat().st_size>8*1024*1024:raise ValueError('Normalized JSON record exceeds 8 MiB')
                recorded=pointer(loads(path.read_text()),binding.get('pointer',''))
                if canonical(recorded)!=canonical(r.get('record',{})):raise ValueError('Runtime summary differs from normalized raw record')
                findings.extend(validate(o['domain'],r.get('record',{})))
            except (OSError,ValueError,KeyError,TypeError) as exc:add('RUNTIME_ARTIFACT',o['domain'],'FAIL',rid,str(exc))
            findings.append(check_review({'id':rid},digest({'obligation':o,'receipt':r}),reviews,root))
    else:
        # Informational only; runtime certification is not claimed by publication.
        pass
    counts=Counter(f.status for f in findings)
    return {'schema':'stratum-verification-report/v1','target':target.get('id','unknown'),'profile':profile,
            'status':aggregate(findings),'generated_at':datetime.now(timezone.utc).isoformat(),
            'lock_sha256':lock,'consumer_sha256':digest(inventory.get('functions',{})),
            'coverage':{'inputs':len(inventory.get('fields',{})),'functions_and_files':len(inventory.get('functions',{})),
                        'scene_contracts':len(inventory.get('scenes',{})),'claims':len(claims),'sources':len(sources),
                        'verified_sources':sum(v=='PASS' for v in source_status.values()),'verified_scientific_claims':sum(claim_status.get(c['id'])=='PASS' and c.get('classification')!='presentation' for c in claims),'passed_presentation_bindings':sum(claim_status.get(c['id'])=='PASS' and c.get('classification')=='presentation' for c in claims)},
            'counts':dict(counts),'domains':{d:aggregate([f for f in findings if f.domain==d]) for d in ('evidence',)+DOMAINS},
            'runtime_certified':profile=='runtime' and aggregate(findings)=='PASS',
            'findings':[f.json() for f in findings]}

def markdown(report):
    lines=[f"# Scientific verification — {report['target']}",f"\n**{report['profile']}: {report['status']}**",f"\nGenerated: {report['generated_at']}",
           '\nThis report distinguishes software/contract tests from evidence-backed science. Missing evidence is not a contradiction, and a green engineering run is not scientific certification.',
           '\n## Coverage','\n| Item | Count |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in report['coverage'].items()]
    lines += ['\n## Gates','\n| Domain | Status |','|---|---|']+[f'| {d} | {s} |' for d,s in report['domains'].items()]
    groups=Counter((f['code'],f['status']) for f in report['findings'])
    lines += ['\n## Findings by failure mode','\n| Code | Status | Count |','|---|---|---:|']+[f'| {c} | {s} | {n} |' for (c,s),n in sorted(groups.items())]
    lines += ['\n## First unresolved findings']
    for f in [x for x in report['findings'] if x['status']!='PASS'][:24]:
        lines += [f"\n**{f['code']} / {f['subject']}** — {f['message']} {f['remedy']}"]
    lines += ['\nThe adjacent JSON report contains every finding and consumer ID. No raw source, signed review, GPU run or deployment measurement is inferred from a URL or a fixture.']
    return '\n'.join(lines)+'\n'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,default='publication');p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args(argv);root=Path(a.root).resolve()
    try:
        inv=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=root,check=True,capture_output=True,text=True,timeout=60).stdout)
        target=loads((root/'verification/manifests/observatory.json').read_text());catalog=loads((root/'verification/manifests/sources.json').read_text())
        def optional(path):
            f=root/path
            return loads(f.read_text()) if f.exists() else {}
        report=audit(root,target,inv,catalog,profile=a.profile,reviews=optional('verification/manifests/reviews.json'),receipt=optional('verification/reports/engineering.json'),runtime=optional('verification/manifests/runtime.json'))
    except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
        print(f'MANIFEST/TOOL ERROR: {exc}',file=sys.stderr);return 1
    out=root/'verification/reports';out.mkdir(exist_ok=True)
    (out/f'{a.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/f'{a.profile}.md').write_text(markdown(report))
    print(f"{a.profile}: {report['status']} | {report['coverage']} | {report['counts']}")
    return exit_code(report['status'])
if __name__=='__main__':raise SystemExit(main())
