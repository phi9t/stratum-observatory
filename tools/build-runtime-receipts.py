#!/usr/bin/env python3
"""Assemble runtime receipts from read-only fleet probes.

Writes a CANDIDATE, never the accepted lock, mirroring verification/sources.py: a fresh
capture is not an approval. Every environment fingerprint names the checksummed artifact it
was read out of, and the normalized record is written as its own artifact so the receipt's
summary is reproducible from bytes rather than asserted.

This attests device attributes actually reported by the driver on a named host. It does not
attest kernel execution, achieved bandwidth or model correctness.
"""
import hashlib,json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verification import hardware
from verification.evidence import digest,loads
from verification.runner import contract_sha256

# SKU -> (obligation id, contract id, kind, architecture, cc, gpu_count, cpu_count).
# Counts are the SKU's own scope: a GB300 superchip is two GPUs and one CPU, whatever the
# host happens to hold. GPU_ID in domains/contracts.py enforces exactly this.
SKUS={
 'gpu.H100-SXM':('runtime.gpu.H100-SXM','gpu.H100-SXM','H100-SXM','gpu','hopper','9.0',1,0),
 'gpu.B200':('runtime.gpu.B200','gpu.B200','B200','gpu','blackwell','10.0',1,0),
 'gpu.GB300':('runtime.gpu.GB300','gpu.GB300','GB300','superchip','blackwell-ultra','10.3',2,1),
}
HOSTS=json.loads((ROOT/'evidence/runtime/hosts.json').read_text()) if (ROOT/'evidence/runtime/hosts.json').exists() else {}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_INVENTORY=None
def inventory():
    """The live consumer inventory, derived from the running code exactly once per build."""
    global _INVENTORY
    if _INVENTORY is None:
        _INVENTORY=loads(subprocess.run(['node','tools/export-verification.mjs'],cwd=ROOT,
                                        capture_output=True,text=True,check=True).stdout)
    return _INVENTORY


def contract_by_id(cid):
    target=loads((ROOT/'verification/manifests/observatory.json').read_text())
    found=next((c for c in target['contracts'] if c['id']==cid),None)
    if found is None:raise SystemExit(f'{cid}: no such contract in observatory.json')
    return found


def first_gpu(text):
    for line in text.splitlines():
        parts=[p.strip() for p in line.split(',')]
        if len(parts)>=9 and parts[1].startswith('GPU-'):return parts
    raise SystemExit('No GPU row in the capture')


def build(evidence_dir,key):
    oid,contract_id,sku,kind,arch,cc,ngpu,ncpu=SKUS[key]
    d=Path(evidence_dir)
    probe=loads((d/'manifest.json').read_text())
    name,uuid,bus,devid,driver,mem_mib,compute,power,mig=first_gpu((d/'gpu.txt').read_text())[:9]
    smi=(d/'smi.txt').read_text()
    cuda=re.search(r'CUDA Version:\s*([0-9.]+)',smi)
    if not cuda:raise SystemExit(f'{key}: no CUDA version in the driver banner')
    rel=lambda p:str(Path(p).relative_to(ROOT))
    artifacts=[{'path':rel(p),'sha256':sha(p)} for p in sorted(d.iterdir()) if p.name!='record.json']
    record={'id':contract_id,'sku':sku,'kind':kind,'architecture':arch,'compute_capability':cc,
            'gpu_count':ngpu,'cpu_count':ncpu,
            # Measured allocatable capacity, not the vendor's nominal figure. The contract
            # distinguishes the two precisely so a probe cannot be mistaken for a datasheet.
            'memory':{'value':round(int(mem_mib)/1024,2),'unit':'GiB','scope':'per_gpu','kind':'allocatable_measured'}}
    (d/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    artifacts.append({'path':rel(d/'record.json'),'sha256':sha(d/'record.json')})
    gpu_art=rel(d/'gpu.txt');smi_art=rel(d/'smi.txt')
    env={'sku':(name,gpu_art),'pci_device_id':(devid,gpu_art),'gpu_uuid':(uuid,gpu_art),
         'driver':(driver,gpu_art),'cuda':(cuda.group(1),smi_art),
         'compute_capability':(compute,gpu_art),'mig':(mig,gpu_art)}
    # What a hardware-attribute receipt binds, each derived from bytes and never written by hand:
    # the data model the capture was made under, the contract record it vouches for with the
    # application's own constants applied, and the attested half of the capture itself. See
    # docs/verifiers/EVIDENCE.md. hardware.load() re-derives hardware_sha256 from the attested
    # half and refuses a fixture, so a doctored capture cannot be carried into a receipt here.
    captured=hardware.load(d/'hardware.json')
    return oid,{'executed_at':probe['generated_at'],'exit_code':0,
      'lock_sha256':digest({'target':loads((ROOT/'verification/manifests/observatory.json').read_text()),
                            'catalog':loads((ROOT/'verification/manifests/sources.json').read_text())}),
      'consumer_sha256':digest(inventory()['functions']),
      'probe_schema_sha256':hardware.schema_sha256(),
      'contract_sha256':contract_sha256(contract_by_id(contract_id),inventory().get('fields',{})),
      'hardware_sha256':hardware.hardware_sha256(captured),
      'environment':{k:{'value':v,'method':'probed','source_artifact':src} for k,(v,src) in env.items()},
      'record':record,'artifacts':artifacts,
      'record_artifact':{'path':rel(d/'record.json'),'pointer':''},
      'hardware_artifact':{'path':rel(d/'hardware.json')}}


def main():
    receipts={}
    for key in SKUS:
        d=ROOT/'evidence/runtime'/key
        if not d.exists():print(f'skip {key}: no evidence captured');continue
        oid,receipt=build(d,key);receipts[oid]=receipt
        print(f"{oid:26} {receipt['record']['memory']['value']} GiB/gpu  driver {receipt['environment']['driver']['value']}  "
              f"cuda {receipt['environment']['cuda']['value']}  {len(receipt['artifacts'])} artifacts")
    out=ROOT/'verification/candidates';out.mkdir(exist_ok=True)
    (out/'runtime.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(f'\nWrote {out/"runtime.json"} as a candidate. Promotion into verification/manifests/runtime.json is a separate deliberate act.')
    return 0


if __name__=='__main__':raise SystemExit(main())
