#!/usr/bin/env python3
"""Read-only local inventory capture. This is not a GPU/RDMA benchmark or attestation.

No sudo, fabric reconfiguration, model loading, shell evaluation or outbound
requests. Logs may contain device UUIDs and interface addresses: review before
sharing. Missing utilities/devices produce NOT_RUN, never a successful probe.
"""
import argparse,hashlib,json,os,platform,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
COMMANDS={
 'gpu':['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,compute_cap,power.limit','--format=csv,noheader,nounits'],
 'topology':['nvidia-smi','topo','-m'], 'nvlink':['nvidia-smi','nvlink','--status'],
 'cuda':['nvcc','--version'], 'verbs':['ibv_devinfo','-v'],
 'rdma':['rdma','link','show'], 'pci':['lspci','-nn'],
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
 if out.exists():raise SystemExit('Refusing to overwrite an existing probe directory')
 out.mkdir(parents=True);results=[]
 for id,cmd in COMMANDS.items():
  if not shutil.which(cmd[0]):results.append({'id':id,'status':'NOT_RUN','reason':'utility not installed','argv':cmd});continue
  try:
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=False)
   raw=(r.stdout+'\nSTDERR:\n'+r.stderr).encode();path=out/(id+'.txt');path.write_bytes(raw)
   results.append({'id':id,'status':'PASS' if r.returncode==0 and r.stdout.strip() else 'NOT_RUN','exit':r.returncode,'argv':cmd,'path':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
  except (OSError,subprocess.TimeoutExpired) as exc:results.append({'id':id,'status':'NOT_RUN','reason':str(exc),'argv':cmd})
 result={'schema':'stratum-readonly-probe/v1','generated_at':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
   'scope':'Inventory capture only. Does not prove kernel support, GPUDirect, remote visibility, measured bandwidth or correctness.',
   'environment':{k:os.environ[k] for k in ['NCCL_ALGO','NCCL_PROTO','NCCL_NET','NCCL_IB_HCA','NVSHMEM_REMOTE_TRANSPORT'] if k in os.environ},'results':results}
 (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 return 0 if results and all(r['status']=='PASS' for r in results) else 2
if __name__=='__main__':raise SystemExit(main())
