"""Read safetensors headers only with strict HTTP ranges; never read tensor payloads."""
import hashlib
import json
import struct
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sources=json.loads((ROOT/'models/kimi-k2/sources.json').read_text())
rev=sources['checkpoint']['revision']
def inspect(shard):
 url=f'https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/{rev}/{shard}'+('' if shard=='model-1-of-61.safetensors' else '?download=true')
 def read(start,end):
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=12) as response:
   if response.status!=206 or not response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'):
    raise RuntimeError('Server did not honor metadata-only range')
   return response.read(end-start+1)
 print(shard,flush=True)
 length=struct.unpack('<Q',read(0,7))[0]
 header=read(8,length+7);metadata=json.loads(header)
 return dict(shard=shard,url=url,header_bytes=length,header_sha256=hashlib.sha256(header).hexdigest(),
             tensors={k:dict(shape=v['shape'],dtype=v['dtype']) for k,v in metadata.items() if k!='__metadata__'})
records=[]
errors=[]
for shard in ['model-1-of-61.safetensors','model-2-of-61.safetensors']:
 try:
  records.append(inspect(shard))
 except Exception as error:
  errors.append(dict(shard=shard,error=str(error)))

if errors:
 raise RuntimeError(f'Metadata fetch incomplete; existing evidence preserved: {errors}')
path=ROOT/'models/kimi-k2/storage-evidence.json'
path.write_text(json.dumps(dict(checkpoint=sources['checkpoint'],scope='safetensors JSON headers only; no weight payloads fetched',shards=records,errors=errors),separators=(',',':'))+'\n')
print(f'Wrote metadata for {len(records)} shards, {sum(len(r["tensors"]) for r in records)} tensors')

summary=dict(checkpoint=sources['checkpoint'],inspected_shards=[r['shard'] for r in records],header_sha256=[r['header_sha256'] for r in records],observed_frequency_widths=[t['shape'][0] for r in records for name,t in r['tensors'].items() if name.endswith('inv_freq')],scope='Headers only; remaining shards not inspected. Repeated frequency width inferred and reconciled against pinned Hub total.')
(ROOT/'models/kimi-k2/storage-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
