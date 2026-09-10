#!/usr/bin/env python3
"""Read-only local inventory capture. This is not a GPU/RDMA benchmark or attestation.

No sudo, fabric reconfiguration, model loading, shell evaluation or outbound
requests. Logs may contain device UUIDs and interface addresses: review before
sharing. Missing utilities/devices produce NOT_RUN, never a successful probe.

Two outputs. `manifest.json` is the raw command ledger, unchanged: what ran, what it
returned, and the checksum of the bytes it returned. `hardware.json` is the versioned
record described by verification/schemas/hardware.schema.json, split into

  attested   what the machine IS   -- stable device and fabric attributes; the digest
  transient  what it is DOING      -- telemetry, recorded for context, never evidence

`hardware_sha256` covers the attested half alone, so two probes of an unchanged machine
agree on it however busy the machine happened to be. A field a utility did not report is
listed in `unavailable` with the reason it is missing; it is never guessed from a datasheet.

This file runs on hosts that have no copy of this repository, so it carries its own
canonicalisation rather than importing verification.evidence. The two are held together by
verification/tests/test_hardware.py, which asserts they produce the same digest.
"""
import argparse,hashlib,json,os,platform,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path

SCHEMA='stratum-hardware/v1'

# Field order matters: the CSV is positional. Splitting the stable half from the live half
# also keeps one unsupported field name on an older driver from losing the other half.
ATTESTED_GPU_FIELDS=('index,name,uuid,serial,pci.bus_id,pci.device_id,pci.sub_device_id,'
 'pci.link.gen.max,pci.link.width.max,driver_version,vbios_version,compute_cap,memory.total,'
 'memory.reserved,ecc.mode.current,ecc.mode.pending,persistence_mode,compute_mode,'
 'accounting.mode,mig.mode.current,mig.mode.pending,clocks.max.graphics,clocks.max.sm,'
 'clocks.max.memory,power.max_limit,power.min_limit,power.default_limit,power.limit')
TRANSIENT_GPU_FIELDS=('index,uuid,pstate,temperature.gpu,fan.speed,power.draw,utilization.gpu,'
 'utilization.memory,memory.used,memory.free,clocks.sm,clocks.memory,pci.link.gen.current,'
 'pci.link.width.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total')

# Every queried field and the per-GPU record field it feeds. nvidia-smi refuses an entire query
# over one name it does not implement, so the probe drops the refused names and asks again; this
# map is what lets the resulting absence be reported against the record field a reader is looking
# for rather than against a CSV column. `driver_version` is queried per device as a cross-check
# and the record reads the driver from the richer `nvidia-smi -q` output instead.
ATTESTED_GPU_PATHS={'index':'index','name':'name','uuid':'uuid','serial':'serial',
 'pci.bus_id':'pci.bus_id','pci.device_id':'pci.device_id','pci.sub_device_id':'pci.sub_device_id',
 'pci.link.gen.max':'pci.link_gen_max','pci.link.width.max':'pci.link_width_max',
 'driver_version':'driver_version','vbios_version':'vbios_version','compute_cap':'compute_capability',
 'memory.total':'memory.total_mib','memory.reserved':'memory.reserved_mib',
 'ecc.mode.current':'ecc_mode.current','ecc.mode.pending':'ecc_mode.pending',
 'persistence_mode':'persistence_mode','compute_mode':'compute_mode','accounting.mode':'accounting_mode',
 'mig.mode.current':'mig_mode.current','mig.mode.pending':'mig_mode.pending',
 'clocks.max.graphics':'clocks_max.graphics_mhz','clocks.max.sm':'clocks_max.sm_mhz',
 'clocks.max.memory':'clocks_max.memory_mhz','power.max_limit':'power.max_limit_w',
 'power.min_limit':'power.min_limit_w','power.default_limit':'power.default_limit_w',
 'power.limit':'power.enforced_limit_w'}
TRANSIENT_GPU_PATHS={'index':'index','uuid':'uuid','pstate':'pstate','temperature.gpu':'temperature_c',
 'fan.speed':'fan_speed_pct','power.draw':'power_draw_w','utilization.gpu':'utilization_gpu_pct',
 'utilization.memory':'utilization_memory_pct','memory.used':'memory_used_mib',
 'memory.free':'memory_free_mib','clocks.sm':'clocks_sm_mhz','clocks.memory':'clocks_memory_mhz',
 'pci.link.gen.current':'link_gen_current','pci.link.width.current':'link_width_current',
 'ecc.errors.corrected.volatile.total':'ecc_errors_corrected_volatile',
 'ecc.errors.uncorrected.volatile.total':'ecc_errors_uncorrected_volatile'}

COMMANDS={
 # Fields chosen to cover exactly the environment keys the runtime obligations declare:
 # sku, pci_device_id, gpu_uuid, driver, compute_capability and mig. Receipts already in
 # verification/manifests/runtime.json read this artifact positionally: do not reorder it.
 'gpu':['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,pci.device_id,driver_version,memory.total,compute_cap,power.limit,mig.mode.current','--format=csv,noheader,nounits'],
 'gpu_attested':['nvidia-smi','--query-gpu='+ATTESTED_GPU_FIELDS,'--format=csv,noheader,nounits'],
 'gpu_transient':['nvidia-smi','--query-gpu='+TRANSIENT_GPU_FIELDS,'--format=csv,noheader,nounits'],
 # Full header carries the driver's supported CUDA version, which is not a toolkit version.
 'smi':['nvidia-smi'],
 # Board part numbers, inforom, module id, remapped rows: attributes with no CSV query field.
 'smiq':['nvidia-smi','-q'],
 'mig':['nvidia-smi','mig','-lgip'],
 'topology':['nvidia-smi','topo','-m'], 'nvlink':['nvidia-smi','nvlink','--status'],
 'nvlink_cap':['nvidia-smi','nvlink','-c'],
 'cuda':['nvcc','--version'], 'verbs':['ibv_devinfo','-v'],
 'rdma':['rdma','link','show'], 'ofed':['ofed_info','-s'],
 'lscpu':['lscpu'], 'numa':['numactl','--hardware'], 'kernel':['uname','-r'],
 'pci':['lspci','-nn'],
}

# Sections of the record and the command each is read out of. A section whose command did
# not run, or ran and reported nothing usable, becomes an entry in `unavailable`.
SECTIONS=[('attested.driver','smiq'),('attested.kernel','kernel'),('attested.cpu','lscpu'),
          ('attested.numa','numa'),('attested.gpus','gpu_attested'),('attested.topology','topology'),
          ('attested.nvlink','nvlink_cap'),('attested.mig_profiles','mig'),('attested.rdma','verbs'),
          ('attested.ofed','ofed'),('attested.cuda_toolkit','cuda'),
          ('transient.gpus','gpu_transient'),('transient.nvlink','nvlink'),
          ('transient.rdma','rdma'),('transient.numa','numa'),('transient.cpu','lscpu')]

NA={'','N/A','[N/A]','Not Supported','[Not Supported]','Unknown Error','[Unknown Error]',
    'Insufficient Permissions','[Insufficient Permissions]'}
ANSI=re.compile(r'\x1b\[[0-9;]*m')
KV=re.compile(r'^(\s*)(\S.*?)\s+:\s?(.*)$')
# nvidia-smi prints this on stdout and exits 2 when a --query-gpu name is not implemented by the
# installed driver. It refuses the whole query, so the probe reads the names back out of its own
# message rather than guessing which of ours it disliked.
REJECTION=re.compile(r'Field "[^"]+" is not a valid field to query\.')
FIELD_REJECTED=re.compile(r'Field "([^"]+)" is not a valid field to query\.')


# --- canonical form -----------------------------------------------------------------
def canonical(value):
    """Byte-for-byte the same as verification.evidence.canonical; see the module docstring."""
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


# --- small helpers ------------------------------------------------------------------
def body(text):return text.split('\nSTDERR:\n')[0]
def clean(value):
    value=(value or '').strip()
    return None if value in NA else value
def as_int(value):
    value=clean(value)
    if value is None:return None
    m=re.match(r'^-?\d+',value.replace(',',''))
    return int(m.group(0)) if m else None
def as_float(value):
    value=clean(value)
    if value is None:return None
    m=re.match(r'^-?\d+(?:\.\d+)?',value.replace(',',''))
    return float(m.group(0)) if m else None
def put(target,key,value):
    """Record a value, or record nothing. A field the tool did not report stays absent."""
    if value is not None:target[key]=value
def sub(target,key,pairs):
    node={}
    for k,v in pairs:put(node,k,v)
    if node:target[key]=node
def ranges(values):
    """[0,1,2,3,8,9,10,11] -> '0-3,8-11'. A normalisation of what numactl printed, not a claim."""
    out=[];start=prev=None
    for v in sorted(values):
        if start is None:start=prev=v;continue
        if v==prev+1:prev=v;continue
        out.append((start,prev));start=prev=v
    if start is not None:out.append((start,prev))
    return ','.join(str(a) if a==b else f'{a}-{b}' for a,b in out)
def slug(text):return re.sub(r'_+','_',re.sub(r'[^a-z0-9]+','_',text.lower())).strip('_')

def rejections(text):
    """The driver's refusal sentences, verbatim and de-duplicated."""
    return list(dict.fromkeys(REJECTION.findall(text)))
def rejected_fields(text):
    """The --query-gpu names this driver refused, in the order it named them."""
    return list(dict.fromkeys(FIELD_REJECTED.findall(text)))
def query_fields(argv):
    """The --query-gpu field list a command actually ran with, read back out of its own argv."""
    for arg in argv or ():
        if str(arg).startswith('--query-gpu='):return str(arg)[len('--query-gpu='):].split(',')
    return []
def without_fields(argv,drop):
    """The same command minus those fields, or None if nothing would be left to ask for."""
    keep=[f for f in query_fields(argv) if f not in drop]
    if not keep:return None
    return [('--query-gpu='+','.join(keep)) if str(a).startswith('--query-gpu=') else a for a in argv]


# --- parsers ------------------------------------------------------------------------
def parse_csv(text,fields):
    """nvidia-smi --format=csv,noheader,nounits -> one dict per device row."""
    names=fields.split(',') if isinstance(fields,str) else list(fields);rows=[]
    for line in body(text).splitlines():
        if not line.strip():continue
        parts=[p.strip() for p in line.split(',')]
        if len(parts)!=len(names):continue
        rows.append(dict(zip(names,parts)))
    return rows

def parse_smi_query(text):
    """`nvidia-smi -q` indented key/value blocks -> nested dicts, verbatim strings."""
    root={};stack=[(-1,root)]
    for raw in body(text).splitlines():
        stripped=raw.strip()
        if not stripped or stripped.startswith('=='):continue
        indent=len(raw)-len(raw.lstrip(' '))
        while stack[-1][0]>=indent:stack.pop()
        parent=stack[-1][1]
        match=KV.match(raw)
        if match:parent[match.group(2).strip()]=match.group(3).strip()
        else:
            node=parent.setdefault(stripped,{})
            if isinstance(node,dict):stack.append((indent,node))
    return root

def sm_count(gpu):
    """Read a multiprocessor count if this driver reports one. Most do not; then it is absent."""
    for key in ('Number of Multiprocessors','Multiprocessor Count','MultiProcessor Count','SM Count'):
        if key in gpu:return as_int(gpu[key])
    return None

def parse_lscpu(text):
    out={}
    for line in body(text).splitlines():
        if ':' not in line:continue
        key,_,value=line.partition(':')
        out[key.strip()]=value.strip()
    return out

def parse_numactl(text):
    nodes={};distances=[];count=None;in_distances=False
    for line in body(text).splitlines():
        m=re.match(r'^available:\s*(\d+)\s+nodes',line)
        if m:count=int(m.group(1));continue
        m=re.match(r'^node\s+(\d+)\s+cpus:\s*(.*)$',line)
        if m:nodes.setdefault(int(m.group(1)),{})['cpus']=ranges(int(c) for c in m.group(2).split());continue
        m=re.match(r'^node\s+(\d+)\s+size:\s*(\d+)\s*MB',line)
        if m:nodes.setdefault(int(m.group(1)),{})['size_mb']=int(m.group(2));continue
        m=re.match(r'^node\s+(\d+)\s+free:\s*(\d+)\s*MB',line)
        if m:nodes.setdefault(int(m.group(1)),{})['free_mb']=int(m.group(2));continue
        if line.startswith('node distances'):in_distances=True;continue
        if in_distances:
            m=re.match(r'^\s*(\d+):\s+(.*)$',line)
            if m:distances.append([int(v) for v in m.group(2).split()])
    return count,nodes,distances

def parse_nvlink_status(text):
    gpus=[];current=None
    for line in body(text).splitlines():
        m=re.match(r'^GPU (\d+): (.*) \(UUID: (\S+)\)\s*$',line)
        if m:current={'index':int(m.group(1)),'uuid':m.group(3),'links':[]};gpus.append(current);continue
        m=re.match(r'^\s*Link (\d+):\s*(.*)$',line)
        if m and current is not None:
            value=m.group(2).strip();link={'link':int(m.group(1))}
            if value.startswith('<'):link['state']=value.strip('<>')
            else:
                link['state']='active'
                put(link,'rate_gb_s',as_float(value))
            current['links'].append(link)
    return gpus

def parse_nvlink_caps(text):
    gpus=[];current=None;links={}
    for line in body(text).splitlines():
        m=re.match(r'^GPU (\d+): (.*) \(UUID: (\S+)\)\s*$',line)
        if m:
            links={};current={'index':int(m.group(1)),'uuid':m.group(3),'_links':links};gpus.append(current);continue
        m=re.match(r'^\s*Link (\d+),\s*(.*?):\s*(\S+)\s*$',line)
        if m and current is not None:
            links.setdefault(int(m.group(1)),{})[CAPS.get(m.group(2).strip(),slug(m.group(2)))]=m.group(3).strip().lower()=='true'
    out=[]
    for gpu in gpus:
        links=gpu.pop('_links')
        gpu['link_count']=len(links)
        gpu['links']=[{'link':i,'capabilities':links[i]} for i in sorted(links)]
        out.append(gpu)
    return out
CAPS={'P2P is supported':'p2p_supported','Access to system memory supported':'system_memory_access_supported',
      'P2P atomics supported':'p2p_atomics_supported','System memory atomics supported':'system_memory_atomics_supported',
      'SLI is supported':'sli_supported','Link is supported':'link_supported'}

def parse_topology(text):
    lines=ANSI.sub('',body(text)).splitlines()
    labels=[];rows=[];legend={};in_legend=False
    for line in lines:
        m=re.match(r'^\s+(NIC\d+):\s*(\S+)\s*$',line)
        if in_legend and m:legend[m.group(1)]=m.group(2);continue
        if line.startswith('NIC Legend'):in_legend=True;continue
        if '\t' not in line:continue
        cells=line.split('\t')
        if not labels and not cells[0].strip():
            head=[c.strip() for c in cells[1:]]
            labels=head[:head.index('CPU Affinity')] if 'CPU Affinity' in head else head
            continue
        if not labels or not cells[0].strip():continue
        row={'name':cells[0].strip(),'peers':{}}
        for label,value in zip(labels,cells[1:1+len(labels)]):
            if value.strip():row['peers'][label]=value.strip()
        trailing=[c.strip() for c in cells[1+len(labels):] if c.strip()]
        for key,value in zip(('cpu_affinity','numa_affinity','gpu_numa_id'),trailing):row[key]=value
        rows.append(row)
    return labels,rows,legend

VERBS_DEVICE=('transport','fw_ver','node_guid','sys_image_guid','vendor_id','vendor_part_id',
              'hw_ver','board_id','phys_port_cnt','max_mr_size','max_qp','max_sge','max_cq',
              'max_pd','atomic_cap','max_pkeys','num_comp_vectors')
VERBS_PORT_ATTESTED=('max_mtu','link_layer','max_msg_sz','gid_tbl_len','pkey_tbl_len')
# `rdma link show` and `ibv_devinfo` both report link state, in different vocabularies
# ('DOWN' vs 'PORT_DOWN (1)'). rdma link show wins; verbs only fills a port it did not list.
VERBS_PORT_TRANSIENT=('state','phys_state','active_mtu','active_speed','active_width','port_lid','sm_lid')
VERBS_PORT_FALLBACK={'state':'state','phys_state':'physical_state'}

def parse_verbs(text):
    """ibv_devinfo -v -> per-HCA attributes and per-port attributes, verbatim."""
    devices=[];device=None;port=None;port_indent=None
    for raw in body(text).splitlines():
        expanded=raw.expandtabs(8)
        if not expanded.strip():continue
        indent=len(expanded)-len(expanded.lstrip(' '))
        m=re.match(r'^\s*([A-Za-z_0-9 \[\]]+?):\s*(.*)$',expanded)
        if not m:continue
        key,value=m.group(1).strip(),m.group(2).strip()
        if key=='hca_id':
            device={'hca':value,'attributes':{},'ports':[]};devices.append(device);port=None;port_indent=None;continue
        if device is None:continue
        if key=='port':
            port={'port':as_int(value),'attested':{},'transient':{}};device['ports'].append(port);port_indent=indent;continue
        if port is not None and port_indent is not None and indent<=port_indent:port=None
        target=port if port is not None else device
        if port is not None:
            if key in VERBS_PORT_ATTESTED:port['attested'][key]=value
            elif key in VERBS_PORT_TRANSIENT:port['transient'][key]=value
        elif key in VERBS_DEVICE:device['attributes'][key]=value
    return devices

def parse_rdma_link(text):
    out=[]
    for line in body(text).splitlines():
        m=re.match(r'^\S+/\d+:\s*(\S+)/(\d+):\s*state\s+(\S+)\s+physical_state\s+(\S+)(?:\s+netdev\s+(\S+))?',line)
        if m:out.append({'hca':m.group(1),'port':int(m.group(2)),'state':m.group(3),
                         'physical_state':m.group(4),'netdev':m.group(5)})
    return out

MIG_ROW=re.compile(r'^\|\s+(\d+)\s+(MIG\s+\S+)\s+(\d+)\s+(\d+)/(\d+)\s+([\d.]+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\|\s*$')
def parse_mig_profiles(text):
    out=[]
    for line in body(text).splitlines():
        m=MIG_ROW.match(line)
        if m:out.append({'gpu':int(m.group(1)),'name':re.sub(r'\s+',' ',m.group(2)),'profile_id':int(m.group(3)),
                         'instances_free':int(m.group(4)),'instances_total':int(m.group(5)),
                         'memory_gib':float(m.group(6)),'p2p':m.group(7),'sm_count':int(m.group(8)),
                         'decoders':int(m.group(9)),'encoders':int(m.group(10))})
    return out


# --- record assembly ----------------------------------------------------------------
def read_artifacts(directory):
    """Load a capture directory, refusing any artifact whose bytes no longer match the ledger."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    texts={}
    for result in manifest.get('results',[]):
        if not result.get('path'):continue
        path=directory/result['path']
        if not path.exists():raise ValueError(f'{directory.name}: missing artifact {result["path"]}')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=result.get('sha256'):
            raise ValueError(f'{directory.name}: {result["path"]} does not match its recorded digest')
        # A refused query keeps its artifact and its checksum, but it reported nothing usable and
        # must never be read as if it had.
        if result.get('status')=='PASS':texts[result['id']]=raw.decode('utf-8')
    return manifest,texts

def build_record(directory,sku=None):
    """Sort one capture into what the machine IS and what it was doing. Never invents a value."""
    manifest,texts=read_artifacts(directory)
    status={r['id']:r for r in manifest.get('results',[])}
    attested={'model':SCHEMA};transient={};
    put(transient,'captured_at',manifest.get('generated_at'))

    query=parse_smi_query(texts['smiq']) if 'smiq' in texts else {}
    gpu_sections={k:v for k,v in query.items() if k.startswith('GPU ') and isinstance(v,dict)}
    by_bus={k[4:].strip():v for k,v in gpu_sections.items()}

    # driver -------------------------------------------------------------------------
    if query:
        sub(attested,'driver',[('version',clean(query.get('Driver Version'))),
                               ('cuda_driver_version',clean(query.get('CUDA Version'))),
                               ('attached_gpus',as_int(query.get('Attached GPUs')))])
        if 'driver' in attested:attested['driver']['sources']=['smiq.txt']
    if 'driver' not in attested and 'smi' in texts:
        banner=body(texts['smi'])
        version=re.search(r'Driver Version:\s*(\S+)',banner);cuda=re.search(r'CUDA Version:\s*(\S+)',banner)
        sub(attested,'driver',[('version',version.group(1) if version else None),
                               ('cuda_driver_version',cuda.group(1) if cuda else None)])
        if 'driver' in attested:attested['driver']['sources']=['smi.txt']

    # gpus ---------------------------------------------------------------------------
    gpus=[];live=[]
    # The CSV is positional, so what each column means is whatever the command actually asked
    # for -- read back out of its own argv, not assumed from the declared list. A driver that
    # refused a name was re-asked without it, and every column after it has shifted.
    attested_fields=query_fields(status.get('gpu_attested',{}).get('argv',[])) or ATTESTED_GPU_FIELDS
    transient_fields=query_fields(status.get('gpu_transient',{}).get('argv',[])) or TRANSIENT_GPU_FIELDS
    for row in parse_csv(texts.get('gpu_attested',''),attested_fields):
        section=by_bus.get(clean(row.get('pci.bus_id')) or '',{})
        gpu={'sources':['gpu_attested.txt']+(['smiq.txt'] if section else [])}
        put(gpu,'index',as_int(row.get('index')));put(gpu,'name',clean(row.get('name')))
        put(gpu,'uuid',clean(row.get('uuid')));put(gpu,'serial',clean(row.get('serial')))
        put(gpu,'vbios_version',clean(row.get('vbios_version')))
        put(gpu,'compute_capability',clean(row.get('compute_cap')))
        put(gpu,'persistence_mode',clean(row.get('persistence_mode')))
        put(gpu,'compute_mode',clean(row.get('compute_mode')))
        put(gpu,'accounting_mode',clean(row.get('accounting.mode')))
        sub(gpu,'pci',[('bus_id',clean(row.get('pci.bus_id'))),('device_id',clean(row.get('pci.device_id'))),
                       ('sub_device_id',clean(row.get('pci.sub_device_id'))),
                       ('link_gen_max',as_int(row.get('pci.link.gen.max'))),
                       ('link_width_max',as_int(row.get('pci.link.width.max')))])
        sub(gpu,'memory',[('total_mib',as_int(row.get('memory.total'))),('reserved_mib',as_int(row.get('memory.reserved')))])
        sub(gpu,'clocks_max',[('graphics_mhz',as_int(row.get('clocks.max.graphics'))),
                              ('sm_mhz',as_int(row.get('clocks.max.sm'))),('memory_mhz',as_int(row.get('clocks.max.memory')))])
        sub(gpu,'power',[('max_limit_w',as_float(row.get('power.max_limit'))),('min_limit_w',as_float(row.get('power.min_limit'))),
                         ('default_limit_w',as_float(row.get('power.default_limit'))),
                         ('enforced_limit_w',as_float(row.get('power.limit')))])
        sub(gpu,'ecc_mode',[('current',clean(row.get('ecc.mode.current'))),('pending',clean(row.get('ecc.mode.pending')))])
        sub(gpu,'mig_mode',[('current',clean(row.get('mig.mode.current'))),('pending',clean(row.get('mig.mode.pending')))])
        put(gpu,'brand',clean(section.get('Product Brand')))
        put(gpu,'architecture',clean(section.get('Product Architecture')))
        put(gpu,'minor_number',as_int(section.get('Minor Number')))
        put(gpu,'module_id',as_int(section.get('Module ID')))
        put(gpu,'board_id',clean(section.get('Board ID')))
        put(gpu,'board_part_number',clean(section.get('Board Part Number')))
        put(gpu,'gpu_part_number',clean(section.get('GPU Part Number')))
        put(gpu,'multigpu_board',clean(section.get('MultiGPU Board')))
        put(gpu,'gsp_firmware_version',clean(section.get('GSP Firmware Version')))
        put(gpu,'inforom_image_version',clean((section.get('Inforom Version') or {}).get('Image Version')))
        put(gpu,'virtualization_mode',clean((section.get('GPU Virtualization Mode') or {}).get('Virtualization Mode')))
        put(gpu,'multiprocessor_count',sm_count(section))
        gpus.append(gpu)
    if gpus:attested['gpus']=sorted(gpus,key=lambda g:g.get('index',0))

    for row in parse_csv(texts.get('gpu_transient',''),transient_fields):
        entry={}
        put(entry,'index',as_int(row.get('index')));put(entry,'uuid',clean(row.get('uuid')))
        put(entry,'pstate',clean(row.get('pstate')));put(entry,'temperature_c',as_int(row.get('temperature.gpu')))
        put(entry,'fan_speed_pct',as_int(row.get('fan.speed')));put(entry,'power_draw_w',as_float(row.get('power.draw')))
        put(entry,'utilization_gpu_pct',as_int(row.get('utilization.gpu')))
        put(entry,'utilization_memory_pct',as_int(row.get('utilization.memory')))
        put(entry,'memory_used_mib',as_int(row.get('memory.used')));put(entry,'memory_free_mib',as_int(row.get('memory.free')))
        put(entry,'clocks_sm_mhz',as_int(row.get('clocks.sm')));put(entry,'clocks_memory_mhz',as_int(row.get('clocks.memory')))
        put(entry,'link_gen_current',as_int(row.get('pci.link.gen.current')))
        put(entry,'link_width_current',as_int(row.get('pci.link.width.current')))
        put(entry,'ecc_errors_corrected_volatile',as_int(row.get('ecc.errors.corrected.volatile.total')))
        put(entry,'ecc_errors_uncorrected_volatile',as_int(row.get('ecc.errors.uncorrected.volatile.total')))
        live.append(entry)
    for entry in live:
        section=next((v for k,v in by_bus.items() if v.get('GPU UUID')==entry.get('uuid')),{})
        rows=(section.get('Retired Pages') or {}).get('Remapped Rows') or {}
        put(entry,'remapped_rows_correctable',as_int(rows.get('Correctable Error')))
        put(entry,'remapped_rows_uncorrectable',as_int(rows.get('Uncorrectable Error')))
        put(entry,'remapped_rows_pending',clean(rows.get('Pending')))
    if live:transient['gpus']=sorted(live,key=lambda g:g.get('index',0))
    if 'smiq' in texts:
        transient['compute_process_count']=len(re.findall(r'^\s*Process ID\s+:',body(texts['smiq']),re.M))

    # topology, nvlink, mig ----------------------------------------------------------
    if 'topology' in texts:
        labels,rows,legend=parse_topology(texts['topology'])
        if rows:
            attested['topology']={'labels':labels,'rows':rows,'sources':['topology.txt']}
            if legend:attested['topology']['nic_legend']=legend
    if 'nvlink_cap' in texts:
        caps=parse_nvlink_caps(texts['nvlink_cap'])
        if caps:attested['nvlink']={'gpus':caps,'sources':['nvlink_cap.txt']}
    if 'nvlink' in texts:
        live_links=parse_nvlink_status(texts['nvlink'])
        if live_links:transient['nvlink']={'gpus':live_links}
    if 'mig' in texts:
        profiles=parse_mig_profiles(texts['mig'])
        if profiles:
            grouped={}
            for p in profiles:grouped.setdefault(p['gpu'],[]).append(p)
            attested['mig_profiles']={'gpus':[{'index':g,'profiles':[
                {'name':p['name'],'profile_id':p['profile_id'],'instances_total':p['instances_total'],
                 'memory_gib':p['memory_gib'],'p2p':p['p2p'],'sm_count':p['sm_count'],
                 'decoders':p['decoders'],'encoders':p['encoders']}
                for p in sorted(grouped[g],key=lambda p:p['profile_id'])]} for g in sorted(grouped)],
                'sources':['mig.txt']}
            transient['mig_instances']=[{'gpu':p['gpu'],'profile_id':p['profile_id'],
                                         'instances_free':p['instances_free']}
                                        for p in sorted(profiles,key=lambda p:(p['gpu'],p['profile_id']))]

    # cpu, numa, kernel, fabric software ---------------------------------------------
    if 'lscpu' in texts:
        cpu=parse_lscpu(texts['lscpu'])
        node={'sources':['lscpu.txt']}
        put(node,'model',clean(cpu.get('Model name')));put(node,'architecture',clean(cpu.get('Architecture')))
        put(node,'vendor_id',clean(cpu.get('Vendor ID')));put(node,'logical_cpus',as_int(cpu.get('CPU(s)')))
        put(node,'online_cpus',clean(cpu.get('On-line CPU(s) list')))
        put(node,'sockets',as_int(cpu.get('Socket(s)')))
        put(node,'cores_per_socket',as_int(cpu.get('Core(s) per socket')))
        put(node,'threads_per_core',as_int(cpu.get('Thread(s) per core')))
        put(node,'numa_nodes',as_int(cpu.get('NUMA node(s)')))
        put(node,'family',clean(cpu.get('CPU family')));put(node,'model_id',clean(cpu.get('Model')))
        put(node,'stepping',clean(cpu.get('Stepping')));put(node,'byte_order',clean(cpu.get('Byte Order')))
        put(node,'virtualization',clean(cpu.get('Virtualization')))
        put(node,'max_mhz',as_float(cpu.get('CPU max MHz')));put(node,'min_mhz',as_float(cpu.get('CPU min MHz')))
        for label,key in [('L1d cache','l1d_cache'),('L1i cache','l1i_cache'),('L2 cache','l2_cache'),('L3 cache','l3_cache')]:
            put(node,key,clean(cpu.get(label)))
        if len(node)>1:attested['cpu']=node
        sub(transient,'cpu',[('current_mhz',as_float(cpu.get('CPU MHz')))])
    if 'numa' in texts:
        count,nodes,distances=parse_numactl(texts['numa'])
        stable=[{k:v for k,v in dict(node,node=index).items() if k!='free_mb'} for index,node in sorted(nodes.items())]
        if stable:
            attested['numa']={'nodes':stable,'sources':['numa.txt']}
            put(attested['numa'],'node_count',count)
            if distances:attested['numa']['distances']=distances
            free=[{'node':index,'free_mb':node['free_mb']} for index,node in sorted(nodes.items()) if 'free_mb' in node]
            if free:transient['numa']={'nodes':free}
    if 'kernel' in texts:
        release=clean(body(texts['kernel']))
        if release:attested['kernel']={'release':release.splitlines()[0].strip(),'sources':['kernel.txt']}
    if 'ofed' in texts:
        version=clean(body(texts['ofed']))
        if version:attested['ofed']={'version':version.splitlines()[0].strip(),'sources':['ofed.txt']}
    if 'cuda' in texts:
        match=re.search(r'release\s+([0-9.]+)',body(texts['cuda']))
        if match:attested['cuda_toolkit']={'version':match.group(1),'sources':['cuda.txt']}

    # rdma ----------------------------------------------------------------------------
    links={(l['hca'],l['port']):l for l in parse_rdma_link(texts.get('rdma',''))}
    netdev={};ports_live=[]
    for (hca,port),link in links.items():
        put(netdev,hca,link.get('netdev'))
        ports_live.append({'hca':hca,'port':port,'state':link['state'],'physical_state':link['physical_state']})
    devices=[]
    for device in parse_verbs(texts.get('verbs','')):
        entry={'hca':device['hca'],'sources':['verbs.txt']+(['rdma.txt'] if device['hca'] in netdev else [])}
        for key in VERBS_DEVICE:put(entry,key,clean(device['attributes'].get(key)))
        put(entry,'netdev',netdev.get(device['hca']))
        ports=[]
        for port in device['ports']:
            item={}
            put(item,'port',port['port'])
            for key in VERBS_PORT_ATTESTED:put(item,key,clean(port['attested'].get(key)))
            if item:ports.append(item)
            existing=next((p for p in ports_live if p['hca']==device['hca'] and p['port']==port['port']),None)
            if existing is None:
                existing={'hca':device['hca'],'port':port['port']};ports_live.append(existing)
            for key in VERBS_PORT_TRANSIENT:
                target=VERBS_PORT_FALLBACK.get(key,key)
                if target in existing:continue
                put(existing,target,clean(port['transient'].get(key)))
        if ports:entry['ports']=ports
        devices.append(entry)
    if devices:attested['rdma']={'devices':sorted(devices,key=lambda d:d['hca']),'sources':['verbs.txt']}
    if ports_live:transient['rdma']={'ports':sorted(ports_live,key=lambda p:(p['hca'],p['port']))}

    # what is missing, and why ---------------------------------------------------------
    unavailable=[]
    for field,command in SECTIONS:
        half,_,key=field.partition('.')
        if key in (attested if half=='attested' else transient):continue
        unavailable.append({'field':field,'command':command,'reason':reason_for(status.get(command),command)})
    if 'gpus' in attested and not any('multiprocessor_count' in g for g in attested['gpus']):
        unavailable.append({'field':'attested.gpus[].multiprocessor_count','command':'smiq',
                            'reason':reason_for(status.get('smiq'),'smiq')})
    # A field this driver refused by name. The query was re-asked without it, so the rest of the
    # capture is real; this is where the missing one is named, in the driver's own words.
    for command,declared,paths,half in (('gpu_attested',ATTESTED_GPU_FIELDS,ATTESTED_GPU_PATHS,'attested'),
                                        ('gpu_transient',TRANSIENT_GPU_FIELDS,TRANSIENT_GPU_PATHS,'transient')):
        asked=set(query_fields((status.get(command) or {}).get('argv',[])))
        if not asked:continue  # the query never ran at all; the whole section is reported above
        refused={}
        for result in manifest.get('results',[]):
            if not str(result.get('id','')).startswith(command+'_rejected'):continue
            for sentence in rejections(result.get('reason','')):
                refused.setdefault(rejected_fields(sentence)[0],(str(result['id']),sentence))
        for field in declared.split(','):
            if field in asked:continue
            source,reason=refused.get(field,(command,'not requested by this capture'))
            unavailable.append({'field':f'{half}.gpus[].{paths[field]}','command':source,'reason':reason})

    record={'schema':SCHEMA,'hardware_sha256':digest(attested),'attested':attested,
            'transient':transient,'commands':manifest.get('results',[]),
            'unavailable':sorted(unavailable,key=lambda u:u['field'])}
    put(record,'sku',sku or manifest.get('sku'))
    if manifest.get('fixture'):record['fixture']=True
    return record

def reason_for(result,command):
    if result is None:return f'{command} was not attempted by this probe'
    if result.get('status')!='PASS':
        return result.get('reason') or f'exited {result.get("exit")}'
    return f'not reported by {result.get("path",command)}'


# --- capture ------------------------------------------------------------------------
def capture(out):
    results=[]
    for id,cmd in COMMANDS.items():
        if not shutil.which(cmd[0]):results.append({'id':id,'status':'NOT_RUN','reason':'utility not installed','argv':cmd});continue
        argv=list(cmd);attempt=0
        while True:
            try:
                r=subprocess.run(argv,capture_output=True,text=True,timeout=20,check=False)
            except (OSError,subprocess.TimeoutExpired) as exc:
                results.append({'id':id,'status':'NOT_RUN','reason':str(exc),'argv':argv});break
            raw=(r.stdout+'\nSTDERR:\n'+r.stderr).encode()
            refusals=rejections(r.stdout+r.stderr) if r.returncode!=0 else []
            reduced=without_fields(argv,rejected_fields(' '.join(refusals))) if refusals else None
            if reduced is None:
                path=out/(id+'.txt');path.write_bytes(raw)
                results.append({'id':id,'status':'PASS' if r.returncode==0 and r.stdout.strip() else 'NOT_RUN',
                                'exit':r.returncode,'argv':argv,'path':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
                break
            # This driver does not implement a field name we asked for, and refuses the whole
            # query over it. Keep the refusal as a checksummed artifact, drop exactly the names
            # it refused and ask again: one unknown name must not cost every other attribute.
            attempt+=1
            path=out/f'{id}.rejected.{attempt}.txt';path.write_bytes(raw)
            results.append({'id':f'{id}_rejected_{attempt}','status':'NOT_RUN','exit':r.returncode,
                            'reason':' '.join(refusals),'argv':argv,'path':path.name,
                            'sha256':hashlib.sha256(raw).hexdigest()})
            argv=reduced
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out');p.add_argument('--sku')
    p.add_argument('--replay',help='rebuild the hardware record from an existing capture directory; runs nothing')
    args=p.parse_args()
    if args.replay:
        record=build_record(Path(args.replay),sku=args.sku)
        print(json.dumps(record,indent=2))
        return 0
    if not args.out:raise SystemExit('--out is required unless --replay is given')
    out=Path(args.out)
    if out.exists():raise SystemExit('Refusing to overwrite an existing probe directory')
    out.mkdir(parents=True);results=capture(out)
    result={'schema':'stratum-readonly-probe/v1','generated_at':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
      'scope':'Inventory capture only. Does not prove kernel support, GPUDirect, remote visibility, measured bandwidth or correctness.',
      'environment':{k:os.environ[k] for k in ['NCCL_ALGO','NCCL_PROTO','NCCL_NET','NCCL_IB_HCA','NVSHMEM_REMOTE_TRANSPORT'] if k in os.environ},'results':results}
    if args.sku:result['sku']=args.sku
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        record=build_record(out,sku=args.sku)
        (out/'hardware.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'hardware_sha256':record['hardware_sha256'],
                          'unavailable':[u['field'] for u in record['unavailable']]},indent=2))
    except Exception as exc:
        # The raw capture is already on disk and is the evidence; a parser failure must not
        # destroy it, and must not be reported as a successful record either.
        print(f'hardware record NOT built: {exc}')
    print(json.dumps(result,indent=2))
    return 0 if results and all(r['status']=='PASS' for r in results) else 2
if __name__=='__main__':raise SystemExit(main())
