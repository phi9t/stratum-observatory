"""Deterministic domain-contract checks; these are not hardware execution claims.

Each validator checks an explicit record. Missing inputs block rather than pass.
Source and review verification are separate, mandatory publication gates.
"""
from __future__ import annotations
import math
from collections import Counter
from ..evidence import HEX40, HEX64
from ..model import Finding

DOMAINS = ('model','gpu','scale_up','scale_out','communication','training','serving','numerics','storage','measurement','visualization','release')

class Check:
    def __init__(self,domain,record): self.domain=domain; self.record=record; self.out=[]
    def require(self,*keys):
        missing=[k for k in keys if k not in self.record or self.record[k] is None]
        if missing:
            self.out.append(Finding('DOMAIN_INPUT',self.domain,'BLOCKED',self.record.get('id',self.domain),'Missing contract inputs: '+', '.join(missing),'Supply a source-bound contract; do not use empty objects as skipped tests.'))
        return not missing
    def check(self,condition,code,message):
        if not condition:self.out.append(Finding(code,self.domain,'FAIL',self.record.get('id',self.domain),message,'Correct the underlying model/units/state/order, then rerun this domain and its dependents.'))
    def finish(self):
        return self.out or [Finding('CONTRACT_CONSISTENT',self.domain,'PASS',self.record.get('id',self.domain),'Checked contract is internally consistent; this is not an empirical conformance result.')]

def integer(n,positive=True):return type(n)==int and (n>0 if positive else n>=0)
def finite(n):return type(n) in (int,float) and math.isfinite(n)
def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)

def model(c,x):
    if not c.require('layers','layer_groups','experts','top_k','shared','d','latent','heads','kv_heads','head_dim','index_base'):return
    c.check(all(integer(x[k]) for k in ('layers','d','latent','heads','kv_heads','head_dim')),'MODEL_DIMENSION','Tensor dimensions and layer counts must be positive integers.')
    c.check(x['index_base'] in (0,1),'MODEL_INDEX_BASE','Layer indexing convention must be explicit.')
    all_layers=[i for values in x['layer_groups'].values() for i in values]
    expected=list(range(x['index_base'],x['index_base']+x['layers']))
    c.check(sorted(all_layers)==expected,'MODEL_LAYER_PARTITION','Attention families must partition the decoder layers exactly once.')
    c.check(integer(x['experts'],False) and integer(x['top_k'],False) and ((x['experts']==0 and x['top_k']==0 and x['shared']==0) or (x['experts']>0 and 0<x['top_k']<=x['experts'])) and integer(x['shared'],False),'MODEL_ROUTING','Routed top-k must fit the expert population; shared count is separate.')
    c.check(x['heads']%x['kv_heads']==0,'MODEL_GQA','Query heads must form integer groups over KV heads.')
    c.check(x['latent']<=x['d'],'MODEL_LATENT','A declared compressed expert path cannot exceed its input width.')
    if 'tensors' in x:
        tensors=x['tensors']
        for name,t in tensors.items():
            c.check(t.get('dtype') in ('fp32','bf16','fp16','fp8','fp4','int8','int4'),'MODEL_DTYPE',f'{name} lacks an explicit supported representation.')
            c.check(all(integer(d) for d in t['shape']),'MODEL_SHAPE',f'Invalid shape for {name}.')
        for op in x.get('ops',[]):
            if op['kind']=='linear':
                a,b=tensors[op['input']]['shape'],tensors[op['output']]['shape'];w=op['weight_shape']
                c.check(a[:-1]==b[:-1] and w==[b[-1],a[-1]],'MODEL_CONTRACTION','Linear contraction shape is inconsistent.')
            elif op['kind']=='add':
                shapes=[tensors[k]['shape'] for k in op['inputs']]+[tensors[op['output']]['shape']]
                c.check(all(s==shapes[0] for s in shapes),'MODEL_RESIDUAL','Residual add requires equal shapes or a documented projection.')

GPU_ID={
 'H100-SXM':('gpu','hopper','9.0',1,0),
 'H200-SXM':('gpu','hopper','9.0',1,0),
 'B200':('gpu','blackwell','10.0',1,0),
 'B300':('gpu','blackwell-ultra','10.3',1,0),
 'GB300':('superchip','blackwell-ultra','10.3',2,1),
 'GB300-NVL72':('rack','blackwell-ultra','10.3',72,36),
}
def gpu(c,x):
    if not c.require('sku','kind','architecture','compute_capability','gpu_count','cpu_count','memory'):return
    spec=GPU_ID.get(x['sku'])
    if spec is None:c.check(False,'GPU_SKU','Unknown SKU: no family-name extrapolation is allowed.');return
    kind,arch,cc,ng,nc=spec
    c.check((x['kind'],x['gpu_count'],x['cpu_count'])==(kind,ng,nc),'GPU_IDENTITY','GPU, superchip, and NVL rack are distinct scopes. GB300 is not one GPU.')
    c.check((x['architecture'],x['compute_capability'])==(arch,cc),'GPU_CAPABILITY','Wrong silicon family or compute capability for this SKU.')
    m=x['memory']
    c.check(m.get('unit') in ('GB','GiB') and m.get('scope') in ('per_gpu','system') and m.get('kind') in ('nominal','allocatable_measured') and finite(m.get('value')) and m['value']>0,'GPU_MEMORY','HBM capacity needs units, scope, and nominal-versus-measured status.')
    if arch=='hopper': c.check(not ({'fp4','mxfp4','nvfp4'} & set(x.get('native_formats',[]))),'GPU_NATIVE_FORMAT','Hopper must not be labeled as having native FP4 tensor-core execution.')
    for name,rate in x.get('rates',{}).items():
        c.check(rate.get('unit') in ('GB/s','Gb/s','TFLOP/s') and rate.get('scope') in ('per_gpu','per_link','per_port','system') and rate.get('direction') in ('one_way','bidirectional_aggregate','not_applicable'),'GPU_RATE_SCOPE',f'{name} has ambiguous units/aggregation/direction.')
        if rate.get('unit')=='TFLOP/s':c.check(rate.get('precision') is not None and rate.get('sparsity') in ('dense','2:4'),'GPU_PEAK_MODE','Compute peaks require precision and sparsity; no implicit sparse-to-dense comparison.')
    if 'kernel' in x:
        k=x['kernel'];limits=k.get('limits',{})
        required=('threads_per_block','registers_per_thread','shared_bytes','cluster_size','features','native_math','barrier_before_consume')
        c.check(all(key in k for key in required) and bool(limits),'GPU_KERNEL_SCHEMA','Kernel resource and instruction limits must be explicit and source-backed.')
        if all(key in k for key in required) and limits:
            c.check(0<k['threads_per_block']<=limits['threads_per_block'] and 0<k['registers_per_thread']<=limits['registers_per_thread'] and k['threads_per_block']*k['registers_per_thread']<=limits['registers_per_sm'] and 0<=k['shared_bytes']<=limits['shared_bytes_per_block'],'GPU_KERNEL_RESOURCES','Kernel exceeds a per-block or per-SM necessary resource bound; passing is not an occupancy prediction.')
            c.check(k['cluster_size']<=limits['portable_cluster'] or k.get('nonportable_optin') is True and k['cluster_size']<=limits['nonportable_cluster'],'GPU_CLUSTER_OPTIN','A nonportable thread-block cluster needs explicit opt-in and a device-specific limit.')
            if 'tma' in k['features']:c.check(k['barrier_before_consume'] is True,'GPU_ASYNC_ORDER','TMA completion/visibility must precede consumers; an async launch is not completion.')
            if arch=='hopper':c.check(k['native_math'] not in ('fp4','mxfp4','nvfp4'),'GPU_NATIVE_FORMAT','Stored FP4 weights do not imply native Hopper FP4 arithmetic.')
    if 'kernel_target' in x:
        allowed={'9.0':{'sm_90','sm_90a'},'10.0':{'sm_100','sm_100a','sm_100f'},'10.3':{'sm_103','sm_103a','sm_103f','sm_100f'}}[cc]
        c.check(x['kernel_target'] in allowed,'GPU_KERNEL_TARGET','Kernel target needs architecture-specific compatibility evidence; suffixes are not interchangeable.')

def scale_up(c,x):
    if not c.require('system','compute_trays','gpus_per_tray','cpus_per_tray','total_gpus','total_cpus','scale_up','scale_out'):return
    c.check(x['compute_trays']*x['gpus_per_tray']==x['total_gpus'] and x['compute_trays']*x['cpus_per_tray']==x['total_cpus'],'TOPOLOGY_CONSERVATION','Per-tray components do not reconcile with the system total.')
    c.check(x['scale_up']=='NVLink' and x['scale_out'] in ('InfiniBand','RoCEv2','Ethernet'),'TOPOLOGY_FABRIC','NVLink domain and scale-out fabric must be represented separately.')
    if x['system']=='GB300-NVL72':c.check(x['total_gpus']==72 and x['total_cpus']==36,'TOPOLOGY_NVL72','NVL72 rack identity requires 72 GPUs and 36 Grace CPUs.')
    if 'switch_count' in x:
        expected={'DGX-H100':(4,'asic'),'DGX-H200':(4,'asic'),'DGX-B200':(2,'asic'),'GB300-NVL72':(9,'tray')}.get(x['system'])
        if expected:c.check((x['switch_count'],x.get('switch_count_scope'))==expected,'TOPOLOGY_SWITCHES','Wrong switch count or ASIC-versus-tray scope for the named system; do not reuse a Hopper topology unchanged.')
    if 'gpu_nic_map' in x:
        c.check(set(x['gpu_nic_map'])=={str(i) for i in range(x['total_gpus'])},'TOPOLOGY_AFFINITY','Every logical GPU needs a declared NIC/rail affinity; do not infer it from slot order.')
    if x.get('peer_access_claim')=='available':c.check(x.get('peer_probe_sha256') is not None,'TOPOLOGY_PEER_ACCESS','A topology drawing alone does not establish permitted peer access, IMEX domain, or fabric partition.')

def scale_out(c,x):
    if not c.require('fabric','link_gbps','direction','scope','payload_GBs','mtu','route_mtu','ports','uplink_ports','uplink_gbps','oversubscription'):return
    c.check(finite(x['link_gbps']) and x['link_gbps']>0 and close(x['payload_GBs'],x['link_gbps']/8),'NETWORK_RATE','Per-port line rate converts bits to bytes by /8; payload goodput requires additional overhead/efficiency.')
    c.check(x['direction']=='one_way' and x['scope']=='per_port','NETWORK_DIRECTION','A one-way transfer bound cannot use bidirectional aggregate or total-switch bandwidth.')
    c.check(integer(x['mtu']) and integer(x['route_mtu']) and x['mtu']<=x['route_mtu'],'NETWORK_MTU','RDMA path MTU exceeds a route segment limit.')
    if x['uplink_ports']<=0 or x['uplink_gbps']<=0:c.check(False,'NETWORK_PORTS','Uplink capacity must be positive.')
    else:c.check(close(x['oversubscription'],max(1,x['ports']*x['link_gbps']/(x['uplink_ports']*x['uplink_gbps']))),'NETWORK_OVERSUB','Oversubscription does not match downlink/uplink port capacity.')
    if x['fabric']=='RoCEv2':
        if not c.require('mode','pfc','ecn','gid_type','traffic_class'):return
        c.check(x['mode'] in ('lossy','lossless'),'ROCE_MODE','RoCE mode must be explicit; PFC is not universally mandatory.')
        if x['mode']=='lossless':c.check(x['pfc'] is True,'ROCE_PFC','This lossless-PFC profile requires PFC along the path, not merely on one host.')
        c.check(x['ecn'] is True,'ROCE_ECN','This RoCEv2 profile requires ECN/CNP configuration evidence.')
        c.check(x['gid_type']=='RoCEv2' and type(x['traffic_class'])==int and 0<=x['traffic_class']<=255,'ROCE_ADDRESSING','RoCEv2 GID, routing and traffic-class mapping must be consistent.')
    elif x['fabric']=='InfiniBand':
        c.check(bool(x.get('subnet_manager')) and x.get('pkey') is not None and x.get('sl') is not None,'IB_CONTROL_PLANE','InfiniBand requires declared SM/partition/SL policy, not Ethernet QoS assumptions.')
        if 'pkey' in x:c.check(type(x['pkey'])==int and 0<x['pkey']<=65535 and (x['pkey'] & 0x7fff)!=0,'IB_PARTITION','P_Key must have a nonzero 15-bit partition identifier and an explicit membership bit.')
        if 'sl' in x:c.check(type(x['sl'])==int and 0<=x['sl']<=15,'IB_SERVICE_LEVEL','InfiniBand service level is a 4-bit field; do not substitute an Ethernet traffic-class byte.')
    else:c.check(False,'NETWORK_FABRIC','Unsupported scale-out fabric; do not equate TCP with RDMA.')
    if 'health' in x:
        h=x['health']
        c.check(h.get('uncorrectable_errors',0)==0 and h.get('timeout_ms',0)>0 and (h.get('abort_completed') is True or h.get('degraded') is False and h.get('retries',0)==0),'NETWORK_HEALTH','Transport errors, timeout/abort failure or retry degradation must be surfaced, not excluded from healthy-path claims.')
    if x.get('gpudirect')=='verified':c.check(bool(x.get('gpudirect_probe')) and bool(x.get('registration_backend')),'NETWORK_GDR','GPUDirect needs actual registration/data-path evidence, not an RDMA-capable NIC label.')

def communication(c,x):
    if not c.require('library','version','api_role','backend','source_refs','memory_registered','submitted','completed','released','remote_ready_after_complete','producer_event','consumer_wait','progress','completion'):return
    roles={'NCCL':'collectives','NVSHMEM':'pgas','NIXL':'transfer'}
    c.check(x['library'] in roles and x['api_role']==roles.get(x['library']),'COMM_ROLE','NCCL collectives, NVSHMEM PGAS and NIXL transfers are not interchangeable abstractions.')
    c.check(bool(x['version']) and x['version'] not in ('main','latest') and bool(x['source_refs']),'COMM_VERSION','Bind the selected API/backend to a pinned implementation version.')
    c.check(bool(x['backend']) and x['progress'] in ('poll','thread','device','stream'),'COMM_PROGRESS','Selected transport and progress mechanism must be explicit.')
    c.check(x['memory_registered'] is True,'COMM_REGISTRATION','Buffers must satisfy the selected backend registration/lifetime contract.')
    c.check(all(integer(x[k],False) for k in ('submitted','completed','released')) and x['released']<=x['completed']<=x['submitted'],'COMM_LIFETIME','Release/registration lifetime must extend beyond transfer completion.')
    c.check(x['remote_ready_after_complete'] is True,'COMM_REMOTE_READY','Enqueued or locally submitted does not imply remotely consumable.')
    c.check(x['producer_event'] is True and x['consumer_wait'] is True,'COMM_CUDA_ORDER','Producer and consumer CUDA ordering must be established.')
    if x['library']=='NVSHMEM':c.check(x['completion'] in ('quiet','barrier_completion','signal_wait_protocol'),'COMM_COMPLETION','Fence establishes ordering, not completion; prove visibility with the appropriate completion protocol.')
    if x['library']=='NIXL':c.check(x['completion'] not in ('enqueue','submitted','fence'),'COMM_COMPLETION','NIXL transfer completion must be observed before buffer reuse.')
    if x['library']=='NCCL' and 'algorithm' in x:
        c.check(x['algorithm'].get('kind') in ('assumed','observed'),'NCCL_ALGORITHM','Ring/tree selection is either an explicit analytic assumption or observed in runtime evidence.')
        if x['algorithm'].get('kind')=='observed':c.check(bool(x['algorithm'].get('trace_sha256')),'NCCL_ALGORITHM','Observed NCCL algorithm requires a trace, not a payload formula.')

def acyclic(nodes):
    if len({n['id'] for n in nodes})!=len(nodes):return False
    graph={n['id']:n.get('after',[]) for n in nodes};seen={}
    def dfs(k):
        if k not in graph:return False
        if seen.get(k)==1:return False
        if seen.get(k)==2:return True
        seen[k]=1
        if not all(dfs(p) for p in graph[k]):return False
        seen[k]=2;return True
    return all(dfs(k) for k in graph)

def training(c,x):
    if not c.require('world_size','rank_coords','mesh_axes','microbatch','accumulation','dp_size','global_batch','optimizer','optimizer_bytes','optimizer_total','schedule'):return
    ranks=x['rank_coords'];axes=x['mesh_axes'];coords=[tuple(r[a] for a in axes) for r in ranks]
    mesh=math.prod(len({r[a] for r in ranks}) for a in axes)
    c.check(len(ranks)==x['world_size'] and {r['rank'] for r in ranks}==set(range(x['world_size'])) and len(set(coords))==len(coords) and mesh==x['world_size'],'TRAIN_RANK_MESH','Declared independent rank coordinates must be a bijective mesh; EP/CP subgroups are not extra world-size factors by default.')
    if 'ep_group' in x:c.check(set(x['ep_group'])<=set(range(x['world_size'])),'TRAIN_GROUP','Expert subgroup contains a nonexistent rank.')
    c.check(all(integer(x[k]) for k in ('microbatch','accumulation','dp_size')) and x['global_batch']==x['microbatch']*x['accumulation']*x['dp_size'],'TRAIN_BATCH','Global examples/step = microbatch × accumulation × data-parallel degree, not tensor/expert rank count.')
    c.check(all(finite(v) and v>=0 for v in x['optimizer_bytes'].values()) and close(sum(x['optimizer_bytes'].values()),x['optimizer_total']),'TRAIN_MEMORY','State bytes must reconcile under the named optimizer/precision/offload scenario.')
    c.check(bool(x['schedule']) and acyclic(x['schedule']),'TRAIN_SCHEDULE','Schedule contains a missing dependency, duplicate task or same-step cycle.')
    if x.get('loss_normalization')=='per_rank_mean' and x.get('unequal_tokens_per_rank'):c.check(False,'TRAIN_LOSS_WEIGHTING','Unweighted rank-mean loss is not a token-weighted global objective for uneven token counts.')

def serving(c,x):
    if not c.require('context','prefix','layer_states','required_layers','namespace','private_mutable_state','handoff','weights_replicas','placement','estimated_fit','memory_components','logical_total'):return
    c.check(integer(x['context'],False) and integer(x['prefix'],False) and x['prefix']<=x['context'],'SERVING_LENGTH','Prefix boundary cannot exceed the requested prompt.')
    states=x['layer_states']
    c.check(len(states)==len(x['required_layers']) and {s['layer'] for s in states}==set(x['required_layers']),'SERVING_STATE_COMPLETE','Prefix restore requires all attention/recurrent state groups, not only token K/V.')
    c.check(all(s['boundary']==x['prefix'] for s in states),'SERVING_PREFIX','All restored state must belong to the same token boundary.')
    c.check(all(s['namespace']==x['namespace'] for s in states) and x['private_mutable_state'] is True,'SERVING_ISOLATION','Mutable recurrent state must be request-private or explicitly copy-on-write.')
    h=x['handoff'];order=['registered','producer_done','submitted','completed','consumer_start','released']
    c.check(all(k in h and finite(h[k]) for k in order) and all(h[a]<=h[b] for a,b in zip(order,order[1:])),'SERVING_HANDOFF','Producer readiness, transfer completion, consumer launch and release are out of order.')
    if x['placement']=='disaggregated':c.check(x['weights_replicas']>=2,'SERVING_REPLICAS','Separate prefill/decode pools require independent weight residency unless a sourced alternate arrangement is declared.')
    c.check(all(finite(v) and v>=0 for v in x['memory_components'].values()) and close(sum(x['memory_components'].values()),x['logical_total']),'SERVING_MEMORY','Logical cache/weight components do not reconcile.')
    if x['estimated_fit']:c.check(bool(x.get('peak_allocation_measurement')) and bool(x.get('workspace_reserve')),'SERVING_FIT','Logical payload alone is not a fit guarantee.')

def numerics(c,x):
    if not c.require('format','numel','weight_bits','scale_bytes','group_size','payload_bytes','accumulator','reference','actual','atol','rtol','tolerance_policy','finite_required'):return
    c.check(integer(x['numel'],False) and integer(x['weight_bits']) and integer(x['group_size']) and integer(x['scale_bytes'],False),'NUMERICS_SHAPE','Quantization group/payload dimensions are invalid.')
    if x['group_size']<=0:return
    size=math.ceil(x['numel']*x['weight_bits']/8)+math.ceil(x['numel']/x['group_size'])*x['scale_bytes']
    c.check(x['payload_bytes']==size,'NUMERICS_STORAGE','Packed values and all partial-group scales must be counted; this formula excludes explicitly separate metadata/padding.')
    if x['format']=='mxfp4':c.check((x['weight_bits'],x['group_size'],x['scale_bytes'])==(4,32,1),'NUMERICS_FORMAT','This MXFP4 profile uses 4-bit weights and one 8-bit scale per 32 values; it is not NVFP4.')
    p=x['tolerance_policy']
    c.check(finite(x['atol']) and finite(x['rtol']) and 0<=x['atol']<=p['atol_max'] and 0<=x['rtol']<=p['rtol_max'],'NUMERICS_TOLERANCE','Tolerance exceeds the precommitted numerical-error policy.')
    a,b=x['reference'],x['actual']
    c.check(bool(a) and len(a)==len(b),'NUMERICS_OUTPUT','Differential outputs must be nonempty and identically shaped.')
    c.check(all(finite(v) for v in a+b),'NUMERICS_FINITE','NaN/Inf in a required-finite output is a failure, not an excluded sample.')
    if len(a)==len(b) and all(finite(v) for v in a+b):c.check(all(abs(u-v)<=x['atol']+x['rtol']*abs(u) for u,v in zip(a,b)),'NUMERICS_DIFFERENTIAL','Numerical outputs exceed the approved reference-relative error budget.')

def storage(c,x):
    if not c.require('checkpoint_id','expected_shards','shards','state_keys','commit_published','persisted_shards','manifest_total','tokenizer_revision','dataset_revision'):return
    ids=[s['id'] for s in x['shards']]
    c.check(len(ids)==len(set(ids)) and set(ids)==set(x['expected_shards']),'STORAGE_SHARDS','Missing, duplicated or unexpected checkpoint shards.')
    c.check(all(HEX64.fullmatch(str(s.get('sha256',''))) and integer(s.get('size'),False) for s in x['shards']),'STORAGE_DIGEST','Each shard needs a content hash and byte length.')
    c.check(sum(s['size'] for s in x['shards'])==x['manifest_total'],'STORAGE_SIZE','Checkpoint manifest byte total does not reconcile.')
    if x['commit_published']:c.check(set(x['persisted_shards'])==set(x['expected_shards']),'STORAGE_ATOMICITY','Commit marker became visible before all required shards were durable.')
    c.check({'weights','optimizer','rng','sampler','step'}<=set(x['state_keys']),'STORAGE_RESTART','Exact training restart requires optimizer, RNG, data-position and step state, not only weights.')
    c.check(bool(HEX40.fullmatch(x['tokenizer_revision'])) and bool(HEX40.fullmatch(x['dataset_revision'])),'STORAGE_DATA_PIN','Tokenizer and dataset/preprocessing identity must be version-pinned.')

def measurement(c,x):
    if not c.require('kind','metric','samples','warmup','iterations','unit','environment','summary','statistic','timeouts','failures','request_count'):return
    c.check(x['kind'] in ('measured','assumed','derived'),'MEASURE_KIND','Evidence class must distinguish observation from estimate.')
    if x['kind']=='measured':
        c.check(bool(x.get('raw_artifact')) and bool(HEX64.fullmatch(str(x.get('raw_sha256','')))),'MEASURE_TRACE','Measured claims need raw outputs bound by checksum and environment.')
        c.check(x.get('method')!='payload_over_rate','MEASURE_ESTIMATE','A bandwidth-only lower bound is not a measured latency.')
        c.check({'gpu','driver','cuda','framework_commit','clocks'}<=set(x['environment']),'MEASURE_ENV','Performance comparison needs exact GPU/software/clock/environment identity.')
    samples=x['samples'];c.check(bool(samples) and all(finite(v) and v>=0 for v in samples),'MEASURE_SAMPLES','Samples must be nonempty, finite and nonnegative.')
    c.check(len(samples)==x['iterations'] and len(samples)+x['timeouts']+x['failures']==x['request_count'],'MEASURE_DENOMINATOR','Failed/timed-out requests must stay in the declared denominator.')
    if samples and x['statistic']=='mean':c.check(close(sum(samples)/len(samples),x['summary']),'MEASURE_SUMMARY','Summary does not agree with raw samples.')
    else:c.check(x['statistic']=='mean','MEASURE_STATISTIC','Unsupported estimator; add a tested quantile/bootstrap implementation before using this statistic.')

def visualization(c,x):
    if not c.require('claim_ids','nodes','edges','canvas'):return
    nodes=x['nodes'];ids={n['id'] for n in nodes}
    c.check(bool(nodes) and len(ids)==len(nodes),'VISUAL_NODES','Diagram must contain unique nodes.')
    known=set(x['claim_ids']);width,height=x['canvas']
    for n in nodes:
        c.check(bool(n.get('claims')) and set(n['claims'])<=known,'VISUAL_PROVENANCE',f'{n["id"]} has an unbound scientific label.')
        a,b,w,h=n['bounds'];tx,ty,tw,th=n['text_bounds']
        c.check(0<=a and 0<=b and a+w<=width and b+h<=height and a<=tx and b<=ty and tx+tw<=a+w and ty+th<=b+h,'VISUAL_OVERFLOW',f'{n["id"]} overflows its container or canvas.')
    for e in x['edges']:
        pts=e['points'];c.check(all(a[0]==b[0] or a[1]==b[1] for a,b in zip(pts,pts[1:])),'VISUAL_ORTHOGONAL','Wires must use orthogonal segments.')
        c.check(e['source'] in ids and e['target'] in ids,'VISUAL_EDGE','Diagram edge references nonexistent nodes.')

def release(c,x):
    if not c.require('source_sha256','built_from_sha256','artifact_sha256','repeat_build_sha256','external_requests','geometry_defects','runtime_errors','commands','source_lock_sha256','receipt_lock_sha256'):return
    c.check(x['source_sha256']==x['built_from_sha256'],'RELEASE_STALE','Build receipt belongs to a different source tree.')
    c.check(x['artifact_sha256']==x['repeat_build_sha256'],'RELEASE_DETERMINISM','Repeated builds produced different portable HTML.')
    c.check(x['source_lock_sha256']==x['receipt_lock_sha256'],'RELEASE_LOCK','Evidence receipt refers to a different source lock.')
    c.check(x['external_requests']==x['geometry_defects']==x['runtime_errors']==0,'RELEASE_BROWSER','Browser/runtime/geometry gate is not clean.')
    c.check({'node','browser'}<=set(x['commands']) and all(v==0 for v in x['commands'].values()),'RELEASE_COMMAND','Required verification command is missing or failed.')

VALIDATORS={d:globals()[d] for d in DOMAINS}
def validate(domain: str, record: dict) -> list[Finding]:
    if domain not in VALIDATORS:return [Finding('DOMAIN_UNKNOWN',domain,'FAIL',domain,'Unknown verifier domain.','Register a tested verifier; never skip an unknown domain.')]
    c=Check(domain,record)
    try:VALIDATORS[domain](c,record)
    except (KeyError,TypeError,ValueError,ZeroDivisionError,IndexError,OverflowError) as exc:
        c.out.append(Finding('DOMAIN_SCHEMA',domain,'FAIL',record.get('id',domain),f'Invalid typed contract: {exc}','Fix the contract schema; this is not a skipped check.'))
    return c.finish()
