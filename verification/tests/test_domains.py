import copy
import unittest
from verification.domains import contracts

def codes(domain, record): return {f.code for f in contracts.validate(domain,record) if f.status!='PASS'}

class DomainTests(unittest.TestCase):
    def test_api_exists(self): self.assertTrue(callable(getattr(contracts,'validate',None)))
    def test_unknown_domain_fails(self): self.assertIn('DOMAIN_UNKNOWN', codes('invented',{}))
    def test_empty_domain_cannot_pass(self):
        for d in contracts.DOMAINS:
            with self.subTest(domain=d): self.assertTrue(codes(d,{}))
    def test_model_partition(self):
        x=dict(layers=4,layer_groups={'kda':[1,2,3],'mla':[4]},experts=8,top_k=2,shared=1,d=16,latent=8,heads=4,kv_heads=2,head_dim=4,index_base=1)
        self.assertFalse(codes('model',x));x['layer_groups']['mla']=[3,4]
        self.assertIn('MODEL_LAYER_PARTITION',codes('model',x))
    def test_top_k_cannot_exceed_experts(self):
        x=dict(layers=1,layer_groups={'dense':[1]},experts=8,top_k=9,shared=1,d=16,latent=8,heads=4,kv_heads=2,head_dim=4,index_base=1)
        self.assertIn('MODEL_ROUTING',codes('model',x))
    def test_gb300_is_not_single_gpu(self):
        x=dict(sku='GB300',kind='gpu',architecture='blackwell-ultra',compute_capability='10.3',gpu_count=1,cpu_count=0,memory={'value':288,'unit':'GB','scope':'per_gpu','kind':'nominal'})
        self.assertIn('GPU_IDENTITY',codes('gpu',x))
    def test_gb300_superchip_counts(self):
        x=dict(sku='GB300',kind='superchip',architecture='blackwell-ultra',compute_capability='10.3',gpu_count=2,cpu_count=1,memory={'value':288,'unit':'GB','scope':'per_gpu','kind':'nominal'})
        self.assertFalse(codes('gpu',x))
    def test_hopper_not_native_fp4(self):
        x=dict(sku='H100-SXM',kind='gpu',architecture='hopper',compute_capability='9.0',gpu_count=1,cpu_count=0,native_formats=['bf16','fp4'],memory={'value':80,'unit':'GB','scope':'per_gpu','kind':'nominal'})
        self.assertIn('GPU_NATIVE_FORMAT',codes('gpu',x))
    def test_wrong_capability(self):
        x=dict(sku='B200',kind='gpu',architecture='blackwell',compute_capability='9.0',gpu_count=1,cpu_count=0,memory={'value':180,'unit':'GB','scope':'per_gpu','kind':'nominal'})
        self.assertIn('GPU_CAPABILITY',codes('gpu',x))
    def test_rack_gpu_conservation(self):
        x=dict(system='GB300-NVL72',compute_trays=18,gpus_per_tray=4,cpus_per_tray=2,total_gpus=72,total_cpus=36,scale_up='NVLink',scale_out='InfiniBand')
        self.assertFalse(codes('scale_up',x));x['total_gpus']=144
        self.assertIn('TOPOLOGY_CONSERVATION',codes('scale_up',x))
    def net(self): return dict(fabric='RoCEv2',mode='lossless',link_gbps=400,direction='one_way',scope='per_port',payload_GBs=50,mtu=4096,route_mtu=9000,pfc=True,ecn=True,gid_type='RoCEv2',traffic_class=3,ports=8,uplink_ports=8,uplink_gbps=400,oversubscription=1)
    def test_400_gbps_is_50_gbytes(self): self.assertFalse(codes('scale_out',self.net()))
    def test_bits_bytes_mismatch(self):
        x=self.net();x['payload_GBs']=400
        self.assertIn('NETWORK_RATE',codes('scale_out',x))
    def test_duplex_not_unidirectional_payload(self):
        x=self.net();x['direction']='bidirectional_aggregate'
        self.assertIn('NETWORK_DIRECTION',codes('scale_out',x))
    def test_roce_lossy_can_disable_pfc(self):
        x=self.net();x.update(mode='lossy',pfc=False)
        self.assertFalse(codes('scale_out',x))
    def test_roce_lossless_without_pfc_not_certified(self):
        x=self.net();x['pfc']=False
        self.assertIn('ROCE_PFC',codes('scale_out',x))
    def test_ib_requires_partition_and_sm(self):
        x=self.net();x['fabric']='InfiniBand'
        self.assertIn('IB_CONTROL_PLANE',codes('scale_out',x))
    def test_mtu_path_limit(self):
        x=self.net();x['route_mtu']=2048
        self.assertIn('NETWORK_MTU',codes('scale_out',x))
    def test_oversubscription_port_conservation(self):
        x=self.net();x['uplink_ports']=4
        self.assertIn('NETWORK_OVERSUB',codes('scale_out',x))
    def comm(self,lib):return dict(library=lib,version='1.2.3',api_role={'NCCL':'collectives','NVSHMEM':'pgas','NIXL':'transfer'}[lib],backend='ucx',source_refs=['api'],memory_registered=True,submitted=10,completed=9,released=8,remote_ready_after_complete=True,producer_event=True,consumer_wait=True,progress='poll',completion='quiet' if lib=='NVSHMEM' else 'completion')
    def test_three_communication_libraries_are_not_interchangeable(self):
        for lib in ['NCCL','NVSHMEM','NIXL']:
            self.assertFalse(codes('communication',self.comm(lib)))
        x=self.comm('NIXL');x['api_role']='collectives'
        self.assertIn('COMM_ROLE',codes('communication',x))
    def test_fence_does_not_prove_nvshmem_completion(self):
        x=self.comm('NVSHMEM');x['completion']='fence'
        self.assertIn('COMM_COMPLETION',codes('communication',x))
    def test_enqueue_does_not_prove_remote_readiness(self):
        x=self.comm('NIXL');x['remote_ready_after_complete']=False
        self.assertIn('COMM_REMOTE_READY',codes('communication',x))
    def test_memory_not_released_before_completion(self):
        x=self.comm('NIXL');x['released']=10
        self.assertIn('COMM_LIFETIME',codes('communication',x))
    def training(self): return dict(world_size=8,rank_coords=[{'rank':i,'dp':i//4,'tp':i%4,'pp':0} for i in range(8)],mesh_axes=['dp','tp','pp'],microbatch=2,accumulation=4,dp_size=2,global_batch=16,ep_group=[0,1,2,3],optimizer='Muon',optimizer_bytes={'weights':2,'grads':2,'master':4,'state':8},optimizer_total=16,schedule=[{'id':'gather','after':[]},{'id':'forward','after':['gather']},{'id':'backward','after':['forward']},{'id':'update','after':['backward']}])
    def test_training_batch_formula(self):
        x=self.training();self.assertFalse(codes('training',x));x['global_batch']=32
        self.assertIn('TRAIN_BATCH',codes('training',x))
    def test_overlap_groups_not_multiplied_as_extra_world_dimension(self):
        x=self.training();x['rank_coords'][1]=dict(x['rank_coords'][0])
        self.assertIn('TRAIN_RANK_MESH',codes('training',x))
    def test_schedule_dependency_cycle(self):
        x=self.training();x['schedule'][0]['after']=['update']
        self.assertIn('TRAIN_SCHEDULE',codes('training',x))
    def serving(self): return dict(context=1024,prefix=512,layer_states=[{'layer':0,'boundary':512,'namespace':'r1','dtype':'fp32'},{'layer':1,'boundary':512,'namespace':'r1','dtype':'bf16'}],required_layers=[0,1],namespace='r1',private_mutable_state=True,handoff={'registered':0,'producer_done':1,'submitted':2,'completed':3,'consumer_start':4,'released':5},weights_replicas=2,placement='disaggregated',estimated_fit=False,memory_components={'weights':100,'mla':10,'kda':5,'conv':2},logical_total=117)
    def test_serving_state_consistent(self):self.assertFalse(codes('serving',self.serving()))
    def test_prefix_all_layers_required(self):
        x=self.serving();x['layer_states'].pop()
        self.assertIn('SERVING_STATE_COMPLETE',codes('serving',x))
    def test_prefix_same_boundary(self):
        x=self.serving();x['layer_states'][1]['boundary']=256
        self.assertIn('SERVING_PREFIX',codes('serving',x))
    def test_request_namespace_isolation(self):
        x=self.serving();x['layer_states'][0]['namespace']='other'
        self.assertIn('SERVING_ISOLATION',codes('serving',x))
    def test_handoff_completion_before_consumer(self):
        x=self.serving();x['handoff']['consumer_start']=2
        self.assertIn('SERVING_HANDOFF',codes('serving',x))
    def test_two_weight_replicas_in_split_serving(self):
        x=self.serving();x['weights_replicas']=1
        self.assertIn('SERVING_REPLICAS',codes('serving',x))
    def numeric(self):return dict(format='mxfp4',numel=64,weight_bits=4,scale_bytes=1,group_size=32,payload_bytes=34,accumulator='fp32',reference=[1.0,0.0],actual=[1.000001,0.0],atol=1e-5,rtol=1e-5,tolerance_policy={'atol_max':1e-4,'rtol_max':1e-4},finite_required=True)
    def test_mxfp4_includes_scale_bytes(self):self.assertFalse(codes('numerics',self.numeric()))
    def test_scales_not_free(self):
        x=self.numeric();x['payload_bytes']=32
        self.assertIn('NUMERICS_STORAGE',codes('numerics',x))
    def test_partial_groups_round_up(self):
        x=self.numeric();x['numel']=33;x['payload_bytes']=19
        self.assertFalse(codes('numerics',x))
    def test_nan_never_passes(self):
        x=self.numeric();x['actual'][1]=float('nan')
        self.assertIn('NUMERICS_FINITE',codes('numerics',x))
    def test_cannot_relax_tolerance_to_pass(self):
        x=self.numeric();x['atol']=1e9
        self.assertIn('NUMERICS_TOLERANCE',codes('numerics',x))
    def storage(self): return dict(checkpoint_id='cp1',expected_shards=['0','1'],shards=[{'id':'0','sha256':'a'*64,'size':10},{'id':'1','sha256':'b'*64,'size':12}],state_keys=['weights','optimizer','rng','sampler','step'],commit_published=True,persisted_shards=['0','1'],manifest_total=22,tokenizer_revision='c'*40,dataset_revision='d'*40)
    def test_checkpoint_complete_before_commit(self):
        x=self.storage();self.assertFalse(codes('storage',x));x['persisted_shards']=['0']
        self.assertIn('STORAGE_ATOMICITY',codes('storage',x))
    def test_checkpoint_requires_rng_and_sampler(self):
        x=self.storage();x['state_keys'].remove('rng')
        self.assertIn('STORAGE_RESTART',codes('storage',x))
    def perf(self):return dict(kind='measured',metric='decode_tokens_per_second',samples=[10,11,12],warmup=2,iterations=3,unit='tokens/s',raw_artifact='trace.json',raw_sha256='a'*64,environment={'gpu':'H100-SXM','driver':'x','cuda':'y','framework_commit':'a'*40,'clocks':'default'},summary=11,statistic='mean',timeouts=0,failures=0,request_count=3)
    def test_measurement_requires_trace(self):
        x=self.perf();self.assertFalse(codes('measurement',x));x.pop('raw_sha256')
        self.assertIn('MEASURE_TRACE',codes('measurement',x))
    def test_failed_requests_not_silently_removed(self):
        x=self.perf();x['failures']=1
        self.assertIn('MEASURE_DENOMINATOR',codes('measurement',x))
    def test_payload_over_rate_cannot_be_measured_latency(self):
        x=self.perf();x['method']='payload_over_rate'
        self.assertIn('MEASURE_ESTIMATE',codes('measurement',x))
    def test_visual_label_requires_claim(self):
        x=dict(claim_ids=['x'],nodes=[{'id':'n','claims':[],'bounds':[0,0,10,10],'text_bounds':[1,1,8,8]}],edges=[],canvas=[100,100])
        self.assertIn('VISUAL_PROVENANCE',codes('visualization',x))
    def test_visual_overflow(self):
        x=dict(claim_ids=['x'],nodes=[{'id':'n','claims':['x'],'bounds':[0,0,10,10],'text_bounds':[1,1,12,8]}],edges=[],canvas=[100,100])
        self.assertIn('VISUAL_OVERFLOW',codes('visualization',x))
    def test_release_no_stale_artifact(self):
        x=dict(source_sha256='a'*64,built_from_sha256='b'*64,artifact_sha256='c'*64,repeat_build_sha256='c'*64,external_requests=0,geometry_defects=0,runtime_errors=0,commands={'node':0,'browser':0},source_lock_sha256='d'*64,receipt_lock_sha256='d'*64)
        self.assertIn('RELEASE_STALE',codes('release',x))

if __name__=='__main__':unittest.main()
