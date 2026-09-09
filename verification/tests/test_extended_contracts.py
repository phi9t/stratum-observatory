import copy,unittest
from verification.domains.contracts import validate

def errors(domain,x):return {f.code for f in validate(domain,x) if f.status!='PASS'}
class ExtendedContractTests(unittest.TestCase):
    def gpu(self):
        return dict(sku='H100-SXM',kind='gpu',architecture='hopper',compute_capability='9.0',gpu_count=1,cpu_count=0,memory={'value':80,'unit':'GB','scope':'per_gpu','kind':'nominal'},kernel={'threads_per_block':256,'registers_per_thread':64,'shared_bytes':65536,'cluster_size':8,'nonportable_optin':False,'limits':{'threads_per_block':1024,'registers_per_thread':255,'registers_per_sm':65536,'shared_bytes_per_block':232448,'portable_cluster':8,'nonportable_cluster':16},'features':['tma'],'barrier_before_consume':True,'native_math':'bf16'})
    def test_gpu_resource_contract_consistent(self):self.assertFalse(errors('gpu',self.gpu()))
    def test_shared_memory_limit_is_per_block_not_per_sm(self):
        x=self.gpu();x['kernel']['shared_bytes']=999999
        self.assertIn('GPU_KERNEL_RESOURCES',errors('gpu',x))
    def test_tensor_transfer_needs_consumer_barrier(self):
        x=self.gpu();x['kernel']['barrier_before_consume']=False
        self.assertIn('GPU_ASYNC_ORDER',errors('gpu',x))
    def test_nonportable_cluster_needs_optin(self):
        x=self.gpu();x['kernel']['cluster_size']=16
        self.assertIn('GPU_CLUSTER_OPTIN',errors('gpu',x))
    def test_fp4_emulation_not_native_hopper_instruction(self):
        x=self.gpu();x['kernel']['native_math']='fp4'
        self.assertIn('GPU_NATIVE_FORMAT',errors('gpu',x))
    def test_compiler_arch_specific_suffix_not_cross_family(self):
        x=self.gpu();x['kernel_target']='sm_100a'
        self.assertIn('GPU_KERNEL_TARGET',errors('gpu',x))
    def test_peak_without_dense_sparse_label_fails(self):
        x=self.gpu();x['rates']={'compute':{'unit':'TFLOP/s','scope':'per_gpu','direction':'not_applicable','value':989,'precision':'bf16'}}
        self.assertIn('GPU_PEAK_MODE',errors('gpu',x))
    def test_wrong_nvswitch_count_is_not_hopper_template_copy(self):
        x=dict(system='DGX-B200',compute_trays=1,gpus_per_tray=8,cpus_per_tray=2,total_gpus=8,total_cpus=2,scale_up='NVLink',scale_out='InfiniBand',switch_count=4,switch_count_scope='asic')
        self.assertIn('TOPOLOGY_SWITCHES',errors('scale_up',x))
    def network(self):return dict(fabric='InfiniBand',link_gbps=400,direction='one_way',scope='per_port',payload_GBs=50,mtu=4096,route_mtu=4096,ports=8,uplink_ports=8,uplink_gbps=400,oversubscription=1,subnet_manager=True,pkey=0xffff,sl=0)
    def test_valid_ib_control_plane(self):self.assertFalse(errors('scale_out',self.network()))
    def test_bad_service_level(self):
        x=self.network();x['sl']=256
        self.assertIn('IB_SERVICE_LEVEL',errors('scale_out',x))
    def test_zero_partition_rejected(self):
        x=self.network();x['pkey']=0x8000
        self.assertIn('IB_PARTITION',errors('scale_out',x))
    def test_bad_reliability_timeout_not_hidden(self):
        x=self.network();x['health']={'retries':0,'uncorrectable_errors':1,'timeout_ms':5000,'abort_completed':False,'fallback':'none','degraded':False}
        self.assertIn('NETWORK_HEALTH',errors('scale_out',x))
