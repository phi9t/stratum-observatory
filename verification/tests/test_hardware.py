"""The hardware record: a versioned model whose attested half is stable across probes.

The property every downstream binding rests on is that two honest probes of an unchanged
machine agree on `hardware_sha256`. These tests prove it in both directions from recorded
fixtures, never from live hardware: transient-only differences must not move the digest, and
every attested leaf must.
"""
import importlib.util,json,unittest
from pathlib import Path

from verification import hardware
from verification.evidence import canonical,digest,loads
from verification.schema import validate as shape_validate

ROOT=Path(__file__).resolve().parents[2]
FIXTURES=ROOT/'verification/tests/fixtures/hardware'

def _probe():
 spec=importlib.util.spec_from_file_location('probe_system',ROOT/'tools/probe-system.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
probe=_probe()


def leaves(value,path='$'):
 """Every scalar leaf in a record, as (path, container, key) so a test can mutate one."""
 if isinstance(value,dict):
  for k,v in value.items():
   if isinstance(v,(dict,list)):yield from leaves(v,f'{path}.{k}')
   else:yield f'{path}.{k}',value,k
 elif isinstance(value,list):
  for i,v in enumerate(value):
   if isinstance(v,(dict,list)):yield from leaves(v,f'{path}[{i}]')
   else:yield f'{path}[{i}]',value,i


class RecordShapeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.schema=loads((ROOT/'verification/schemas/hardware.schema.json').read_text())
  cls.baseline=probe.build_record(FIXTURES/'probe-baseline')

 def test_the_schema_is_versioned_and_declares_both_halves(self):
  self.assertEqual(self.schema['$id'],'stratum-hardware/v1')
  self.assertIn('attested',self.schema['properties'])
  self.assertIn('transient',self.schema['properties'])

 def test_the_schema_uses_only_keywords_this_validator_implements(self):
  # A constraint the house validator silently ignores is worse than one never written.
  self.assertEqual(shape_validate({'schema':'stratum-hardware/v1'},self.schema)[:0],[])
  found=[]
  def walk(node,path='$'):
   if isinstance(node,dict):
    for k,v in node.items():
     if k in shape_validate.__globals__['UNSUPPORTED']:found.append(f'{path}.{k}')
     walk(v,f'{path}.{k}')
   elif isinstance(node,list):
    for i,v in enumerate(node):walk(v,f'{path}[{i}]')
  walk(self.schema)
  self.assertEqual(found,[],'schema uses keywords verification/schema.py does not implement')

 def test_every_pattern_in_the_schema_is_anchored(self):
  found=[]
  def walk(node,path='$'):
   if isinstance(node,dict):
    if isinstance(node.get('pattern'),str) and not (node['pattern'].startswith('^') and node['pattern'].endswith('$')):
     found.append(path)
    for k,v in node.items():walk(v,f'{path}.{k}')
   elif isinstance(node,list):
    for i,v in enumerate(node):walk(v,f'{path}[{i}]')
  walk(self.schema)
  self.assertEqual(found,[])

 def test_a_recorded_capture_conforms_to_the_schema(self):
  errors=shape_validate(self.baseline,self.schema)
  self.assertEqual(errors,[],'\n'.join(errors[:8]))

 def test_every_fixture_capture_conforms_to_the_schema(self):
  for d in sorted(FIXTURES.iterdir()):
   if not d.is_dir():continue
   errors=shape_validate(probe.build_record(d),self.schema)
   self.assertEqual(errors,[],f'{d.name}: '+'\n'.join(errors[:8]))

 def test_the_record_carries_the_comprehensive_field_set(self):
  a=self.baseline['attested']
  gpu=a['gpus'][0]
  self.assertEqual(gpu['clocks_max']['sm_mhz'],1980)
  self.assertEqual(gpu['clocks_max']['memory_mhz'],2619)
  self.assertEqual(gpu['memory']['total_mib'],81559)
  self.assertEqual(gpu['memory']['reserved_mib'],573)
  self.assertEqual(gpu['ecc_mode']['current'],'Enabled')
  self.assertEqual(gpu['persistence_mode'],'Enabled')
  self.assertEqual(gpu['power']['max_limit_w'],700.0)
  self.assertEqual(gpu['pci']['link_gen_max'],5)
  self.assertEqual(gpu['pci']['link_width_max'],16)
  self.assertEqual(gpu['vbios_version'],'96.00.74.00.11')
  self.assertEqual(gpu['serial'],'1650000000001')
  self.assertEqual(gpu['architecture'],'Hopper')
  self.assertEqual(gpu['mig_mode']['current'],'Enabled')
  profiles={p['name']:p for p in a['mig_profiles']['gpus'][0]['profiles']}
  self.assertEqual(profiles['MIG 1g.10gb']['instances_total'],7)
  self.assertEqual(profiles['MIG 1g.10gb']['sm_count'],16)
  self.assertEqual(profiles['MIG 7g.80gb']['memory_gib'],79.25)
  self.assertTrue(a['nvlink']['gpus'][0]['links'][0]['capabilities']['p2p_supported'])
  self.assertEqual(a['nvlink']['gpus'][0]['link_count'],2)
  self.assertEqual(a['cpu']['model'],'Intel(R) Xeon(R) Platinum 8480C')
  self.assertEqual(a['cpu']['logical_cpus'],16)
  self.assertEqual(a['cpu']['sockets'],2)
  self.assertEqual(a['numa']['node_count'],2)
  self.assertEqual(a['numa']['nodes'][0]['cpus'],'0-3,8-11')
  self.assertEqual(a['numa']['distances'],[[10,21],[21,10]])
  self.assertEqual(a['kernel']['release'],'5.15.120.bsk.6-amd64')
  self.assertEqual(a['rdma']['devices'][0]['hca'],'mlx5_0')
  self.assertEqual(a['rdma']['devices'][0]['fw_ver'],'28.39.2500')
  self.assertEqual(a['rdma']['devices'][0]['ports'][0]['max_mtu'],'4096 (5)')
  self.assertEqual(a['rdma']['devices'][0]['netdev'],'eth0')
  self.assertEqual(a['ofed']['version'],'MLNX_OFED_LINUX-23.10-1.1.9.0:')
  self.assertEqual(a['driver']['version'],'535.161.08')
  self.assertEqual(a['topology']['rows'][0]['peers']['GPU1'],'NV18')
  self.assertEqual(a['topology']['nic_legend']['NIC0'],'mlx5_0')

 def test_a_memory_only_numa_node_is_recorded_with_an_empty_cpu_list(self):
  # A Grace-Blackwell host exposes device memory as CPU-less NUMA nodes: numactl prints
  # `node 2 cpus:` with nothing after it. An empty CPU list is a fact the tool reported,
  # not a value it withheld, so the model has to be able to carry it.
  _,nodes,_=probe.parse_numactl('available: 3 nodes (0-2)\nnode 0 cpus: 0 1\n'
                                'node 0 size: 10 MB\nnode 2 cpus:\nnode 2 size: 0 MB\n')
  self.assertEqual(nodes[2]['cpus'],'')
  record=json.loads(json.dumps(self.baseline))
  record['attested']['numa']['nodes'].append({'node':9,'cpus':'','size_mb':0})
  self.assertEqual(shape_validate(record,self.schema),[])

 def test_telemetry_is_recorded_but_kept_out_of_the_attested_half(self):
  t=self.baseline['transient']
  self.assertEqual(t['gpus'][0]['temperature_c'],30)
  self.assertEqual(t['gpus'][0]['power_draw_w'],71.0)
  self.assertEqual(t['gpus'][0]['utilization_gpu_pct'],0)
  self.assertEqual(t['gpus'][0]['memory_used_mib'],0)
  self.assertEqual(t['gpus'][0]['pstate'],'P0')
  attested_keys={p.rsplit('.',1)[-1] for p,_,_ in leaves(self.baseline['attested'])}
  for telemetry in ('temperature_c','power_draw_w','utilization_gpu_pct','utilization_memory_pct',
                    'memory_used_mib','memory_free_mib','fan_speed_pct','pstate','captured_at',
                    'clocks_sm_mhz','clocks_memory_mhz','link_gen_current','link_width_current',
                    'ecc_errors_corrected_volatile','remapped_rows_correctable','state',
                    'physical_state','active_mtu','active_speed','port_lid','instances_free',
                    'compute_process_count','free_mib','current_mhz'):
   self.assertNotIn(telemetry,attested_keys,f'{telemetry} is telemetry and must not be attested')


class DigestStabilityTests(unittest.TestCase):
 """The property the whole design rests on."""
 @classmethod
 def setUpClass(cls):
  cls.a=probe.build_record(FIXTURES/'probe-baseline')
  cls.b=probe.build_record(FIXTURES/'probe-reprobe')
  cls.c=probe.build_record(FIXTURES/'probe-driver-bump')

 def test_the_fixtures_really_do_differ_in_their_transient_half(self):
  # Guards against a vacuous stability proof: if the two captures were identical the
  # digest test below would pass while measuring nothing.
  self.assertNotEqual(canonical(self.a['transient']),canonical(self.b['transient']))
  self.assertEqual(self.b['transient']['gpus'][0]['utilization_gpu_pct'],96)
  self.assertEqual(self.b['transient']['rdma']['ports'][0]['state'],'DOWN')
  self.assertEqual(self.b['transient']['nvlink']['gpus'][0]['links'][1]['state'],'inactive')

 def test_a_reprobe_of_unchanged_hardware_keeps_the_same_digest(self):
  self.assertEqual(self.a['hardware_sha256'],self.b['hardware_sha256'])
  self.assertEqual(canonical(self.a['attested']),canonical(self.b['attested']))

 def test_a_changed_attested_field_moves_the_digest(self):
  self.assertNotEqual(self.a['hardware_sha256'],self.c['hardware_sha256'])
  self.assertEqual(self.c['attested']['driver']['version'],'550.144.03')

 def test_every_attested_leaf_moves_the_digest(self):
  # "Sensitive to every attested field" is a claim about all of them, not a sample.
  base=hardware.hardware_sha256(self.a)
  paths=list(leaves(self.a['attested']))
  self.assertGreater(len(paths),80,'the attested half is not comprehensive')
  for path,container,key in paths:
   original=container[key]
   container[key]='__mutated__' if not isinstance(original,str) else original+'__mutated__'
   try:
    self.assertNotEqual(hardware.hardware_sha256(self.a),base,f'{path} does not reach the digest')
   finally:
    container[key]=original
  self.assertEqual(hardware.hardware_sha256(self.a),base)

 def test_no_transient_leaf_moves_the_digest(self):
  base=hardware.hardware_sha256(self.a)
  for path,container,key in leaves(self.a['transient']):
   original=container[key]
   container[key]='__mutated__' if not isinstance(original,str) else original+'__mutated__'
   try:
    self.assertEqual(hardware.hardware_sha256(self.a),base,f'{path} reaches the digest')
   finally:
    container[key]=original

 def test_the_probe_digest_agrees_with_the_house_digest_function(self):
  # The probe runs on a machine that has no copy of this repository, so it carries its own
  # canonicalisation. This is the cross-check that the two never drift apart.
  for record in (self.a,self.b,self.c):
   self.assertEqual(record['hardware_sha256'],digest(record['attested']))
   self.assertEqual(record['hardware_sha256'],hardware.hardware_sha256(record))


class AbsenceTests(unittest.TestCase):
 """A command that did not run is a recorded absence, never a fabricated value."""
 @classmethod
 def setUpClass(cls):
  cls.baseline=probe.build_record(FIXTURES/'probe-baseline')
  cls.partial=probe.build_record(FIXTURES/'probe-partial')

 def test_an_absent_utility_is_recorded_not_run_with_its_reason(self):
  by_id={c['id']:c for c in self.partial['commands']}
  self.assertEqual(by_id['numa']['status'],'NOT_RUN')
  self.assertEqual(by_id['numa']['reason'],'utility not installed')
  self.assertEqual(by_id['ofed']['status'],'NOT_RUN')

 def test_fields_from_an_absent_utility_are_absent_not_zero(self):
  self.assertNotIn('numa',self.partial['attested'])
  self.assertNotIn('ofed',self.partial['attested'])
  self.assertIn('numa',self.baseline['attested'])

 def test_an_absent_field_is_named_with_the_reason_it_is_missing(self):
  missing={u['field']:u for u in self.partial['unavailable']}
  self.assertIn('attested.numa',missing)
  self.assertEqual(missing['attested.numa']['reason'],'utility not installed')
  self.assertEqual(missing['attested.numa']['command'],'numa')

 def test_a_field_the_tool_never_reported_is_named_rather_than_invented(self):
  # No shipped nvidia-smi reports a multiprocessor count, so the field must appear as a
  # recorded absence. Inventing one from a datasheet is the defect this test exists for.
  missing={u['field'] for u in self.baseline['unavailable']}
  self.assertIn('attested.gpus[].multiprocessor_count',missing)
  for gpu in self.baseline['attested']['gpus']:
   self.assertNotIn('multiprocessor_count',gpu)

 def test_a_multiprocessor_count_is_read_when_a_driver_does_report_one(self):
  parsed=probe.parse_smi_query('GPU 00000000:0F:00.0\n    Number of Multiprocessors             : 132\n')
  self.assertEqual(parsed['GPU 00000000:0F:00.0']['Number of Multiprocessors'],'132')
  self.assertEqual(probe.sm_count(parsed['GPU 00000000:0F:00.0']),132)

 def test_an_absent_capture_does_not_silently_shrink_the_digest_to_the_same_value(self):
  self.assertNotEqual(self.partial['hardware_sha256'],self.baseline['hardware_sha256'])


class RejectedFieldTests(unittest.TestCase):
 """A --query-gpu name this driver does not implement.

 nvidia-smi refuses the whole query over one unknown field name, so a single wrong name would
 otherwise cost every GPU attribute in the capture. The probe drops exactly the names the
 driver refused, keeps the refusal itself as a checksummed artifact, and names each dropped
 field in `unavailable` with the driver's own words. Nothing is guessed in its place.
 """
 @classmethod
 def setUpClass(cls):
  cls.baseline=probe.build_record(FIXTURES/'probe-baseline')
  cls.reduced=probe.build_record(FIXTURES/'probe-rejected-field')

 def test_the_driver_s_refusal_names_the_field_it_refused(self):
  self.assertEqual(probe.rejected_fields('Field "pci.link.gen.max" is not a valid field to query.\n'),
                   ['pci.link.gen.max'])
  self.assertEqual(probe.rejected_fields('no GPUs were found\n'),[])

 def test_a_refused_field_is_dropped_and_the_rest_of_the_query_survives(self):
  argv=['nvidia-smi','--query-gpu=index,pci.link.gen.max,name','--format=csv,noheader,nounits']
  self.assertEqual(probe.query_fields(argv),['index','pci.link.gen.max','name'])
  self.assertEqual(probe.without_fields(argv,['pci.link.gen.max']),
                   ['nvidia-smi','--query-gpu=index,name','--format=csv,noheader,nounits'])
  self.assertIsNone(probe.without_fields(argv,['index','pci.link.gen.max','name']))

 def test_the_surviving_fields_are_still_read_out_of_the_right_columns(self):
  # The CSV is positional: dropping a field shifts every column after it. Read the effective
  # field list back out of the ledger's argv or every remaining value is silently wrong.
  gpu=self.reduced['attested']['gpus'][0]
  self.assertEqual(gpu['memory']['total_mib'],81559)
  self.assertEqual(gpu['memory']['reserved_mib'],573)
  self.assertEqual(gpu['vbios_version'],'96.00.74.00.11')
  self.assertEqual(gpu['power']['max_limit_w'],700.0)
  self.assertEqual(gpu['ecc_mode']['current'],'Enabled')
  self.assertEqual(gpu['pci']['sub_device_id'],'0x16C110DE')
  self.assertEqual(self.reduced['transient']['gpus'][0]['memory_free_mib'],80986)
  self.assertEqual(self.reduced['transient']['gpus'][0]['clocks_sm_mhz'],345)

 def test_a_refused_field_is_absent_from_the_record_never_defaulted(self):
  gpu=self.reduced['attested']['gpus'][0]
  self.assertNotIn('link_gen_max',gpu['pci'])
  self.assertNotIn('link_width_max',gpu['pci'])
  self.assertNotIn('clocks_memory_mhz',self.reduced['transient']['gpus'][0])
  self.assertNotIn('link_gen_current',self.reduced['transient']['gpus'][0])
  # and the baseline, where the driver accepted them, does carry them
  self.assertEqual(self.baseline['attested']['gpus'][0]['pci']['link_gen_max'],5)

 def test_a_refused_field_is_named_in_unavailable_with_the_driver_s_reason(self):
  missing={u['field']:u for u in self.reduced['unavailable']}
  for field,reason in (('attested.gpus[].pci.link_gen_max','pci.link.gen.max'),
                       ('attested.gpus[].pci.link_width_max','pci.link.width.max'),
                       ('transient.gpus[].clocks_memory_mhz','clocks.memory'),
                       ('transient.gpus[].link_gen_current','pci.link.gen.current'),
                       ('transient.gpus[].link_width_current','pci.link.width.current')):
   self.assertIn(field,missing)
   self.assertEqual(missing[field]['reason'],f'Field "{reason}" is not a valid field to query.')
   self.assertTrue(missing[field]['command'].endswith(tuple('12345')),missing[field]['command'])

 def test_the_refusal_itself_is_kept_as_a_checksummed_artifact(self):
  ledger={c['id']:c for c in self.reduced['commands']}
  self.assertEqual(ledger['gpu_attested_rejected_1']['status'],'NOT_RUN')
  self.assertEqual(ledger['gpu_attested_rejected_1']['exit'],2)
  self.assertIn('sha256',ledger['gpu_attested_rejected_1'])
  self.assertIn('is not a valid field to query',ledger['gpu_attested_rejected_1']['reason'])

 def test_every_queried_gpu_field_names_the_record_field_it_feeds(self):
  # The map is what lets a refusal be reported against a record field rather than a CSV
  # column. It must not drift from the query strings on either side.
  self.assertEqual(sorted(probe.ATTESTED_GPU_FIELDS.split(',')),sorted(probe.ATTESTED_GPU_PATHS))
  self.assertEqual(sorted(probe.TRANSIENT_GPU_FIELDS.split(',')),sorted(probe.TRANSIENT_GPU_PATHS))

 def test_a_reduced_capture_does_not_collide_with_the_full_one(self):
  self.assertNotEqual(self.reduced['hardware_sha256'],self.baseline['hardware_sha256'])


class FixtureContainmentTests(unittest.TestCase):
 """A fixture must never be mistakable for production evidence."""
 def test_every_fixture_capture_is_marked_in_its_manifest_and_its_record(self):
  for d in sorted(FIXTURES.iterdir()):
   if not d.is_dir():continue
   self.assertIs(loads((d/'manifest.json').read_text()).get('fixture'),True,f'{d.name} manifest')
   self.assertIs(probe.build_record(d).get('fixture'),True,f'{d.name} record')

 def test_loading_a_fixture_as_production_evidence_is_refused(self):
  record=probe.build_record(FIXTURES/'probe-baseline')
  with self.assertRaises(ValueError):hardware.check(record)
  self.assertEqual(hardware.check(record,allow_fixtures=True),record)

 def test_no_captured_evidence_record_is_fixture_marked(self):
  for path in sorted((ROOT/'evidence').rglob('hardware.json')):
   record=loads(path.read_text())
   self.assertNotIn('fixture',record,f'{path} is marked as a fixture but sits in evidence/')
   self.assertEqual(hardware.check(record),record)

 def test_a_record_whose_digest_does_not_match_its_attested_half_is_refused(self):
  record=probe.build_record(FIXTURES/'probe-baseline')
  record['attested']['kernel']['release']='6.6.0-not-what-was-probed'
  with self.assertRaises(ValueError):hardware.check(record,allow_fixtures=True)


class ReplayIntegrityTests(unittest.TestCase):
 def test_replaying_a_capture_whose_artifact_drifted_is_refused(self):
  import shutil,tempfile
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp)/'capture'
   shutil.copytree(FIXTURES/'probe-baseline',d)
   (d/'kernel.txt').write_bytes(b'6.6.0-tampered\n\nSTDERR:\n')
   with self.assertRaises(ValueError):probe.build_record(d)

 def test_replay_is_deterministic(self):
  self.assertEqual(canonical(probe.build_record(FIXTURES/'probe-baseline')),
                   canonical(probe.build_record(FIXTURES/'probe-baseline')))


class ReadOnlyCommandTests(unittest.TestCase):
 """The probe touches other people's machines. Every command must be inspectably read-only."""
 BINARIES={'nvidia-smi','nvcc','lscpu','numactl','uname','ibv_devinfo','rdma','ofed_info','lspci'}
 MUTATING={'-pm','--persistence-mode','-e','--ecc-config','-p','--reset-ecc-errors','-ac',
           '-rac','-pl','--power-limit','-lgc','-rgc','--gpu-reset','-r','-cc','-caa','-gom',
           '-am','-cgi','-dgi','-cci','-dci','--auto-boost-default','-i'}

 def test_every_command_is_a_known_read_only_utility(self):
  for id,argv in probe.COMMANDS.items():
   self.assertIn(argv[0],self.BINARIES,f'{id} runs an unvetted binary')
   self.assertNotIn('sudo',argv,f'{id} escalates')
   # argv[0] is already pinned to the allowlist above, so a bare `-c` is a utility flag
   # (`nvidia-smi nvlink -c` lists link capabilities), never a shell's command string.
   for shellish in ('sh','bash','&&','|',';','>'):
    self.assertNotIn(shellish,argv,f'{id} looks like shell evaluation')

 def test_no_nvidia_smi_command_can_change_the_device(self):
  for id,argv in probe.COMMANDS.items():
   if argv[0]!='nvidia-smi':continue
   for arg in argv[1:]:
    self.assertNotIn(arg,self.MUTATING,f'{id} passes the mutating flag {arg}')
   for verb in ('create','destroy','set','reset'):
    self.assertNotIn(verb,argv,f'{id} names a mutating verb')

 def test_the_existing_obligation_fields_are_still_captured(self):
  # The receipts in verification/manifests/runtime.json read these out of gpu.txt. Losing
  # any of them would retire real probed evidence.
  self.assertEqual(probe.COMMANDS['gpu'],
   ['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,pci.device_id,driver_version,memory.total,'
    'compute_cap,power.limit,mig.mode.current','--format=csv,noheader,nounits'])


if __name__=='__main__':unittest.main()
