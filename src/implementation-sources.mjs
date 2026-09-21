import { MODELS, SOURCES } from './models.mjs';

// Scope is deliberately independent of catalog role: Transformers is a framework,
// even when a historical catalog entry calls it publisher_code.
const implementationMap = {
 ds2: { publisher: ['ds2-code'], transformers: [], serving: ['ds-runtime'] },
 ds3: { publisher: ['ds3-code'], transformers: [], serving: ['ds-runtime'] },
 q38: { publisher: [], transformers: ['q3-code'], serving: ['q3-runtime', 'q3-vllm'] },
 q330: { publisher: [], transformers: ['q3-moe-code'], serving: ['q3-moe-runtime', 'q3-moe-vllm'] },
 q3235: { publisher: [], transformers: ['q3-moe-code'], serving: ['q3-moe-runtime', 'q3-moe-vllm'] },
 q3827: { publisher: [], transformers: ['q35-code'], serving: ['q35-runtime', 'q35-vllm'] },
 q38flash: { publisher: [], transformers: ['q4-code'], serving: ['q4-runtime'] },
 kimi3: { publisher: ['k3-code'], transformers: [], serving: ['k3-runtime'] },
 dsv41: { publisher: ['dsv41-code', 'dsv41-engram', 'dsv41-kernel', 'dsv41-vision'], transformers: [], serving: ['dsv41-sglang-config', 'dsv41-sglang', 'dsv41-sglang-cache'] }
};
// Qualified symbols inspected in the catalog's SHA-256 checked source cache.
const symbols = {
 'ds2-code': ['DeepseekV2Attention', 'DeepseekV2MoE', 'MoEGate'],
 'ds3-code': ['DeepseekV3Attention', 'DeepseekV3MoE', 'MoEGate'],
 'ds-runtime': ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepseekV2AttentionMLA'],
 'q3-code': ['Qwen3Attention', 'Qwen3ForCausalLM'],
 'q3-moe-code': ['Qwen3MoeAttention', 'Qwen3MoeSparseMoeBlock'],
 'q3-runtime': ['Qwen3Attention', 'Qwen3ForCausalLM'],
 'q3-vllm': ['Qwen3Attention', 'Qwen3ForCausalLM'],
 'q3-moe-runtime': ['Qwen3MoeSparseMoeBlock', 'Qwen3MoeForCausalLM'],
 'q3-moe-vllm': ['Qwen3MoeSparseMoeBlock', 'Qwen3MoeForCausalLM'],
 'q35-code': ['Qwen3_5GatedDeltaNet', 'Qwen3_5Attention', 'Qwen3_5ForConditionalGeneration'],
 'q35-runtime': ['Qwen3_5GatedDeltaNet', 'Qwen3_5ForConditionalGeneration'],
 'q35-vllm': ['Qwen3_5ForConditionalGeneration'],
 'q4-code': ['Qwen4ExpTextAttention', 'Qwen4ExpTextPLELayer', 'Qwen4ExpTextGatedResidual'],
 'q4-runtime': ['Qwen4ExpQSAAttention'],
 'k3-code': ['KimiDeltaAttention', 'KimiMLAAttention', 'KimiSparseMoeBlock', '_apply_attn_res'],
 'k3-runtime': ['KimiK3LinearForCausalLM', 'KimiK3ForConditionalGeneration'],
 'dsv41-code': ['Attention', 'Compressor', 'Indexer', 'Block.hc_mixes', 'Transformer'],
 'dsv41-engram': ['EngramLayout', 'NgramHashState'],
 'dsv41-kernel': ['sparse_attn', 'act_quant', 'hc_split_sinkhorn'],
 'dsv41-sglang-config': ['normalize_deepseek_v41_config'],
 'dsv41-sglang': ['DeepseekV4ForCausalLM', 'DeepseekV4Model'],
 'dsv41-sglang-cache': ['DeepSeekV4TokenToKVPool._init_compressed_pools'],
 'megatron-ddp': ['DistributedDataParallel.start_grad_sync', 'DistributedDataParallel.finish_grad_sync'],
 'megatron-optimizer': ['DistributedOptimizer'],
 'megatron-parallel-layers': ['ColumnParallelLinear', 'RowParallelLinear', 'VocabParallelEmbedding'],
 'megatron-expert-dispatch': ['MoEAlltoAllTokenDispatcher.token_dispatch', 'MoEAlltoAllTokenDispatcher.token_combine'],
 'megatron-schedules': ['get_forward_backward_func', 'forward_backward_pipelining_without_interleaving'],
 'dsv41-vision': ['ViT', 'Aligner']
};

// Operator IDs are semantic scene identities, not keywords inferred from prose.
const operations = {
 attention: ['attention','attention-gate','attn-sink','q-position','q-rope','k-rope','qk-position','kv-build','inverse-rope','index-proj','index-blocks','index-select','index-compress','csa2-select','csa2-candidates','csa2-union'],
 projection: ['q-proj','k-proj','v-proj','qkv-proj','q-down','q-up','kv-down','kv-up','out-proj','ffn-up','ffn-down','expert-up','expert-down','shared-up','shared-down','latent-down','latent-up','lm-head'],
 embedding: ['embedding','text-input'],
 ffn: ['ffn-activation','expert-activation','shared-activation','router','top-k','latent-combine','routed-shared','shared-gate'],
 dispatch: ['dispatch','combine'],
 residual: ['norm1','norm2','merge-attn','merge-ffn','expand','collapse','incoming','attention-read','attention-mixer','attention-write','ffn-read','ffn-mixer','ffn-write','outgoing','residual-collapse'],
 cache: ['kv-cache','window-kv','compress-kv','state-bundle','state-history','state-recurrent','state-conv','state-indexer','state-window','csa2-window','csa2-full','csa2-owners'],
 forward: ['prefill','decode','overview-block','forward','backward','activation-policy'],
 vision: ['vision-input','patches','vision-norm1','vision-qkv','vision-rope','vision-attention','vision-add1','vision-ffn','vision-add2','vision-merge'],
 engram: ['engram-table','engram-project','engram-gate','engram-write'],
 optimizer: ['optimizer'], ddp: ['gradient-sync','param-gather','backward-gather'],
 schedule: ['microbatch'],
};
const trainingMechanisms = {projection:'megatron-parallel-layers',embedding:'megatron-parallel-layers',dispatch:'megatron-expert-dispatch',optimizer:'megatron-optimizer',ddp:'megatron-ddp',schedule:'megatron-schedules',forward:'megatron-schedules'};
const trainingIds = [...new Set(Object.values(trainingMechanisms))];

// Class names below were inspected in the same pinned cache files as the overview symbols.
const symbolProfiles = {
 'ds2-code': {attention:['DeepseekV2Attention'],ffn:['DeepseekV2MoE','DeepseekV2MLP'],residual:['DeepseekV2DecoderLayer'],forward:['DeepseekV2ForCausalLM']},
 'ds3-code': {attention:['DeepseekV3Attention'],ffn:['DeepseekV3MoE','DeepseekV3MLP'],residual:['DeepseekV3DecoderLayer'],forward:['DeepseekV3ForCausalLM']},
 'q3-code': {attention:['Qwen3Attention'],ffn:['Qwen3MLP'],residual:['Qwen3DecoderLayer'],forward:['Qwen3ForCausalLM']},
 'q3-moe-code': {attention:['Qwen3MoeAttention'],ffn:['Qwen3MoeSparseMoeBlock'],residual:['Qwen3MoeDecoderLayer'],forward:['Qwen3MoeForCausalLM']},
 'q35-code': {attention:['Qwen3_5Attention','Qwen3_5GatedDeltaNet'],ffn:['Qwen3_5MLP'],residual:['Qwen3_5DecoderLayer'],vision:['Qwen3_5VisionModel'],forward:['Qwen3_5ForConditionalGeneration']},
 'q4-code': {attention:['Qwen4ExpTextAttention','Qwen4ExpTextQSAIndexer'],ffn:['Qwen4ExpTextSparseMoeBlock'],residual:['Qwen4ExpTextGatedResidual'],vision:['Qwen4ExpVisionModel'],forward:['Qwen4ExpForConditionalGeneration']},
 'k3-code': {attention:['KimiMLAAttention','KimiDeltaAttention'],ffn:['KimiSparseMoeBlock'],residual:['_apply_attn_res'],forward:['KimiLinearForCausalLM']},
 'ds-runtime': {attention:['DeepseekV2AttentionMLA'],ffn:['DeepseekV2MoE'],residual:['DeepseekV2DecoderLayer']},
 'q3-runtime': {attention:['Qwen3Attention'],residual:['Qwen3DecoderLayer']},
 'q3-vllm': {attention:['Qwen3Attention'],residual:['Qwen3DecoderLayer']},
 'q3-moe-runtime': {attention:['Qwen3MoeAttention'],ffn:['Qwen3MoeSparseMoeBlock'],residual:['Qwen3MoeDecoderLayer']},
 'q3-moe-vllm': {attention:['Qwen3MoeAttention'],ffn:['Qwen3MoeSparseMoeBlock'],residual:['Qwen3MoeDecoderLayer']},
 'k3-runtime': {attention:['KimiK3MLAAttention','KimiK3DeltaAttention'],ffn:['KimiK3MoE'],residual:['KimiK3DecoderLayer']},
};

function operationSymbols(id, topic, operatorId) {
 const mechanism = topic === 'projection' ? /^(ffn|expert|shared|latent)-/.test(operatorId) ? 'ffn' : 'attention' : topic === 'dispatch' ? 'ffn' : topic === 'cache' ? 'attention' : topic;
 if (symbolProfiles[id]?.[mechanism]) return symbolProfiles[id][mechanism];
 if (id === 'dsv41-code') return ({attention:['Attention','Indexer'],projection:['ColumnParallelLinear','RowParallelLinear'],embedding:['ParallelEmbedding'],ffn:['Gate','MoE','Expert'],dispatch:['MoE'],residual:['Block.hc_mixes'],cache:['Attention','Compressor','Indexer'],engram:['Engram'],forward:['Transformer']})[topic] || symbols[id];
 if (id === 'q4-code') return ({attention:['Qwen4ExpTextAttention','Qwen4ExpTextQSAIndexer'],ffn:['Qwen4ExpTextSparseMoeBlock'],dispatch:['Qwen4ExpTextSparseMoeBlock'],residual:['Qwen4ExpTextGatedResidual'],engram:['Qwen4ExpTextPLELayer']})[topic] || symbols[id];
 if (id === 'dsv41-kernel') return topic === 'residual' ? ['hc_split_sinkhorn'] : ['sparse_attn','act_quant'];
 if (id === 'megatron-ddp' && ['param-gather','backward-gather'].includes(operatorId)) return ['DistributedDataParallel.start_param_sync'];
 if (id === 'megatron-parallel-layers') return topic === 'embedding' ? ['VocabParallelEmbedding'] : ['ColumnParallelLinear','RowParallelLinear'];
 return symbols[id];
}

/** Source navigation, never support certification. An omitted nodeInfo gives an overview.
 * For operation filtering pass nodeInfo.operatorId alongside the existing source IDs.
 * Unknown operations get their cited baseline chain, not inferred framework support.
 */
export function implementationReferences(model, view = 'model', nodeInfo = null) {
 const m = MODELS[typeof model === 'string' ? model : model?.id];
 const mapping = m && implementationMap[m.id];
 if (!mapping) return { baseline: [], serving: [], training: [] };
 const selected = new Set(Array.isArray(nodeInfo?.sources) ? nodeInfo.sources : []);
 const overview = nodeInfo === null;
 const topic = Object.keys(operations).find(key => operations[key].includes(nodeInfo?.operatorId));
 const absentMechanism = ['state-recurrent','state-conv'].includes(nodeInfo?.operatorId) && !m.linearHeads || nodeInfo?.operatorId === 'state-indexer' && !m.indexer;
 const record = (sourceId, group, provider, kind, scope, explanation) => {
  const source = SOURCES[sourceId];
  return { sourceId, title: source.title, url: source.url, revision: source.revision,
   symbols: [...(operationSymbols(sourceId,topic,nodeInfo?.operatorId) || ['architectures', 'model_type'])], provider, kind, scope,
   status: 'reference-only', executed: false, selected: selected.has(sourceId),
   relevant: group === (view === 'serving' || view === 'training' ? view : 'baseline'), explanation };
 };
 const configId = m.sources.find(id => SOURCES[id]?.role === 'publisher_config');
 const baseline = [record(configId, 'baseline', 'Hugging Face publisher', 'configuration', 'exact-checkpoint',
  `Pinned publisher configuration for ${m.repo}. Checkpoint identity and architecture declaration; this file is not executable model code.`)];
 const baselineApplies = id => {
  if (overview) return true;
  if (absentMechanism) return false;
  if (!topic) return !nodeInfo.operatorId && selected.has(id);
  if (['optimizer','ddp','schedule'].includes(topic)) return false;
  if (id === 'dsv41-engram') return topic === 'engram';
  if (id === 'dsv41-vision') return topic === 'vision';
  if (id === 'dsv41-kernel') return ['attention','residual'].includes(topic);
  if (id === 'dsv41-code' && topic === 'vision') return false;
  return true;
 };
 for (const id of mapping.publisher.filter(baselineApplies)) baseline.push(record(id, 'baseline', 'Hugging Face publisher', 'implementation', 'exact-checkpoint',
  'Publisher code from the same pinned checkpoint repository. Source inspection does not establish checkpoint parity or successful execution.'));
 for (const id of mapping.transformers.filter(baselineApplies)) baseline.push(record(id, 'baseline', 'Transformers', 'implementation', 'architecture-family',
  'Separate Transformers implementation matching the architecture declared by the publisher configuration. No publisher modeling file is assumed; weight loading and exact-checkpoint execution have not been verified.'));
 const servingApplies = id => {
  if (view === 'training') return false;
  if (overview) return true;
  if (absentMechanism) return false;
  if (!topic) return !nodeInfo.operatorId && selected.has(id);
  if (['optimizer','ddp','schedule','engram','vision'].includes(topic)) return false;
  if (id === 'q4-runtime') return topic === 'attention' || (topic === 'cache' && !['state-recurrent','state-conv'].includes(nodeInfo.operatorId));
  if (id === 'dsv41-sglang-config') return false;
  if (id === 'dsv41-sglang-cache') return topic === 'cache';
  if (id === 'dsv41-sglang' && topic === 'cache') return false;
  return true;
 };
 const serving = mapping.serving.filter(servingApplies).map(id => record(id, 'serving', SOURCES[id].url.includes('/vllm-project/') ? 'vLLM' : 'SGLang', 'implementation',
  id === 'q4-runtime' ? 'component-only' : 'architecture-family', id === 'q4-runtime'
   ? 'QSA attention component only. This file does not establish complete Flash-Next serving, PLE, residual, vision or weight-loading support.'
   : 'Pinned serving implementation or dispatch/cache path for the architecture family. Exact checkpoint loading, backend compatibility and performance remain unverified.'));
 const mechanism = trainingMechanisms[topic];
 const applicableTraining = view === 'serving' ? [] : overview ? trainingIds.filter(id => m.experts || id !== 'megatron-expert-dispatch') : mechanism && (topic !== 'dispatch' || m.experts) ? [mechanism] : trainingIds.filter(id => selected.has(id));
 const training = applicableTraining.map(id => record(id, 'training', 'Megatron-LM', 'implementation', 'generic-mechanism',
  'Generic distributed training mechanism for comparison with this operation. This is not the implementation of the selected FSDP scenario. No exact-checkpoint adapter, publisher training recipe or executed training support is established.'));
 return { baseline, serving, training };
}
