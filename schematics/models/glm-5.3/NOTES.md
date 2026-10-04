# Source reading notes

Exact checkpoint and all revisions are recorded in sources.json. Read modeling_glm_moe_dsa.py in full, configuration_glm_moe_dsa.py and NVIDIA serving/model.py plus attention.py. No publisher modeling file/auto_map in config; Transformers is the reference. No verified checkpoint-specific public training implementation identified; training is outside scope.

## Module inventory and forward order
- GlmMoeDsaRMSNorm: float32 mean square, reciprocal square root with epsilon, cast then multiply learned gain. Decoder norms use config epsilon; q_a and kv_a constructors omit epsilon and therefore use 1e-6, unlike vLLM which supplies 1e-5.
- GlmMoeDsaRotaryEmbedding: config class rewrites head_dim to qk_rope_head_dim (64), despite raw config head_dim=192. Default theta=8000000; frequency theta^(-2i/64), no YaRN branch. apply_rotary_pos_emb_interleave reads alternating pairs but returns concatenated even and odd rotated slices, not alternating output storage.
- GlmMoeDsaIndexer: q_resid [B,S,2048] → wq_b [B,S,32,128]; hidden → wk [B,S,128] → LayerNorm (gain+bias, eps 1e-6). Split each into rotary 64 and passing 64; rotate and concatenate. Cache single index key per token. q·k/sqrt(128), ReLU, multiply learned signed head weights hidden→32 divided by sqrt(32), sum heads, causal mask, min(2048,T) top-k indices. No softmax in index scoring.
- GlmMoeDsaAttention: q_a 6144→2048; RMSNorm; q_b→64×256 and split 192+64. kv_a 6144→576, split 512+64; RMSNorm only on 512. Rotate q/k slices. Cache normalized 512 latent + rotated 64 key BEFORE expansion. kv_b 512→64×448, split per head key 192 + value 256; broadcast rotary key; concatenate key. Full indexer computes indices; shared layer requires preceding indices, errors if missing. Eager path scatters exclusion mask and intersects causal mask; scaled qk product, masked softmax, weighted values, merge heads to 16384, o_proj→6144. All attention projections have no bias.
- GlmMoeDsaMLP: independent gate/up projections, SiLU(gate) × up, down. Dense intermediate 12288; shared expert 2048. No bias.
- GlmMoeDsaTopkRouter: flatten tokens, float32 linear to 256 logits, sigmoid, correction BUFFER for choices only; group score sum of best two, retain best groups (one group here, vacuous), top eight choices, gather ORIGINAL sigmoid scores, divide selected sum+1e-20, multiply 2.5. Bias is not an additive contribution to mixture weights.
- GlmMoeDsaExperts: packed gate_up [256,4096,6144], down [256,6144,2048]; dispatch selected tokens, SiLU gate × up, down, mixture weight multiply, index_add to original token positions.
- GlmMoeDsaMoE: routed branch plus always-on shared expert from same input; shared expert has no router logit.
- GlmMoeDsaDecoderLayer: input RMSNorm, attention, residual add, post-attention RMSNorm, dense/MoE, residual add. mlp_layer_types is authoritative.
- GlmMoeDsaPreTrainedModel: framework glue/initialization; ignores checkpoint layer 78 (MTP). Excluded as no separate forward.
- GlmMoeDsaModel: embedding [154880,6144]; 78 layers; absolute positions, causal mask, rotary embeddings; loop propagates top-k each layer; final RMSNorm.
- GlmMoeDsaForCausalLM: untied bias-free head [154880,6144], optionally select positions, logits. Loss/generation excluded.

## Layer variation
Explicit indexer_types: layers 0,1,2 full, then 6,10,...,74 full; others shared. Explicit mlp_layer_types: 0..2 dense, 3..77 sparse. Sharing is selections, not KV tensors or weights. Each layer has its own latent cache.

## Serving comparison
Registry maps GlmMoeDsaForCausalLM to vllm.models.deepseek_v32. NVIDIA implementation aliases architecture at package level. Decoder distinguishes dense by first_k_dense_replace and moe_layer_freq (same actual partition). Fuses residual add/all-reduce/RMSNorm and sometimes embedding gather+norm. Loader fuses q_a and kv_a, gate/up, and index wk/head weights. attention computes query in latent key space via W_UK_T, sparse MQA on compressed cache, then W_UV expands attended latent; avoids materializing expanded KV in decode. Index keys stored FP8 plus scale metadata, whereas reference caches unquantized tensor. Schedule uses freq/offset rather than explicit list: check equality on this config. Norm epsilon mismatch retained as reference, not averaged. Hardware kernels/quantization are not proved numerically by CPU parity.

## Ledger boundary
Count unquantized decoder trainable parameters only; correction buffers, FP8 scales and auxiliary MTP excluded. Hub safetensors.parameters.total includes auxiliary layer 78 and correction buffers, but excludes FP8 scale tensors. The names-only tensor index and pinned MTP constructors reconcile every tensor shape; including the scale tensors reproduces metadata.total_size in bytes. See reference/storage-audit.json for the exact breakdown. No weights were downloaded.

## CPU parity and review decisions
The preinstalled Transformers version rejected this checkpoint’s layer-type vocabulary. An isolated pinned Transformers source package under ignored runs/ supplied support modules; the fixture imports the pinned reference files themselves. Eager attention and eager experts avoid unsupported CPU grouped-matmul execution. All parameters and correction buffers are perturbed before recording logits, norms, attention, index selection and router weights. Toy index-head count is eight to avoid a degenerate all-zero ReLU top-k tie in the selected fixture. Top-k tie ordering remains backend-dependent; tests compare valid selected sets, not masked future fillers. Real dimensions are tested in shape mode, not executed numerically.

Screenshot review replaced the narrow stock router widget with a spaced custom SVG and the mobile serving table with responsive facts. Routed expert inspectors describe packed parameters rather than standalone dense MLP names. No shared expert consumes router logits or weights.
