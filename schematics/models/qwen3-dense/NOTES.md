# Pinned source reading

Reference: transformers v4.51.3, matching the checkpoint's loading era. Read all classes and forward paths, including non-shipped task heads. Serving: vLLM qwen3.py; inherited Qwen2 implementation is added as a separate source before citing inherited behavior.

- Qwen3RMSNorm: float32 mean square over last axis, reciprocal square root with epsilon, cast back then learned gain. No centering or bias.
- Qwen3MLP: gate projection → SiLU; independent up projection; elementwise product; down projection. All bias-free. Per-position, no cross-token communication.
- Qwen3Attention: separate Q/K/V projections from normalized residual. Reshape [B,S,H,D]; RMSNorm Q/K on D only, shared gains across heads; transpose to [B,H,S,D]. Rotate Q and K with shared cos/sin; V untouched. Cache update stores rotated K and raw V before replication. Eager attention repeats each KV head for its query group, scores QKᵀ scaled D^-0.5, adds sliced causal/padding mask, float32 softmax, dropout (zero here), weighted V, transpose/merge, output projection.
- Qwen3DecoderLayer: save residual; input RMSNorm; attention; add; save updated residual; post-attention RMSNorm; MLP; add. Full attention at every index: use_sliding_window=false and sliding_window=null disable the index-dependent window branch.
- Qwen3RotaryEmbedding: default unscaled RoPE because rope_scaling=null; inverse frequencies nonpersistent. Positions multiplied by frequencies; concatenate frequency vector with itself; cosine/sine computed in float32 and cast. rotate_half pairs the first and second halves, not adjacent coordinates.
- Qwen3Model: token embedding; initialize DynamicCache if requested; absolute positions start at past length; prepare causal/padding mask; compute rotary cos/sin once shared across all layers; sequential decoder loop; final RMSNorm. Static/sliding caches and backend-specific mask plumbing are alternate execution paths, not new checkpoint layers.
- Qwen3PreTrainedModel: framework metadata/initialization, no forward circuit.
- KwargsForCausalLM: typing-only keyword container.
- Qwen3ForCausalLM: decoder → optional sequence slice → bias-free untied LM head; labels optionally activate training loss (excluded).
- Qwen3ForSequenceClassification: last nonpad-token classification, not shipped.
- Qwen3ForTokenClassification: dropout and per-token classifier, not shipped.
- Qwen3ForQuestionAnswering: span start/end classifier, not shipped.

Shapes at this checkpoint: residual [B,S,4096]; Q [B,32,S,128]; K/V [B,8,S,128]; FFN branches [B,S,12288]; logits [B,S,151936]. Q/K gain vectors each [128], not [4096] or [1024]. All layers have independent weights and identical structure.

Serving comparison: fused QKVParallelLinear split into Q/K/V; headwise Q/K RMSNorm before RoPE retained; tensor-parallel KV partitioning or replication; backend Attention owns cache; fused residual add + RMSNorm carries residual separately across calls. Packed gate/up weights are declared in LM wrapper. No MoE, no learned position table, no tied vocabulary matrices.

Decisions: full unpadded prefill to logits in clean reference; decode cache explained and drawn but not numerically executed. CPU float32 parity checks arithmetic, not bfloat16 kernel rounding or checkpoint behavior. No weights downloaded.

Training source: NVIDIA NeMo AutoModel dense Qwen3. Read model.py in full. It inherits HF attention/layer constructors, overrides attention for packed THD [T,H] tensors and Transformer Engine, keeps Q/K normalization before rotation, rejects KV cache on the packed path, applies document-boundary metadata rather than padded masks, and supports context-parallel position metadata. Padded mode delegates to HF. It is not numerically exercised here; its imported helpers are outside the source-reading scope. No training parity or kernel claim is made.

Upgrade decisions: preserve the validated toy config and upstream fixture (four query heads, two KV heads), and explicitly distinguish its two-query grouping from the checkpoint's four-query grouping. Gain four applies to all seeded browser weights, including synthetic norm gains; it is not a checkpoint fact or trained-model claim. Add raw, normalized and rotated trace checkpoints and diagnostic scaled dot products without changing the executed forward path. The normalization-only and projection-only scores are counterfactual comparisons on identical projected inputs, not alternate checkpoint modes. Show group probabilities and exact repeated KV vectors together; show signed scores in a table, not a probability heatmap. Rotary coordinates use actual half-split pairs from the forward trace. Preserve parity fixture bytes.
