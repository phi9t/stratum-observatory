# GLM-5.3-Flash schematic notes

## Scope

Model page slug: `glm-5.3-flash`.

Checkpoint/config:

- Hugging Face checkpoint `zai-org/GLM-5.3-Flash-BF16`
- revision `a5b45eb41df6402735dedc900be14a42e8d5e538`
- config path `config.json`

Reference implementation:

- `huggingface/transformers`
- commit `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`
- path `src/transformers/models/glm5_next/modeling_glm5_next.py`
- config path `src/transformers/models/glm5_next/configuration_glm5_next.py`

Serving implementation:

- `vllm-project/vllm`
- commit `273a04964ed9d1e389c4a7cb9d660e8294782f8d`
- paths under `vllm/models/glm5next/common/`: `model.py`, `attention.py`,
  `kda.py`, `sparse_indexer.py`, `mtp.py`

Training evidence, read only:

- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-02-report.md @ 968462a4f`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-05-report.md @ 5eccb1492`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-06-report.md @ 1e707e0b6`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-13-report.md @ 3aa4f8048`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-13b-report.md @ 2228c3877`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-13c-report.md @ d4da5c208`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-17-report.md @ 23e39799e`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-18-report.md @ 4d55df899`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/task-n01b-report.md @ 3fca80581`
- `torchtitan: .scratch/glm53-flash-sft-ladder/sdd/progress.md @ d9ec5c537`
- `torchtitan local: experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01/logs/`
- `torchtitan local: .scratch/glm53-flash-sft-ladder/evidence/t07b/full_size_param_counts.json`

The TorchTitan files are not in `sources.json`, because the fetch/cache tool only
accepts Hugging Face and GitHub sources. The page cites TorchTitan report paths
as measured local evidence, not as `code(...)` facts. No TorchTitan command was
run. The parameter-count JSON is not git-tracked; it remains local evidence.

## Architecture facts checked against pinned code

- The checkpoint is `model_type: "glm5_next"` with text subtype
  `glm5_next_text` and architecture `Glm5NextForConditionalGeneration`.
- Text hidden size is `4096`, vocabulary is `154880`, and the text model has
  `45` decoder layers.
- The layer schedule is a 3:1 hybrid pattern. Layers `3, 7, 11, 15, 19, 23,
  27, 31, 35, 39, 43` are `deepseek_sparse_attention`; all other base layers
  are `linear_attention` KDA.
- The first `3` layers use dense MLP. Layers `3` through `44` use MoE.
- The checkpoint has one next-token prediction layer:
  `num_nextn_predict_layers: 1`.
- Text positional encoding is NoPE: `qk_rope_head_dim: 0`,
  `mla_use_nope: true`, and the text model calls sparse attention with
  `position_embeddings=None`.
- Sparse attention is MLA with `q_lora_rank: 1536`, `kv_lora_rank: 512`,
  `qk_nope_head_dim: 256`, `v_head_dim: 256`, and `64` heads.
- The sparse indexer has `32` heads of width `128`, top-k `2048`,
  `index_kpool: 4`, k-pool compression enabled, and tail selection enabled.
  `get_pooled_states` groups keys into pools of four and produces one weighted
  pooled key per complete pool.
- Important correction: every base sparse layer in this checkpoint runs its own
  indexer. `indexer_types` is `"full"` for all `45` entries. The reference code
  supports `"shared"` cross-layer top-k reuse, but this checkpoint does not use
  it. The only checked reuse flag is MTP serving reuse:
  `index_share_for_mtp_iteration: true`.
- KDA is implemented as `Glm5NextTextLinearAttention` with `64` heads,
  head dim `128`, causal depthwise conv kernel `4`, a bounded forget gate with
  lower bound `-5`, beta sigmoid, recurrent/chunk KDA kernels, gated RMSNorm,
  and output projection.
- mHC is enabled with `hc_mult: 4`, `hc_eps: 1e-6`, and `hc_sinkhorn_iters: 20`.
  The reference expands token streams to `[B, S, 4, 4096]`, collapses to
  `[B, S, 4096]` before each attention/MLP, and expands/mixes after.
- The MoE router is fp32, uses sigmoid scores, adds
  `e_score_correction_bias` only for selecting the top experts, gathers the
  original sigmoid scores for weights, normalizes them, and scales by `2.5`.
- MoE has `288` routed experts, top-8 routing, and `1` shared expert. The shared
  expert is added after the routed sum and does not consume router logits.
- Dense and expert MLPs clamp gate/up activations with `swiglu_limit: 10` and use
  SiLU gate times up projection.
- vLLM validates `index_kpool_always_select_tail=True`, reserves a top-k buffer
  of `index_topk + index_kpool - 1` rounded to 128-column sparse tiles, stores
  compressed indexer state at one state per `index_kpool` tokens, keeps raw tail
  cache for incomplete pools, fuses KDA projections, and fuses deferred mHC
  post/pre work across adjacent layers.

## Parameter accounting

The Hugging Face fetch reports `safetensors.total = 321323031390`.

The reconciled stored-tensor ledger used by the page includes:

- embedding and untied head, `1268776960`
- KDA attention, `4682897792`
- sparse MLA attention, `1291868160`
- sparse indexers, `82190592`
- dense MLP, `452984832`
- routed experts in base MoE layers, `304405807104`
- shared experts in base MoE layers, `1056964608`
- base router weights plus correction buffers, `49557312`
- base mHC tensors, `35391870`
- base layer/final norms, `372736`
- MTP stored tensors, `7432592416`
- vision tower/projector, `563627008`

These sum exactly to `321323031390`. The separate TorchTitan JSON
`full_size_param_counts.json` reports text decoder without MTP/vision as
`313326799870`; that report does not use the same "stored tensor including
router correction buffers" boundary as the Hub total reconciliation.

## Training and scaling evidence

All measured values are from the local TorchTitan campaign, not from the
Transformers/vLLM source cache.

- `full_size_param_counts.json` gives the LoRA rank-16 memory plan:
  base bf16 shard `72.9520804877393` GiB/GPU at world size 8, LoRA optimizer
  shard `0.09386444091796875` GiB/GPU, static floor
  `73.04594492865726` GiB/GPU, and static plus largest all-gather
  `86.85580561636016` GiB/GPU. World size 4 static floor is
  `146.09188985731453` GiB/GPU.
- `task-13c-report.md` gives the full-depth EP8 measured run: 15 steps,
  mean `29013.4` ms/optimizer step, peak printed CUDA memory `81.46` GiB/rank,
  and exact HDFS resume equality for steps `11` to `15` after staging the
  step-10 checkpoint. This is measured on 8x B200 TorchTitan.
- T08f logs under
  `experiments/glm53_flash_sft_ladder/results/t08f-20261014T213158Z/attempt-01/logs/`
  give rung-4 EP8 training: seed 42 trained 94 steps with loss
  `1.67393` to `0.005254015326499939`, peak `81.46` GiB/GPU and roughly
  `37` s/step under shared load; seed 43 trained 94 steps with final loss
  `0.007782145403325558`. The step-62 trainable-only checkpoint was published
  to HDFS, staged back, and resumed to step 94 with exact equality for steps
  `63..94`; `resume_loss_compare.md` reports max abs delta `0.0`.
- T08f adapter export logs exist for both seeds with `906` exported tensors, but
  there is no completed task report and no Countdown eval pass@1 result in the
  inspected logs, so the page marks eval pending.
- `task-13-report.md` gives first-8-layer FSDP loop timing around
  `110` seconds/step and `67.1171875` GiB/rank, plus an early EP4 result.
- `task-13b-report.md` gives first-8-layer grouped baseline
  `29579.53493` ms/step at `66.9765625` GiB/rank and EP8 grouped
  `5652.821390` ms/step at `17.3046875` GiB/rank.
- `task-17-report.md` gives one released-width MoE layer on one B200:
  loop `1352.5309903294935` ms, grouped `34.2193393347164` ms, speedup
  `39.52533908091311x`.
- `task-18-report.md` records DeepEP as not ready for GLM yet. The DeepEP source
  commit is `01dc3aaac82068020353dce2c302e38153c0bfaa`, but GLM DeepEP loss,
  grad, and ms/step are missing after the post-fix rerun was blocked by GPU
  occupancy. The completed T18 all-to-all row is `7820.945087` ms/step under
  shared load.
- `task-n01b-report.md` gives the DeepEP v2 build context and Qwen smoke, not a
  GLM DeepEP performance claim.

## Long-context estimates

The page labels these as estimates derived from config, not measurements.

- Full activation checkpointing still has to retain the mHC stream boundary:
  `45 x 4 x 4096 x 2` bytes = `1474560` bytes/token, about `1.41` MiB/token.
  At `16384` tokens this is about `22.5` GiB for the whole model, one
  sequence on one GPU. A per-layer value is `4 x 4096 x 2` bytes/token, about
  `0.5` GiB at `16384` tokens.
- Current indexer score storage remains the O(L^2) term:
-  `[L, 32 heads, L / 4]` per sparse layer.
- EP8 per-GPU activation room is estimated as `178.35` GiB B200 capacity minus
  `73.04594492865726` GiB static floor minus `6` GiB buffers, about `99.3`
  GiB/GPU.
- Working limits to display:
  `16-24k` tokens/GPU today, `32-48k` with chunked indexer and chunked loss,
  about `256k` with 8-way context parallelism, and about `1M` with CPU offload
  of saved activations.

## SFT ladder status

`task-08f-report.md` was checked and does not exist, so T08f is cited from logs
and `progress.md`, not from a completed report.

Current ladder evidence:

- Rung 0 Qwen3 0.6B/1.7B completed: dev pass@1 0.6B base `0.004` to
  `0.254`/`0.326`, and 1.7B base `0.024` to `0.370`/`0.228`
  (`task-02-report.md @ 968462a4f`).
- Rung 1 Qwen3 8B completed: dev pass@1 base `0.028` to `0.474`/`0.488`
  (`task-05-report.md @ 5eccb1492`).
- Rung 2 Qwen3 30B-A3B completed: dev pass@1 base `0.006` to attention LoRA
  `0.440`/`0.476` across two seeds, and attention+expert LoRA `0.600` for one
  seed; exact HDFS resume and router-stat check found no collapse
  (`task-06-report.md @ 1e707e0b6`).
- Rung 4 GLM-5.3-Flash T08f trained seed 42 and seed 43 to 94 steps on EP8 and
  proved exact step-62 HDFS resume equality over steps `63..94`. Adapter export
  logs exist, but the Countdown eval result remains pending in the inspected
  evidence.

## Reference implementation decision

The local clean reference will be shape-only. It enumerates the exact stored
tensors, layer schedule, major tensor shapes, and parameter count without
downloading or executing the model weights. Whole-model numerical parity against
Transformers/vLLM is unverified and is stated on the page and in `REPORT.md`.
