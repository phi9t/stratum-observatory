# Implementation reference mapping

`implementationReferences(model, view, nodeInfo)` in `src/implementation-sources.mjs`
returns publisher baseline, serving and training groups. The renderer receives pinned
URLs, source IDs, symbol names, provider, scope and explicit execution status. A selected
node passes its stable scene identity as `nodeInfo.operatorId` alongside `sources`.
The API narrows files and symbols to that operation and suppresses unrelated training
or serving groups. Unknown operators retain checkpoint configuration identity and
leave implementation coverage empty; the UI keeps their authored reference chain
in a separate disclosure. Omitting node information gives a checkpoint overview.
Selection cannot introduce a different checkpoint or upgrade source scope.

This is source navigation. Every record remains `reference-only` and `executed: false`.
Neither the mapping nor the inspection below supplies source acceptance, independent
review, a weight-loading experiment or a runtime receipt.

## Publisher baseline and implementation ownership

The baseline starts with each preset's own pinned Hugging Face configuration from
`MODELS.sources`. The application continues to own checkpoint identities in `models.mjs`.

| Preset | Implementation relationship |
| --- | --- |
| DeepSeek V2/V3 | Publisher `modeling_deepseek.py` in the exact configuration revision |
| Qwen3 dense | HF configuration declares `Qwen3ForCausalLM`; separate native Transformers code |
| Qwen3 MoE | HF configurations declare `Qwen3MoeForCausalLM`; separate native Transformers code |
| Qwen3.8 27B | HF configuration declares `Qwen3_5ForConditionalGeneration`; separate Transformers architecture implementation |
| Qwen3.8 Flash-Next | HF configuration declares `Qwen4ExpForConditionalGeneration`; separate Transformers architecture implementation |
| Kimi K3 | Publisher `modeling_kimi_linear.py` text backbone at the exact configuration revision |
| DeepSeek V4.1 Flash | Publisher inference implementation, Engram and kernels at the exact configuration revision |

The Qwen configurations inspected have no `auto_map`. The mapping therefore does not
invent a publisher modeling file or silently relabel Transformers as publisher code.
The historical `publisher_code` role on some Transformers catalog entries remains
unchanged; display ownership comes from the actual repository.

## Serving and training scope

SGLang DeepSeek, Qwen and Kimi references are architecture-family implementation
references. V4.1 includes the normalization, shared implementation and cache files.
vLLM Qwen references retain architecture-family scope. The Flash-Next vLLM source is
only `Qwen4ExpQSAAttention`; it is explicitly component-only and does not certify the
rest of that checkpoint's serving path.

Megatron-LM `DistributedDataParallel` synchronization, `DistributedOptimizer`, tensor
parallel layers, expert dispatch/combine and forward/backward schedules are generic
training mechanisms. Projection and embedding operators link to parallel layers;
expert dispatch links to the token dispatcher; optimizer, synchronization and
microbatch/forward/backward operators link to their corresponding mechanisms. No inspected source establishes an exact-checkpoint
Megatron adapter, publisher training recipe or executed support for these presets.
Every generic reference carries that limit and explicitly distinguishes Megatron
from the selected FSDP scenario. A data-stream operator does not inherit optimizer
or distributed-layer references just because it appears in a training view.

## Inspection evidence

The mapping was audited against the existing `verification/manifests/sources.json`
source cache. Each inspected cache file was checked against its manifest SHA-256;
Python files were parsed without executing them, and displayed class/function names
were checked against the parsed definitions. Publisher configurations were decoded
and their architecture/model-type/auto-map declarations inspected. No cached source bytes or signatures were changed or redistributed.

Additional display catalog entries (`q3-vllm`, `q3-moe-vllm`, `q35-vllm`,
`megatron-ddp`, `megatron-optimizer`) copy the repository, revision and file identities
already present in the source manifest. Their browser URLs use those exact commits.
These existing entries required no upstream acquisition. The additional generic
Megatron files below were retrieved from their pinned primary-source URLs, parsed
without execution and stored only in the ignored local source cache. They are new
reference candidates with acquired byte identities and AST anchors in the source catalog. They do not add signed reviews or establish scientific acceptance.
No latest-version or executed-support claim is implied.

`tests/implementation-sources.test.mjs` covers exact publisher baseline identity,
Transformers separation, generic/partial support limits, unrelated-checkpoint
exclusion and independent result records. Cache inspection is an authoring audit,
not an offline runtime dependency for the browser or Node tests.

## Additional Megatron source inspection

All three files use revision `56ef6ed61ccb3edffa138ec581c212d479d9ec06`.

| File | Inspected symbols | Retrieved SHA-256 |
| --- | --- | --- |
| [Tensor parallel layers](https://github.com/NVIDIA/Megatron-LM/blob/56ef6ed61ccb3edffa138ec581c212d479d9ec06/megatron/core/tensor_parallel/layers.py) | `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding` | `6f571d15161fed1e0ccb04b34569cdc55fc5aad085778ee24b36f81ac6796b4e` |
| [Expert dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/56ef6ed61ccb3edffa138ec581c212d479d9ec06/megatron/core/transformer/moe/token_dispatcher.py) | `MoEAlltoAllTokenDispatcher.token_dispatch`, `.token_combine` | `18f9b0ef848ab36006d2cee2ac94da1be16426bbcc1200a327bcaf0ddb97d7de` |
| [Pipeline schedules](https://github.com/NVIDIA/Megatron-LM/blob/56ef6ed61ccb3edffa138ec581c212d479d9ec06/megatron/core/pipeline_parallel/schedules.py) | `get_forward_backward_func`, `forward_backward_pipelining_without_interleaving` | `a003ff59b4d28766a9b38b722c0df76a36054697f4cce40c385a2f5b9828f2cb` |

Operation tests reject Engram links on residual operators, QSA links on expert
operators, Megatron links on unrelated data-stream operators, and cache implementation
links for mechanisms absent from the checkpoint. Exact checkpoint support remains
unclaimed for every framework reference.
