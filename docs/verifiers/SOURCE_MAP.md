# Observatory reference map — acquisition and review pending

Current target: `verification/manifests/observatory.json`. The atlas links publisher configuration and model implementations, independent SGLang/vLLM paths, hardware documents and communication libraries. Full commit IDs identify candidates; **none of the entries below is an accepted byte snapshot**. Per-claim selectors, active call paths, shapes/guards and independent reviews remain obligations in the publication report.

`evidence/reference-registry.json` is the inspectable UI reference map. `verification/manifests/sources.json` additionally contains the stricter source catalog and retained cross-domain requirements. Source roles cannot be substituted: configuration ≠ implementation; documentation ≠ measured execution.

## Source catalog

| ID | Role | Repository / version | Reference |
|---|---|---|---|
| `hf-config` | publisher_config | moonshotai/Kimi-K3 · `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` | [config.json](https://huggingface.co/moonshotai/Kimi-K3/raw/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/config.json) |
| `hf-model` | publisher_code | moonshotai/Kimi-K3 · `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` | [modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/raw/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py) |
| `sglang-model` | framework_code | sgl-project/sglang · `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | [python/sglang/srt/models/kimi_k3.py](https://raw.githubusercontent.com/sgl-project/sglang/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/models/kimi_k3.py) |
| `sglang-config` | framework_code | sgl-project/sglang · `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | [python/sglang/srt/configs/kimi_k3.py](https://raw.githubusercontent.com/sgl-project/sglang/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/configs/kimi_k3.py) |
| `sglang-vision` | framework_code | sgl-project/sglang · `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | [python/sglang/srt/models/kimi_k3_vl.py](https://raw.githubusercontent.com/sgl-project/sglang/28457f0dcab4ccf748d60f2387a1a4a7fdb5a110/python/sglang/srt/models/kimi_k3_vl.py) |
| `nccl-api` | library_code | NVIDIA/nccl · `fd168324a3dc0c9080fd4881b6c7f4bb252a95a2` | [src/nccl.h.in](https://raw.githubusercontent.com/NVIDIA/nccl/fd168324a3dc0c9080fd4881b6c7f4bb252a95a2/src/nccl.h.in) |
| `nccl-collectives` | library_code | NVIDIA/nccl · `fd168324a3dc0c9080fd4881b6c7f4bb252a95a2` | [src/collectives.cc](https://raw.githubusercontent.com/NVIDIA/nccl/fd168324a3dc0c9080fd4881b6c7f4bb252a95a2/src/collectives.cc) |
| `nvshmem-api` | library_code | NVIDIA/nvshmem · `b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2` | [src/include/host/nvshmem_api.h](https://raw.githubusercontent.com/NVIDIA/nvshmem/b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2/src/include/host/nvshmem_api.h) |
| `nvshmem-quiet` | library_code | NVIDIA/nvshmem · `b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2` | [src/host/comm/quiet.cpp](https://raw.githubusercontent.com/NVIDIA/nvshmem/b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2/src/host/comm/quiet.cpp) |
| `nixl-api` | library_code | ai-dynamo/nixl · `ceb05d70fef664fe65b0a6a04167db1f29462fa5` | [src/api/cpp/nixl.h](https://raw.githubusercontent.com/ai-dynamo/nixl/ceb05d70fef664fe65b0a6a04167db1f29462fa5/src/api/cpp/nixl.h) |
| `nixl-agent` | library_code | ai-dynamo/nixl · `ceb05d70fef664fe65b0a6a04167db1f29462fa5` | [src/core/nixl_agent.cpp](https://raw.githubusercontent.com/ai-dynamo/nixl/ceb05d70fef664fe65b0a6a04167db1f29462fa5/src/core/nixl_agent.cpp) |
| `cuda-cc` | vendor_document | document · `snapshot requested 2026-09-06` | [document](https://developer.nvidia.com/cuda/gpus) |
| `hopper-tuning` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| `blackwell-tuning` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) |
| `dgx-hopper` | vendor_document | document · `DGX H100/H200 User Guide; snapshot requested 2026-09-06` | [document](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) |
| `dgx-b200` | vendor_document | document · `DGX B200 User Guide; snapshot requested 2026-09-06` | [document](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) |
| `gb300-product` | vendor_document | document · `GB300 NVL72 specification; snapshot requested 2026-09-06` | [document](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| `nvl72-hardware` | vendor_document | document · `GB200/GB300 conditional hardware guide; snapshot requested 2026-09-06` | [document](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html) |
| `roce-cumulus` | vendor_document | document · `Cumulus Linux 5.13` | [document](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux-513/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/) |
| `ib-opensm` | vendor_document | document · `DOCA 3.2.1 archive` | [document](https://docs.nvidia.com/doca/archive/3-2-1/nvidia-sm/index.html) |
| `nccl-doc` | vendor_document | document · `NCCL 2.31.2; snapshot requested 2026-09-06` | [document](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) |
| `nvshmem-ordering` | vendor_document | document · `Latest content requires exact snapshot + version extraction; not a locked release yet` | [document](https://docs.nvidia.com/nvshmem/api/latest/using.html) |
| `k3-paper` | paper | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://arxiv.org/html/2607.24653v1) |
| `dynamo-nixl` | vendor_document | document · `Dynamo v1.1.1` | [document](https://docs.nvidia.com/dynamo/v1.1.1/backends/sg-lang/disaggregation) |
| `cuda-compatibility` | vendor_document | document · `13.3; snapshot requested 2026-09-06` | [document](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) |
| `ds2-config` | publisher_config | deepseek-ai/DeepSeek-V2 · `44f7caf9e95112ea265b13f837bd43d520253548` | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/44f7caf9e95112ea265b13f837bd43d520253548/config.json) |
| `ds2-code` | publisher_code | deepseek-ai/DeepSeek-V2 · `44f7caf9e95112ea265b13f837bd43d520253548` | [modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/44f7caf9e95112ea265b13f837bd43d520253548/modeling_deepseek.py) |
| `ds2-paper` | paper | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://arxiv.org/abs/2405.04434) |
| `ds3-config` | publisher_config | deepseek-ai/DeepSeek-V3 · `e815299b0bcbac849fa540c768ef21845365c9eb` | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) |
| `ds3-code` | publisher_code | deepseek-ai/DeepSeek-V3 · `e815299b0bcbac849fa540c768ef21845365c9eb` | [modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py) |
| `ds3-paper` | paper | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://arxiv.org/abs/2412.19437) |
| `ds-runtime` | framework_code | sgl-project/sglang · `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | [python/sglang/srt/models/deepseek_v2.py](https://raw.githubusercontent.com/sgl-project/sglang/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/models/deepseek_v2.py) |
| `q3-code` | publisher_code | huggingface/transformers · `c93057d4835cd31752bb56f59989dd27696eb45b` | [src/transformers/models/qwen3/modeling_qwen3.py](https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3/modeling_qwen3.py) |
| `q3-moe-code` | publisher_code | huggingface/transformers · `c93057d4835cd31752bb56f59989dd27696eb45b` | [src/transformers/models/qwen3_moe/modeling_qwen3_moe.py](https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) |
| `q3-runtime` | framework_code | sgl-project/sglang · `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | [python/sglang/srt/models/qwen3.py](https://raw.githubusercontent.com/sgl-project/sglang/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/models/qwen3.py) |
| `q3-moe-runtime` | framework_code | sgl-project/sglang · `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | [python/sglang/srt/models/qwen3_moe.py](https://raw.githubusercontent.com/sgl-project/sglang/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/models/qwen3_moe.py) |
| `q3-paper` | paper | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://arxiv.org/abs/2505.09388) |
| `q38-config` | publisher_config | Qwen/Qwen3-8B · `b968826d9c46dd6066d109eabc6255188de91218` | [config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/b968826d9c46dd6066d109eabc6255188de91218/config.json) |
| `q330-config` | publisher_config | Qwen/Qwen3-30B-A3B · `d47d535f78ec44bd57128f8e8aeba17eeb0285ea` | [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/d47d535f78ec44bd57128f8e8aeba17eeb0285ea/config.json) |
| `q3235-config` | publisher_config | Qwen/Qwen3-235B-A22B · `d0aa57eb283a2d2489020470fde653a310b4ec6a` | [config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/d0aa57eb283a2d2489020470fde653a310b4ec6a/config.json) |
| `q3827-config` | publisher_config | Qwen/Qwen3.8-27B · `72a217afab8029b39e4af1c7273a829995a3dbaf` | [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/72a217afab8029b39e4af1c7273a829995a3dbaf/config.json) |
| `q35-code` | publisher_code | huggingface/transformers · `c93057d4835cd31752bb56f59989dd27696eb45b` | [src/transformers/models/qwen3_5/modeling_qwen3_5.py](https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| `q35-runtime` | framework_code | sgl-project/sglang · `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | [python/sglang/srt/models/qwen3_5.py](https://raw.githubusercontent.com/sgl-project/sglang/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/models/qwen3_5.py) |
| `q38flash-config` | publisher_config | Qwen/Qwen3.8-Flash-Next · `34567a4712bc9766c4449e2e98e4468bfa24d915` | [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/34567a4712bc9766c4449e2e98e4468bfa24d915/config.json) |
| `q4-code` | publisher_code | huggingface/transformers · `c93057d4835cd31752bb56f59989dd27696eb45b` | [src/transformers/models/qwen4_exp/modular_qwen4_exp.py](https://raw.githubusercontent.com/huggingface/transformers/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen4_exp/modular_qwen4_exp.py) |
| `q4-runtime` | framework_code | vllm-project/vllm · `6865e67f0be02d53694517f6f71d7fb96492792d` | [vllm/models/qwen4_exp/nvidia/qsa.py](https://raw.githubusercontent.com/vllm-project/vllm/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/qwen4_exp/nvidia/qsa.py) |
| `hopper-system` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) |
| `h100` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://www.nvidia.com/en-us/data-center/h100/) |
| `h200` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://www.nvidia.com/en-us/data-center/h200/) |
| `b200-system` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html) |
| `b200` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| `gb300` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) |
| `nccl` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) |
| `nvshmem` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/nvshmem/api/latest/using.html) |
| `roce` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/Layer-1-and-Switch-Ports/Quality-of-Service/RDMA-over-Converged-Ethernet-RoCE/) |
| `ib` | vendor_document | document · `Reference inspected 2026-09-06; exact document snapshot pending` | [document](https://www.nvidia.com/en-us/networking/products/infiniband/) |

## Evidence admission

Read `EVIDENCE.md`. Acquire raw bytes into a new candidate, verify hashes and immutable revision, resolve exact selectors/symbols, and review the consumer interpretation. Do not copy parsed web text into a file and call it a raw-source snapshot. Do not replace an accepted digest merely to erase drift.

For a released model with an unavailable implementation, leave the code-backed claim BLOCKED. Scope a documentation-only explanation explicitly; it does not fulfill the dual-code publication policy. For GPU/fabric claims, distinguish device, board, superchip, node, rack and transport/backend versions. For scheduling and recovery claims, static source review is not an executed experiment.

The migration added seven model contracts to the existing Kimi contract and retained the other domains. It did not carry forward old artifact receipts or create reviewer signatures. The historical `kimi-k3.json` target and `archive/kimi-baseline/` do not certify this new interface.
