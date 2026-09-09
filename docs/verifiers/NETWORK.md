# Scale-out networking and communication verifier

A valid networking claim identifies **physical fabric → addressing/control plane → registered memory → chosen transport/plugin → API operation → CUDA ordering → completion → consumer**. A familiar library name or an RDMA-capable NIC is insufficient.

## Physical scale-out contract

Record InfiniBand generation/port speed or Ethernet/RoCE version, HCA SKU/firmware, link width/rate, switch ports and uplinks, rail mapping, routing policy, MTU on every relevant path, oversubscription, RDMA registration backend and PCI/NUMA path. Keep per-port one-way line capacity separate from full-duplex, per-node, switch-total and application-goodput rates. A 400 Gb/s port has a 50 GB/s line-byte upper bound before protocol and utilization corrections; this is not measured payload throughput.

The checker reconciles bits/bytes, scope/direction, MTU bounds and downlink/uplink oversubscription. `payload_GBs` in its initial record schema denotes this ideal line-byte bound, not empirical goodput; efficiencies belong in a separately reviewed cost model.

**InfiniBand:** capture active subnet manager, partition identifier and membership bit, service level, link state, path MTU, routing and congestion configuration. Do not substitute Ethernet PFC/DSCP vocabulary for IB credit/QoS semantics. The verifier rejects missing SM/P_Key/SL, zero partition identifiers, invalid SL and inconsistent rate/MTU records.

**RoCEv2:** capture GID/type, L3 reachability, VLAN/traffic-class/DSCP mapping, path MTU, ECN/CNP configuration, selected lossless-PFC or explicitly supported lossy policy, and per-hop configuration/counters. PFC is not universally required by every valid RoCE deployment; the lossless-PFC profile requires it, while the lossy profile does not. A host setting alone is not fabric-wide evidence.

GPUDirect claims need actual registration and data-path results, including driver, DMA-BUF or peer-memory path and IOMMU/ACS/topology implications. Retry, error, timeout, abort and fallback states must remain visible. A fallback TCP path must not silently inherit an RDMA label or bandwidth model.

## Library-specific semantics

| Library | Verify against pinned code | Required operational evidence |
|---|---|---|
| NCCL | `src/nccl.h.in`, `src/collectives.cc`, selected transport/plugin and device algorithm | Rank/count/type/op/root agreement; stream dependencies; observed algorithm/protocol/rail choice; group/async errors and communicator recovery |
| NVSHMEM | `src/include/host/nvshmem_api.h`, host/device ordering implementation and chosen transport | PE/team participation; symmetric/registered memory; issuer scope; progress; put/get/signal order; quiet/visibility and reuse protocol |
| NIXL | `src/api/cpp/nixl.h`, `src/core/nixl_agent.cpp`, selected plugin | Registration and descriptor/metadata exchange; transfer lifetime; progress/completion; remote readiness; deregistration/free ordering |

The actual immutable commits are in `verification/manifests/sources.json`. API headers alone are not proof that a backend or fast path was selected.

NCCL is not interchangeable with NIXL. NVSHMEM offers PGAS operations and also collectives; its interface category does not imply it implements every NCCL operation identically. NVSHMEM `fence` orders relevant operations but does not establish completion. Host, device and on-stream issuer scopes must be respected: a host-side ordering call is not automatically a completion barrier for device-issued traffic. Remote visibility and local source reuse are distinct. NIXL enqueue or local submission is not a remote-ready signal.

## Executable checks and runtime obligations

Static contracts check role/version/backend, progress mode, registration lifetime, `released ≤ completed ≤ submitted`, producer/consumer ordering, and the invalid fence/enqueue-as-completion cases. They do not simulate a transport or prove distributed happens-before from counts alone.

Runtime receipts must cover every registered IB/RoCE scenario and each library, not one generic “network passed” result. Run two-host correctness before throughput; include small/large/zero payloads where supported, uneven peer sizes, delayed receiver, buffer reuse stress, cancellation, rank/connection failure and recovery. Capture raw outputs, CUDA events, NCCL/plugin logs, verbs/HCA and switch counters, affinity and exact versions. Hardware stress/fault injection is opt-in; the supplied probe script is read-only and performs none of these tests automatically.

Payload formulas must state their algorithm and message convention. Ring all-reduce, all-gather and expert all-to-all are different byte-accounting problems. NIC line rate divided into payload is an analytic bound, not measured collective latency, TTFT, ITL or application throughput.
