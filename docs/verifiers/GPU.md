# GPU and scale-up system verifier

## Identity is part of the type

| Profile | Scope | CUDA CC policy | Do not conflate with |
|---|---|---|---|
| H100-SXM | One Hopper GPU | 9.0 | H100 PCIe, NVL, a whole DGX |
| H200-SXM | One Hopper GPU | 9.0 | A new Tensor Core architecture inferred from larger HBM |
| B200 | One Blackwell GPU, exact platform SKU | 10.0 | Two dies as two independent CUDA GPUs; GB200 superchip |
| B300 | One Blackwell Ultra GPU | 10.3 | GB300 CPU+GPU superchip |
| GB300 | NVL72 compute-tray superchip scope: two GPUs + one Grace CPU | 10.3 on GPU | The rack or a separately specified DGX Station variant |
| GB300-NVL72 | Rack: 72 GPUs + 36 Grace CPUs | GPU CC 10.3 | 72 superchips, or 72 independent scale-out nodes |

The authoritative CC table, Hopper/Blackwell tuning/compatibility guides and DGX hardware guides are catalogued primary sources. These profile rules are not a claim that any device was tested here. B300/GB300 per-GPU HBM snapshot values remain unresolved rather than being back-solved from a rounded rack marketing total.

## Required source-backed fields

Record silicon and platform SKU, CUDA-visible GPU count, die/package scope, SM/device limits, physical and allocatable HBM, memory generation, L2/shared/register resource scope, native math/accumulator format, dense versus structured-sparse peak, clocks and power assumptions. All bandwidths include units, direction, aggregation and endpoint scope. “GPU memory”, coherent CPU memory and addressability are not interchangeable capacity guarantees.

For each kernel bind its source/compiler version, virtual/native target and suffix, selected dtype, tensor layout/alignment, threads, registers, shared memory, cluster size, async copy/compute barriers and feature use. The executable verifier checks necessary resource bounds, native FP4 mislabeling on Hopper, TMA consumer ordering, nonportable cluster opt-in, and the selected target profile. Passing necessary bounds is not a measured occupancy or performance prediction. Generic PTX JIT compatibility and architecture-specific `a` features need their own compiler/launch evidence; the target list is not a universal binary-compatibility solver.

## Scale-up topology

Reconcile GPU/CPU/tray totals, switch ASIC versus switch-tray counts, NIC ports and device ownership. The DGX-B200 template must not inherit Hopper's four-switch count: the catalogued DGX B200 guide describes two NVLink switches. NVL72's nine switch trays are not nine GPUs or necessarily the same ASIC-count abstraction. Keep NVLink/NVSwitch scale-up distinct from InfiniBand/RoCE scale-out, and NVLink-C2C distinct from GPU-GPU links.

Require an explicit GPU→PCI bus→NUMA→NIC/rail mapping. Do not infer affinity from array or slot ordering. Capture active MIG configuration, peer-access matrix, fabric partitions, Fabric Manager/IMEX prerequisites and disabled/degraded links before claiming reachability.

## Execution and failure cases

`python3 tools/probe-system.py --out <new-directory>` collects a bounded, read-only local inventory. It does not change clocks, network settings or driver configuration, and it does not run kernels or certify GPUDirect. Review device identifiers before sharing logs.

The stronger runtime gate needs separate receipts for each registered SKU/system. Add compiled feature and memory kernels, device attribute outputs, actual peer transfers and calibrated memory/compute measurements. Negative tests include wrong GPU scope, unsupported architecture-specific kernel, sparse/dense peak confusion, asynchronous consumption before completion, too much shared memory, nonportable clusters without opt-in, wrong switch count and topology labels that claim inaccessible peers.
