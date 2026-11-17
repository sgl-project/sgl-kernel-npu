# Split-KV Host/Device transfers

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | Supported | Supported |
| `memfabric_aicpu` | Requires compatible MemFabric sparse copy | Requires compatible MemFabric sparse copy |
| `torch_native` | Supported | Supported |

The AIV implementation targets Ascend A3 (arch22) and A5 (arch35). Availability depends on a compatible runtime and registered Host memory.

## Signatures

Public import path: `sgl_kernel_npu.operators.mem_cache`.

```python
def transfer_kv_per_layer_kv_split_pf_lf(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, device_layer_id, host_layer_id, device_index_layer_id=None, host_index_layer_id=None,
    impl="auto",
) -> None: ...

def transfer_kv_all_layer_kv_split_lf_pf(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, host_layer_start=0,
    impl="auto",
) -> None: ...

def transfer_kv_per_layer_kv_split(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, device_layer_id, host_layer_id, device_index_layer_id=None, host_index_layer_id=None,
    impl="auto",
) -> None: ...

def transfer_kv_all_layer_kv_split(
    device_indices, host_indices, device_k, host_k, device_v, host_v,
    device_index_k=None, host_index_k=None,
    device_index_k_scale=None, host_index_k_scale=None, page_size=128,
    *, host_layer_start=0,
    impl="auto",
) -> None: ...


```

`auto` uses `ascendc_aiv` for both directions with ACL-registered Host memory. With MemFabric-registered Host memory, per-layer H2D uses `memfabric_aicpu` to build address tables on the AICPU and submit MemFabric `sparse_copy`. All-layer D2H uses `ascendc_aiv` with the device virtual addresses (DVAs) obtained during MemFabric registration. The explicit choices are `ascendc_aiv`, `memfabric_aicpu`, and `torch_native`.

## Description

Device buffers always use `[layer,page,token,head,dim]`. The per-layer `pf_lf` API performs H2D and the all-layer `lf_pf` API performs D2H with PF Host `[page,layer,token,head,dim]`. APIs without a layout suffix use LF Host `[layer,page,token,head,dim]`.

Per-layer APIs transfer only the specified Device/Host KV layer and explicitly selected Indexer slot. Omitting both Indexer IDs disables Indexer transfer; providing only one raises. All-layer APIs transfer every KV layer in this Device pool into Host layers beginning at `host_layer_start`; Indexer components use their own complete layer space.

Indices pair Device/Host pages in order. Payload bytes are unchanged, with no quantization or dequantization. Destination pages must not overlap. Inputs must consist of aligned complete contiguous token-slot runs, each containing `page_size` entries. Production kernels use the first entry of each run to determine the page; the caller ensures continuity of the remaining entries.

## Parameters and return value

| Parameter | Type, default and constraints |
| --- | --- |
| `device_indices` / `host_indices` | 1-D int64 tensors on the same NPU as K; equal lengths divisible by page_size; token slots, not page numbers or pointers; pages must fit every participating component |
| `device_k` / `host_k` | Nonempty 5-D tensors; NPU LF Device and CPU PF/LF Host; matching dtype and per-page `[T,H,D]` |
| `device_v` / `host_v` | Tensor pair; nonempty Device V follows KV layer selection; empty Device V denotes no separate packed-K V payload, with an unused Host placeholder allowed |
| `device_index_k` / `host_index_k` | Optional, default None; transferred only when both operands exist, retaining legacy semantics; Indexer layer count can differ from KV |
| `device_index_k_scale` / `host_index_k_scale` | Optional FP32 scale, default None; copied unchanged when both exist; Device may be `[I,P,T,1]`, Host is `[P,I,T,1,1]` or `[I,P,T,1,1]` |
| `page_size` | Positive integer, default 128; matches the token dimension of every selected component |
| `device_layer_id` / `host_layer_id` | Required nonnegative pool-local IDs for per-layer APIs; may differ |
| `device_index_layer_id` / `host_index_layer_id` | Optional nonnegative independent Indexer slots, default None; provide both or neither |
| `host_layer_start` | All-layer Host KV offset, default 0; sufficient Host layers must remain |
| `impl` | Keyword-only, default auto; see backend descriptions |
| Return value | None; H2D mutates Device, D2H mutates Host |

Each `(page,layer)` payload must be contiguous. Gaps between pages in Host layer views are allowed. The Host allocation and registration requirements depend on the selected implementation.

## ascendc_aiv

Resolves registered Host device addresses through the operator-local host-memory helper and calls the CANN `kv_exchange` operator (`torch.ops.npu.kv_exchange`). The Torch adapter passes component addresses, page and layer strides, and traversal parameters as an ACLNN attribute. CANN tiling delivers the description to the AIV kernel, which runs on the caller's stream.

Build the CANN operator with `python3 setup.py --arch-version 2201 --ops kv_exchange` for A3 or use `--arch-version 3510` for A5.

For MemFabric, create a `memfabric_hybrid.offload.Offload` instance, initialize `Scene.LOCAL` with `alloc_size == reserve_size`, publish it with `host_memory.set_memfabric_pool`, allocate Host buffers with `pool.empty`, and call `host_memory.register_host_memory` once per full allocation before passing its views to the transfer APIs. For ACL, allocate CPU tensors and call `host_memory.register_acl_host_memory` once per full allocation.

## memfabric_aicpu

The four APIs accept explicit `impl="memfabric_aicpu"` in wheels containing
`kv_sparse_metadata`. This command builds the metadata operator and the AIV operator on A3.

    python3 setup.py --arch-version 2201 --ops kv_exchange,kv_sparse_metadata --artifact wheel

The default `--ops all` also discovers both operators. Follow the upstream
[AICPU build prerequisites](../../../../../python/sgl_kernel_npu/README.md#building-and-installing-aicpu-operators).

One AICPU task reads the live NPU page-start indices and generates
`src` and `dst` (int64 address bits) and `lengths` and `count` (int32).
Records are ordered by component, page, and layer. The compact component
description is an ACLNN integer array attribute, so it needs no staging tensor or
separate H2D copy. CANN serializes and delivers the task parameters.

Python passes the initialized pool's `sparse_copy` method, published through
`host_memory.set_memfabric_pool`, to a Host C++ submission entry. On the caller's
current stream, this entry submits AICPU metadata generation, calls back into the
same pool's `sparse_copy`, then records the internal transfer completion event.
If no pool was published, it uses the initialized legacy module-level
`offload.sparse_copy`. Internal event management moves to C++; the event is retained,
and the caller's completion events are unchanged.

The device AICPU builds the address tables from the NPU indices. The installed
MemFabric package selects the execution unit for `sparse_copy`.
The selected copy must support the generated original Host virtual addresses (HVAs),
using buffers from the same pool. Batches above 65,536 records are rejected before
launch. If a page start is invalid, unaligned, or out of range, the builder publishes
an empty batch and reports an error.

The AICPU builder's address checks can be run without an MF entity:

    python -m pytest tests/python/operators/mem_cache/test_kv_sparse_metadata_aicpu.py -v

End-to-end checks also require compatible MemFabric sparse copy and a matching
Host pool.

## torch_native

An independent synchronous reference directly copies original Tensor layer/page slices. It does not reuse production view selection or address generation. Use for correctness comparison, not transfer-performance evaluation.

## Streams and lifetimes

AIV enqueues on the caller's current NPU stream. The C++ adapter retains the NPU tensors through the queued launch. Contiguous index copies are created and used on the submission stream. The caller provides cross-stream dependencies, keeps its NPU inputs and registered Host buffers alive through completion, and waits before modifying or reusing transfer storage. Synchronize outstanding transfers before releasing the MemFabric pool.

AICPU enqueues metadata generation and MemFabric copy on the caller's current NPU stream. Its C++ adapter records the Device components and original indices on that stream. The address tables are allocated and consumed on the same stream. In eager execution, the Host C++ submission entry records the internal event after MF copy submission and orders MemFabric submissions across streams on the same device. Python no longer creates, records, or waits on internal events. The caller owns the Host buffers and provides dependencies between producers, transfers, and consumers.

## NPU graph capture

The AIV path requires eager execution. The AICPU path can capture metadata generation and `sparse_copy` on the capture stream with a graph-compatible MemFabric runtime. The caller keeps the index and KV tensors and the registered MF pool alive at fixed addresses for capture and replay. Update index values in place with unchanged shapes. The caller orders replays against other MemFabric submissions that share the pool, since captured transfers bypass the eager per-device completion event.

## Example

```python
import torch
import torch_npu
from memfabric_hybrid import offload
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
    register_host_memory,
    set_memfabric_pool,
    uninitialize_host_memory,
)
from sgl_kernel_npu.operators.mem_cache import (
    transfer_kv_all_layer_kv_split_lf_pf as backup,
    transfer_kv_per_layer_kv_split_pf_lf as load,
)

torch.npu.set_device(0)
pool = offload.Offload()
config = offload.OffloadConfig()
config.device_id = 0
config.reserve_size = 1 << 30
config.alloc_size = 1 << 30
config.scene = offload.Scene.LOCAL
assert pool.initialize(config) == 0
set_memfabric_pool(pool)
ps = 4
original = torch.arange(3 * 4 * ps * 8, device="cpu").reshape(3, 4, ps, 1, 8).to(torch.bfloat16)
dev = original.npu()
host = pool.empty([6, 5, ps, 1, 8], dtype=torch.bfloat16)
register_host_memory(host)
empty_dev, empty_host = torch.empty(0, device="npu"), torch.empty(0, device="cpu")
source = torch.arange(ps, 2 * ps, dtype=torch.int64, device="npu")
saved = torch.arange(2 * ps, 3 * ps, dtype=torch.int64, device="npu")
target = torch.arange(3 * ps, 4 * ps, dtype=torch.int64, device="npu")
backup(source, saved, dev, host, empty_dev, empty_host, page_size=ps,
       host_layer_start=1, impl="ascendc_aiv")
load(target, saved, dev, host, empty_dev, empty_host, page_size=ps,
     device_layer_id=2, host_layer_id=1, impl="ascendc_aiv")
torch.npu.synchronize()
assert torch.equal(dev[2, 3].cpu(), original[0, 1])
del host
uninitialize_host_memory()

```

For LF Host, allocate `[5,6,ps,1,8]`, meaning `[host_layers,host_pages,ps,1,8]`, and use the APIs without layout suffixes.

## Validation

```bash
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl ascendc_aiv --impl-ref torch_native --enable-perf=false -v -s -rs
```

The fixture initializes the LOCAL MemFabric pool and registers Host allocations. With `--impl auto`, it exercises `memfabric_aicpu` for H2D and `ascendc_aiv` for D2H. Tests isolate in-place destinations, compare exact bytes, and then use `make_executor` / `check_performance`. Shared profiling reports summed NPU-task duration, excluding full CPU submission overhead; measure end-to-end overhead separately. The tests require an available NPU and matching CANN, Torch, and MemFabric builds with `kv_exchange`. With `--impl auto`, they also require `kv_sparse_metadata` and compatible MemFabric `sparse_copy`.
