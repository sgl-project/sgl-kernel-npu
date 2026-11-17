# Split-KV Host/Device transfers

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | Supported | Supported |
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

`auto` selects `ascendc_aiv` for all four APIs, for both per-layer H2D and all-layer D2H. Explicit choices are `ascendc_aiv` and `torch_native`.

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

Each `(page,layer)` payload must be contiguous; gaps between pages in Host layer views are allowed. Callers register the complete Host allocation through MemFabric or ACL before passing its views.

## ascendc_aiv

Resolves registered Host device addresses through the operator-local host-memory helper and calls the CANN `kv_exchange` operator (`torch.ops.npu.kv_exchange`). The Torch adapter passes component addresses, page and layer strides, and traversal parameters as an ACLNN attribute. CANN tiling delivers the description to the AIV kernel, which runs on the caller's stream.

Build the CANN operator with `python3 setup.py --arch-version 2201 --ops kv_exchange` for A3 or use `--arch-version 3510` for A5.

For MemFabric, create a `memfabric_hybrid.offload.Offload` instance, initialize `Scene.LOCAL` with `alloc_size == reserve_size`, publish it with `host_memory.set_memfabric_pool`, allocate Host buffers with `pool.empty`, and call `host_memory.register_host_memory` once per full allocation before passing its views to the transfer APIs. For ACL, allocate CPU tensors and call `host_memory.register_acl_host_memory` once per full allocation.

## torch_native

An independent synchronous reference directly copies original Tensor layer/page slices. It does not reuse production view selection or address generation. Use for correctness comparison, not transfer-performance evaluation.

## Streams and lifetimes

AIV enqueues on the caller's current NPU stream. The C++ adapter uses `record_stream` on the Device components and both index tensors passed to it, so the NPU caching allocator keeps their storage alive until the transfer completes. Contiguous index copies are created and used on the submission stream. The caller provides cross-stream dependencies, keeps Host buffers and their registrations alive through completion, and waits before modifying or reusing transfer storage. Synchronize outstanding transfers before releasing the MemFabric pool.

## NPU graph capture

The AIV transfer path does not support NPU graph capture.

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

The fixture initializes the LOCAL host-memory entry and registers Host allocations for both `auto` and `ascendc_aiv`. Tests isolate in-place destinations, compare exact bytes, and then use `make_executor` / `check_performance`. Shared profiling reports summed NPU-task duration, excluding full CPU submission overhead; measure end-to-end overhead separately. An available NPU and matching CANN/Torch/MemFabric build containing the in-tree `kv_exchange` operator are required.
