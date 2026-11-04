# Split-KV Host/Device transfers

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `aicpu` | MF A3/910C SDMA path; hardware regression pending | MF A5/950 URMA path; hardware regression pending |
| `aiv` | Unverified; registration enabled | Declared; PF/LF refactor needs hardware regression |
| `torch_native` | Unverified | Synchronous correctness reference |

The wrappers do not restrict registration by hardware generation. Actual availability depends on the matching MemFabric backend and runtime. Passing the original URMA tests does not establish correctness of this refactor.

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

`auto` defaults to `aicpu` for per-layer H2D and `aiv` for all-layer D2H, independently of the MemFabric scene. Explicit choices are `aicpu`, `aiv`, and `torch_native`; both production implementations retain support for either direction.

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

Each `(page,layer)` payload must be contiguous; gaps between pages in Host layer views are allowed. Callers supply Host storage allocated/registered through the matching MemFabric path.

## aicpu

Triton constructs the final pointer tables for all active components in one launch, followed by one five-argument `offload.sparse_copy(src, dst, lengths, count, device)` call. The original 2-D grid and constexpr LAYOUTS are retained. The component list has fixed K, V, Indexer K and Indexer scale positions, with None for absent payloads. Flat scalar parameters use these component names; no component dictionary or role-to-slot mapping is constructed. Tuples are formed only inside the kernel, avoiding affected Ascend launchers' runtime tuple expansion. Missing components have zero layers in LAYOUTS and are skipped at compile time; the grid covers only active components. Runtime addresses, page capacities and the page count are not value-specialized. Components write directly into the final tables, without temporary tables or concatenation. Initialize MemFabric with `Scene.AICPU`, `flags=0`, and nonzero equal GiB-aligned `reserve_size` / `alloc_size`. The MF backend selects AICPU + SDMA on Ascend910C (A3), or AICPU + URMA on Ascend950 (A5).

Pass original Host addresses from the MF pool and NPU payload addresses; do not substitute AIV DVAs. MF determines H2D/D2H from each record's Host-pool endpoint at execution time. The wrapper still uses its direction to construct source/destination tables. Up to 65536 records with positive int32 byte lengths are supported; both index tensors must already reside on the KV buffer's NPU. The old `LOCAL_URMA`, `CopyDirection` and private `_get_scene` interfaces are not required.

Host libraries, Python bindings, launch extension and the MF AICPU kernel package must be updated together. On A5, one sparse_copy currently enqueues nine transfer tiles with notify waits, then one status-check kernel, including small batches. A3 enqueues a transfer kernel, notify wait and status check. Profile the complete submission sequence when comparing performance.

## aiv

Resolves registered Host device addresses through `host_memory.resolve_host_dva` and submits through the in-tree `kv_exchange` CANN operator (`torch.ops.npu.kv_exchange`): the C++ adapter builds the device-resident 44-slot metadata -- actual layer/page strides, full-page byte width, PF/LF traversal mode and per-component layer ranges, mirroring the aicpu `build_kv_sparse_metadata` contract -- and launches the AIV kernel on the caller's stream. The pinned staging that carries the metadata upload is returned to the wrapper and retained until the completion event.

The AIV path still requires MF `host_register` / `host_get_device_pointer` support. The AICPU-only MF checkout does not export those APIs; AIV or mixed `auto` integration requires a build with a compatible shared Host-pool registration contract.

## torch_native

An independent synchronous reference directly copies original Tensor layer/page slices. It does not reuse production view selection or address generation. Use for correctness comparison, not transfer-performance evaluation.

## Streams and lifetimes

Production implementations enqueue on the caller's current NPU stream without creating a transfer stream or synchronizing. The caller provides producer dependencies and consumer completion events, and prevents storage mutation/reuse until completion. Wrappers retain original inputs, sparse-copy pointer tables and AIV pinned staging until completion events retire. New contiguous index copies are allocated and consumed on the submission stream, so their storage lifetime relies on allocator ordering. The sparse_copy wrapper serializes submissions across streams to protect shared resources; callers provide AIV cross-stream data dependencies. Synchronize all transfers and end their use before releasing the MemFabric entity.

## NPU graph capture

Use explicit `impl="aicpu"` for both directions, with initialized graph-capable MF resources and a warmup of the same transfer shapes before capture. Both index tensors must already be on the KV buffer's NPU. Addresses, shapes, page count, layer selection and direction are fixed by capture; index and payload contents may be updated in place before replay.

The pointer tables are allocated during capture with `torch.empty` and belong to the graph's private memory pool. Capture bypasses eager event queries, retirement and shared-resource event bookkeeping. Callers retain static input tensors and Host/MF storage for the graph lifetime, and explicitly order all graph replays and eager transfers sharing the MF resource, including across streams. Drain replay work and destroy graphs before releasing MF resources. This does not add AIV graph support.

Run `python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py -k graph --impl aicpu --enable-perf=false -v -s` for PF/LF, packed/separate-KV roundtrips with changing NPU index contents, in-graph consumers and externally ordered alternating replay streams. Hardware validation is still required.

## Example

```python
import torch
import torch_npu
from memfabric_hybrid import offload
from sgl_kernel_npu.operators.mem_cache import (
    transfer_kv_all_layer_kv_split_lf_pf as backup,
    transfer_kv_per_layer_kv_split_pf_lf as load,
)

torch.npu.set_device(0)
config = offload.OffloadConfig()
config.device_id = 0
config.reserve_size = config.alloc_size = 1 << 30
config.flags = 0
config.scene = offload.Scene.AICPU
assert offload.initialize(config) == 0
ps = 4
original = torch.arange(3 * 4 * ps * 8, device="cpu").reshape(3, 4, ps, 1, 8).to(torch.bfloat16)
dev = original.npu()
host = offload.empty([6, 5, ps, 1, 8], dtype=torch.bfloat16)
empty_dev, empty_host = torch.empty(0, device="npu"), torch.empty(0, device="cpu")
source = torch.arange(ps, 2 * ps, dtype=torch.int64, device="npu")
saved = torch.arange(2 * ps, 3 * ps, dtype=torch.int64, device="npu")
target = torch.arange(3 * ps, 4 * ps, dtype=torch.int64, device="npu")
backup(source, saved, dev, host, empty_dev, empty_host, page_size=ps,
       host_layer_start=1, impl="aicpu")
load(target, saved, dev, host, empty_dev, empty_host, page_size=ps,
     device_layer_id=2, host_layer_id=1, impl="aicpu")
torch.npu.synchronize()
assert torch.equal(dev[2, 3].cpu(), original[0, 1])
del host
offload.uninitialize()

```

For LF Host, allocate `[5,6,ps,1,8]`, meaning `[host_layers,host_pages,ps,1,8]`, and use the APIs without layout suffixes.

## Validation

```bash
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aicpu --impl-ref torch_native --enable-perf=false -v -s -rs
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aiv --impl-ref torch_native --enable-perf=false -v -s -rs
python -m pytest tests/python/operators/mem_cache/test_kv_transfer.py --impl aicpu --impl-ref torch_native --prof-dir results/kv --csv-path results/kv.csv -v -s
```

The test fixture initializes LOCAL for an explicit aiv choice and AICPU otherwise. Use `--impl aicpu` with the new AICPU-only MF package. The auto default calls sparse_copy for H2D and the kv_exchange AIV kernel for D2H, requiring MF support plus a build containing the in-tree `kv_exchange` operator. Only service-sized cases enable performance by default; `--enable-perf` overrides this. Tests isolate in-place destinations, compare exact bytes, and then use `make_executor` / `check_performance`. Add `-m extra_checks` for supplementary pointer-table checks. Shared profiling reports mean summed NPU-task duration, excluding full CPU submission overhead; measure end-to-end overhead separately when comparing the old implementation. An available A3/910C or A5/950 and matching CANN/Torch/MemFabric are required; verify imports use the current build.
