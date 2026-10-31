# cache_loc_assign

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign

result = cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset,
                         out_cache_loc, impl="ascendc")
```

The public signature accepts five positional tensors and keyword-only
`impl="auto"`. `auto` is an alias for `ascendc`, without fallback.
`torch_native` selects an independent PyTorch implementation. The native binding
remains available as `torch.ops.npu.cache_loc_assign`.

## Parameters and behavior

| Parameter | Contract |
| --- | --- |
| `req_pool_indices` | NPU request rows `[B]`, int32 or int64. |
| `token_pool` | NPU int32 pool `[pool_rows, row_width]`, with a dense storage layout. |
| `start_offset`, `end_offset` | NPU int64 arrays `[B]`; each interval is `[start, end)`. |
| `out_cache_loc` | Caller-owned contiguous NPU int32 tensor, addressed as a flat packed buffer. A contiguous multidimensional view retains its original shape. |
| `impl` | `auto`, `ascendc`, or `torch_native`. |

Let `length[b] = end_offset[b] - start_offset[b]`, and
`offset[b] = sum(length[:b])`. For `0 <= j < length[b]`:

```text
token_pool[req_pool_indices[b], start_offset[b] + j] = out_cache_loc.flat[offset[b] + j]
```

The operator mutates **token_pool** and returns a Tensor aliasing its same
storage, dtype, shape and device. Request indices and offsets are read-only.
`out_cache_loc` is read-only; unselected pool rows and positions remain unchanged.
There is no output allocation, automatic contiguous copy, index cast or clamp.

The common public checks validate dtype combinations and
`out_cache_loc` contiguity. Other index/shape/storage requirements remain
caller preconditions, rather than newly added runtime validation.

The CANN adapter does not repeat the public dtype/contiguity checks.
Direct `torch.ops.npu.cache_loc_assign` callers must satisfy these conditions.
The loaded library supplies the implementation of this Torch operator.

## AscendC implementation and support

The independent CANN operator is `SglCacheLocAssign`. The Torch
adapter invokes its generated ACLNN workspace-query and execution APIs, on the
current NPU stream. Assign retains the full AIV core count.
The two operators share the Kernel computation and sixteen-field
Tiling structure. `cacheLocSize` uses the logical `out_cache_loc.numel()`, not
`B * 16` or the backing allocation size.

The Kernel uses `MAX_STEP=16`, a UB capacity limit and aligned reads. Numerical
test inputs use unique request rows, nonoverlapping source/destination
storage and intervals of at most 16 tokens. Random-start cases include positions near row ends. The 16-element
loads require readable storage; numerical coverage does not establish
allocation-end read safety. Separate NPU memory diagnostics are required. Empty batches, conflicting duplicate requests, invalid
indices and aliased input/output storage are not newly guaranteed or repaired.

UB failures are reported through the CANN/ACLNN error path.

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Declared; CANN build and real NPU validation pending | Declared; CANN build and real NPU validation pending |
| `torch_native` | Implemented; real NPU validation pending | Implemented; real NPU validation pending |

The native Python implementation performs PyTorch slice copies on the input
device. It may synchronize to obtain index metadata for its Python loop; it is
not a CPU NumPy golden or a latency substitute for AscendC. No floating-point
promotion is needed for these integer copies. Graph capture and autograd support
are not claimed.

## Build and test

```bash
python3 setup.py --make_clean --arch-version 2201 \
  --ops cache_loc_assign,cache_loc_update --artifact wheel
# Install the exact wheel from output/ in a matching Linux/CANN/Torch environment.
python3 -m pytest tests/python/operators/mem_cache/test_cache_locations.py \
  --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false -v -s
```

A selected wheel contains only the native operators selected for its build.
The package loads its native library at
import; missing native artifacts are not isolated from other backends.

The two cache-location operators are built together. Selecting either ID with `setup.py --ops` includes both; direct CMake builds must select both IDs or `ALL`.
