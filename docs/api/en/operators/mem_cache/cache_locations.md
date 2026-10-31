# cache_loc_assign / cache_loc_update

Copy cache-slot indices between a request-to-token mapping table and a packed
location buffer. `cache_loc_assign` writes indices into the table;
`cache_loc_update` gathers indices from it. Neither operator copies KV vectors.

## Architecture support

Generation names follow the shared operators framework: `arch22` and `arch35`,
with architecture IDs `2201` and `3510`. Implementation scope and hardware
validation status are listed separately.

| Implementation | arch22 / 2201 | arch35 / 3510 |
| --- | --- | --- |
| `ascendc` | Implemented; CANN build and NPU validation pending | Implemented; CANN build and NPU validation pending |
| `torch_native` | Implemented; NPU validation pending | Implemented; NPU validation pending |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset,
                 out_cache_loc, impl="auto")
cache_loc_update(req_pool_indices, token_pool, start_offset, end_offset,
                 out_cache_loc, impl="auto")
```

Both interfaces take five tensors and keyword-only `impl`, which defaults to
`auto`. `auto` and `ascendc` call the corresponding `torch.ops.npu.cache_loc_assign`
or `torch.ops.npu.cache_loc_update` binding, without automatic fallback.
Select `torch_native` explicitly for the PyTorch implementation. The native
Torch bindings remain directly callable.

## Parameters and semantics

| Parameter | Shape, type and constraints |
| --- | --- |
| `req_pool_indices` | NPU Tensor `[B]`, int32 or int64; request row indices in `token_pool`. |
| `token_pool` | Dense NPU int32 Tensor `[pool_rows, row_width]`, storing cache-slot indices for each request. |
| `start_offset` | NPU int64 Tensor `[B]`; inclusive start of each request interval. |
| `end_offset` | NPU int64 Tensor `[B]`; exclusive end of each request interval. |
| `out_cache_loc` | Caller-allocated contiguous NPU int32 Tensor, addressed as a flat packed buffer with at least the sum of all interval lengths. Contiguous multidimensional tensors retain their shape. Input for assign; output for update. |
| `impl` | `auto`, `ascendc`, or `torch_native`; defaults to `auto`. |

Let `length[b] = end_offset[b] - start_offset[b]` and
`offset[b] = sum(length[:b])`. Request intervals are packed consecutively in
`out_cache_loc`, in request order; their lengths may differ.

### cache_loc_assign

For each `0 <= j < length[b]`:

```text
token_pool[req_pool_indices[b], start_offset[b] + j] = out_cache_loc.flat[offset[b] + j]
```

**Mutates `token_pool` in place and returns a Tensor sharing its storage**,
with the same shape, dtype and device. Unselected rows and positions outside the
target intervals remain unchanged, as do the other four inputs. The AscendC
implementation skips zero-length intervals before reading the token pool.

### cache_loc_update

For each `0 <= j < length[b]`:

```text
out_cache_loc.flat[offset[b] + j] = token_pool[req_pool_indices[b], start_offset[b] + j]
```

**Mutates `out_cache_loc` in place and returns a Tensor sharing its storage**,
with the same shape, dtype and device. Elements beyond the packed output length
remain unchanged, as do the other four inputs.

Neither interface automatically makes inputs contiguous, casts indices or clips
indices. Public entries check the dtype combinations above and `out_cache_loc`
contiguity. Direct `torch.ops.npu` callers must ensure these conditions because
the adapter does not repeat the public checks.

## Implementation and limitations

The CANN types are `SglCacheLocAssign` and `SglCacheLocUpdate`, invoked through
ACLNN on the current NPU stream. Assign retains scheduling across all AIV cores;
update retains single-core scheduling. Both share the computation and a
sixteen-field Tiling structure. Location-buffer sizing uses its logical `numel()`.

AscendC interval lengths are limited to `MAX_STEP=16`; UB capacity and aligned
reads also constrain inputs. Numerical tests use unique request rows, valid
indices and nonoverlapping input/output storage. Empty batches, conflicting
duplicate requests, invalid indices and overlapping storage have not been
validated. Fixed 16-element reads require readable storage; numerical cases near
row ends do not replace allocation-boundary memory diagnostics. UB capacity
failures are reported through the CANN/ACLNN error path.

`torch_native` uses PyTorch slice copies on the input device. Reading index
metadata for its Python loop may synchronize execution. Integer copying needs
no floating-point promotion. Graph capture and autograd have not been validated.

## Build and test

Operator selection is passed through unchanged; selecting one ID does not
automatically add the other. Building, installing and running a single selected
operator still requires CANN/NPU validation. The example below selects both IDs.

```bash
python3 setup.py --make_clean --arch-version 2201 \
  --ops cache_loc_assign,cache_loc_update --artifact wheel
# Install the wheel produced in output/ in a matching Linux/CANN/Torch environment.
python3 -m pytest tests/python/operators/mem_cache/test_cache_locations.py \
  --impl=ascendc --impl-ref=torch_native --skip-acc-check=false --enable-perf=false -v -s
```

Use `--arch-version 3510` for arch35. The 52 cases cover individual accuracy,
in-place writes, read-only inputs, untouched regions and changed-value round
trips. Omit `--enable-perf=false` to run the performance workloads selected by
the tests.

The package loads its native library at import time; missing native artifacts
are not fully isolated from other backends.
