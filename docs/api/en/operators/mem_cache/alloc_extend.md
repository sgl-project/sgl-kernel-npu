# alloc_extend

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `cpu` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import alloc_extend

def alloc_extend(pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values, *, impl="auto"):
    ...
```

## Description

Allocates packed token-slot indices for extending paged requests. It mutates the
caller-owned `out_indices` and conditionally `values[0]`, and returns `None`.
`auto` selects `ascendc` without fallback; `torch_native` computes on the input
device; `cpu` computes on CPU and copies results back into the caller's buffers.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `pre_lens` / `seq_lens` / `last_loc` | `torch.Tensor` | `[B]` | int64 | Required | Existing lengths, target lengths and last occupied slots; dense vectors. |
| `free_pages` | `torch.Tensor` | `[P]` | int64 | Required | Available page IDs in allocation order. |
| `page_size` | `int` | — | — | Required | Positive page length, represented as int32 in Tiling. |
| `out_indices` | `torch.Tensor` | `[sum(seq_lens-pre_lens)]` | int64 | Required | Caller-owned dense output, written in place. |
| `values` | `torch.Tensor` | At least 1 element | int64 | Required | Conditionally writes values[0]; preserves other elements. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; auto / ascendc / torch_native / cpu. auto selects ascendc without fallback. |

Returns `None`; outputs are written into caller-owned buffers in place.

B must be positive; prefixes cannot exceed sequences, free pages and storage must be sufficient, and last locations must be valid.

The public entry checks the dtype of the first four Tensor inputs and
`out_indices`. It does not add shape, contiguity or `values` dtype checks. Direct
`torch.ops.npu.alloc_extend` calls bypass those Python checks. The CANN definition
accepts int64 tensors. No implicit contiguous copies or input casts are applied
by the AscendC adapter.

For each request, reuse the remaining slots after `last_loc` in its partial page,
then fill full and partial new pages from `free_pages`. Prefix sums determine
which output slice and free pages belong to each request. Inputs and free-page
ownership are not modified by this operator.

`values[0]` is written with the total number of new pages only when the final
request reaches a newly allocated partial page. It remains unchanged when the
final request finishes in its existing page or at a full-page boundary. This
buffer is **not a general page-count API**; the other elements remain unchanged.
The reference implementations preserve this condition rather than fabricating a
page-count result on every invocation.

## Backend implementations

`SglAllocExtend` declares two in-place Ref outputs. Its generated ACLNN query
receives six tensor descriptors followed by the page-size attribute; the Torch
binding keeps the public argument order with `page_size` before the two outputs.
The adapter uses the current NPU stream and retains descriptor, input and
workspace lifetimes across queued execution. Tiling keeps `min(AIV cores, B)`
scheduling and the four-field data layout.

CPU reference computation follows the supplied CPU golden's allocation semantics,
using int64 output buffers and explicitly preserving the conditional `values`
write. `torch_native` uses vectorized PyTorch indexing. Neither path uses NumPy or
floating-point promotion. CPU transfers and native metadata inspection may
synchronize. Package import
still requires the NPU runtime and installed native artifacts, including when
selecting `cpu`.

## Build and test

```bash
python3 setup.py --make_clean --arch-version 2201 --ops alloc_extend --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_alloc_extend.py \
  --impl=ascendc --impl-ref=cpu --skip-acc-check=false --enable-perf=false -v -s
# Exercise the other callable implementation against the same CPU reference:
python3 -m pytest tests/python/operators/mem_cache/test_alloc_extend.py \
  --impl=torch_native --impl-ref=cpu --skip-acc-check=false --enable-perf=false -v -s
```

Use `3510` for arch35. Tests explicitly fix `impl_ref="cpu"`; they compare private
output buffers, check unchanged inputs and exercise 12 workloads. Omit the
performance override for the selected long-prefill and B100 workloads. A wheel
built with `--ops alloc_extend` contains only that selected native operator;
confirm the installed library source when testing a shared `torch.ops` name.
