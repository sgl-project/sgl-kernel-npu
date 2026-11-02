# assign_cache_op

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

def assign_cache_op(dst, src, dst_start, dst_end, src_start, src_end, *, impl="auto"):
    ...
```

## Description

`dst` is dense `[batch, width]`, `src` is a packed 1D tensor, and the four index arrays use int64. Each row copies `src[src_start[i]:src_end[i]]` into `dst[i, dst_start[i]:...]`. The source interval determines the length; callers supply matching destination intervals and sufficient storage. Index arrays may include a terminal offset or unused suffix. A successful enqueue returns `True`; insufficient index-array lengths or rank-zero inputs return `False` without writing.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `dst` | `torch.Tensor` | `[B, W]` | 1/2/4/8-byte elements | Required | Dense destination, written in place; copies bit patterns. |
| `src` | `torch.Tensor` | `[N]` | Compatible with copy byte lengths | Required | Packed one-dimensional source. |
| `dst_start` / `dst_end` | `torch.Tensor` | At least B elements | int64 | Required | Destination boundaries; lengths must match source intervals. |
| `src_start` / `src_end` | `torch.Tensor` | At least B elements | int64 | Required | Source boundaries that determine each copy length. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; auto / ascendc / torch_native. auto selects ascendc without fallback. |

Returns `bool`: `True` on successful enqueue, or `False` when the public entry rejects invalid ranks or insufficient index-array lengths.

Payload dispatch uses the destination element width, exactly as the kernel: 1/2/4/8-byte elements are copied through integer carriers. The adapter does not maintain a business-dtype whitelist: it describes destination bit patterns using an equal-width integer ACL descriptor and describes the source as bytes, without copying storage or changing Tensor dtype. Other widths have no copy branch and return True without writing; in particular, 16-byte complex128 is not a supported copy operation. No dtype conversion or contiguous copy is inserted by the public entry. The AscendC path needs storage for its 32-byte rounded DMA access, including the final source interval.

## Backend implementations

`auto` selects `ascendc` on ARCH22/ARCH35 without fallback. The implementation directly calls `torch.ops.npu.assign_cache_op`; CANN type `SglAssignCache` owns the Ref destination. The adapter allocates a zeroed synchronization tensor and retains it through asynchronous launch. `torch_native` performs vectorized indexed copies on the input device. There is no CPU-only backend because the source test computes its reference on NPU.

## Build and test

The 12 workload tests cover int8/int16/int32/int64 and `(gap, width) = (1,1024), (2,1024), (50000,100000)`, including the 48 active rows in a 96-row allocation. They compare the whole destination, preserve read-only inputs, require successful return, and benchmark the int32 cases using the test framework. Each workload runs 32 iterations, synchronizing and checking the full output, input buffers and in-place identity after every call. The functional loop remains enabled when profiling is disabled. The tests do not import another test suite.

```bash
python3 setup.py --make_clean --arch-version 2201 --ops assign_cache_op --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_assign_cache_op.py \
  --impl=ascendc --impl-ref=torch_native --enable-perf=false -v -s
```
