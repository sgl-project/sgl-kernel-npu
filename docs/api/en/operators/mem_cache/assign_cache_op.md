# assign_cache_op

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

def assign_cache_op(
    dst, src, dst_start, dst_end, src_start, src_end, *, impl="auto",
): ...
```

`auto` selects `ascendc`.

## Description

Copy variable-length segments from one packed source buffer into specified column ranges of destination rows. This general-purpose operation can update a request-to-token mapping after allocating new KV-cache slots, or copy other dense payloads using the same segment layout. Data is copied bit for bit using 1-, 2-, 4- or 8-byte elements, without numerical dtype conversion. The destination is updated in place and the function returns a boolean status.

## Parameters and return value

Use tensors on the same NPU.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `dst` | `torch.Tensor` | `[B, W]` | By element width | Required | Dense destination. Payloads with element widths of 1, 2, 4 or 8 bytes are copied bit for bit. |
| `src` | `torch.Tensor` | `[N]` | By element width | Required | Dense packed source. Use matching element widths for element-wise copies. |
| `dst_start` | `torch.Tensor` | [L], L ≥ B | `int64` | Required | Inclusive destination column per row. |
| `dst_end` | `torch.Tensor` | [L], L ≥ B | `int64` | Required | Exclusive destination end column. |
| `src_start` | `torch.Tensor` | [L], L ≥ B | `int64` | Required | Inclusive source offset per row. |
| `src_end` | `torch.Tensor` | [L], L ≥ B | `int64` | Required | Exclusive source end offset per row. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. Choices are `"auto"`, `"ascendc"`, `"torch_native"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `ok` | `bool` | False if an input is missing or scalar, or an index vector has fewer than B entries. Otherwise a normal return is True, including when an unsupported element width causes no copy. |

## ascendc backend

Use positive segment lengths and provide accessible storage for copies rounded up to 32 bytes at the end of each segment.

## torch_native backend

Copies on the input device and updates the same destination tensor.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mem_cache import assign_cache_op

dst = torch.zeros((2, 16), device="npu", dtype=torch.int32)
src = torch.arange(16, device="npu", dtype=torch.int32)
starts = torch.tensor([0, 8], device="npu", dtype=torch.int64)
cols = torch.zeros_like(starts)
ok = assign_cache_op(dst, src, cols, cols + 8, starts, starts + 8, impl="ascendc")
print(ok, dst[:, 0].tolist())  # True, [0, 8]
```
