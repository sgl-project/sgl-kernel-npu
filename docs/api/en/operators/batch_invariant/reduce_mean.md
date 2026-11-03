# reduce_mean

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import reduce_mean_batch_invariant

def reduce_mean_batch_invariant(input, dim, keepdim=False, *, impl="auto"):
    ...
```

The operator forwards to `torch.ops.batch_invariant_ops.npu_reduce_mean_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Computes the mean over one dimension. The reduction order does not depend on
how many rows are present, so reducing a feature dimension gives bitwise
identical rows whether the input holds one row or many. Reducing the batch
dimension itself mixes rows and is expected to change with the batch size; the
batch-invariance contract only applies to feature dimensions.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | any | `torch.float16`, `torch.bfloat16`, `torch.float32` | Required | Input on NPU. |
| `dim` | `int` | — | — | Required | Single dimension to reduce; negative values count from the end. A dimension list is not accepted. |
| `keepdim` | `bool` | — | — | `False` | Keeps the reduced dimension with size 1. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | Input shape without `dim`, or with it set to 1 when `keepdim` is True | Same as input | Newly allocated; inputs are not modified. |

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import reduce_mean_batch_invariant

x = torch.randn(8, 23, 529, device="npu", dtype=torch.float32)

full = reduce_mean_batch_invariant(x, -1, keepdim=True)
single = reduce_mean_batch_invariant(x[:1], -1, keepdim=True)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
