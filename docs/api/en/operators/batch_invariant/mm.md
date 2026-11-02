# mm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

def mm_batch_invariant(input, mat2, *, impl="auto"):
    ...
```

The operator forwards to `torch.ops.batch_invariant_ops.npu_mm_batch_invariant`.
Install the CANN op package and the torch extension first; see
[installation](installation.md).

## Description

Computes $Y = XW$ for $X \in \mathbb{R}^{N\times K}$ and
$W \in \mathbb{R}^{K\times M}$. The reduction order over $K$ is independent of
$N$, so each output row is bitwise identical whether it is computed alone or
inside a larger batch. That is the batch-invariance property; the numeric
result is an ordinary matrix product. Promoting or lowering coverage to other
shapes is not claimed.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[N, K]` | `torch.float16`, `torch.bfloat16`, `torch.float32` | Required | Left operand on NPU; `N > 0`, `K > 0`. |
| `mat2` | `torch.Tensor` | `[K, M]` | Same as input | Required | Right operand on the same device; a transposed, non-contiguous view is accepted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, M]` | Same as input | Newly allocated; inputs are not modified. |

The public operator forwards arguments without validation. If the external
package is missing, the first call raises an import error instead of falling
back to another implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

a = torch.randn(257, 7168, device="npu", dtype=torch.bfloat16)
b = torch.randn(16384, 7168, device="npu", dtype=torch.bfloat16).transpose(0, 1)

full = mm_batch_invariant(a, b)
single = mm_batch_invariant(a[:1], b)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
