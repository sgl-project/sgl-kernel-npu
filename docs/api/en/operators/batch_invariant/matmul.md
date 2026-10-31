# matmul

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

def matmul_batch_invariant(input, other, *, impl="auto"):
    ...
```

The operator forwards to `torch.ops.batch_invariant_ops.npu_matmul_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Batched matrix multiplication with broadcasting. For
$X \in \mathbb{R}^{\ldots \times N\times K}$ and
$W \in \mathbb{R}^{\ldots \times K\times M}$, leading batch dimensions follow
the usual broadcast rules. The reduction order over $K$ is independent of the
batch size, so each matrix result is bitwise identical whether it is computed
alone or with a larger batch. Reducing the batch dimension in the caller is
still the caller's responsibility.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[..., N, K]` | `torch.float16`, `torch.bfloat16`, `torch.float32` | Required | Left operand on NPU. |
| `other` | `torch.Tensor` | `[..., K, M]` | Same as input | Required | Right operand; leading dimensions broadcast against `input`. A transposed, non-contiguous view is accepted. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., N, M]` | Same as input | Newly allocated; inputs are not modified. |

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

a = torch.randn(8, 1, 4096, device="npu", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)

full = matmul_batch_invariant(a, b)
single = matmul_batch_invariant(a[:1], b)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
