# softmax

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import softmax_batch_invariant

def softmax_batch_invariant(input, dim, *, impl="auto"):
    ...
```

The operator forwards to `torch.ops.batch_invariant_ops.npu_softmax_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Softmax along one dimension, typically the vocabulary dimension of logits. The
reduction order does not depend on how many rows are present, so a row's
probabilities are bitwise identical whether it is normalized alone or inside a
larger batch.

The kernel only accepts the **last dimension**. Passing any other `dim` raises
an aclnn parameter error.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[..., V]` | `torch.float16`, `torch.bfloat16`, `torch.float32` | Required | Input on NPU. |
| `dim` | `int` | — | — | Required | Must be the last dimension (`-1` or `input.dim() - 1`). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | Same as input | Same as input | Newly allocated; inputs are not modified. |

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import softmax_batch_invariant

logits = torch.randn(257, 16384, device="npu", dtype=torch.bfloat16)

full = softmax_batch_invariant(logits, -1)
single = softmax_batch_invariant(logits[:1], -1)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
