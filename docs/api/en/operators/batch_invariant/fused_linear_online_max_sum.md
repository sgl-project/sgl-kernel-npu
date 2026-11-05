# fused_linear_online_max_sum

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_linear_online_max_sum_batch_invariant,
)

def fused_linear_online_max_sum_batch_invariant(
    input,
    weight,
    target,
    vocab_start_index,
    vocab_end_index,
    return_logits=False,
    *,
    impl="auto",
):
    ...
```

The operator forwards to
`torch.ops.batch_invariant_ops.npu_fused_linear_online_max_sum_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Computes this rank's logits for a vocabulary-parallel linear layer and the
online max/sum statistics used by the cross entropy. For
$X \in \mathbb{R}^{M\times K}$, local weight $W \in \mathbb{R}^{N\times K}$:

$$
L = XW^{\mathsf{T}}, \qquad
m_i = \max_j L_{ij}, \qquad
s_i = \sum_j e^{L_{ij}}
$$

The reduction order does not depend on $M$, so each token's statistics are
bitwise identical whether the token is processed alone or inside a larger
batch. The kernel follows its own online formulation, so the statistics are
meant to be consumed as a pair and are not required to match a naive
`max`/`exp` reference bitwise.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[M, K]` | `torch.float16`, `torch.bfloat16` | Required | Hidden states on NPU. |
| `weight` | `torch.Tensor` | `[N, K]` | Same as `input` | Required | Local vocabulary slice, kept in `[N, K]` layout. |
| `target` | `torch.Tensor` | `[M]` | Integer | Required | Target token ids in the global vocabulary. |
| `vocab_start_index` | `int` | — | — | Required | Global index of the first local logit column. |
| `vocab_end_index` | `int` | — | — | Required | Global index one past the last local logit column. |
| `return_logits` | `bool` | — | — | `False` | Also return the local logits. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `logits_max` | `torch.Tensor` | `[M]` | `torch.float32` | Row-wise logits maximum. |
| `sum_exp_logits` | `torch.Tensor` | `[M]` | `torch.float32` | Row-wise sum of exponentials. |
| `predicted_logits` | `torch.Tensor` | `[M]` | `torch.float32` | Logits at the target positions. |
| `target_mask` | `torch.Tensor` | `[(M + 7) // 8]` | `torch.uint8` | Bit-packed per-token mask; eight tokens per byte. |
| `masked_target` | `torch.Tensor` | `[M]` | Same as `target` | Local target index used by the loss. |
| `vocab_parallel_logits` | `torch.Tensor` or undefined | `[M, N]` | Same as `input` | Local logits when `return_logits` is True; otherwise an undefined tensor. |

Inputs are not modified.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import (
    fused_linear_online_max_sum_batch_invariant,
)

hidden = torch.randn(257, 7168, device="npu", dtype=torch.bfloat16)
weight = torch.randn(8192, 7168, device="npu", dtype=torch.bfloat16)
target = torch.randint(0, 32768, (257,), device="npu", dtype=torch.int64)

full = fused_linear_online_max_sum_batch_invariant(
    hidden, weight, target, 4096, 12288, True
)
single = fused_linear_online_max_sum_batch_invariant(
    hidden[:1], weight, target[:1], 4096, 12288, True
)
for full_out, single_out in zip(full, single):
    torch.testing.assert_close(full_out[:1], single_out, atol=0, rtol=0)
```
