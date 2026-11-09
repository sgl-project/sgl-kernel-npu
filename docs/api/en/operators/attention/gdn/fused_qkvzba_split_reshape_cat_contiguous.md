# fused_qkvzba_split_reshape_cat_contiguous

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat_contiguous

def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    *,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

`auto` and `triton` resolve to the same implementation; `torch_native` is an independent Torch reference. Backends are imported lazily and there is no automatic fallback.

## Description

Splits Qwen3.5's segment-wise projections `[all q | all k | all v | all z]` and `[all b | all a]` into q/k/v, z, b and a. The output layout is identical to `fused_qkvzba_split_reshape_cat`; only the input row layout differs (no per-head-group interleaving here).

With `QK = num_heads_qk`, `V = num_heads_v`, `dq = head_qk`, `dv = head_v`:

```text
mixed_qkvz row = [ all q (QK*dq) | all k (QK*dq) | all v (V*dv) | all z (V*dv) ]
mixed_ba   row = [ all b (V) | all a (V) ]
```

Output layout:

```text
mixed_qkv = [ all q | all k | all v ]      width 2*QK*dq + V*dv
z         = [rows, V, dv]
b, a      = [rows, V]
```

Mapping:

```text
mixed_qkv[t, 0 : 2*QK*dq + V*dv] = mixed_qkvz[t, 0 : 2*QK*dq + V*dv]
z[t, h]                          = mixed_qkvz[t, 2*QK*dq + V*dv + h*dv : ... + dv]
b[t, h]                          = mixed_ba[t, h]
a[t, h]                          = mixed_ba[t, V + h]
```

This is a data-reordering operation: it does not change element values, but the output shapes differ from the inputs.

## Parameters and return values

| Parameter | Type / shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- |
| `mixed_qkvz` | Tensor `[rows, 2*num_heads_qk*head_qk + 2*num_heads_v*head_v]`, floating point | Required | 2-D and contiguous; the row width must equal `2*num_heads_qk*head_qk + 2*num_heads_v*head_v`. |
| `mixed_ba` | Tensor `[rows, 2*num_heads_v]`, floating point | Required | 2-D and contiguous; may use a different dtype than `mixed_qkvz`. |
| `num_heads_qk` | `int` | Required | Number of q/k heads; must be positive. |
| `num_heads_v` | `int` | Required | Number of v heads; must be positive and divisible by `num_heads_qk`. |
| `head_qk` | `int` | Required | Per-head q/k dimension; must be positive. |
| `head_v` | `int` | Required | Per-head v/z dimension; must be positive. |
| `impl` | `str` | `"auto"` | Keyword-only; `auto`, `triton` or `torch_native`. |

| Return value | Shape / dtype | Meaning |
| --- | --- | --- |
| `mixed_qkv` | `[rows, 2*num_heads_qk*head_qk + num_heads_v*head_v]`, `mixed_qkvz.dtype` | Concatenated q/k/v. |
| `z` | `[rows, num_heads_v, head_v]`, `mixed_qkvz.dtype` | z expanded per head. |
| `b` | `[rows, num_heads_v]`, `mixed_ba.dtype` | b. |
| `a` | `[rows, num_heads_v]`, `mixed_ba.dtype` | a. |

**Asserted constraints**: both inputs are 2-D, share the same row count, are contiguous, `num_heads_v % num_heads_qk == 0`, and both row widths match the formulas above. Checks run before the kernel launch and report the actual shape/stride.

**Caller requirements**: pass both inputs on the same NPU. Inputs are not modified. Triton allocates all outputs; the torch_native `z` result is a view of `mixed_qkvz`, while the other outputs are materialized.

## triton backend

Requires contiguous inputs; the wrapper checks this before launch. Empty batches are supported and return correctly shaped empty tensors: `[0, width]`, `[0, V, dv]`, `[0, V]` and `[0, V]`.

## torch_native backend

The independent PyTorch reference uses column slicing, reshape and concatenation. Its `z` result is a view into `mixed_qkvz`; `mixed_qkv`, `b` and `a` are materialized outputs.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat_contiguous

QK, V, HEAD = 2, 4, 128  # Qwen3.5 TP=8 shard
rows = 1
width = 2 * QK * HEAD + 2 * V * HEAD
qkvz = torch.arange(rows * width, dtype=torch.float32).reshape(rows, -1).to("npu:0")
ba = torch.arange(rows * 2 * V, dtype=torch.float32).reshape(rows, -1).to("npu:0")

mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
    qkvz, ba, QK, V, HEAD, HEAD, impl="triton"
)
# mixed_qkv: [1, 1024]; z: [1, 4, 128]; b/a: [1, 4]
assert torch.equal(mixed_qkv, qkvz[:, : 2 * QK * HEAD + V * HEAD])
assert torch.equal(z, qkvz[:, 2 * QK * HEAD + V * HEAD :].reshape(rows, V, HEAD))
assert torch.equal(b, ba[:, :V])
assert torch.equal(a, ba[:, V:])
```
