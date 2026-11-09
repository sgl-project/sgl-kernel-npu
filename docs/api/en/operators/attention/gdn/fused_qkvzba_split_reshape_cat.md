# fused_qkvzba_split_reshape_cat

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat

def fused_qkvzba_split_reshape_cat(
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

Splits the fused Qwen3-Next `in_proj_qkvz` / `in_proj_ba` outputs per head group into q, k, v, z, b and a, and repacks them into the layout consumed by downstream GDN attention. This is a data-reordering operation: it does not change element values, but the output order and shapes differ from the inputs.

Source rows are packed per head group, with `QK = num_heads_qk`, `V = num_heads_v`, `r = V // QK`, `dq = head_qk`, `dv = head_v`:

```text
mixed_qkvz row = [ g0: q(dq) | k(dq) | v(r*dv) | z(r*dv) ][ g1: ... ] ...   (QK groups, QK*(2*dq+2*r*dv))
mixed_ba   row = [ g0: b(r) | a(r) ][ g1: ... ] ...                         (2*V)
```

Output layout:

```text
mixed_qkv = [ all q | all k | all v ]      width 2*QK*dq + V*dv
z         = [rows, V, dv]
b, a      = [rows, V]
```

Mapping (`g ∈ [0,QK)`, `j ∈ [0,r)`, `i ∈ [0,dq)`):

```text
mixed_qkv[t, g*dq + i]             = mixed_qkvz[t, g*(2*dq+2*r*dv) + i]
mixed_qkv[t, QK*dq + g*dq + i]     = mixed_qkvz[t, g*(2*dq+2*r*dv) + dq + i]
mixed_qkv[t, 2*QK*dq + g*r*dv + m] = mixed_qkvz[t, g*(2*dq+2*r*dv) + 2*dq + m]
z[t, g*r + j]                      = mixed_qkvz[t, g*(2*dq+2*r*dv) + 2*dq + r*dv + j*dv : ... + dv]
b[t, g*r + j]                      = mixed_ba[t, g*2*r + j]
a[t, g*r + j]                      = mixed_ba[t, g*2*r + r + j]
```

## Parameters and return values

| Parameter | Type / shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- |
| `mixed_qkvz` | Tensor `[rows, num_heads_qk*(2*head_qk + 2*(num_heads_v//num_heads_qk)*head_v)]`, floating point | Required | 2-D and contiguous; the row width must match the head parameters. |
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

**Asserted constraints**: both inputs are 2-D, share the same row count, are contiguous, `num_heads_v % num_heads_qk == 0`, both row widths match the formulas above, and `rows > 0`. Checks run before the kernel launch and report the actual shape/stride.

**Caller requirements**: pass both inputs on the same NPU. Inputs are not modified; the Triton backend allocates all four outputs.

## triton backend

Requires contiguous inputs and `rows > 0`; the wrapper checks both conditions before launch. Output tensors are newly allocated on the corresponding input devices.

## torch_native backend

The independent PyTorch reference uses reshape, slicing and concatenation; like Triton, it requires contiguous inputs.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import fused_qkvzba_split_reshape_cat

QK, V, HEAD = 2, 4, 128  # Qwen3-Next TP=8 shard
rows = 1
qkvz = torch.arange(rows * QK * (2 * HEAD + 2 * (V // QK) * HEAD), dtype=torch.float32)
qkvz = qkvz.reshape(rows, -1).to("npu:0")
ba = torch.arange(rows * 2 * V, dtype=torch.float32).reshape(rows, -1).to("npu:0")

mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
    qkvz, ba, QK, V, HEAD, HEAD, impl="triton"
)
# mixed_qkv: [1, 1024]; z: [1, 4, 128]; b/a: [1, 4]
assert torch.equal(mixed_qkv[:, 128:256], qkvz[:, 768:896])  # q of g1
assert torch.equal(mixed_qkv[:, 256:384], qkvz[:, 128:256])  # k of g0
assert torch.equal(z[:, 3], qkvz[:, 1408:1536])  # z of the second v head of g1
assert torch.equal(b[:, 2], ba[:, 4])  # b0 of g1
assert torch.equal(a[:, 0], ba[:, 2])  # a0 of g0
```
