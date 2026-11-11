# qsa_sparse_attention

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.attention import qsa_sparse_attention

def qsa_sparse_attention(q, k, v, slots, softmax_scale=None, *, impl="auto"):
    ...
```

## Description

Sparse GQA over physical slots[R,S]. BF16 Q[R,Hq,256], K/V[N,Hkv,256]; local heads 24:2,12:1,6:1,3:1. Returns contiguous BF16 shaped as Q; all-padding rows are zero. auto selects the Triton/native hybrid; zero scale retains the legacy default 1/16.

Slot width S=(B+1)*c-1+t, B=512/2048, c>=2, with t=0 or 3<=t<=c. Nonnegative slots must be below N; duplicate slots retain duplicate contributions. Q and slots may have row gaps, while K/V must be contiguous pool views. All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

For each query head, select its GQA key/value head and the keys named by slots, then compute `softmax(scale * QK.T) @ V`. The independent oracle computes selected-key attention in CPU FP64 and rounds directly to BF16.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [R,Hq,256], BF16 | required | NPU; last dimension contiguous; row/head gaps allowed. |
| `k / v` | Tensor | [N,Hkv,256], BF16 | required | Contiguous pools on q.device; storage offsets allowed. |
| `slots` | Tensor | [R,S], int32 | required | Same NPU, contiguous columns; negative means padding; nonnegative values <N. |
| `softmax_scale` | float or None | — | None | None and numeric zero select 1/16. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | q.shape | BF16 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU FP64 reference; returns results to the input device. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import qsa_sparse_attention

q = torch.randn(1, 3, 256, device="npu", dtype=torch.bfloat16)
k = torch.randn(16, 1, 256, device="npu", dtype=q.dtype)
v = torch.randn_like(k)
slots = torch.full((1, 2051), -1, device="npu", dtype=torch.int32)
slots[:, :16] = torch.arange(16, device="npu", dtype=torch.int32)
y = qsa_sparse_attention(q, k, v, slots, impl="auto")
```
