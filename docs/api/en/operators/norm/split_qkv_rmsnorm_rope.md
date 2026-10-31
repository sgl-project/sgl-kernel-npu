# split_qkv_rmsnorm_rope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
    *,
    impl="auto",
):
    ...
```

`auto` selects the same lazily imported entry as `triton`; there is no fallback.
The implementations declare all known generations, but hardware execution remains unverified.

## Description

Split concatenated Q/K/V. If eps is not None, normalize Q/K independently per head and apply optional paired biases. Rotate the first R features of each head and preserve its remaining features. V is copied without normalization or rotation.

$$
A(x,w,b)=\begin{cases}x,&\epsilon=\mathrm{None},\\x\,\mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)w+b,&\text{otherwise},\end{cases}\quad Y_{q,k}=\mathrm{RoPE}_R(A(X_{q,k})),\quad Y_v=X_v.
$$

For each head, rotation is $y_i=x_i c_i+ho(x)_i s_i$ on its first R features; $ho$ pairs halves for NeoX and adjacent features for GPT-J. The tail is copied after normalization.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | Tensor | [N,Q+2K] | floating | required | Contiguous concatenated Q/K/V. |
| `sin` | Tensor | [N,R] or [N,1,1,R] | floating | required | Contiguous duplicated-half sine table; exactly N*R values. |
| `cos` | Tensor | [N,R] or [N,1,1,R] | floating | required | Contiguous cosine table matching sin's rotary width R. |
| `q_hidden_size` | int | — | — | required | Positive Q; divisible by K. |
| `kv_hidden_size` | int | — | — | required | Positive K; divisible by head_dim. |
| `head_dim` | int | — | — | required | Positive per-head dimension D; excluding bool. |
| `eps` | int/float/None | — | — | None | None skips normalization; otherwise a finite positive Python scalar. |
| `q_weight` | Tensor/None | at least D values | floating | None | Required when eps is set, contiguous and on input device. |
| `k_weight` | Tensor/None | at least D values | floating | None | Required when eps is set, contiguous and on input device. |
| `q_bias` | Tensor/None | at least D values | floating | None | Optional, paired with k_bias; applied only during normalization. |
| `k_bias` | Tensor/None | at least D values | floating | None | Optional, paired with q_bias; contiguous and on input device. |
| `is_neox_style` | bool | — | — | True | True pairs halves; False pairs adjacent features using first-half frequencies. |
| `impl` | str | — | — | `"auto"` | Keyword-only: auto, triton, torch_native. |

Floating means float16, bfloat16 or float32; read tensors must share the input device. Norm vectors are contiguous and may use a different floating dtype. N may be zero. Use finite participating values for accuracy comparisons. Positive integer options exclude bool.

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | New contiguous output on the input device. |
| `k` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |
| `v` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |

No input tensor, weight or bias is modified. Each backend validates metadata with its own norm/norm_utils.py; the implementations have no shared validation dependency. Triton additionally requires an NPU. Invalid metadata raises AssertionError before a launch; no device values are read back for validation.

## triton backend

Retains split_qkv_rmsnorm_rope_kernel and its per-KV-head column grid. Q/K normalization and RoPE are FP32; casts occur at output storage. No intermediate activation cast is added. Empty token batches avoid a launch.

The original script under `sgl_kernel_npu.norm` is unchanged. The checks documented here belong to the migrated implementation.

## torch_native backend

Uses independent PyTorch FP32 RMSNorm and rotary arithmetic. NeoX pairs the two halves; GPT-J pairs adjacent channels and repeats frequencies from the first half of each duplicated-half table. V is cloned, including when normalization is disabled.

No Triton kernel is called. The reference is usable on CPU as well as NPU. Reduction order, rsqrt approximation and multiply-add fusion may cause numerical differences; dtype-specific tolerance checks do not prove bitwise equivalence to the framework unfused path.

CPU arithmetic support does not change package initialization: ordinary public
imports still require torch_npu, the native library and ACL hardware discovery.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

x = torch.randn(4, 256, device='npu', dtype=torch.bfloat16)
sin = torch.zeros(4, 1, 1, 32, device='npu')
cos = torch.ones_like(sin)
wq = torch.ones(64, device='npu')
wk = torch.ones_like(wq)
args = (x, sin, cos, 128, 64, 64)
options = dict(eps=1e-6, q_weight=wq, k_weight=wk)
outputs = split_qkv_rmsnorm_rope(*args, **options)
reference = split_qkv_rmsnorm_rope(*args, impl="torch_native", **options)
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
