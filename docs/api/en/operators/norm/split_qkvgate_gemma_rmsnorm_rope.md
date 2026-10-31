# split_qkvgate_gemma_rmsnorm_rope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    *,
    impl="auto",
):
    ...
```

`auto` selects the same lazily imported entry as `triton`; there is no fallback.
The implementations declare all known generations, but hardware execution remains unverified.

## Description

Deinterleave the per-head Q/gate pairs, normalize Q/K per head using the Gemma multiplier 1+weight, and apply NeoX RoPE to the first R features. The input is [q0,gate0,q1,gate1,...,K,V], not a concatenation of all Q followed by all gates. Gate and V are copied without normalization, activation or rotation.

$$
G(x,w)=x\,\mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)(1+w),\qquad Y_q=\mathrm{RoPE}_R(G(Q,w_q)),\quad Y_k=\mathrm{RoPE}_R(G(K,w_k)),\quad Y_v=V,\quad Y_g=\mathrm{gate}.
$$

For each head, rotation is $y_i=x_i c_i+ho(x)_i s_i$ on its first R features; $ho$ pairs halves for NeoX and adjacent features for GPT-J. The tail is copied after normalization.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `input` | Tensor | [N,2Q+2K] | floating | required | Contiguous per-head Q/gate pairs, followed by K and V. |
| `sin` | Tensor | [N,R] or [N,1,1,R] | floating | required | Contiguous duplicated-half sine table; exactly N*R values. |
| `cos` | Tensor | [N,R] or [N,1,1,R] | floating | required | Contiguous cosine table matching sin's rotary width R. |
| `q_hidden_size` | int | — | — | required | Positive Q; divisible by K. |
| `kv_hidden_size` | int | — | — | required | Positive K; divisible by head_dim. |
| `head_dim` | int | — | — | required | Positive per-head dimension D; excluding bool. |
| `rope_dim` | int | — | — | required | Positive even R <= D; must match sin/cos width. |
| `eps` | int/float | — | — | required | Finite positive Python scalar; normalization is mandatory. |
| `q_weight` | Tensor | at least D values | floating | required | Raw Gemma offset; multiplier is 1+q_weight. |
| `k_weight` | Tensor | at least D values | floating | required | Raw Gemma offset; multiplier is 1+k_weight. |
| `impl` | str | — | — | `"auto"` | Keyword-only: auto, triton, torch_native. |

Floating means float16, bfloat16 or float32; read tensors must share the input device. Norm vectors are contiguous and may use a different floating dtype. N may be zero. Use finite participating values for accuracy comparisons. Positive integer options exclude bool.

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `q` | Tensor | [N,Q] | input.dtype | New contiguous output on the input device. |
| `k` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |
| `v` | Tensor | [N,K] | input.dtype | New contiguous output on the input device. |
| `gate` | Tensor | [N,Q] | input.dtype | New contiguous output on the input device. |

No input tensor, weight or bias is modified. Each backend validates metadata with its own norm/norm_utils.py; the implementations have no shared validation dependency. Triton additionally requires an NPU. Invalid metadata raises AssertionError before a launch; no device values are read back for validation.

## triton backend

Retains split_qkvgate_gemma_rmsnorm_rope_kernel. head_dim must be a power of two, as in the original wrapper. Each column handles one KV head and its associated Q/gate group. The required model cases use partial RoPE with D=256 and R=64. Empty token batches avoid a launch.

The original script under `sgl_kernel_npu.norm` is unchanged. The checks documented here belong to the migrated implementation.

## torch_native backend

Reshapes Q/gate to [N,Hq,2,D], then uses unbind to obtain the two independent head streams. FP32 normalization adds 1 to the raw weight exactly once. FP32 NeoX rotation preserves the non-rotary tail; V/gate are cloned into contiguous outputs.

No Triton kernel is called. The reference is usable on CPU as well as NPU. Reduction order, rsqrt approximation and multiply-add fusion may cause numerical differences; dtype-specific tolerance checks do not prove bitwise equivalence to the framework unfused path.

CPU arithmetic support does not change package initialization: ordinary public
imports still require torch_npu, the native library and ACL hardware discovery.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkvgate_gemma_rmsnorm_rope

x = torch.randn(4, 384, device='npu', dtype=torch.bfloat16)
sin = torch.zeros(4, 1, 1, 32, device='npu')
cos = torch.ones_like(sin)
wq = torch.zeros(64, device='npu')
wk = torch.zeros_like(wq)
args = (x, sin, cos, 128, 64, 64, 32, 1e-6, wq, wk)
outputs = split_qkvgate_gemma_rmsnorm_rope(*args)
reference = split_qkvgate_gemma_rmsnorm_rope(*args, impl="torch_native")
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
