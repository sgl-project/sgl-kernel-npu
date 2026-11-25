# chunk_fwd_o

## Hardware support

| Implementation | ARCH22 | ARCH35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64, *, impl="auto"):
    ...
```

`auto` selects `triton`.

## Description

Reconstruct Gated DeltaNet outputs after the chunk state update. For each chunk,
combine queries projected through its entering state with causal within-chunk
attention over the transformed values `v`:

$$
O_c = s\left[\operatorname{diag}(e^{G_c})Q_cH_c
       +\operatorname{tril}\left((Q_cK_c^T)\odot E_c\right)V_c\right],
\quad E_{ij}=\begin{cases}e^{G_i-G_j}&G_i\le G_j\\0&\text{otherwise}.\end{cases}
$$

`g` is already chunk-local cumulative log decay. `h` contains entering states,
not the final state or the initial state alone. `v` is the value residual from
the state-update stage, not necessarily the model's original V projection.
Each group of `H/Hg` value heads shares a query/key head. The local attention
matrix is converted to v.dtype before its value product; outputs use v.dtype.

Packed sequences restart chunk numbering at each boundary and store entering
states in sequence/chunk order. For equal-length B>1, gate addressing follows
the existing flattened `[B,H,T]` offset `b*T + head*T + token`; use packed B=1
sequences when supplying independent per-sequence gates.

## Parameters and return value

| Parameter | Type / shape | Dtype / default | Description and constraints |
| --- | --- | --- | --- |
| `q`, `k` | Tensor `[B,T,Hg,K]` | FP16/BF16/FP32; required | Contiguous NPU queries/keys with matching dtype/device. |
| `v` | Tensor `[B,T,H,V]` | q.dtype; required | Contiguous transformed values; H must be a multiple of Hg. |
| `h` | Tensor `[B,ceil(T/C),H,K,V]` | q.dtype; required | Contiguous entering chunk states. Packed input uses `[1,total_chunks,H,K,V]`. |
| `g` | Tensor `[B,T,H]` | Floating point, usually FP32; `None` | Chunk-local cumulative log decay on q's device. A Tensor is required in practice: passing None raises AttributeError. |
| `scale` | float or `None` | `None` | Output multiplier, defaults to `K**-0.5`. |
| `cu_seqlens` | Tensor `[N+1]` or `None` | int32/int64; `None` | Packed B=1 boundaries, starting at 0 and ending at T; positive sequence lengths. |
| `chunk_size` | int | `64` | Positive power-of-two C for Triton. |
| `impl` | str | `"auto"` | `auto`, `triton`, `torch_native`. |

Inputs are not modified. Supply finite values and enough entering states for
all chunks. Triton caches packed metadata; do not mutate cu_seqlens in place.

| Return | Type / shape | Dtype / device | Description |
| --- | --- | --- | --- |
| Output | Tensor `[B,T,H,V]` | v.dtype / v.device | Newly allocated reconstructed output. |

## triton backend

Supports equal-length and packed inputs, grouped query/key heads, and feature
and sequence tails. Both input forms retain the caller-provided chunk boundaries.

## torch_native backend

Computes the entering-state and causal local terms using PyTorch matrix products
in FP32, with the local attention conversion to v.dtype preserved. Packed boundary
reads synchronize when cu_seqlens is on the NPU. No state update is performed.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

q = torch.randn(1, 65, 2, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(1, 65, 4, 128, device="npu", dtype=q.dtype)
h = torch.zeros(1, 2, 4, 128, 128, device="npu", dtype=q.dtype)
g = torch.zeros(1, 65, 4, device="npu", dtype=torch.float32)
o = chunk_fwd_o(q, k, v, h, g=g, impl="triton")
```
