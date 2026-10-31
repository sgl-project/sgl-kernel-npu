# recompute_w_u_fwd

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

def recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None, *, impl="auto"):
    ...
```

## Description

Compute the Gated DeltaNet WY representation for each sequence, chunk and value head:

$$
W_c=A_c\operatorname{diag}(\beta_c\odot\exp(G_c))K_c^{\uparrow},
\qquad
U_c=A_c\operatorname{diag}(\beta_c)V_c.
$$

$A_c$ is the inverse returned by `solve_tril`, rather than its strictly lower
triangular input. $G_c$ is `g_cumsum`, the already accumulated chunk-local log
decay; this operator does not accumulate it again. $\beta_c$ contains update
weights; $\odot$ and $\exp$ are elementwise.
$K_c^{\uparrow}$ reuses keys by head group: value head $h$ uses key head
$\lfloor h/(H/H_g)\rfloor$. Callers do not need to repeat keys beforehand.

Sequences are split into chunks of width C; only valid rows of the final chunk
are returned. Chunk numbering restarts at each `cu_seqlens` boundary.
This stage follows triangular inversion and precedes chunk state updates in GDN.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `k` | `torch.Tensor` | `[B,T,Hg,K]` | FP16/BF16/FP32 | Required | Nonempty keys. Triton requires contiguous NPU input. |
| `v` | `torch.Tensor` | `[B,T,H,V]` | FP16/BF16/FP32 | Required | Nonempty values; H must be a multiple of Hg. Triton requires contiguous input. |
| `beta` | `torch.Tensor` | `[B,T,H]` | FP16/BF16/FP32 | Required | Per-token, per-value-head update weights. |
| `g_cumsum` | `torch.Tensor` | `[B,T,H]` | FP16/BF16/FP32 | Required | Chunk-local cumulative log decay, typically FP32. |
| `A` | `torch.Tensor` | `[B,T,H,C]` | FP16/BF16/FP32 | Required | Nonempty inverse blocks from `solve_tril`. Triton requires contiguous input and C in {16,32,64}; torch_native accepts any positive integer C. |
| `cu_seqlens` | `torch.Tensor` or `None` | `[N+1]` | int32/int64 | `None` | Boundaries start at 0, end at T and strictly increase; packed input requires B=1. Triton requires contiguous boundaries on k's device; torch_native also accepts CPU or noncontiguous boundaries. |
| `impl` | `str` | — | — | `"auto"` | Keyword only: `auto`, `triton`, `torch_native`; `auto` defaults to Triton. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `w` | `torch.Tensor` | `[B,T,H,K]` | k.dtype | Newly allocated on k's device. |
| `u` | `torch.Tensor` | `[B,T,H,V]` | v.dtype | Newly allocated on v's device; return order is `(w, u)`. |

`k`, `v`, `beta`, `g_cumsum` and `A` must share a device and are not modified.
Callers must supply finite values and avoid overflow in `exp(g_cumsum)`.
Triton does not check boundary values; torch_native checks the start, end and
increasing order. Triton caches chunk indices, so do not modify `cu_seqlens`
in place after use.

## triton backend

Converts beta and cumulative decay to contiguous `[B,H,T]` layout before launching
one program per chunk and batch. Computes in FP32, tiling K and V by 128 elements;
supports feature tails and partial sequence chunks. Handles equal-length batches,
packed sequences and grouped key/value heads. Converts outputs to k/v dtypes.

## torch_native backend

Independent PyTorch exponentiation, elementwise multiplication and matrix
multiplication in FP32. Supports noncontiguous tensors and any positive chunk
size. Converts packed boundaries to a Python list and validates them;
`tolist()` synchronizes when boundaries are on the NPU.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
v = torch.randn(1, 64, 4, 128, device="npu", dtype=torch.bfloat16)
beta = torch.rand(1, 64, 4, device="npu")
g_cumsum = -torch.rand(1, 64, 4, device="npu").cumsum(1)
# Identity is a valid solve_tril output; each head uses its own matrix.
A = torch.eye(64, device="npu", dtype=torch.bfloat16)[None, :, None, :]
A = A.expand(1, 64, 4, 64).contiguous()
w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A)
```
