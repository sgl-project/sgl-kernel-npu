# solve_tril

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import solve_tril

def solve_tril(A, cu_seqlens=None, output_dtype=torch.float, *, impl="auto"):
    ...
```

## Description

Compute the inverse of a unit lower triangular matrix for each sequence chunk
and head:

$$
R_c=(I_C+A_c)^{-1},\qquad (I_C+A_c)R_c=I_C.
$$

$A_c$ is a strictly lower triangular $C\times C$ block and $I_C$ is identity.
The input packs consecutive block rows in `[B, T, H, C]`; its last dimension is
the chunk width, not the key feature dimension. The final partial chunk is
extended with zero rows before inversion and only valid rows are returned.
With `cu_seqlens`, chunk numbering restarts at each sequence boundary.

This is the triangular inverse stage of chunked Gated DeltaNet linear attention,
used between `chunk_scaled_dot_kkt_fwd` and `recompute_w_u_fwd` in
`sgl_kernel_npu.fla.chunk`. SGLang's NPU `TritonGDNKernel.extend` calls this
chunk pipeline for GDN prefill/extend, including Qwen3-Next. For a chunk,
$A_{ij}=\beta_i\exp(G_i-G_j)k_i^\top k_j$ for $i>j$, where $G$ is the cumulative
log decay. The inverse resolves dependencies between delta-rule updates.
The separate recurrent decode path does not require this inverse.
`GDN_USE_MEGA_GDN=1` bypasses the independent kernels in the chunk pipeline.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `A` | `torch.Tensor` | `[B, T, H, C]` | FP16/BF16/FP32 | Required | Nonempty, `C>0`; strictly lower triangular per chunk, finite values. Triton requires contiguous NPU input with `C` in `{16, 32, 64}`; torch_native accepts other positive chunk sizes and noncontiguous input. |
| `cu_seqlens` | `torch.Tensor` or `None` | `[N+1]` | int32/int64 | `None` | Packed input requires `B=1`. Triton requires contiguous boundaries on A's device; torch_native also accepts CPU and noncontiguous boundaries. Starts at 0, ends at T, strictly increasing (no empty sequences). None means B equal-length sequences. |
| `output_dtype` | `torch.dtype` | — | FP16/BF16/FP32 | `torch.float32` | Output conversion after FP32 computation. |
| `impl` | `str` | — | — | `"auto"` | Keyword only: `auto`, `triton`, `torch_native`; `auto` defaults to Triton. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `R` | `torch.Tensor` | Same as A | `output_dtype` | Newly allocated on A's device; inputs are not modified. |

Usage notes:

1. Both backends check properties such as shape and dtype. Only Triton requires contiguous input.
2. The caller must ensure that blocks are strictly lower triangular and values are finite; neither backend checks these element by element. Triton also relies on the caller to supply correct sequence boundaries. torch_native checks that boundaries start at 0, end at T, and strictly increase.
3. Triton caches chunk indices, so modifying `cu_seqlens` in place can cause stale indices to be reused. torch_native does not use this cache.
4. Ill-conditioned matrices can produce large numerical errors.

## triton backend

First inverts 16×16
diagonal blocks in FP32, then merges them into 32×32 or 64×64 inverses using
block matrix multiplication. Converts output with round-to-nearest-even.
Requires contiguous NPU inputs.

## torch_native backend

Independent PyTorch implementation using FP32 `torch.linalg.solve_triangular`
with an identity right-hand side, followed by conversion to `output_dtype`.
Handles each sequence and pads partial chunks independently. Supports any
positive chunk size, noncontiguous input, and CPU or noncontiguous boundaries.
Empty input remains unsupported. Input and output types are FP16, BF16 or FP32;
computation always uses FP32. After converting boundaries to a Python list,
checks the start, end and increasing order. Reading NPU boundaries with
`tolist()` synchronizes the device.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import solve_tril

A = (0.01 * torch.randn(1, 2, 64, 64, device="npu")).tril(-1)
# Two chunks, one head: [B, chunks, rows, columns] -> [B, T, H, C].
A = A.reshape(1, 128, 1, 64).contiguous()
R = solve_tril(A, output_dtype=torch.float32)
R_native = solve_tril(A, impl="torch_native")
```
