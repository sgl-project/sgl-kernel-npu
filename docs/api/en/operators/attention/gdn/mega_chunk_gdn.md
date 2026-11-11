# mega_chunk_gdn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

Omitting `impl` or passing `"auto"` selects `"ascendc"` without automatic fallback.
On ARCH35, use `"torch_native"` explicitly for reference computation; the native
mega implementation is not supported.

```python
from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

def mega_chunk_gdn(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, *, impl="auto",
):
    ...
```

## Description

Computes the gated delta rule forward pass with 128-token chunks and a head
dimension of 128. The native PTO kernel fuses chunk-local cumulative sum, scaled
KKT, triangular inverse, WY recomputation, state update and output reconstruction.
It returns attention output, final states and intermediate results.

Each group of `H/Hg` value heads shares one query/key head. Packed sequences
maintain independent states. Supply nonpositive log decay in `g` and update
weights in `(0,1]` in `beta`. The operator does not normalize Q/K.

Q/K/V, beta and initial states are converted to FP16 before native computation.
Results are restored to the dtypes described below; final states are returned
as FP32 after FP16 rounding. `scale=None` uses `128**-0.5`; an explicit zero
scale is preserved.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q`, `k` | `torch.Tensor` | `[1,T,Hg,128]` | `torch.float16`, `torch.bfloat16` | Required | Query and key tensors; converted to FP16. |
| `v` | `torch.Tensor` | `[1,T,H,128]` | `torch.float16`, `torch.bfloat16` | Required | Value tensor; `H` must be one of 8, 12, 16, 24, 32, 48, 64 and divisible by `Hg`. |
| `g` | `torch.Tensor` | `[1,T,H]` | `torch.float32` | Required | Per-token log decay, before cumulative summation. |
| `beta` | `torch.Tensor` | `[1,T,H]` | `torch.float16`, `torch.bfloat16`, `torch.float32` | Required | State-update weights; converted to FP16. |
| `scale` | `float` or `None` | — | — | `None` | Output scale; `None` uses `128**-0.5`. |
| `initial_state` | `torch.Tensor` or `None` | `[N,H,128,128]` | `torch.float16`, `torch.bfloat16`, `torch.float32` | `None` | Initial states; converted to FP16. `None` starts from zero. |
| `output_final_state` | `bool` | — | — | `True` | Whether to return final states. |
| `cu_seqlens` | `torch.Tensor` or `None` | `[N+1]` | `torch.int32`, `torch.int64` | `None` | Packed sequence boundaries, converted to INT32: first value 0, last value `T`, positive sequence lengths. `None` means one sequence. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `g_sum` | `torch.Tensor` | `[1,T,H]` | `torch.float32` | Log decay summed locally within each chunk. |
| `o` | `torch.Tensor` | `[1,T,H,128]` | `q.dtype` | Attention output; scale is applied in FP16 before dtype restoration. |
| `A_inv` | `torch.Tensor` | `[1,T,H,128]` | `k.dtype` | Inverse chunk matrices packed by token and head. |
| `final_state` | `torch.Tensor` or `None` | `[N,H,128,128]` | `torch.float32` | Final states after FP16 rounding; `None` when `output_final_state=False`. |
| `w` | `torch.Tensor` | `[1,T,H,128]` | `k.dtype` | WY key transform. |
| `h` | `torch.Tensor` | `[1,NT,H,128,128]` | `k.dtype` | States entering each chunk, ordered by sequence and chunk. |
| `v_new` | `torch.Tensor` | `[1,T,H,128]` | `v.dtype` | Residual values after state projection. |

`N` is the number of sequences; without `cu_seqlens`, `N=1`. `NT` is the sum
of `ceil(sequence_length / 128)` over sequences. Outputs are on `q.device`;
inputs are not modified. Native inputs must be contiguous and on the same NPU.
These requirements are caller preconditions; only some are explicitly validated.

## ascendc backend

- Uses the fused PTO computation through the CANN/ACLNN operator on ARCH22.
- Requires a wheel built with the native `mega_chunk_gdn` operator. Source builds
  must initialize the pinned `third_party/pto-isa` submodule before compilation.
- Preserves FP16 computation, intermediate rounding and exponential clamping.
  Inputs are not automatically made contiguous; select the NPU holding them
  before calling this interface directly.

## torch_native backend

- Uses independent chunk algebra on CPU in FP64 after the same entry FP16 casts,
  then restores output dtypes and device.
- Provides a correctness reference on both supported architectures; computation
  runs on CPU even when the inputs and outputs are on an NPU.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

q = torch.randn(1, 129, 4, 128, device="npu", dtype=torch.float16)
k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
v = torch.randn(1, 129, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 129, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 129, 8, device="npu", dtype=q.dtype) * 0.5 + 0.5
cu_seqlens = torch.tensor([0, 22, 129], device="npu", dtype=torch.int32)
g_sum, o, A_inv, final_state, w, h, v_new = mega_chunk_gdn(
    q, k, v, g, beta, cu_seqlens=cu_seqlens,
)
```
