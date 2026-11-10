# mega_chunk_gdn

## Hardware support

| Implementation | ARCH22 | ARCH35 |
| --- | --- | --- |
| `ascendc` (also `auto`) | Declared; CANN build/device validation pending | Unsupported |
| `torch_native` | Reference; device validation pending | Reference; device validation pending |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

def mega_chunk_gdn(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, *, impl="auto",
):
    ...
```

`auto` selects `ascendc`, without fallback. The implementation uses the existing
PTO computation through the standalone CANN/ACLNN build; it introduces no external
Python dependency. No support extension to ARCH35 is included.

## Description

Fuse chunk-local cumsum, scaled KKT, triangular inverse, WY recomputation,
state update and output reconstruction. The original PTO stage sources and
Python helper are copied unchanged. Chunks contain 128 tokens; head dimension
is 128. Each H/Hg value heads shares a Q/K head.

The mathematical recurrence is
`S_t = exp(g_t) * S_(t-1) + beta_t * k_t ⊗ (v_t - k_t^T exp(g_t) S_(t-1))`,
`o_t = scale * q_t^T S_t`. Supply nonpositive log decay and beta in (0,1].
The native implementation preserves its original FP16 rounding and exponential
clamping. It does not add Q/K normalization.

## Parameters and return value

| Parameter | Type / shape | Default | Meaning |
| --- | --- | --- | --- |
| `q`, `k` | Tensor `[1,T,Hg,128]` | Required | Contiguous FP16/BF16 NPU inputs; converted to FP16. |
| `v` | Tensor `[1,T,H,128]` | Required | Same device; converted to FP16. H in {8,12,16,24,32,48,64}, divisible by Hg. |
| `g` | FP32 Tensor `[1,T,H]` | Required | Contiguous per-token log decay. |
| `beta` | Tensor `[1,T,H]` | Required | Contiguous update weights; converted to FP16. |
| `scale` | float or None | None | None means `128**-0.5`; preserve explicit zero. |
| `initial_state` | Tensor `[N,H,128,128]` or None | None | Converted to FP16; None means zero state. |
| `output_final_state` | bool | True | Return final state, or None. |
| `cu_seqlens` | Tensor `[N+1]` or None | None | Contiguous boundaries converted to INT32; first=0, last=T, positive lengths. None means [0,T]. |
| `impl` | str | `"auto"` | `auto`, `ascendc` or `torch_native`. |

All tensors must be on the same NPU and contiguous; this helper does not add an
input guard. Inputs are not modified. FP16 native results are converted back to
the original dtypes at the same points as the legacy helper.

| Return | Shape / dtype | Meaning |
| --- | --- | --- |
| `g_sum` | `[1,T,H]`, FP32 | Log decay summed locally within each chunk. |
| `o` | `[1,T,H,128]`, q.dtype | Output; scale is applied in FP16 before dtype restoration. |
| `A_inv` | `[1,T,H,128]`, k.dtype | Inverse chunk matrices packed by token/head. |
| `final_state` | `[N,H,128,128]`, FP32 or None | Final states, rounded through native FP16. |
| `w` | `[1,T,H,128]`, k.dtype | WY key transform. |
| `h` | `[1,total_chunks,H,128,128]`, k.dtype | Entering chunk states in sequence/chunk order. |
| `v_new` | `[1,T,H,128]`, v.dtype | Residual values after state projection. |

## Ascend C implementation

Build the repository-owned native operator independently of the legacy build:

```bash
git submodule update --init third_party/pto-isa
python3 setup.py --arch-version 2201 --ops mega_chunk_gdn --without-third-party
```

This uses the already pinned PTO-ISA headers. Separate OpDef, inference, Tiling,
Kernel entry and Torch adapter follow the existing attention-domain CANN operators.
The original scalar launch metadata is serialized in Tiling; block count and
PTO dispatch stay unchanged. Same-name CANN Ref ports retain all 19 mutable
buffers, including the helper's initial zeros and scratch buffers. The copied
computation includes cross-core synchronization; full SDK compilation, launch,
stream/graph behavior and device accuracy still require ARCH22 verification.
The existing `torch.ops.npu.mega_chunk_gdn` schema is preserved in the standalone
adapter; legacy and standalone wheels remain alternative installations.

## PyTorch reference

`torch_native` computes independent CPU FP64 chunk algebra after the same entry
FP16 casts, then restores output dtypes/device. It is a correctness reference,
not a performance backend or bitwise emulation of PTO instructions. Tests keep
the legacy maximum-absolute-or-relative-RMS thresholds. The reference is also
checked against an independent tokenwise recurrence.

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
cu = torch.tensor([0, 22, 129], device="npu", dtype=torch.int32)
g_sum, o, A, final_state, w, h, v_new = mega_chunk_gdn(
    q, k, v, g, beta, cu_seqlens=cu, impl="ascendc",
)
```
