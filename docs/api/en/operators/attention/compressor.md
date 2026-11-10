# compressor

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import compressor

compressor(
    x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    rope_head_dim=64, cmp_ratio=4, coff=1, norm_eps=1e-6,
    rotary_mode=2, cache_mode=1, state_cache_stride_dim0=0, *, impl="auto",
)
```

`auto` selects `ascendc` without fallback. `rotary_mode` defaults to `2`
(interleaved), the only compiled mode. The legacy default was `1`; omitting
this argument now selects mode 2. Explicit mode 1 is rejected before launch.

Mode 1 has an intentional error-handling difference: for otherwise valid,
nonempty inputs, the legacy entry could reach a device `Trap` because it had no
matching computation branch; the new entry rejects the call before launch.
For empty inputs that passed the old checks, the legacy path could return
without computation even with mode 1. The new entry rejects mode 1 for empty
inputs too. Mode 2 computation is unchanged.

## Description

DeepSeek-V4 KV compression combines projected tokens over groups, applies
RMSNorm and interleaved RoPE, and updates the KV/score state cache in place.
Let `r=cmp_ratio`, `c=coff`, and `D=len(norm_weight)`. In FP32:

\[
 K_t=x_t W_{kv}^{T},\qquad
 G_t=x_t W_{gate}^{T}+APE_{p_t\bmod r}.
\]

For a completed group ending at absolute position `e` (exclusive, a multiple
of `r`), overlap lane `j=0,...,c-1` uses positions
`e-(c-j)r,...,e-(c-j-1)r-1` and feature slice `jD:(j+1)D`.
For each feature, softmax over these `c*r` scores weights the corresponding
keys. Positions before zero contribute zero keys and negative-infinite scores;
positions before this call's start come from the state cache.

\[
 u_d=\sum_{j,t}\operatorname{softmax}_{j,t}(G_{t,jD+d})K_{t,jD+d},\quad
 v_d=\gamma_d u_d/\sqrt{D^{-1}\sum_i u_i^2+\epsilon}.
\]

Interleaved RoPE rotates adjacent pairs in the last 64 features using the
provided sine/cosine values, then casts the output to `x.dtype`.
For request start `p` and used length `n`, exactly
`floor((p+n)/r)-floor(p/r)` output rows are valid. BSH stores them at the
beginning of each request's output; TH packs all requests' valid rows together.
Unused output rows are unspecified and must not be read.

Paged mode saves new KV/score rows except writes to dummy page 0. Mode 2
saves a suffix of the current raw tokens. Let `E=p+n`, `q=floor(E/r)*r`, and
`keep=max(q-(c-1)*r,0)`. The arch22 and arch35 NORMAL implementations also
retain verification tail rows: when `L>c*r`, set `pad=L-c*r+2` and lower
`keep` to `min(keep,max(E-pad,0))`. The arch35 short-BSH FULL_LOAD path
(`S<=4` and `B*S<=256`) uses the original `keep` without this pad.
New rows with absolute position at least `keep` are written, including a
partial compression chunk. In arch35 NORMAL mode with overlap, earlier page
tails are also written so a later call can resume from a cached SWA page.

## Parameters and return value

All tensors must be contiguous and on the same NPU. `H` is a multiple of 512
in `[1024,10240]`, and `D` is 128 or 512. Define `W=c*D`,
`Sr=ceil(S/r)` for BSH and `Sr=min(T,T//r+B)` for TH.

| Parameter | Type | Meaning and constraints |
| --- | --- | --- |
| `x` | Tensor | FP16/BF16 `[B,S,H]` (BSH) or `[T,H]` (TH). Token axes may be zero. |
| `wkv` | Tensor | Same dtype as `x`, `[W,H]`, KV projection. |
| `wgate` | Tensor | Same dtype as `x`, `[W,H]`, score projection. |
| `state_cache` | Tensor | Mutable FP32 `[P,L,2*W]`, `L>=1`; KV followed by scores. Provide valid history for nonzero start positions. |
| `ape` | Tensor | FP32 `[r,W]`, position-dependent score bias. |
| `norm_weight` | Tensor | FP32 `[D]`, RMSNorm scale. |
| `rope_sin` | Tensor | FP32 `[B,Sr,64]` for BSH or `[Sr,64]` for TH, using the output row layout. |
| `rope_cos` | Tensor | FP32, same shape as `rope_sin`. |
| `state_block_table` | Tensor | Required INT32 tensor. Mode 1 takes `[B,M]` page IDs and also accepts the arch22 single-page `[B]` form. Mode 2 takes `[B,c*r+capacity]` physical state row IDs on both architectures, with column `c*r+t-p` for absolute position `t`. Supply at least one current-token column, enough columns for all accessed positions, and row IDs within the state cache. |
| `cu_seqlens` | Tensor or None | INT32 `[B+1]`, default `None`. Required for TH, forbidden for BSH; nondecreasing, first 0, last T. Differences give input capacities. `[0]` represents zero requests when `T=0`. |
| `seqused` | Tensor or None | INT32 `[B]`, default `None` uses full capacities. Values in `[0,capacity]`. |
| `start_pos` | Tensor or None | INT32 `[B]`, nonnegative absolute starts; default `None` means zero. |
| `rope_head_dim` | int | Must be 64; default 64. |
| `cmp_ratio` | int | 2, 4, 8, 16, 32, 64 or 128; default 4. |
| `coff` | int | 1 or 2; default 1. The latter includes one previous group. |
| `norm_eps` | float | RMSNorm epsilon; default `1e-6`. |
| `rotary_mode` | int | Default 2 (interleaved), the only supported mode. |
| `cache_mode` | int | 1 (paged) or 2 (explicit state rows). Default 1. |
| `state_cache_stride_dim0` | int | Default 0 derives `L*2*W`; an explicit value must equal it and fit signed INT32. |
| `impl` | str | `auto` (default), `ascendc`, or `torch_native`. |

| Return | Type | Meaning |
| --- | --- | --- |
| Compressed KV | Tensor | Contiguous, same device/dtype as `x`; `[B,Sr,D]` or `[Sr,D]`. Only valid rows described above are defined. `state_cache` is updated in place. |

## ascendc backend

Uses the existing architecture-specific Ascend C computation through CANN/ACLNN.
Supports FP16/BF16, TH/BSH, overlap and paged or explicitly addressed caches.
Both architectures use the same mode-2 state table format.
Input values and table entries must describe valid, nonconflicting requests;
metadata validation does not inspect device index values.

Python selects the backend and normalizes legacy cache-table views. The Torch
adapter checks device, storage, dtype and metadata needed to allocate the output.
The original Host tiling checks supported parameter values and tensor-shape
relations through CANN. Empty inputs follow the ACLNN query/execution path too.
Absent optional tensors use CANN's missing-input representation, avoiding the
legacy helper's non-null empty shape for `None`.

## torch_native backend

This reference checks its supported modes and required metadata, without
repeating native hardware and tiling constraints.

An independent reference uses CPU FP32 PyTorch operations and copies the output
and updated cache to the input device. It is intended for correctness comparison
and does not support NPU graph capture. Small floating-point differences from
Ascend C reductions and matrix multiplication are expected.

## Example

This paged example works with either cache-table architecture.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import compressor

device = "npu"
B, S, H, D, r = 1, 4, 1024, 128, 4
x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16) * 0.02
wkv = torch.randn(D, H, device=device, dtype=x.dtype) * 0.02
wgate = torch.randn_like(wkv) * 0.02
state = torch.zeros(2, 16, 2 * D, device=device)
out = compressor(
    x, wkv, wgate, state,
    torch.zeros(r, D, device=device), torch.ones(D, device=device),
    torch.zeros(B, 1, 64, device=device), torch.ones(B, 1, 64, device=device),
    state_block_table=torch.ones(B, 1, device=device, dtype=torch.int32),
    cmp_ratio=r, rotary_mode=2, cache_mode=1, impl="ascendc",
)
print(out.shape)  # [1, 1, 128]; state is updated in place.
```
