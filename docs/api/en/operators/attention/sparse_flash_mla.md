# sparse_flash_mla

This is a non-official operator introduced in CANN 9.2.0.

Upstream documentation: [Sparse Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/7df757adc542c75733f9702af97ee7a14fca9f51/attention/sparse_flash_mla/docs/torchapi_sparse_flash_mla.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` selects `cann_ops_transformers_cann920` on CANN 9.2.0 and arch22.

```python
from sgl_kernel_npu.operators.attention import sparse_flash_mla

def sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

## Description

Computes sparse MLA attention over shared KV (the key and the value are the same
input) in three scenarios:

- **SWA** uses only `ori_kv` with a sliding window, passes no `cmp_kv`, and sets
  `cmp_ratio=1`.
- **CSA** combines the `ori_kv` window with the compressed tokens selected by
  `cmp_sparse_indices`; `cmp_ratio` is 1, 2 or 4.
- **HCA** combines the `ori_kv` window with a contiguous compressed segment;
  `cmp_ratio` is 128.

With valid query and KV lengths $L_q$, $L_{ori}$, $L_{cmp}$, and the pre-compression
length $L_{restored}=L_{cmp}\cdot r+\rho$ ($r$ is `cmp_ratio`, $\rho$ is
`cmp_residual_kv`), query position $i$ attends to

$$
\theta_i = L_{ori} - L_q + i + 1,\qquad
\Omega_i = \left[\max(\theta_i - w_{left} - 1, 0),\ \min(\max(\theta_i + w_{right}, 0), L_{ori})\right)
$$

on the original side, where $w_{left}$ and $w_{right}$ are `ori_win_left` and
`ori_win_right`, and to

$$
\kappa_i = \min\left(\max\left(\left\lfloor\frac{L_{restored} - L_q + i + 1}{r}\right\rfloor, 0\right), L_{cmp}\right)
$$

compressed tokens: `cmp_mask_mode=3` keeps tokens below $\kappa_i$ and `0` keeps all
$L_{cmp}$. CSA gathers the token ids from `cmp_sparse_indices` and stops at `-1`;
HCA reads $[0, \kappa_i)$. The softmax key set is
$K = K_{ori}[\Omega_i] \Vert K_{cmp}[\text{selected}]$ and the value equals the key:

$$
A_{ij}=\alpha Q_iK_j^\top,\qquad
Z_i=\sum_j e^{A_{ij}}+\delta,\qquad
O_i=\frac{\sum_j e^{A_{ij}}V_j}{Z_i},\qquad
\mathrm{LSE}_i=\log Z_i.
$$

$\alpha$ is `softmax_scale`. $\delta=0$ without sinks; with sink logit $s_n$ for head
$n$, $\delta=e^{s_n}$. A query position with no visible key produces zeros.

## Parameters and return value

Use tensors on the same device; the CANN implementation requires NPU tensors. Below,
`B` is the request count, `T` the packed query-token count, `S` the padded query
length, `N1`/`N2` the query/KV head counts, `D` the head width, and `P` the physical
page count.

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `q` | Tensor; `[T,N1,D]` (TND) or `[B,S,N1,D]`; BF16/FP16 | Required | Query; nonempty. CANN requires `D=512`. |
| `ori_kv` | Tensor; `[T,N2,D]`, `[B,S,N2,D]`, or `[P,page_size,N2,D]` per `layout_kv`; same dtype as `q` | Required | Uncompressed shared K/V. |
| `cmp_kv` | Same layouts as `ori_kv`; optional | `None` | Compressed shared K/V; omitted for SWA. |
| `ori_sparse_indices` | Optional Tensor; int32 | `None` | Unsupported on Atlas A2/A3; must be omitted. |
| `cmp_sparse_indices` | Optional Tensor; int32; `[T,N2,K]` (TND) or `[B,S,N2,K]` | `None` | Required for CSA; ids index compressed tokens and `-1` terminates. |
| `ori_block_table`, `cmp_block_table` | Optional Tensor `[B,L]`; int32 | `None` | Required for `PA_BBND`. |
| `cu_seqlens_q`, `cu_seqlens_ori_kv`, `cu_seqlens_cmp_kv` | Optional Tensor `[B+1]`; int32 | `None` | Packed TND boundaries starting at zero. |
| `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv` | Optional Tensor `[B]`; int32 | `None` | Valid length per request; KV side is required for `PA_BBND`. |
| `cmp_residual_kv` | Tensor `[B]`; int32 | `None` | Compression remainder, each value below `cmp_ratio`; required with `cmp_mask_mode=3` and `cmp_ratio!=1`, rejected otherwise. |
| `ori_topk_length` | Optional Tensor; int32 | `None` | Valid length for `ori_sparse_indices`; unused on A2/A3. |
| `cmp_topk_length` | Optional Tensor; int32; `[T,N2]` (TND) or `[B,S,N2]` | `None` | Valid length of `cmp_sparse_indices`; reserved on Atlas A2/A3 and must be omitted there. |
| `sinks` | Tensor `[N1]`; float32 | `None` | Per-head softmax sink logits; the Atlas A2/A3 kernel requires it, so it cannot be omitted or empty. |
| `metadata` | Tensor `[2048]`; int32 | Required | Scheduling data from [sparse_flash_mla_metadata](sparse_flash_mla_metadata.md). |
| `softmax_scale` | `float` | `1.0` | QK score scale; never inferred from the head width. |
| `cmp_ratio` | `int` | `1` | Compression ratio: 1 for SWA, 1/2/4 for CSA, 128 for HCA. |
| `ori_mask_mode` | `int` | `0` | `0` unmasked, `3` causal, `4` sliding window. |
| `cmp_mask_mode` | `int` | `0` | `0` unmasked, `3` causal; SWA uses 0 and CSA/HCA use 3. |
| `ori_win_left`, `ori_win_right` | `int` | `-1` | Window bounds for mode 4; Atlas A3 requires 127 and 0. |
| `layout_q` | `str` | `"BSND"` | `BSND` or `TND`. |
| `layout_kv` | `str` | `"BSND"` | `BSND`, `TND`, or `PA_BBND`. |
| `topk_value_mode` | `int` | `1` | Index convention; only 1 on Atlas A2/A3. |
| `return_softmax_lse` | `bool` | `False` | Return FP32 log-sum-exp. |
| `impl` | `str`, keyword only | `"auto"` | `auto`, `cann_ops_transformers_cann920`, or `torch_native`. |

Returns `(attention_out, softmax_lse)` and does not modify its inputs.

| Value | Type / shape / dtype | Meaning |
| --- | --- | --- |
| `attention_out` | Tensor; same shape, dtype and device as `q` | Attention output. |
| `softmax_lse` | Tensor; float32; `[N2,T,N1/N2]` (TND) or `[B,N2,S,N1/N2]` (BSND) | Log-sum-exp when enabled; an empty `[0]` tensor otherwise. |

## Constraints

- `metadata` must come from [sparse_flash_mla_metadata](sparse_flash_mla_metadata.md)
  with exactly the same layout, length, compression, TopK, mask, window and KV
  presence arguments; mismatches have undefined behaviour.
- On Atlas A2/A3, `num_heads_q` is limited to 1, 2, 4, 8, 16, 32, 64 or 128,
  `num_heads_kv` must be 1, and `head_dim` must be 512.
- On Atlas A2/A3, `ori_win_left` must be 127 and `ori_win_right` must be 0;
  `cmp_ratio` must be 1 for SWA, 1/2/4 for CSA and 128 for HCA.
- Atlas A2/A3 does not support `ori_sparse_indices` (`ori_topk` must be 0) and
  accepts only `topk_value_mode=1`.
- `sinks` must be a nonempty one-dimensional float32 tensor of length
  `num_heads_q`; the Atlas A2/A3 kernel rejects `None`.
- `cmp_topk_length` is reserved on Atlas A2/A3 and must be omitted even for CSA.
- `cmp_residual_kv[i] < cmp_ratio`; it is required when `cmp_kv` is present with
  `cmp_mask_mode=3` and `cmp_ratio!=1`, and rejected when `cmp_mask_mode=0` or
  `cmp_ratio=1`.
- `PA_BBND` requires `seqused_ori_kv` and `ori_block_table` (plus the matching
  compressed-side tensors); block size is a multiple of 16, at most 1024.

## cann_ops_transformers_cann920 backend

Requires CANN 9.2.0 and the ops-transformer dependency bundled in the wheel. The
first call may compile adaptor code, so keep a working C++ toolchain.

Do not install a separately built ops-transformer and its vendor in the same
environment: both declare the same operator and override each other.

CANN 9.2.0 ships a built-in operator with the same name. Import
`sgl_kernel_npu` before `torch_npu` (or set `ASCEND_CUSTOM_OPP_PATH` to the
vendors inside the wheel before the process starts); otherwise the built-in
implementation may be selected and reject the `metadata` size.

## torch_native backend

Used for accuracy comparison; it computes in FP32 on the input device and casts the
output to `q.dtype`. It supports `BSND`/`TND` queries, `BSND`/`TND`/`PA_BBND` KV,
SWA/CSA/HCA, sink logits and `return_softmax_lse`, but not `ori_sparse_indices` or
`topk_value_mode!=1`. `metadata` is ignored as opaque scheduling data and the
implementation is not graph-safe. Floating-point rounding may differ from CANN.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)

device = "npu"
q = torch.randn((16, 8, 512), device=device, dtype=torch.bfloat16) * 0.1
ori_kv = torch.randn((1, 128, 1, 512), device=device, dtype=torch.bfloat16) * 0.1
ori_table = torch.tensor([[0]], device=device, dtype=torch.int32)
seqused_ori = torch.tensor([128], device=device, dtype=torch.int32)
sinks = torch.zeros((8,), device=device, dtype=torch.float32)
boundaries = torch.tensor([0, 16], device=device, dtype=torch.int32)

metadata = sparse_flash_mla_metadata(
    8, 1, 512,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    max_seqlen_q=16,
    max_seqlen_ori_kv=128,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)
out, lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    ori_block_table=ori_table,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=512**-0.5,
    ori_mask_mode=4,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=True,
)
print(out.shape, lse.shape)  # [16, 8, 512], [1, 16, 8]
```
