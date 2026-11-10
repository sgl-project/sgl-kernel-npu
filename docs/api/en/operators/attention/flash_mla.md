# flash_mla

This is a non-official operator introduced in CANN 9.2.0.

Upstream documentation: [Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/37c98efe33ee7faf7d17882532dc1305221c727e/attention/flash_mla/docs/torchapi_flash_mla.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Unsupported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` selects `cann_ops_transformers_cann920` on arch35 with CANN 9.2.0.

```python
from sgl_kernel_npu.operators.attention import flash_mla

def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    *,
    impl="auto",
):
    ...
```

## Description

Computes non-quantized MLA attention from queries and a paged shared KV cache. Each request reads its valid cache tokens through `block_table`; the key uses all 576 channels and the value uses the first 512 channels.

For each query head, with gathered key $K$, value $V=K[:, :D_v]$, scale $\alpha$, and additive mask $M$:

$$
A_{ij}=\alpha Q_iK_j^\top+M_{ij},\qquad
Z_i=\sum_j e^{A_{ij}}+\delta,\qquad
O_i=\frac{\sum_j e^{A_{ij}}V_j}{Z_i},\qquad
\mathrm{LSE}_i=\log Z_i.
$$

Here $\delta=0$ without a sink and $\delta=e^{s_n}$ with sink logit $s_n$ for head $n$. Allowed mask positions have value zero; excluded positions have value $-\infty$. Mode 3 is right-aligned causal attention: query position $i$ may attend to key positions $j\le L_{kv}-L_q+i$.

## Parameters and return value

Use tensors on the same device; the CANN implementation requires NPU tensors. Below, `B` is the request count, `T` the packed query-token count, `S` the padded query length, `N` the local query-head count, `P` the physical page count, `L` the page-table width, and `D_v` the value width. CANN uses QK width 576, value width 512, one KV head, and page size 128.

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `q` | Tensor; `[T,N,576]` (TND), `[B,S,N,576]` (BSND), or `[B,N,S,576]` (BNSD); BF16/FP16 | Required | Query; nonempty, with `1 <= N <= 128`. |
| `k_cache` | Tensor; `[P,128,1,576]` (PA_BBND), `[P,1,128,576]` (PA_BNBD), or `[P,1,36,128,16]` (PA_NZ); same dtype as `q` | Required | Shared key/value cache in the declared physical layout. |
| `block_table` | Optional Tensor `[B,L]`; int32 | `None` | Required for paged attention. Each used entry is a valid physical page ID; each row covers its valid KV length. |
| `cache_seqlens` | Optional Tensor `[B]`; int32 | `None` | Required valid KV lengths, within the referenced cache capacity. |
| `cu_seqlens_q` | Optional Tensor `[B+1]`; int32 | `None` | Required for TND, absent for other query layouts. Starts at zero and ends at `T`; differences give physical query lengths. |
| `seqused_q` | Optional Tensor `[B]`; int32 | `None` | Actual query lengths, no larger than the physical segments; omission uses the full segments. |
| `sinks` | Optional Tensor `[N]`; float32 | `None` | Per-head softmax sink logits. CANN also accepts an empty tensor to disable sinks. |
| `attn_mask` | Optional Tensor `[2048,2048]`; int8 | `None` | Required in modes 3/4: standard compressed upper-triangular mask, with ones above the diagonal. Absent in mode 0. |
| `metadata` | Optional Tensor `[M]`; int32 | `None` | Scheduling data from [flash_mla_metadata](flash_mla_metadata.md), on the query device and generated for matching inputs/attributes. |
| `head_dim_v` | `int` | `512` | Value/output width; CANN requires 512. |
| `softmax_scale` | `float` | `1.0` | Multiplier applied to QK scores; not inferred from the head width. |
| `mask_mode` | `int` | `0` | `0`: unmasked; `3`: right-aligned causal; `4`: right-aligned sliding window. |
| `win_left`, `win_right` | `int` | `-1` | Window bounds in mode 4; `-1` means unbounded. Other modes require `-1`. |
| `max_seqlen_q`, `max_seqlen_kv` | `int` | `-1` | Upstream maximum-length attributes; `-1` uses upstream defaults, otherwise nonnegative. |
| `layout_q` | `str` | `"BSND"` | `BSND`, `BNSD`, or `TND`. |
| `layout_kv` | `str` | `"PA_BBND"` | `PA_BBND`, `PA_BNBD`, or `PA_NZ`. |
| `layout_out` | `str` | `"BSND"` | Must match `layout_q`, except TND input also allows NTD output. |
| `return_softmax_lse` | `bool` | `False` | Whether to return the FP32 log-sum-exp. |
| `impl` | `str`, keyword-only | `"auto"` | `auto`, `cann_ops_transformers_cann920`, or `torch_native`. |

The return value is `(out, lse)`. Inputs are not modified.

| Return value | Type / shape / dtype | Meaning |
| --- | --- | --- |
| `out` | Tensor; `[T,N,D_v]`, `[N,T,D_v]`, `[B,S,N,D_v]`, or `[B,N,S,D_v]` according to `layout_out`; same dtype/device as `q` | Attention output. |
| `lse` | Tensor; float32 on `q.device`; `[N,T]` for TND queries or `[B,N,S]` for other query layouts | Log-sum-exp when enabled; an empty `[0]` tensor when disabled. |

## cann_ops_transformers_cann920 backend

Requires CANN 9.2.0 and an sgl-kernel-npu wheel containing Flash MLA. The first call may compile adapter code, so keep a C++ build environment available.

Supports the layouts in the parameter table, sink logits and modes 0/3/4. Prepare the cache in the declared layout, with valid page IDs, sequence lengths and matching metadata.

## torch_native backend

Provides an accuracy reference using FP32 computation on the input device, with outputs cast to `q.dtype`. Supports FP16/BF16/FP32, TND queries, single-head PA_BBND caches and TND/NTD outputs. Query segments and KV lengths must be positive.

Supports modes 0 and 3. Mode 3 requires the standard causal mask and a query length no greater than the KV length for each request. Omit `seqused_q` or set it to the full length of each query segment. Use `sinks=None` and `-1` for all window and maximum-length parameters.

This implementation computes from the input data, ignores `metadata`, and is unsuitable for graph capture. Floating-point rounding may differ from the CANN implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_mla, flash_mla_metadata

device = "npu"
q = torch.randn((1, 64, 576), device=device, dtype=torch.bfloat16) * 0.1
cache = torch.randn((1, 128, 1, 576), device=device, dtype=torch.bfloat16) * 0.1
table = torch.tensor([[0]], device=device, dtype=torch.int32)
lengths = torch.tensor([128], device=device, dtype=torch.int32)
boundaries = torch.tensor([0, 1], device=device, dtype=torch.int32)

metadata = flash_mla_metadata(
    lengths, 64, 1, cu_seqlens_q=boundaries, layout_q="TND"
)
out, lse = flash_mla(
    q, cache, block_table=table, cache_seqlens=lengths,
    cu_seqlens_q=boundaries, metadata=metadata,
    softmax_scale=576**-0.5,
    layout_q="TND", layout_kv="PA_BBND", layout_out="NTD",
    return_softmax_lse=True, impl="cann_ops_transformers_cann920",
)
print(out.shape, lse.shape)  # [64, 1, 512], [64, 1]
```
