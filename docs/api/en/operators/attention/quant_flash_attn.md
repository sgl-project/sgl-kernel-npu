# quant_flash_attn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformer` (`auto`) | Not supported | Supported |
| `torch_native` | Not registered | Supported |

The CANN backend targets Ascend 950 with CANN 9.2.0 and a matching PyTorch/torch_npu installation. The `cann_ops_transformer` Python extension exposing both QFA interfaces must also be installed; installing `sgl_kernel_npu` alone does not provide that extension or the underlying CANN operators.

## Signature

```python
from sgl_kernel_npu.operators.attention import quant_flash_attn

def quant_flash_attn(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer`. The two names refer to the same implementation, with no automatic fallback. Use explicit TND/PA_BNBD layouts for the MXFP8 path described below; the defaults preserve the upstream API.

## Description

Computes quantized grouped-query attention from Q and a paged KV cache. With dequantized inputs $Q_d$, $K_d$, $V_d$, the attention target is

$$
S = \mathrm{softmax\_scale}\, Q_d K_d^T + M,
\qquad O = \mathrm{softmax}(S)V_d.
$$

MXFP8 mode (`quant_mode=1`) uses FP8 E4M3 Q/K/V and E8M0 descales. Q/K descales cover groups of 32 channels; V descales cover groups of 32 sequence positions per channel. The final pair dimension stores adjacent groups. An E8M0 byte $e$ encodes $2^{e-127}$, with byte 255 reserved for NaN. ModelSlim static V scales are represented by repeating the per-channel scale across these sequence groups.

The quantized calculation also rounds the softmax intermediate P to FP8 E4M3 using `p_scale`. The reference models blockwise online softmax, its scaled exponential and FP8 P conversion, including the logarithm-of-two maximum alignment used by the CANN MXFP8 golden. Dequantizing only Q/K/V and calling ordinary high-precision attention does not model that rounding. Floating-point accumulation order can still differ from the device implementation.

For causal mode, query position `i` can attend to key positions `j <= kv_length - query_length + i`. This bottom-right alignment includes the historical prefix in chunked prefill. Inputs, including caches and metadata, are read-only.

## Parameters and return values

These tables describe the MXFP8 subset covered by the independent reference. Other CANN modes and layouts are forwarded to the installed CANN implementation; they are not validated by this contribution's reference.

Notation: `T` is total packed Q positions, `B` is request count, `Nq/Nkv` are local Q/KV head counts, `G=Nq/Nkv`, `D` is head dimension, `P` is physical page count, `S` is page size and `L` is page-table capacity per request. The covered head dimensions are 64/128/256; `Nq` must be divisible by `Nkv`. All tensors are on the same NPU. Use contiguous lengths, Q, Q descales and page tables. Paged cache tensors must obey CANN's supported strides, including no zero strides.

| Parameter | Type / default | Description and constraints |
| --- | --- | --- |
| `q` | Tensor; required | FP8 E4M3 `[T, Nq, D]`. |
| `k`, `v` | Tensor; required | FP8 E4M3 PA_BNBD `[P, Nkv, S, D]`. |
| `q_descale` | Tensor; required | E8M0 `[T, Nq, D/64, 2]` for TND, or `[Nkv, T, G, D/64, 2]` for N2TGD. |
| `k_descale` | Tensor; required | E8M0 `[P, Nkv, S, D/64, 2]`. |
| `v_descale` | Tensor; required | E8M0 `[P, Nkv, S/64, D, 2]`; the pair indexes adjacent 32-token groups. |
| `quant_mode` | int; required | `1` for the MXFP8 reference. CANN receives the original mode unchanged. |
| `block_table` | Tensor or None; `None` | Int32 `[B, L]` physical page indices; required for PA_BNBD. |
| `p_scale` | Tensor or None; `None` | FP32 `[1]` P quantization scale; use a finite positive value. The SGLang integration uses 1. |
| `cu_seqlens_q`, `cu_seqlens_kv` | Tensor or None; `None` | Int32 `[B+1]` cumulative lengths, beginning with zero. TND Q requires Q boundaries. The reference can derive missing `seqused_kv` from KV boundary differences. |
| `seqused_q`, `seqused_kv` | Tensor or None; `None` | Int32 `[B]` valid lengths. `seqused_q=None` uses Q boundary differences; paged KV should provide effective KV lengths. |
| `sinks` | Tensor or None; `None` | Optional upstream input; unsupported by the MXFP8 reference. |
| `attn_mask` | Tensor or None; `None` | CANN causal mask template: int8 `[2048, 2048]`, upper triangle masked. Decode mode 0 uses None. |
| `metadata` | Tensor or None; `None` | Int32 `[2, max_schedule_size]` from `quant_flash_attn_metadata`; required by the CANN QFA path. The numerical reference does not consume the hardware schedule. |
| `softmax_scale` | float; `1.0` | Multiplier of QK scores; explicitly pass `D**-0.5` for standard attention scaling. |
| `mask_mode` | int; `0` | `0`: unmasked; `3`: bottom-right causal. |
| `win_left`, `win_right` | int; `-1`, `-1` | Upstream window attributes; the reference supports the default full window. |
| `max_seqlen_q`, `max_seqlen_kv` | int; `-1`, `-1` | CANN length bounds; `-1` means unspecified. These must agree with metadata preparation. |
| `layout_q`, `layout_out` | str; `"BSND"` | Set both to `"TND"` for this reference. |
| `layout_q_descale` | str; `"BSND"` | Set to `"TND"` or `"N2TGD"`. |
| `layout_kv` | str; `"BSND"` | Set to `"PA_BNBD"` for this reference. |
| `return_softmax_lse` | bool; `False` | Return LSE when True; otherwise return an empty second tensor. |
| `impl` | str; `"auto"` | Keyword-only `"auto"`, `"cann_ops_transformer"` or `"torch_native"`. |

| Return value | Type / dtype | Shape and meaning |
| --- | --- | --- |
| `attn_output` | Tensor / BF16 | `[T, Nq, D]`, newly allocated on Q's device. |
| `softmax_lse` | Tensor / FP32 | `[Nq, T]` when enabled; `[0]` otherwise. |

The reference returns zero output and `+inf` LSE for padded Q positions, zero-length KV and fully masked rows.

## cann_ops_transformer backend

Lazily imports `quant_flash_attn` from `cann_ops_transformer.ops`, caches the callable and forwards the arguments. CANN owns validation, allocation and dispatch. The wrapper introduces no tensor copies, casts, host length reads or duplicate operator registration.

Build the metadata with the same quantization, mask, layout, head and length settings passed to this function. The metadata producer and consumer cannot independently validate all cross-call consistency conditions. See the [CANN 9.2 API](https://gitcode.com/cann/ops-transformer/blob/v9.2.0/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md) for upstream constraints.

## torch_native backend

Provides an independent eager numerical reference for mode 1, TND Q/output, PA_BNBD cache, TND/N2TGD Q descales and mask modes 0/3. It reconstructs logical KV sequences through the page table and expands scales before blockwise attention. Causal mode derives the standard pattern from sequence-length alignment, rather than supporting arbitrary custom masks. Unsupported reference modes are rejected explicitly; CANN's larger interface is not restricted by the reference's scope.

This backend is intended for correctness comparisons, not serving. It copies inputs to CPU for PyTorch reference calculations and returns outputs on Q's device, avoiding a dependency on NPU FP8 indexing kernels. These copies do not occur in the default CANN path. Metadata and maximum-length hints do not affect the reference's mathematical result. Dynamic graph replay is outside this contribution's test coverage.

## Example

Run on an arch35 NPU with the dependencies listed above. This is one decode query over a 128-token cache; E8M0 scales are built from exact byte 127 (scale 1).

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    quant_flash_attn,
    quant_flash_attn_metadata,
)

device = "npu"
q = torch.ones((1, 4, 128), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = torch.ones((1, 2, 128, 128), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
v = k.clone()

def scale(shape):
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)

q_descale = scale((2, 1, 2, 2, 2))  # Nkv, T, G, D/64, 2
k_descale = scale((1, 2, 128, 2, 2))
v_descale = scale((1, 2, 2, 128, 2))
cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
used_kv = torch.tensor([128], dtype=torch.int32, device=device)
pages = torch.tensor([[0]], dtype=torch.int32, device=device)
layout = dict(layout_q="TND", layout_q_descale="N2TGD", layout_kv="PA_BNBD", layout_out="TND")
plan = quant_flash_attn_metadata(
    4, 2, 128, 1, cu_seqlens_q=cu_q, seqused_kv=used_kv,
    max_seqlen_q=1, mask_mode=0, **layout,
)
kwargs = dict(
    block_table=pages, p_scale=torch.ones(1, dtype=torch.float32, device=device),
    cu_seqlens_q=cu_q, seqused_kv=used_kv, metadata=plan,
    softmax_scale=128**-0.5, max_seqlen_q=1, mask_mode=0, **layout,
)
out, lse = quant_flash_attn(q, k, v, q_descale, k_descale, v_descale, 1, **kwargs)
reference, _ = quant_flash_attn(
    q, k, v, q_descale, k_descale, v_descale, 1, impl="torch_native", **kwargs,
)
print(out.shape, lse.shape)  # torch.Size([1, 4, 128]), torch.Size([0])
```
