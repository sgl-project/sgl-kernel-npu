# quant_flash_attn_metadata

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformer` (`auto`) | Not supported | Supported |

Requires CANN 9.2.0 or later, matching PyTorch/torch_npu, and the `cann_ops_transformer` Python extension exposing QFA and its metadata producer. These dependencies are separate from the `sgl_kernel_npu` wheel. This operator has been tested with CANN 9.2.0.

## Signature

```python
from sgl_kernel_npu.operators.attention import quant_flash_attn_metadata

def quant_flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=None,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    is_grad_enabled=False,
    impl="auto",
):
    ...
```

`auto` selects `cann_ops_transformer`; both names use the same callable. No alternate schedule or automatic fallback is provided.

## Description

Builds the device-side work-partition schedule consumed by [quant_flash_attn](quant_flash_attn.md). The output is CANN scheduling metadata, not a copy of sequence lengths or a page table. This wrapper forwards the caller's existing device lengths and attributes; it does not derive lengths, add decode/speculative offsets or build page tables.

Generate the schedule before calling QFA and reuse it across layers with identical relevant dimensions and attributes. Changed lengths require the caller to regenerate the schedule. The two calls must agree on head counts, dimensions, quantization mode, mask, layouts and sequence-length settings. Inputs are not modified.

## Parameters and return value

`B` is request count. Tensor arguments are contiguous int32 NPU tensors on the consumer's device. TND Q uses cumulative Q boundaries; PA_BBND KV uses valid KV lengths.

| Parameter | Type / default | Description and constraints |
| --- | --- | --- |
| `num_heads_q` | int; required | Local Q head count. |
| `num_heads_kv` | int; required | Local KV head count; must divide the Q head count. |
| `head_dim` | int; required | Q/K head dimension; the Qwen3.6-27B tests use 256. |
| `quant_mode` | int; required | `1` for the covered MXFP8 inference path. Other values are forwarded unchanged. |
| `cu_seqlens_q`, `cu_seqlens_kv` | Tensor or None; `None` | `[B+1]` cumulative lengths, beginning with zero. |
| `seqused_q`, `seqused_kv` | Tensor or None; `None` | `[B]` effective lengths; paged KV uses `seqused_kv`. |
| `batch_size` | int or None; `None` | CANN can infer B from Q boundary/length tensor shape. Leave None for TND. |
| `max_seqlen_q`, `max_seqlen_kv` | int; `-1`, `-1` | Length upper bounds; `-1` means unspecified. |
| `head_dim_v` | int or None; `None` | V dimension; None uses `head_dim`. CANN 9.2 requires equal Q/K and V dimensions for this path. |
| `mask_mode` | int; `0` | `0`: unmasked decode; `3`: bottom-right causal. |
| `win_left`, `win_right` | int; `-1`, `-1` | Upstream window attributes; retain defaults for the covered path. |
| `layout_q`, `layout_out` | str; `"BSND"` | Set both to `"TND"` for the covered MXFP8 path. |
| `layout_q_descale` | str; `"BSND"` | `"TND"` or `"N2TGD"` for MXFP8; must match QFA's actual descale layout. |
| `layout_kv` | str; `"BSND"` | Set to `"PA_BBND"` for the covered paged path. |
| `is_grad_enabled` | bool; `False` | CANN backward-schedule option; inference uses False. |
| `impl` | str; `"auto"` | Keyword-only `"auto"` or `"cann_ops_transformer"`. |

| Return value | Type / dtype | Shape and meaning |
| --- | --- | --- |
| `metadata` | Tensor / int32 | `[2, max_schedule_size]` on NPU. CANN determines capacity from batch/head configuration. Pass the opaque result to QFA unchanged. |

## cann_ops_transformer backend

Lazily imports `quant_flash_attn_metadata` from `cann_ops_transformer.ops` and caches the callable. CANN owns output allocation, scheduling and validation. The wrapper adds no host reads, tensor conversions or shape-dependent fallback.

There is no independent `torch_native` scheduler in this contribution: duplicating CANN's scheduling format is outside the numerical reference's scope. Validation uses the metadata through the attention consumer and compares that consumer against an independent attention reference, plus checks of interface forwarding and output properties. `auto` versus `cann_ops_transformer` is not treated as an independent accuracy comparison. Dynamic graph replay is not included in this contribution's tests.

The [CANN 9.2 API](https://gitcode.com/cann/ops-transformer/blob/v9.2.0/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md) defines the full upstream constraints.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import quant_flash_attn_metadata

cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
used_kv = torch.tensor([128, 257], dtype=torch.int32, device="npu")
metadata = quant_flash_attn_metadata(
    num_heads_q=12,
    num_heads_kv=2,
    head_dim=256,
    quant_mode=1,
    cu_seqlens_q=cu_q,
    seqused_kv=used_kv,
    max_seqlen_q=1,
    mask_mode=0,
    layout_q="TND",
    layout_q_descale="N2TGD",
    layout_kv="PA_BBND",
    layout_out="TND",
    impl="cann_ops_transformer",
)
print(metadata.shape, metadata.dtype)
```

For a complete producer/consumer example, see [quant_flash_attn](quant_flash_attn.md#example).
