# flash_mla_metadata

This is a non-official operator introduced in CANN 9.2.0.

Upstream documentation: [Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/37c98efe33ee7faf7d17882532dc1305221c727e/attention/flash_mla/docs/torchapi_flash_mla.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Unsupported | Supported |

## Signature

The default `auto` selects `cann_ops_transformers_cann920` on arch35 with CANN 9.2.0.

```python
from sgl_kernel_npu.operators.attention import flash_mla_metadata

def flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    *,
    impl="auto",
):
    ...
```

## Description

Generates scheduling data for [flash_mla](flash_mla.md). Pass the returned tensor directly to the main operator as `metadata`.

Sequence lengths, head counts, dimensions, mask, window and query layout must match the attention call. Regenerate metadata when these parameters or the device change.

## Parameters and return value

All tensor inputs must be on the current NPU. `B` is the request count and `T` is the physical packed query-token count.

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `cache_seqlens` | Tensor `[B]`; int32 | Required | Valid KV lengths; one entry per request. Must agree with the main operator and its cache capacity. |
| `num_heads_q` | `int` | Required | Local query-head count in `[1,128]`, matching `q`. |
| `num_heads_kv` | `int` | Required | Must be 1 for the current MLA kernel. |
| `cu_seqlens_q` | Optional Tensor `[B+1]`; int32 | `None` | Required for TND and absent for other layouts. Cumulative physical query lengths, starting at zero and ending at `T`. |
| `seqused_q` | Optional Tensor `[B]`; int32 | `None` | Actual query lengths, no larger than the corresponding physical lengths; omission uses full segments. |
| `max_seqlen_q`, `max_seqlen_kv` | `int` | `-1` | Upstream maximum-length attributes; `-1` uses upstream defaults, otherwise nonnegative. |
| `head_dim_qk` | `int` | `576` | Must be 576, including all key channels. |
| `head_dim_v` | `int` | `512` | Must be 512. |
| `mask_mode` | `int` | `0` | `0`: unmasked; `3`: right-aligned causal; `4`: right-aligned sliding window. |
| `win_left`, `win_right` | `int` | `-1` | Window bounds in mode 4, with `-1` meaning unbounded; other modes require `-1`. |
| `layout_q` | `str` | `"BSND"` | `BSND`, `BNSD`, or `TND`; same as the attention call. |
| `impl` | `str`, keyword-only | `"auto"` | `auto` or `cann_ops_transformers_cann920`. |

| Return value | Type / shape / dtype | Meaning |
| --- | --- | --- |
| `metadata` | Tensor `[M]`; int32 on the current NPU | Scheduling data; its length is determined automatically from the inputs and device. Inputs are not modified. |

Pass the result unchanged to `flash_mla(..., metadata=metadata)`.

## cann_ops_transformers_cann920 backend

Requires CANN 9.2.0 and an sgl-kernel-npu wheel containing both Flash MLA and metadata operators. The first call may compile adapter code, so keep a C++ build environment available.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_mla_metadata

lengths = torch.tensor([128], device="npu", dtype=torch.int32)
boundaries = torch.tensor([0, 1], device="npu", dtype=torch.int32)
metadata = flash_mla_metadata(
    lengths,
    num_heads_q=64,
    num_heads_kv=1,
    cu_seqlens_q=boundaries,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    layout_q="TND",
    impl="cann_ops_transformers_cann920",
)
print(metadata.shape, metadata.dtype)
# Pass unchanged to flash_mla with matching lengths and attributes.
```
