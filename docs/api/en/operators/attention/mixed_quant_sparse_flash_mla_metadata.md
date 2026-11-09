# mixed_quant_sparse_flash_mla_metadata

## Purpose and support

Plans full-row or FlashDecode split work on AICPU from the original and compressed TopK lengths.

The DSL targets Ascend 950PR/950DT (`arch35`); validation status is listed below. `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata


def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
    impl="auto"
): ...
```

## Inputs

`ori_topk_length` and `cmp_topk_length` describe valid sparse lengths per query/head. The `has_ori_kv` and `has_cmp_kv` flags must match the main operator. This profile uses `num_heads_q=64`, `num_heads_kv=1`, `head_dim=512`, and `quant_mode=1`.

## Output and ordering

Returns INT32 `[1024]` metadata. Produce it on the same NPU stream before calling `mixed_quant_sparse_flash_mla`.

## Hardware support

| impl | Ascend 950PR | Ascend 950DT | Ascend 910C |
| --- | --- | --- | --- |
| cannbot_dsl | Supported | Unverified | Unsupported |

## Parameters

T is query rows, B requests, P physical pages and S slots per page. Optional tensors default to None; scalar defaults are shown in the signature.

| Parameter | Type, shape and meaning |
| --- | --- |
| `ori_topk_length` | INT32 [T,1], valid original sparse prefix per query; required when original KV is enabled. |
| `cmp_topk_length` | INT32 [T,1], valid compressed sparse prefix per query; required when compressed KV is enabled. |
| `cu_seqlens_q` | Optional INT32 [B+1], cumulative query row boundaries. |
| `seqused_q` | Optional INT32 [B], valid query rows per request. |
| `seqused_ori_kv` | Optional INT32 [B], original KV lengths; retained for API compatibility. |
| `seqused_cmp_kv` | Optional INT32 [B], compressed KV lengths; retained for API compatibility. |
| `batch_size` | Optional int B, request count; infer from query boundaries when omitted. |
| `max_seqlen_q` | int or None, query capacity per request; infer when unspecified. |
| `max_seqlen_ori_kv` | Optional int, original KV capacity (compatibility argument). |
| `max_seqlen_cmp_kv` | Optional int, compressed KV capacity (compatibility argument). |
| `num_heads_q` | int, query heads: 32 for indexers, 64 for MLA. |
| `num_heads_kv` | int, shared MLA KV heads; use 1. |
| `head_dim` | int, logical head dimension: 128 for indexers, 512 for MLA. |
| `quant_mode` | int, use 1 for the supported quantized profile. |
| `layout_q` | str, use TND (flattened query tokens). |
| `layout_kv` | str, use PA_BBND packed paged KV. |
| `has_ori_kv` | bool, whether the consuming attention call has original KV. |
| `has_cmp_kv` | bool, whether the consuming attention call has compressed KV. |
| `impl` | str, auto and cannbot_dsl select the same AICPU schedule implementation. |

## Returns

| Value | Meaning |
| --- | --- |
| metadata | INT32 [1024], backend-specific schedule. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and cannbot-dsl 0.8.0; the project validation target is CANN 9.2.0. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

Metadata encodes a device-specific schedule, with no unique Torch numerical answer. No torch_native alias is provided. The paired compute reference checks output semantics; this entry tests eager/graph parity after length updates.

## Example

Construct a small input and its metadata:

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

q = torch.randn(1, 64, 512, device="npu", dtype=torch.bfloat16)
kv = torch.zeros(1, 128, 1, 544, device="npu", dtype=torch.uint8)
kv[..., 512:] = torch.ones(1, 128, 1, 16, device="npu", dtype=torch.bfloat16).view(
    torch.uint8
)
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata(
    length,
    None,
    cu_seqlens_q=cu,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    quant_mode=1,
    has_cmp_kv=False,
)
```

## Device test

```bash
python -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py --npu-device=0 --enable-perf=false -v -s
```
