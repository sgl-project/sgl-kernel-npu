# mixed_quant_sparse_flash_mla_metadata

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | Unsupported | Supported |

## Signature

`from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata`

```python
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

## Behavior

Plans full-row or FlashDecode split work on AICPU from the original and compressed TopK lengths.

`ori_topk_length` and `cmp_topk_length` describe valid sparse lengths per query/head. The `has_ori_kv` and `has_cmp_kv` flags must match the main operator. This profile uses `num_heads_q=64`, `num_heads_kv=1`, `head_dim=512`, and `quant_mode=1`.

## Parameters

T is query rows, B requests, P physical pages and S slots per page.

| Parameter | Default | Type, shape and meaning |
| --- | --- | --- |
| `ori_topk_length` | Required | Required contiguous INT32 [T,1], valid original sparse prefix per query. |
| `cmp_topk_length` | Required | Required contiguous INT32 [T,1]; pass zeros when has_cmp_kv=False. |
| `cu_seqlens_q` | `None` | Optional INT32 [B+1], cumulative query row boundaries. |
| `seqused_q` | `None` | Optional INT32 [B], compatibility argument; scheduling uses cu_seqlens_q. |
| `seqused_ori_kv` | `None` | Optional INT32 [B], original KV lengths; retained for API compatibility. |
| `seqused_cmp_kv` | `None` | Optional INT32 [B], compressed KV lengths; retained for API compatibility. |
| `batch_size` | `None` | Optional int, compatibility argument; B is derived from cu_seqlens_q (one batch if absent). |
| `max_seqlen_q` | `None` | Optional int, compatibility argument; query rows come from the length tensor shape. |
| `max_seqlen_ori_kv` | `None` | Optional int, original KV capacity (compatibility argument). |
| `max_seqlen_cmp_kv` | `None` | Optional int, compressed KV capacity (compatibility argument). |
| `num_heads_q` | Required | int, query heads; use 64. |
| `num_heads_kv` | Required | int, shared MLA KV heads; use 1. |
| `head_dim` | Required | int, logical head dimension; use 512. |
| `quant_mode` | Required | int, use 1 for the supported quantized profile. |
| `layout_q` | `'TND'` | str, use TND (flattened query tokens). |
| `layout_kv` | `'PA_BBND'` | str, use PA_BBND packed paged KV. |
| `has_ori_kv` | `True` | bool, must be True for this implementation. |
| `has_cmp_kv` | `True` | bool, whether the consuming attention call has compressed KV. |
| `impl` | `'auto'` | str, auto and cannbot_dsl select the same AICPU schedule implementation. |

## Returns

| Value | Meaning |
| --- | --- |
| metadata | INT32 [1024], backend-specific schedule. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and a compatible CANNBot-DSL runtime installed in the environment. This operator does not enforce a runtime version. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

Metadata encodes a device-specific schedule, with no unique Torch numerical answer. No torch_native alias is provided. The paired compute reference checks output semantics; this entry tests eager/graph parity after length updates.

## Example

Construct a small input and its metadata:

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata(
    length,
    torch.zeros_like(length),
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
