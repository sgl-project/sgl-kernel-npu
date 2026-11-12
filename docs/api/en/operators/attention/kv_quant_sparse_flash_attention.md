# kv_quant_sparse_flash_attention

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Unsupported | Unverified |
| `torch_native` | Unsupported | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

def kv_quant_sparse_flash_attention(
    query,
    query_rope,
    packed_kv,
    sparse_indices,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    scale_value,
    *,
    impl="auto",
):
    ...
```

`auto` selects `ascendc` on arch35. The native binding used by SGLang is
`torch.ops.npu.sgl_kv_quant_sparse_flash_attention` with the same eight arguments.

## Description

Compute one DCP rank's MLA attention over explicit, rank-local Top-K indices.
Each KV record stores 512 bytes of FP8 E4M3FN latent, 128 bytes of BF16 RoPE,
and four FP32 scales at byte offset 640. A scale covers 128 contiguous latent
values. Latent multiplication is in FP32 and the result is rounded to BF16
before attention. RoPE is already BF16.

```text
latent[d] = BF16(FP32(fp8_latent[d]) * scale[d // 128])
logit[i] = (dot(query, latent[i]) + dot(query_rope, rope[i])) * scale_value
maximum = max(logit)
total = sum(exp(logit - maximum))
attention_out = sum(exp(logit[i] - maximum) * latent[i]) / total
LSE = maximum + log(total)
```

Trailing `-1` entries end the selection. With an empty selection, all three
outputs for the row are zero and its LSE is negative infinity.

## Parameters and return values

All tensor inputs are on the same NPU. `T` counts query tokens, `H` is the
query head count, `B` counts requests, and `K` is the Top-K capacity.

| Parameter | Type and shape | Description |
| --- | --- | --- |
| `query` | BF16 `[T, H, 512]` | TND query latent. `H` is 1–64 or 128. |
| `query_rope` | BF16 `[T, H, 64]` | Query RoPE. |
| `packed_kv` | UINT8 `[pages, page_size, 1, 656]` | PA_BSND packed KV with one KV head. Page size is a positive multiple of 16 up to 1024. |
| `sparse_indices` | INT32 `[T, 1, K]` | Selected logical positions in each request, followed by `-1` padding. |
| `block_table` | INT32 `[B, max_pages]` | Maps logical pages to physical pages. |
| `actual_seq_lengths_query` | INT32 `[B]` | Cumulative query lengths without a leading zero. |
| `actual_seq_lengths_kv` | INT32 `[B]` | Number of valid KV tokens per request on this rank. |
| `scale_value` | float | Score multiplier applied before softmax. |
| `impl` | str, default `"auto"` | Keyword-only implementation selector. |

| Return | Type and shape | Description |
| --- | --- | --- |
| `attention_out` | BF16 `[T, H, 512]` | Rank-local attention output. |
| `softmax_max` | FP32 `[1, T, H]` | Maximum score per query and head. |
| `softmax_sum` | FP32 `[1, T, H]` | Sum of the exponentials shifted by the maximum. |

## Ascend C backend

The arch35 kernel uses paged KV and consumes the packed cache directly. It
returns both softmax statistics for the framework's cross-rank LSE merge.

## torch_native backend

The accuracy reference decodes the cache on CPU, computes attention with
PyTorch in FP32, and returns tensors on the query device.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

device = "npu"
query = torch.zeros((1, 32, 512), dtype=torch.bfloat16, device=device)
query_rope = torch.zeros((1, 32, 64), dtype=torch.bfloat16, device=device)
packed = torch.cat(
    (
        torch.zeros((1, 128, 1, 512), dtype=torch.uint8, device="cpu"),
        torch.zeros((1, 128, 1, 64), dtype=torch.bfloat16, device="cpu").view(torch.uint8),
        torch.ones((1, 128, 1, 4), dtype=torch.float32, device="cpu").view(torch.uint8),
    ),
    dim=-1,
).to(device)
indices = torch.tensor([[[0]]], dtype=torch.int32, device=device)
table = torch.tensor([[0]], dtype=torch.int32, device=device)
lengths = torch.tensor([1], dtype=torch.int32, device=device)
out, maximum, total = kv_quant_sparse_flash_attention(
    query, query_rope, packed, indices, table, lengths, lengths, 1 / 24, impl="ascendc"
)
```
