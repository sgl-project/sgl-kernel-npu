# kv_quant_sparse_flash_attention

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | INT8 | E4M3 FP8 |
| `torch_native` | INT8 | INT8 and E4M3 FP8 |

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
    return_softmax_lse=True,
    impl="auto",
):
    ...
```

`auto` selects `ascendc` on arch22 and arch35. The native binding used by SGLang is
`torch.ops.npu.kv_quant_sparse_flash_attention` with the same tensors and attention options.

## Description

Compute MLA attention over explicit Top-K indices.
A RoPE64 KV record stores 512 bytes of INT8 or E4M3 FP8 latent, 128 bytes of BF16 RoPE,
and four FP32 scales at byte offset 640. A scale covers 128 contiguous latent
values. Latent multiplication is in FP32 and the result is rounded to BF16
before attention. A RoPE0 record contains 528 bytes, with the four scales at
byte offset 512. Pass an empty final dimension in query_rope for RoPE0.
For causal attention, pass indices that already encode causal visibility.
With return_softmax_lse=False, the two softmax tensors are empty.

```text
latent[d] = BF16(FP32(quantized_latent[d]) * scale[d // 128])
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
| `query_rope` | BF16 `[T, H, 64]` or `[T, H, 0]` | Query RoPE. |
| `packed_kv` | INT8 or UINT8 `[pages, page_size, 1, 656 or 528]` | PA_BSND packed KV with one KV head. Page size is a positive multiple of 16 up to 1024. |
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

Both kernels read paged packed KV and can return softmax statistics for
cross-rank LSE merging.

## torch_native backend

The accuracy reference decodes the cache on CPU and computes attention with
PyTorch. It keeps the softmax statistics in FP32 and rounds the exponential
weights to BF16 before multiplying by V. It returns tensors on the query device.

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
