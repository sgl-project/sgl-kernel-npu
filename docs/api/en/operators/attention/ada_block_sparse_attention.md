# ada_block_sparse_attention

Block-sparse attention operator.

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | FP16 / BF16 | — |
| `torch_native` | FP16 / BF16 | FP16 / BF16 |

## Signature

```python
from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

def ada_block_sparse_attention(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    *,
    input_layout="BNSD",
    sparse_size=128,
    num_heads=1,
    num_key_value_heads=1,
    scale_value=1.0,
    causal=True,
    inner_precise=1,
    pre_tokens=214748647,
    next_tokens=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    impl="auto",
):
    ...
```

`auto` selects `ascendc` on arch22; `torch_native` uses PyTorch operations.
Requesting `ascendc` on arch35 raises instead of switching backends.

## Description

Computes block-sparse attention over BNSD tensors. The sparsity granularity is
`sparse_size`: `sparse_mask[b, h, i, j] != 0` enables every KV position of
block `j` for query block `i`. With `causal=True` the diagonal block is further
restricted to the lower triangle at element granularity.

```text
scores = scale_value * (query @ key^T)
mask   = sparse block mask [+ causal inside the diagonal block]
out    = softmax(scores, mask) @ value
```

Grouped-query attention is supported: `key`/`value` may have fewer heads than
`query`, and each KV head serves `query_heads / kv_heads` query heads. A query
block without any enabled KV block produces zeros.

## Parameters and return values

`B` is the batch size, `Nq`/`Nkv` the query/key head counts, `S1`/`S2` the query
and KV sequence lengths, and `D` the head width.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `query` | `[B, Nq, S1, D]` | FP16 / BF16 | Must be 4D. |
| `key` | `[B, Nkv, S2, D]` | Same as `query` | `Nq % Nkv == 0`. |
| `value` | `[B, Nkv, S2, D]` | Same as `query` | Value tensor. |
| `sparse_mask` | `[B, Nq, ceil(S1/bs), ceil(S2/bs)]` | int8 / bool | Block mask; the last dimension may be padded to a multiple of 32. |
| `sparse_count_table` | `[B, Nq, ceil(S1/bs)]` | int32 | Number of enabled KV blocks per query block. |
| `input_layout` | — | str, `"BNSD"` | BNSD only; TND is rejected. |
| `sparse_size` | — | int, 128 | Sparse block size. |
| `num_heads` | — | int, 1 | Query head count; should equal `Nq`. |
| `num_key_value_heads` | — | int, 1 | KV head count; 0 falls back to `num_heads`. |
| `scale_value` | — | float, 1.0 | Score multiplier. |
| `causal` | — | bool, `True` | Enables causal masking. |
| `inner_precise` | — | int, 1 | High-precision accumulation mode. |
| `pre_tokens` / `next_tokens` | — | int | Sliding-window bounds; defaults mean unrestricted. |
| `actual_seq_lengths` / `actual_seq_lengths_kv` | — | `None` | Variable lengths; the current implementation is dense-length only. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `attention_out` | `[B, Nq, S1, D]` | Same as `query` |

## ascendc backend

- Requires all tensors on the same NPU; noncontiguous inputs are converted to
  contiguous working tensors internally.
- The padded tail of the mask's last dimension must be zero, and
  `sparse_count_table` must equal the number of enabled blocks per row.

## torch_native backend

- Supports BNSD, fixed lengths and unrestricted windows only.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

batch, heads, seq, dim = 1, 2, 256, 128
blocks = seq // 128
query = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
key = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
value = torch.randn(batch, heads, seq, dim, device="npu", dtype=torch.float16)
mask = torch.zeros(batch, heads, blocks, 32, device="npu", dtype=torch.int8)
mask[..., :blocks] = 1
count = mask[..., :blocks].sum(dim=-1, dtype=torch.int32)

out = ada_block_sparse_attention(
    query, key, value, mask, count,
    sparse_size=128,
    num_heads=heads,
    num_key_value_heads=heads,
    scale_value=dim**-0.5,
    causal=False,
)
print(out.shape)  # [1, 2, 256, 128]
```
