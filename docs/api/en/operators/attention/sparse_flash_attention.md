# sparse_flash_attention

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import sparse_flash_attention

def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` selects `ascendc`; `torch_native` uses PyTorch operations.
An unavailable native implementation raises an error without switching backends.

The operator is registered as `torch.ops.npu.sparse_flash_attention`.

## Description

The operator computes DeepSeek sparse attention (MLA) over a selected set of KV
blocks. Each query token carries top-k index entries; the kernel gathers the
`D` non-rope values from `key` and the `D_rope` values from `key_rope`, computes
softmax over the selected positions and multiplies `value`:

```text
positions = pad_truncate(sparse_indices * sparse_block_size, sparse_block_size)
logits    = q @ k[positions].T * scale_value
attn_out  = softmax(logits) @ v[positions]
lse       = logmax(logits) + log(sum(exp(logits - logmax(logits))))
```

`-1` index entries are padding and stop the selection. Positions at or beyond
the visible KV length are ignored.

## Parameters and return values

`B` is the batch size, `S1` the query length, `T` the packed token count for
TND, `N1`/`N2` the query/KV head counts, `D` the non-rope head dimension and
`D_rope` the rope dimension. The native kernel supports `D = 512`,
`D_rope = 64`, `N2 = 1` and `N1` in `1..64` or `128`.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `query` | `[T, N1, D]` or `[B, S1, N1, D]` | FP16 / BF16 | TND or BSND query. |
| `key` | `[Bn, Bs, N2, D]`, `[T, N2, D]` or `[B, S, N2, D]` | Same as `query` | Paged, TND or BSND KV. |
| `value` | Same as `key` | Same as `query` | KV value. |
| `sparse_indices` | `[T, N2, K]` or `[B, S1, N2, K]` | INT32 | Selected KV block indices, `-1` padded. |
| `scale_value` | — | float | Softmax scale. |
| `block_table` | `[B, maxBlocks]` | INT32 / `None` | Required for `PA_BSND`, must be `None` otherwise. |
| `actual_seq_lengths_query` | `[B]` | INT32 / `None` | TND query prefix sums; required for `TND`. |
| `actual_seq_lengths_kv` | `[B]` | INT32 / `None` | Valid KV length (prefix sums for TND KV). |
| `query_rope` | `[T, N1, D_rope]` or `[B, S1, N1, D_rope]` | Same as `query` | Required for `attention_mode=2`. |
| `key_rope` | `[Bn, Bs, N2, D_rope]`, `[T, N2, D_rope]` or `[B, S, N2, D_rope]` | Same as `query` | Required for `attention_mode=2`. |
| `sparse_block_size` | — | int / `1` | KV positions per index entry. |
| `layout_query` | — | `"BSND"` | `BSND` or `TND`. |
| `layout_kv` | — | `"BSND"` | `BSND`, `TND` or `PA_BSND`. |
| `sparse_mode` | — | int / `3` | Sparse selection mode; `0` selects the given top-k only. |
| `pre_tokens`, `next_tokens` | — | int / `INT64_MAX` | Causal window bounds. |
| `attention_mode` | — | int / `2` | `2` (MLA) only. |
| `return_softmax_lse` | — | `False` | Also return the softmax max/sum split. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `attention_out` | Same as `query` | Same as `query` |
| `softmax_max` | `[N2, T, N1/N2]` (TND) or `[B, N2, S1, N1/N2]` (BSND) | FP32 |
| `softmax_sum` | Same as `softmax_max` | FP32 |

Usage notes:

- All tensors must be on the same NPU and contiguous.
- `softmax_max`/`softmax_sum` are empty tensors unless `return_softmax_lse` is
  true; `softmax_max + log(softmax_sum)` is the natural-log LSE.
- The `torch_native` reference covers `sparse_mode=0` and `attention_mode=2`,
  which is the DSA production path.
