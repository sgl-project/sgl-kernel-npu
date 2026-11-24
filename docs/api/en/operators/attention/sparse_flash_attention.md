# sparse_flash_attention

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `cce` | Not supported | Supported |
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

`auto` selects `ascendc`; use `impl="cce"` to select raw CCE explicitly.
`torch_native` uses PyTorch operations. CCE is available only on arch35
(3510, Ascend 950); it is not supported on A3/arch22.
An unavailable native implementation raises an error without switching backends.

The AscendC binding is `torch.ops.npu.sgl_sparse_flash_attention`; the CCE
binding is `torch.ops.cce.sparse_flash_attention`. Use the public function
above to retain generation checks and implementation selection.

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
`D_rope` the rope dimension. The AscendC kernel supports `D = 512`,
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
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `cce`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `attention_out` | Same as `query` | Same as `query` |
| `softmax_max` | `[N2, T, N1/N2]` (TND) or `[B, N2, S1, N1/N2]` (BSND) | FP32 |
| `softmax_sum` | Same as `softmax_max` | FP32 |

Usage notes:

- All tensors must be on the same NPU and contiguous.
- With `ascendc` or `torch_native`, `softmax_max`/`softmax_sum` are empty tensors unless `return_softmax_lse` is
  true; `softmax_max + log(softmax_sum)` is the natural-log LSE.
- The `torch_native` reference covers `sparse_mode=0` and `attention_mode=2`,
  which is the DSA production path.

## CCE backend (arch35 only)

The CCE prefill implementation uses the same public signature with a narrower
input contract. Its output is BF16 and its softmax statistics are FP32.

| Input / setting | CCE requirement |
| --- | --- |
| `query` | BF16 `[B, SQ, H, 512]`, `B >= 1`, `1 <= SQ <= 8192`. |
| `query_rope` | BF16 `[B, SQ, H, 64]`. |
| `H` | `16`, `32`, `64` or `128`. |
| `key`, `value` | The same tensor object (`key is value`), BF16 `[1, 8192, 1, 512]`, shared across the batch. |
| `key_rope` | BF16 `[1, 8192, 1, 64]`, shared across the batch. |
| `sparse_indices` | INT32 `[B, SQ, 1, 2048]`; use indices in `0..8191`. |
| Layout / scale | `layout_query=layout_kv="BSND"`, `scale_value=1/24`. |
| Storage | Same NPU, contiguous base-format tensors (ND/NCHW); BF16 base addresses aligned to 32 bytes. Inputs must not require gradients. |

Use explicit token selection: `sparse_mode=0`, `sparse_block_size=1`,
`attention_mode=2`. Supply valid indices directly; do not rely on the general
`-1` stop-at-padding convention for this backend. CCE masks invalid entries
individually rather than stopping at the first one.

The CCE wrapper does not forward `block_table`, sequence-length overrides,
`sparse_mode`, `sparse_block_size`, `attention_mode`, or `pre_tokens`/`next_tokens`
to the kernel. These options do not provide paging, truncation or causal-window
processing on CCE; encode the intended selection in `sparse_indices` instead.

CCE always returns `(attention_out, softmax_max, softmax_sum)`, even when
`return_softmax_lse=False`. The shapes are `[B, SQ, H, 512]`, `[B, 1, SQ, H]`
and `[B, 1, SQ, H]`. No automatic fallback occurs for unsupported configurations.

### Example

Run on arch35 with the CCE operator included in the installed wheel:

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import sparse_flash_attention

batch, q_len, heads = 1, 2, 16
device = "npu"
query = torch.randn(batch, q_len, heads, 512, device=device, dtype=torch.bfloat16)
query_rope = torch.randn(batch, q_len, heads, 64, device=device, dtype=torch.bfloat16)
kv = torch.randn(1, 8192, 1, 512, device=device, dtype=torch.bfloat16)
key_rope = torch.randn(1, 8192, 1, 64, device=device, dtype=torch.bfloat16)
indices = torch.arange(2048, device=device, dtype=torch.int32)
indices = indices.view(1, 1, 1, 2048).expand(batch, q_len, 1, 2048).contiguous()

out, softmax_max, softmax_sum = sparse_flash_attention(
    query, kv, kv, indices, 1.0 / 24.0,
    query_rope=query_rope, key_rope=key_rope,
    layout_query="BSND", layout_kv="BSND",
    sparse_mode=0, sparse_block_size=1, attention_mode=2,
    return_softmax_lse=True, impl="cce",
)
```

For generation-aware CI, use `check_accuracy` or call `require_test_impl` /
`require_distinct_impls` before invoking CCE directly. These test helpers skip
unsupported generations; a direct public API call raises instead.
