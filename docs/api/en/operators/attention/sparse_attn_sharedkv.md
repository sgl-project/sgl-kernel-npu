# sparse_attn_sharedkv

## Hardware support

| Implementation | arch22 (2201) | arch35 (3510) |
| --- | --- | --- |
| `ascendc` | Supported | Not supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import sparse_attn_sharedkv

def sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=0.0,
    cmp_ratio=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=128,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

`auto` selects `ascendc`; `torch_native` uses PyTorch operations.
An unavailable native implementation raises an error without switching backends.

## Description

The operator computes sparse attention over shared KV. The current
implementation covers the sliding-window (SWA) template with paged KV:

```text
aligned = seqused_kv[b] - q_len + s
left    = max(aligned - ori_win_left, 0)
right   = min(aligned + ori_win_right, seqused_kv[b] - 1)
logits  = [sink_h, q[b, s, h] @ k[left:right+1].T * softmax_scale]
attn_out[b, s, h] = softmax(logits)[1:] @ k[left:right+1]
```

Here `k` is the logical sequence reconstructed from `ori_kv` through
`ori_block_table`, `h // g` selects the KV head, and `g = n1 / n2`.
`sinks` participates as one extra zero-valued virtual token.

`layout_kv` supports `PA_ND` only, `ori_mask_mode` supports `4` only, and
`return_softmax_lse` supports `False` only.

## Parameters and return values

`B` is the batch size, `S1` the query length, `T = B * S1` the token count for
TND, `N1`/`N2` the query/KV head counts, and `D` the head dimension.

| Parameter | Shape | Dtype / default | Description |
| --- | --- | --- | --- |
| `q` | `[B, S1, N1, D]` or `[T, N1, D]` | FP16 / BF16 | BSND or TND layout. |
| `ori_kv` | `[blockNum, blockSize, N2, D]` | Same as `q` | Paged KV cache. |
| `cmp_kv` | Same as `ori_kv` | Same as `q` | Compressed KV; not covered by the reference. |
| `ori_sparse_indices` | Any | INT32 | Optional native input. |
| `cmp_sparse_indices` | Any | INT32 | Compressed-KV sparse indices. |
| `ori_block_table` | `[B, maxBlocks]` | INT32 | Paged KV block table. |
| `cmp_block_table` | `[B, maxBlocks]` | INT32 | Compressed-KV block table. |
| `cu_seqlens_q` | `[B + 1]` | INT32 / `None` | Required for TND; must be `None` for BSND. |
| `cu_seqlens_ori_kv` | `[B + 1]` | INT32 / `None` | Optional native input. |
| `cu_seqlens_cmp_kv` | `[B + 1]` | INT32 / `None` | Compressed-KV cumulative lengths. |
| `seqused_q` | `[B]` | INT32 / `None` | Optional native input. |
| `seqused_kv` | `[B]` | INT32, required | Valid KV length per batch. |
| `sinks` | `[N1]` | FP32, required | Attention sink values. |
| `metadata` | `[1024]` | INT32, required | Core work-partition metadata. |
| `softmax_scale` | — | float / `0.0` | Softmax scale. |
| `cmp_ratio` | — | int / `0` | Compression ratio. |
| `ori_mask_mode` | — | int / `4` | `4` (sliding window) only. |
| `cmp_mask_mode` | — | int / `3` | Compressed-KV mask mode. |
| `ori_win_left`, `ori_win_right` | — | int / `128`, `0` | Window bounds. |
| `layout_q` | — | `"BSND"` | `BSND` or `TND`. |
| `layout_kv` | — | `"PA_ND"` | `PA_ND` only. |
| `return_softmax_lse` | — | `False` | `False` only. |
| `impl` | — | `"auto"` | Keyword-only: `auto`, `ascendc`, `torch_native`. |

| Return | Shape | Dtype |
| --- | --- | --- |
| `attn_out` | Same as `q` | Same as `q` |
| `softmax_lse` | `[0]` | FP32 |

Usage notes:

- All tensors must be on the same NPU; the current native implementation
  requires `N1 = 64` and `N2 = 1` (a single KV head).
- `ori_kv`, `sinks` and `metadata` are required; `ori_block_table` and
  `seqused_kv` are required for `PA_ND`.
- `torch_native` implements SWA with `PA_ND` only; compressed-KV templates are
  out of scope.
