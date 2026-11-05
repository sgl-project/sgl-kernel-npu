# flash_prefill_bnsd_with_topk_idx

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`; omitting `impl` has the same effect. There is no runtime fallback.

```python
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_with_topk_idx

def flash_prefill_bnsd_with_topk_idx(
    q,
    k_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    topk,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    per_query_seq_lens=None,
    *,
    impl="auto",
):
    ...
```

## Description

During prefill, this operator scores each logical KV block for each query and returns the
selected block ids. Unlike `flash_prefill_bnsd_indexer`, it takes no V cache and computes no
attention output.

For request `r`, its packed query rows are `cu_seqlens[r]:cu_seqlens[r + 1]`. Let
`P = block_size_k` and `L_r = seq_lens[r]`. Logical block `b`, with `0 <= b < ceil(L_r / P)`, contains
key positions `p = b * P + j`, where `0 <= j < P`. Index head `h` uses KV head
`floor(h / gqa_group_size)`, where `gqa_group_size = num_idx_heads / num_kv_heads`.

**Visible keys.** The current implementation scores keys satisfying `0 <= p < L_r` and
`p <= t`, where `t` is the global packed query-row index. It neither subtracts the
request's query offset nor adds its cached KV prefix length. `per_query_seq_lens` only
affects subsequent block selection; it does not change this mask. With cached prefixes or multiple requests, this differs from standard
causal attention based on request-local positions.

For packed query row `t` and key position `p`, the log2-domain logit is

```math
x_{t,p} = (q_t \cdot k_p)\,\text{sm\_scale}\,\log_2 e
```

The block score is reduced according to `score_type`. In these formulas, `p` ranges only
over visible keys in the block; head indices are omitted for brevity:

```math
\text{score}[h,t,b] =
\begin{cases}
\max_{p \in b} x_{t,p}, & \text{score\_type} = \text{"max"} \\
\log_2 \sum_{p \in b} 2^{x_{t,p}}, & \text{score\_type} = \text{"lse"}
\end{cases}
```

A block with no visible key has score `-inf`. Scores are internal intermediates and are not
returned.

**Block selection.** Let `K = ceil(max(seq_lens) / P)`. Returned block ids are zero-based
logical block indices within each request, rather than physical page ids. There are two paths:

- **Fused branch**: `per_query_seq_lens` is not `None` and `num_idx_heads == num_kv_heads`.
  For row `t`, let `query_pos = max(per_query_seq_lens[t] - 1, 0)` and
  `local_t = min(query_pos // P, K - 1)`. The local block is the logical KV block containing
  the query. Each round takes the highest score among all remaining blocks, breaking ties
  by the smallest block id. It writes the id only when the score is greater than `-inf` and
  `block_id * P <= query_pos`; otherwise it writes `-1`, without picking a replacement for
  that slot. After `topk` rounds, it appends `local_t`, or `-1` if that block was already selected.
- **Fallback branch**: every other case. `torch.topk` over all `K` columns of `score` returns
  the first `min(topk, K)` block ids; when `topk > K` the tail is padded with `-1`. This branch
  does **not** drop `-inf` columns and does **not** append the local block.

The width of the result therefore depends on the branch: `topk + 1` for the fused branch and
`topk` for the fallback branch. Ties in the fallback branch follow `torch.topk`, which does not guarantee the smallest block id.

## Parameters and return values

`bs` is the request count and `total_q` is the total number of packed query rows. Tensor shapes
below list dimensions in their stored order.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[total_q, num_idx_heads, head_dim]` | `float16`, `bfloat16` | Required | Index queries. The Triton backend requires an NPU tensor. `num_idx_heads` must be divisible by the number of KV heads (asserted). |
| `k_cache_bnsd` | `torch.Tensor` | `[num_pages, page_size, num_kv_heads, head_dim]` | Same as `q` | Required | Paged index-key cache. `page_size == block_size_k`; page ids must be valid (caller convention). |
| `cu_seqlens` | `torch.Tensor` | `[bs + 1]` | `int32` or `int64` | Required | Prefix sums delimiting each request's packed query rows; starts at zero, is non-decreasing, and ends at `total_q` (caller convention). |
| `seq_lens` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Total KV length per request; determines the block count and the score width. Callers must ensure `bs > 0` and positive KV lengths for requests with queries. |
| `req_to_token` | `torch.Tensor` | `[num_reqs, max_context]` | `int32` or `int64` | Required | Maps request token positions to token slots; used only when `qblock_mappings` is `None`. Logical block `b` maps to physical page `req_to_token[req_pool_indices[r], b * block_size_k] // block_size_k`; the table width must cover the block-start columns read. |
| `req_pool_indices` | `torch.Tensor` | `[bs]` | `int32` or `int64` | Required | Row indices into `req_to_token`; used only when `qblock_mappings` is `None`. |
| `block_size_q` | `int` | — | — | Required | Positive query grouping size. Non-powers of two are accepted. |
| `block_size_k` | `int` | — | — | Required | Tokens per logical KV block; positive, equal to `page_size`, and a power of two (caller convention). |
| `topk` | `int` | — | — | Required | Selection slots per query. The fused branch requires `topk > 0` (asserted); the fallback branch accepts `topk == 0` and returns zero-width indices. |
| `sm_scale` | `float` | — | — | Required | Logit scale, commonly `head_dim ** -0.5`. |
| `score_type` | `str` | — | — | `"max"` | Block-score reduction: `"max"` or `"lse"`; other values raise `AssertionError`. |
| `qblock_mappings` | `tuple` or `None` | — | — | `None` | Optional precomputed six-tuple: `qb_to_qstart`, `qb_to_qblock`, `qb_seq_lens`, and `qb_qend` (`int32`, each length `all_seqblock_q`), `block_table` (`int32`, `[all_seqblock_q, max_blocks]`, physical page ids), and Python `int` `all_seqblock_q`. Built internally when omitted. |
| `per_query_seq_lens` | `torch.Tensor` or `None` | `[total_q]` | `int32` | `None` | Per-query KV count including the current token (position plus 1); used only for block selection and the local block, not the scoring mask. Ignored when head counts differ. The fused branch asserts `int32` dtype and length `total_q`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `topk_idx` (fused branch) | `torch.Tensor` | `[num_idx_heads, total_q, topk + 1]` | `int32` | New tensor on `q`'s device. Empty slots are `-1`; the last slot is the local block (or `-1` when already selected). Inputs are not modified. |
| `topk_idx` (fallback branch) | `torch.Tensor` | `[num_idx_heads, total_q, topk]` | `int32` | New tensor on `q`'s device. The tail is `-1` when `topk` exceeds the block count; `-inf` columns are not dropped. Inputs are not modified. |

Precomputed mappings must match the current inputs: `qb_to_qstart` and `qb_qend` are each
request's packed query bounds (end exclusive); `qb_to_qblock` is its query-group index, and
`qb_seq_lens` is its total KV length. `all_seqblock_q` counts query groups, and
`max_blocks >= ceil(max(seq_lens) / block_size_k)`. Reuse `qblock_mappings` only while this
metadata and the page mapping remain unchanged.

Backends check the assertions noted in the table. Callers must ensure the remaining shape,
dtype, block-size, and mapping requirements, and place all tensors on the same device. Triton requires the `per_query_seq_lens` and four one-dimensional
mapping tensors it uses to be contiguous. Use `max(seq_lens) == 0` only for inputs without queries, rather than
nonempty-query block selection.

## triton backend

Triton computes block scores; block selection runs separately.
The fused selection branch uses Triton; the fallback branch uses `torch.topk`.
`MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP` sets the number of consecutive KV blocks processed
per step. It defaults to `1`, accepts `1/2/4`, and asserts on other integer values; it does
not change the operation semantics.

The public entry reads device scalars and synchronizes, so it cannot be used directly during
graph capture. Supplying `qblock_mappings` does not remove all synchronization.

## torch_native backend

An independent PyTorch correctness reference, computing by request and KV block with the same
mask and selection rules. Floating-point rounding
can make the two backends rank nearly tied blocks differently.

## Example

Requires the package that ships this operator and an NPU environment.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_with_topk_idx

page_size = 128
num_pages = 1  # ceil(4 / page_size)
q = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(num_pages, page_size, 1, 128, dtype=q.dtype, device="npu")

# The single request uses physical page 0.
req_to_token = torch.zeros(1, num_pages * page_size, dtype=torch.int32, device="npu")

topk_idx = flash_prefill_bnsd_with_topk_idx(
    q,
    k_cache,
    cu_seqlens=torch.tensor([0, 4], dtype=torch.int32, device="npu"),
    seq_lens=torch.tensor([4], dtype=torch.int32, device="npu"),
    req_to_token=req_to_token,
    req_pool_indices=torch.tensor([0], dtype=torch.int32, device="npu"),
    block_size_q=4,
    block_size_k=page_size,
    topk=2,
    sm_scale=128**-0.5,
    per_query_seq_lens=torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="npu"),
    impl="triton",  # Use "torch_native" for the reference implementation.
)
print(topk_idx.shape, topk_idx.dtype)  # torch.Size([1, 4, 3]), torch.int32
print(topk_idx[0, -1].tolist())        # [0, -1, -1]
```
