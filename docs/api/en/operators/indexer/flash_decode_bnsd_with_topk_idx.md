# flash_decode_bnsd_with_topk_idx

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

`ascendc` is not implemented: the operator only provides `triton` and `torch_native`.

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or
passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

def flash_decode_bnsd_with_topk_idx(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    max_seqlen,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    score_type="max",
    disable_index_value=False,
    num_kv_chunks=None,
    max_num_kv_chunks=8,
    use_triton_topk=None,
    num_topk_chunks=None,
    use_chunked_score=True,
    score_blocks_per_chunk=16,
    score_max_chunks=32,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    packed_seq_lens=False,
    runtime_fill_only=False,
    runtime_score_short_max_blocks=0,
    runtime_score_short_chunks=0,
    fused_append_local=False,
    use_native=True,
    *,
    impl="auto",
):
    ...
```

## Description

Decode attention over a paged KV cache: it scores KV blocks and returns block-level topk indices, or the
attention output together with the indices on the full attention path. Three mutually exclusive branches:

- **① Empty topk**: `disable_index_value=True` and `topk<=0` returns `(None, [QH, B, 0])`.
- **② Score only**: `disable_index_value=True` and `topk>0` scores every KV block per query row and head and
  returns `(None, topk_idx)`; production decode/verify uses this path.
- **③ Full attention**: `disable_index_value=False` produces the attention output and the block scores in one
  KV pass and returns `(o, topk_idx)`.

Supported scope: branches ① and ② (score-only) are fully supported, including `score_type`
`"max"`/`"lse"`, `init_blocks`/`local_blocks` forcing, the direct page source and `block_table`,
`packed_seq_lens`, `sanitize_page_ids` and `fused_append_local`. Branch ③ remains in the Triton
implementation only; it is out of the supported and validated scope and `torch_native` rejects it.

Block score reductions (`max` and `lse`; $D$ is head_dim, $\sigma$ is `sm_scale` and defaults to $D^{-1/2}$):

$$
s_{i,h,b}=\max_{j\in b}\left(\sigma\,q_i k_j^{T}\right),
\qquad
s_{i,h,b}=\log\sum_{j\in b}\exp\left(\sigma\,q_i k_j^{T}\right)
$$

Each query row and head returns the `topk` blocks with the highest scores. Typical use cases:

- **decode**: `disable_index_value=True` and `topk>0`, with the current token's `q` and the paged KV cache;
  returns block indices shaped `[QH, B, topk]`. `fused_append_local=True` is commonly used to append the
  causal local block in the last slot.
- **verify / packed**: `packed_seq_lens=True` with `seq_lens` holding `[B*G]` per-row lengths, so several
  draft tokens of one request are processed together; both `block_table` and the direct page source work.
- **full attention (Triton only)**: `disable_index_value=False` returns the attention output `o` and
  `topk_idx` from one KV pass.

## Parameters and returns

The public operator forwards arguments without input checks; each implementation validates key
constraints such as dtypes, page-source combinations, capacity, the packed `num_kv_heads`
requirement, and the `seq_lens`/`block_table` dimensions. Other shape constraints are the caller's
responsibility.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[B, QH, D]` | `float16`, `bfloat16` | required | Queries on the NPU; `QH` must be a multiple of `KVH` (GQA). |
| `sink` | `torch.Tensor` or `None` | `[QH, D]` | same as `q` | required (may be `None`) | Attention sink, used only by the Triton full attention path; the production scenario always passes `None` and torch_native rejects that branch. |
| `k_cache_bnsd` | `torch.Tensor` | `[num_blocks, block_size, KVH, D]` | same as `q` | required | Key cache read block by block. |
| `v_cache_bnsd` | `torch.Tensor` or `None` | same as `k_cache_bnsd` | same as `q` | required (may be `None` for score-only; required for attention) | Ignored by the score-only path; used only by the Triton full attention path. |
| `block_table` | `torch.Tensor` or `None` | `[B, max_num_blocks]` | `int32`, `int64` | required (may be `None`) | Logical to physical block map; alternative to the direct page source, and the only mode of the full attention path (Triton only); its column count must cover the actual block count, otherwise the table is read out of bounds (not asserted) |
| `seq_lens` | `torch.Tensor` | `[B]` or `[B*QH//KVH]` | `int32`, `int64` | required | Causal KV lengths; per-row lengths when `packed_seq_lens=True`. |
| `max_seqlen` | `int` | — | — | required | KV length upper bound used to size the score chunks; it does not change the selected blocks when scores are distinct; Triton only. |
| `block_size` | `int` | — | — | required | Tokens per block; must match the cache and be a power of two (not asserted; production uses 128). |
| `topk` | `int` | — | — | required | Number of returned block indices. |
| `init_blocks` | `int` | — | — | required | Leading blocks whose candidate score becomes `1e30`; two or more blocks of this kind tie, so their order and the pick beyond `topk` are undefined |
| `local_blocks` | `int` | — | — | required | Trailing blocks whose candidate score becomes `1e29`; wins over `init_blocks`; two or more blocks of this kind tie, so their order and the pick beyond `topk` are undefined |
| `sm_scale` | `float` or `None` | — | — | `None` | Softmax scale; `D ** -0.5` when `None`. |
| `score_type` | `str` | — | — | `"max"` | `"max"` or `"lse"`, the per-block token reduction |
| `disable_index_value` | `bool` | — | — | `False` | `True` selects the score-only path, `False` the full attention path; torch_native supports `True` only. |
| `num_kv_chunks` | `int` or `None` | — | — | `None` | KV chunks of the full attention path; chosen from the vector-core count when `None`; Triton only. |
| `max_num_kv_chunks` | `int` | — | — | `8` | Power-of-two cap of the automatic chunk choice; Triton only. |
| `use_triton_topk` | `bool` or `None` | — | — | `None` | Compatibility parameter, no effect. |
| `num_topk_chunks` | `int` or `None` | — | — | `None` | Compatibility parameter, no effect. |
| `use_chunked_score` | `bool` | — | — | `True` | Compatibility parameter, no effect. |
| `score_blocks_per_chunk` | `int` | — | — | `16` | Target block tiles per score program; Triton only. |
| `score_max_chunks` | `int` | — | — | `32` | Upper bound of the score chunk count; Triton only. |
| `req_to_token` | `torch.Tensor` or `None` | `[num_requests, max_context]` | `int32`, `int64` | `None` | Direct page source holding the token slot per position; requires `req_pool_indices`. |
| `req_pool_indices` | `torch.Tensor` or `None` | `[B]` | `int32`, `int64` | `None` | Batch rows into `req_to_token`. |
| `max_num_blocks` | `int` or `None` | — | — | `None` | Page-table capacity. Always required by the direct mode and must satisfy `max_num_blocks * block_size <= req_to_token.shape[1]` (checked even by the empty-topk branch); with `block_table` it is required only for the score-only path with `fused_append_local=True`. |
| `num_pages` | `int` or `None` | — | — | `None` | Physical page count; required by the direct mode. |
| `sanitize_page_ids` | `bool` | — | — | `False` | Direct mode only: clamps physical block ids to `[0, num_pages-1]`. |
| `packed_seq_lens` | `bool` | — | — | `False` | `seq_lens` holds per-row lengths (verify packing); score-only only and requires `num_kv_heads == 1`. |
| `runtime_fill_only` | `bool` | — | — | `False` | Pick the direct-fill candidate path from the runtime block count; Triton only. |
| `runtime_score_short_max_blocks` | `int` | — | — | `0` | Block-count threshold of the short-context adaptive chunking; when enabled (non-zero) it must be `>0` and given together with the next parameter, and the default 0 disables it; Triton only. |
| `runtime_score_short_chunks` | `int` | — | — | `0` | Chunk count used below the threshold; when enabled (non-zero) it must be a power of two and is truncated to the total chunk count when larger, and the default 0 disables it; Triton only. |
| `fused_append_local` | `bool` | — | — | `False` | Fuse candidate merge with the causal local block append; the last slot is emitted only for ② with `topk>0`. |
| `use_native` | `bool` | — | — | `True` | Compatibility parameter, no effect (there is no native backend). |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; one of `"auto"`, `"triton"`, `"torch_native"`. |

| Return | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `o` | `torch.Tensor` or `None` | `[B, QH, D]` | same as `q` | `None` on the score-only path; a new tensor on the full attention path (Triton only). |
| `topk_idx` | `torch.Tensor` | `[QH, B, topk]`, or `[QH, B, topk+1]` for ② with `topk>0` and `fused_append_local` | `int32` | `-1` marks an invalid, out-of-range or de-duplicated slot; once valid candidates are exhausted block 0 is repeated. The order of equally scored blocks is undefined. |

Additional notes: the empty-topk branch returns `[QH, B, 0]`; the direct page source only supports the
score-only path; `sink` only affects the full attention denominator.

## triton backend

- Uses CANN Triton kernels and covers branches ①②③; branch ③ is kept for compatibility only and is out of
  the supported and validated scope. The number of score chunks follows `max_seqlen` and the runtime block
  count.
- The `init_blocks`/`local_blocks` rewrite only affects topk candidate scores.
- The direct page source gathers logical to physical blocks inside the kernel and can clamp them.
- `packed_seq_lens=True` with `num_kv_heads>1` is unsupported because the score kernel and the fused merge use
  different row mappings (production verify uses `num_kv_heads=1`).

## torch_native backend

- Independent PyTorch implementation; it never calls Triton and implements **only the score-only branch**
  (`disable_index_value=True`): `score_type` `"max"`/`"lse"`, `init_blocks`/`local_blocks` forcing, the
  direct page source and `block_table`, `packed_seq_lens`, `sanitize_page_ids` and `fused_append_local`;
  `topk<=0` returns the empty index tensor.
- The full attention branch (`disable_index_value=False`) has no reference implementation and is rejected
  with `AssertionError`.
- Ignores 12 scheduling/compatibility parameters: `max_seqlen`, `num_kv_chunks`, `max_num_kv_chunks`,
  `score_blocks_per_chunk`, `score_max_chunks`, `runtime_fill_only`, `runtime_score_short_max_blocks`,
  `runtime_score_short_chunks`, `use_native`, `use_triton_topk`, `num_topk_chunks`, `use_chunked_score`.
- Processes the whole KV range in one pass; memory and latency grow linearly with the context length, so it is
  meant for correctness checks only; `o` is always `None`.
- `topk_idx` matches Triton when scores are distinct; the order of equally scored blocks is undefined.
- `packed_seq_lens` combined with `runtime_score_short_max_blocks` may truncate the candidate list when a row
  group crosses the threshold; equivalence with Triton is not guaranteed.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

batch, q_heads, kv_heads, head_dim = 1, 1, 1, 128
block_size, topk = 128, 16
seq_len = 4096
num_blocks = seq_len // block_size

# Distinct block scores keep the int32 indices bit-comparable across implementations.
q = torch.zeros(batch, q_heads, head_dim, device="npu", dtype=torch.bfloat16)
q[:, :, 0] = 1.0
k_cache = torch.zeros(
    num_blocks, block_size, kv_heads, head_dim, device="npu", dtype=torch.bfloat16
)
coefficients = torch.arange(num_blocks, device="npu", dtype=torch.float32)
k_cache[:, :, :, 0] = coefficients.view(-1, 1, 1)
block_table = torch.arange(num_blocks, device="npu", dtype=torch.int32).view(1, -1)
seq_lens = torch.tensor([seq_len], device="npu", dtype=torch.int32)

print(list_impls(flash_decode_bnsd_with_topk_idx))  # ['auto', 'torch_native', 'triton']
_, topk_idx = flash_decode_bnsd_with_topk_idx(
    q, None, k_cache, None, block_table, seq_lens,
    max_seqlen=seq_len, block_size=block_size, topk=topk,
    init_blocks=0, local_blocks=0,
    disable_index_value=True, fused_append_local=True, max_num_blocks=num_blocks,
)
print(topk_idx.shape)  # [1, 1, 17]

_, topk_ref = flash_decode_bnsd_with_topk_idx(
    q, None, k_cache, None, block_table, seq_lens,
    max_seqlen=seq_len, block_size=block_size, topk=topk,
    init_blocks=0, local_blocks=0,
    disable_index_value=True, fused_append_local=True, max_num_blocks=num_blocks,
    impl="torch_native",
)
torch.testing.assert_close(topk_idx, topk_ref)
```
