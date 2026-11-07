# flash_decode_bnsd_with_gqa_share_sparse

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `ascendc` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import flash_decode_bnsd_with_gqa_share_sparse

def flash_decode_bnsd_with_gqa_share_sparse(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    use_native=True,
    *,
    impl="auto",
):
    ...
```

`auto` selects `ascendc` when `use_native=True`, `sink is None`, `D=128`, `block_size=128`
and the native op is available in the current build; otherwise it selects `triton`. Explicit
`triton`, `ascendc` and `torch_native` select that backend directly. Backend execution errors
are propagated without retrying another backend.

## Description

Computes sparse attention over KV blocks selected by the caller; this operator does not select
top-k blocks. Query heads in each GQA group share both a KV head and its selected block list.
`B` is the number of query rows: a decode batch, or flattened queries for verify/prefill.
`QH` and `KVH` are the query and KV head counts, `G=QH/KVH`, `D` is the head dimension,
`BS=block_size`, and `NB` is the number of physical cache pages. Query head `h` uses KV head
`kvh=h//G`.

For row `b`, each selected logical block `lb=topk_idx[kvh,b,t]` contributes token offsets
`n=0..BS-1` with `lb>=0` and `lb*BS+n < seq_lens[b]`. Negative block ids mark empty slots.
Tokens beyond the last partial block are masked. For a `[QH,B,T]` index tensor, only the first
query head of each GQA group supplies the shared block list. Duplicate blocks are counted repeatedly;
there is no de-duplication.

The physical page `phys(t)` comes from one of two page sources:

- **Block table:** `block_table[b,lb]`.
- **Direct map:** `req_to_token[req_pool_indices[b],lb*BS] // BS`. Only block-start slots are
  read; each logical block must map to one physical cache page. With `sanitize_page_ids=True`,
  the resulting page id is clamped to `[0,num_pages-1]`.

Let $\mathcal{S}_{b,kvh}$ contain the valid selected token occurrences described above. For each query head:

$$
s_{tn} = \mathrm{sm\_scale}\; q_{b,h}^{T} K_{\mathrm{phys}(t),n,kvh}, \qquad
z_{\mathrm{sink}} =
\begin{cases}
\exp(\mathrm{sm\_scale}\; q_{b,h}^{T} sink_h), & sink \ne \mathrm{None}, \\
0, & sink = \mathrm{None},
\end{cases}
$$

$$
out_{b,h} =
\frac{\sum_{(t,n)\in\mathcal{S}_{b,kvh}} \exp(s_{tn}) V_{\mathrm{phys}(t),n,kvh}}
{z_{\mathrm{sink}} + \sum_{(t,n)\in\mathcal{S}_{b,kvh}} \exp(s_{tn})}
$$

The sums run only over valid token occurrences. A sink contributes a query-dependent softmax
logit and zero value; it is a vector `[QH,D]`, not a precomputed scalar logit. An empty selection
returns zero, including when a sink is supplied. Output shape and dtype match `q`.

There is no additional causal mask or query-position input. To implement causal verify/prefill,
flatten queries into rows and supply the visible KV prefix length for each row in `seq_lens`.
Queries from the same request may share a `req_pool_indices` entry.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[B, QH, D]` | `torch.float16`, `torch.bfloat16` | required | Queries; `B`, `QH`, `D` must be positive and `QH % KVH == 0`. |
| `sink` | `torch.Tensor` / `None` | `[QH, D]` | same as `q` | required (may be `None`) | Optional sink vector; shape and dtype must match the table. `ascendc` accepts only `None`. |
| `k_cache_bnsd` | `torch.Tensor` | `[NB, BS, KVH, D]` | same as `q` | required | Paged key cache; `NB` is the physical page count. |
| `v_cache_bnsd` | `torch.Tensor` | same as `k_cache_bnsd` | same as `q` | required | Paged value cache; shape and dtype must match the key cache. |
| `block_table` | `torch.Tensor` / `None` | `[B, max_num_blocks]` | `int32`, `int64` | required (or the direct map) | Logical-to-physical page map. Pass `None` when using the direct map. |
| `seq_lens` | `torch.Tensor` | `[B]` | `int32`, `int64` | required | Visible KV token count per row, not the last token index. Contiguous, nonnegative, within page-map capacity and signed int32 range. |
| `block_size` | `int` | — | — | required | Tokens per cache page; must match the cache second dimension. Triton requires a positive power of two; `ascendc` requires 128. |
| `topk_idx` | `torch.Tensor` | `[KVH or QH, B, T]` | `int32`, `int64` | required | Selected logical block ids, `T > 0`; use `-1` for empty slots. Nonnegative ids must be within page-map capacity. A `QH` leading dimension uses the first head of each GQA group. |
| `sm_scale` | `float` / `None` | — | — | `None` | Softmax scale; `None` means `D ** -0.5`. |
| `num_topk_chunks` | `int` / `None` | — | — | `None` | Triton only: positive power of two, at most `T`. `None` uses a heuristic based on batch size, KV heads, `T` and vector-core count; small batches with short lists use one chunk. |
| `max_num_topk_chunks` | `int` | — | — | `8` | Triton only: positive cap for automatic chunk selection; ignored when `num_topk_chunks` is explicit. |
| `req_to_token` | `torch.Tensor` / `None` | `[num_requests, max_context]` | `int32`, `int64` | `None` | Direct map from request and token position to physical token slot. Mutually exclusive with `block_table`; requires `req_pool_indices`. |
| `req_pool_indices` | `torch.Tensor` / `None` | `[B]` | `int32`, `int64` | `None` | Contiguous request-row indices into `req_to_token`, one per query row; supplied exactly when `req_to_token` is. |
| `max_num_blocks` | `int` / `None` | — | — | `None` | Required and positive for the direct map; `max_num_blocks * BS <= req_to_token.shape[1]`. Ignored with `block_table`. |
| `num_pages` | `int` / `None` | — | — | `None` | Required and positive for the direct map; set to `NB`. Used as the clamping bound when sanitizing; ignored with `block_table`. |
| `sanitize_page_ids` | `bool` | — | — | `False` | Clamp direct-map physical page ids to `[0, num_pages-1]`; ignored with a block table. |
| `topk_blocks_per_step` | `int` | — | — | `1` | Triton only: selected blocks fused per step. Effective value is `max(1, int(value))` and must be a power of two. Keep 1 for `D=128, BS=128`; see backend limits below. |
| `prefetch_idx` | `bool` | — | — | `False` | Triton only: prefetch block and page indices; effective only when `topk_blocks_per_step` resolves to 1. |
| `num_warps` | `int` / `None` | — | — | `None` | Triton launch setting; `None` uses 4. |
| `num_stages` | `int` / `None` | — | — | `None` | Triton launch setting; `None` uses 2. |
| `use_native` | `bool` | — | — | `True` | Used only by `auto`: permits the native route under the conditions above. `False` forces `auto` to choose Triton. Ignored by explicit backends. |
| `impl` | `str` | — | — | `"auto"` | Keyword only: `"auto"`, `"triton"`, `"ascendc"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, QH, D]` | same as `q` | New tensor on `q.device`; no input is modified. |

**Caller requirements:** all supplied tensors must be on the same NPU. Use contiguous inputs
for portable calls across backends, especially `seq_lens` and `req_pool_indices`, which Triton
reads without a stride argument. `KVH` and page-map dimensions must be positive; all tensor shapes
must agree with the table. Nonnegative logical block ids must be smaller than
`block_table.shape[1]` or `max_num_blocks`, even if the block lies beyond the visible prefix.
Request-row indices and physical page ids must be in range. Page-id sanitization only clamps
physical pages on the direct path; it does not validate logical ids, request rows or block-table
entries. Use finite queries, sinks and valid K/V values.

**Asserted constraints:** Triton and AscendC check query/cache dtypes and matching K/V
shapes. Both check the topk leading dimension when normalizing GQA indices. Triton additionally checks cache dimensions, batch sizes,
GQA divisibility, page-source arguments and chunk/fusion settings. AscendC additionally requires
`sink=None, D=128, BS=128`. The PyTorch reference does not repeat these assertions; callers
must satisfy the table and shared requirements for every backend.

## triton backend

Uses online softmax over selected blocks. `num_topk_chunks` splits the topk list into parallel
chunks; one chunk avoids a merge launch, while multiple chunks require temporary partial outputs.
Accumulators are fp32; softmax weights are cast to the K/V dtype before multiplication by V,
and partial outputs are stored in `q.dtype`. Chunking can therefore change floating-point results.

For `D=128, BS=128`, `topk_blocks_per_step=2` exceeds the tested A2/A3 UB capacity and fails
compilation with `ub overflow`; keep this setting at 1 for Triton. `auto` does not retry with
different launch settings after a compilation failure.

## ascendc backend

Calls `torch.ops.npu.npu_sparse_attention_score` from `csrc/sparse_attention_score`.
Requires an arch22 build containing the native op, `D=128`, `BS=128`, and `sink=None`.
An explicit request raises an error when the native op is unavailable; unsupported sizes or a
non-None sink are rejected by assertions. This adapter also requires Triton for index preparation.

Accepts either page source and normalizes per-query-head indices to GQA-shared indices. Uses
`inner_precise=4` for high-precision accumulation. Chunking, block fusion, prefetch and launch
settings are accepted but ignored.

## torch_native backend

Independent PyTorch accuracy reference using fp32 scores, weights and accumulation, then casting
the result to `q.dtype`. It gathers selected blocks and loops over query rows and KV heads;
it is intended for correctness comparisons. Chunking, block fusion, prefetch, launch settings
and `use_native` are ignored.

Reduction order, weight rounding and intermediate storage differ across backends, so results
need not be bitwise identical. Cancellation and large V magnitudes can increase the differences;
choose accuracy tolerances for the actual dtype and workload. Triton converts `seq_lens` to
int32, whereas the reference uses int64; valid inputs must stay within signed int32 range.

## Example

Requires this package and the NPU runtime. This example uses a block table; for the direct map,
pass `block_table=None` and supply `req_to_token`, `req_pool_indices`, `max_num_blocks` and `num_pages`.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import flash_decode_bnsd_with_gqa_share_sparse

B, QH, KVH, D, BS, NB = 1, 8, 1, 128, 128, 8

torch.manual_seed(0)
q = torch.randn(B, QH, D, dtype=torch.bfloat16, device="npu")
k_cache = torch.randn(NB, BS, KVH, D, dtype=torch.bfloat16, device="npu")
v_cache = torch.randn_like(k_cache)
topk_idx = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32, device="npu")  # [KVH, B, topk]
seq_lens = torch.tensor([385], dtype=torch.int32, device="npu")

# Logical blocks map to physical pages with the same ids.
block_table = torch.arange(NB, dtype=torch.int32, device="npu").view(B, NB)

out_auto = flash_decode_bnsd_with_gqa_share_sparse(
    q, None, k_cache, v_cache, block_table, seq_lens, BS, topk_idx,
    sm_scale=D**-0.5,
    num_topk_chunks=1,
)
out_ref = flash_decode_bnsd_with_gqa_share_sparse(
    q, None, k_cache, v_cache, block_table, seq_lens, BS, topk_idx,
    sm_scale=D**-0.5,
    num_topk_chunks=1,
    impl="torch_native",
)
print(out_auto.shape, out_auto.dtype)  # torch.Size([1, 8, 128]) torch.bfloat16
torch.testing.assert_close(out_auto, out_ref, atol=5e-2, rtol=5e-2)
```
