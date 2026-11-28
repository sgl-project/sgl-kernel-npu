# paged_attention_sink_prefill

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Supported | Supported |

`arch22` is unverified for the Triton implementation: the kernels pass A5
CV-pipeline compile options, and the page-aggregation path requires the
Triton-Ascend CANN extension. Validate on arch22 before relying on it.

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill


def paged_attention_sink_prefill(
    *,
    q,
    k_cache,
    v_cache,
    cu_q_lens,
    block_tables,
    seqlens_kv=None,
    sinks=None,
    gqa_interleave=False,
    softmax_scale=None,
    impl="auto",
): ...
```

All arguments are keyword-only. `auto` selects `triton` without fallback.

## Description

Causal paged grouped-query attention over packed requests, with one learned
sink logit per query head. The sink participates in the softmax denominator
with a zero value contribution, so it can only reduce attention mass.

For request b, query row i, head h, and KV position j, let P=`page_size`,
g be the KV head of h, Lb=`seqlens_kv[b]` (or the query length when
`seqlens_kv` is omitted), and Cb=Lb-Lq_b the cached prefix length. The query
sits after the cached prefix, so it may read KV positions
`0 <= j <= Cb + i`. With `p = block_tables[b, floor(j / P)]` and `r = j mod P`:

$$
S_{bj} = \mathrm{softmax\_scale}\sum_d q_{bihd}\,k_{p,r,g,d},\qquad
O_{bihd} = \frac{\sum_{j} e^{S_{bj}}\,v_{p,r,g,d}}
{\sum_{j} e^{S_{bj}} + e^{s_h}} .
$$

`g` is `h mod Hkv` when `gqa_interleave` is true, otherwise
`h // (Hq / Hkv)`. Queries with no visible KV positions cannot occur because
each query always reads at least its own position.

Computation uses FP32 online softmax accumulation; the output is cast to the
query dtype. Reduction order and intermediate rounding may differ between
implementations.

## Parameters and return value

All tensors must share one device. Triton requires NPU storage; the reference
also accepts CPU tensors, although importing the package still requires the
configured native runtime, torch_npu, ACL and an available NPU.

| Parameter | Type | Default | Description and constraints |
| --- | --- | --- | --- |
| `q` | Tensor | Required | Contiguous FP16/BF16 `[tokens, Hq, D]` packed queries. |
| `k_cache` | Tensor | Required | Contiguous `[pages, P, Hkv, D]` keys, same dtype as q. |
| `v_cache` | Tensor | Required | Same shape and dtype as `k_cache`. |
| `cu_q_lens` | Tensor | Required | Contiguous int32 `[B+1]`; starts at 0, non-decreasing, ends at `tokens`. Values are validated by the implementations, which read them to build the schedule. |
| `block_tables` | Tensor | Required | Contiguous int32/int64 `[B, max_pages]` physical page IDs. Each row must cover `ceil(seqlens_kv[b] / P)` pages. |
| `seqlens_kv` | Tensor | `None` | Contiguous int32 `[B]` total KV length per request, including its query rows. `None` means every KV length equals its query length. Must be at least the query length. |
| `sinks` | Tensor | `None` | FP16/BF16 `[Hq]` sink logits in the query dtype. `None` disables the sink. |
| `gqa_interleave` | bool | `False` | Use the interleaved query-to-KV head mapping. |
| `softmax_scale` | float | `None` | Finite positive QK scale; `None` uses `D**-0.5`. |
| `impl` | str | `"auto"` | `"auto"`, `"triton"` or `"torch_native"`. |

Additional constraints: `Hq` is positive and divisible by `Hkv`, `P` is a
power of two of at least 16, and `D` is an integer in `[16, 208]`. The tile
bound was validated on arch35; larger head dimensions exceed the per-tile UB
budget of the 128-query-row kernel.

| Return value | Type | Shape | Dtype / device | Description |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[tokens, Hq, D]` | Same as q | Newly allocated result; `tokens=0` returns an empty tensor. |

## triton backend

- Builds a longest-processing-time schedule on the host: each task is one
  (request, 128-row query tile, query head) triple, and tasks are assigned to
  cube cores by estimated KV blocks scanned. The grid is the cube core count,
  so no global atomics are required.
- KV blocks below the cached prefix are processed without a mask; later blocks
  read the causal mask from a shared boolean table instead of recomputing it
  per row. The sink logit seeds the online maximum and the denominator.
- When `page_size < 128` and divides 128, a page-aggregation kernel folds
  128 consecutive KV rows into one cube instruction; otherwise a block-pointer
  kernel scans one `BLOCK_SIZE_N` tile per step.
- Reads all metadata from device buffers except the schedule inputs, which are
  copied to the host once per call. Not NPU Graph capture safe.
- Non-power-of-two head dimensions and GQA groups are supported by
  Triton-Ascend 3.6 on arch35; arch22 remains unverified.

## torch_native backend

- Gathers the valid pages of each request, expands shared KV heads with the
  selected mapping, and evaluates FP32 softmax attention in query-row chunks.
- Concatenates the sink logit as one extra score and drops its value column, so
  it only enlarges the denominator. Empty query rows keep the zero initialised
  output.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

device = "npu:0"
q = torch.randn(129, 64, 64, dtype=torch.bfloat16, device=device)
k_cache = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device=device)
v_cache = torch.randn_like(k_cache)
cu_q_lens = torch.tensor([0, 129], dtype=torch.int32, device=device)
block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
sinks = torch.randn(64, dtype=torch.bfloat16, device=device)

output = paged_attention_sink_prefill(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    cu_q_lens=cu_q_lens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
)
reference = paged_attention_sink_prefill(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    cu_q_lens=cu_q_lens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
    impl="torch_native",
)
torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
```
