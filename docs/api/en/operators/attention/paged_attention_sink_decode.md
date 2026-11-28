# paged_attention_sink_decode

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Supported | Supported |

`arch22` is unverified for the Triton implementation, whose vectors are tuned
with A5 compile options. Validate on arch22 before relying on it.

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention import paged_attention_sink_decode


def paged_attention_sink_decode(
    *,
    q,
    k_cache,
    v_cache,
    seqlens,
    block_tables,
    sinks=None,
    gqa_interleave=False,
    softmax_scale=None,
    impl="auto",
): ...
```

All arguments are keyword-only. `auto` selects `triton` without fallback.

## Description

Paged grouped-query decode attention with one learned sink logit per query
head. Each request contributes exactly one query row, whose position is the
last KV entry, so the whole KV row is visible and no causal mask is needed.
The sink participates in the softmax denominator with a zero value
contribution, so it can only reduce attention mass.

For request b, head h, and KV position j < Lb=`seqlens[b]`, let P=`page_size`,
g the KV head of h, `p = block_tables[b, floor(j / P)]`, and `r = j mod P`:

$$
S_{bj} = \mathrm{softmax\_scale}\sum_d q_{bhd}\,k_{p,r,g,d},\qquad
O_{bhd} = \frac{\sum_{j} e^{S_{bj}}\,v_{p,r,g,d}}
{\sum_{j} e^{S_{bj}} + e^{s_h}} .
$$

`g` is `h mod Hkv` when `gqa_interleave` is true, otherwise
`h // (Hq / Hkv)`. A request with `Lb=0` produces a zero output row.

Computation uses FP32 online softmax accumulation; the output is cast to the
query dtype. Reduction order and intermediate rounding may differ between
implementations, and the KV-split path reorders the reduction over splits.

## Parameters and return value

All tensors must share one device. Triton requires NPU storage; the reference
also accepts CPU tensors, although importing the package still requires the
configured native runtime, torch_npu, ACL and an available NPU.

| Parameter | Type | Default | Description and constraints |
| --- | --- | --- | --- |
| `q` | Tensor | Required | Contiguous FP16/BF16 `[B, Hq, D]` decode queries. |
| `k_cache` | Tensor | Required | Contiguous `[pages, P, Hkv, D]` keys, same dtype as q. |
| `v_cache` | Tensor | Required | Same shape and dtype as `k_cache`. |
| `seqlens` | Tensor | Required | Contiguous int32 `[B]` KV lengths, including the current token. Lengths may differ per request. |
| `block_tables` | Tensor | Required | Contiguous int32/int64 `[B, max_pages]` physical page IDs; row b must cover `ceil(seqlens[b] / P)` pages. |
| `sinks` | Tensor | `None` | FP16/BF16 `[Hq]` sink logits in the query dtype. `None` disables the sink. |
| `gqa_interleave` | bool | `False` | Use the interleaved query-to-KV head mapping. |
| `softmax_scale` | float | `None` | Finite positive QK scale; `None` uses `D**-0.5`. |
| `max_kv_len` | int | `None` | Longest KV row known to the caller. When given, the split decision and the page-table check use it without a device-to-host read; must be positive. |
| `impl` | str | `"auto"` | `"auto"`, `"triton"` or `"torch_native"`. |

Additional constraints: `Hq` is positive and divisible by `Hkv`, `P` is a
power of two of at least 16, and `D` is an integer in `[16, 256]`.

| Return value | Type | Shape | Dtype / device | Description |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[B, Hq, D]` | Same as q | Newly allocated result; `B=0` returns an empty tensor. |

## triton backend

- The default path assigns one task per (request, KV head) to cube cores; the
  `Hq/Hkv` query heads of a group share each K/V tile, and the sink seeds the
  online maximum and denominator. Each task streams its whole KV row.
- Long KV rows on an under-filled grid switch to a two-phase flash decode.
  The split is used when `B * Hkv < 0.4 * cube cores` and the longest KV row is
  at least 256 tokens (at least 2048 for GQA); the split count is
  `max(1, min(cube_core_count // (B * Hkv), max_kv_len // 256))`.
- Phase 1 runs on cube cores and writes a locally normalised partial output
  plus its log-sum-exp per (request, KV head, split); it excludes the sink so
  the sink cannot be counted once per split. Phase 2 runs on vector cores,
  re-weights the splits by their log-sum-exp, and folds the sink in exactly
  once as an extra zero-value denominator term.
- Splits are page aligned so no KV block straddles two logical pages. The
  longest KV length is copied to the host only when the batch is a split
  candidate (`B * Hkv` below 0.4 cube cores) and `max_kv_len` is omitted; that
  read chooses the split count and validates the page-table width. Passing
  `max_kv_len` removes the read entirely. Other batches keep all metadata on the
  device and the kernel clamps page indices to the table width, so callers must
  size `block_tables` for their lengths. The split path allocates FP32
  workspace per call and is not NPU Graph capture safe.
- Non-power-of-two head dimensions and GQA groups are supported by
  Triton-Ascend 3.6 on arch35; arch22 remains unverified.

## torch_native backend

- Gathers the valid pages of each request, expands shared KV heads with the
  selected mapping, and evaluates FP32 softmax attention per query row.
- Concatenates the sink logit as one extra score and drops its value column, so
  it only enlarges the denominator. Requests with `Lb=0` keep a zero output row.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

device = "npu:0"
q = torch.randn(2, 16, 64, dtype=torch.bfloat16, device=device)
k_cache = torch.randn(6, 128, 4, 64, dtype=torch.bfloat16, device=device)
v_cache = torch.randn_like(k_cache)
seqlens = torch.tensor([129, 300], dtype=torch.int32, device=device)
block_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
sinks = torch.randn(16, dtype=torch.bfloat16, device=device)

output = paged_attention_sink_decode(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    seqlens=seqlens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
)
reference = paged_attention_sink_decode(
    q=q,
    k_cache=k_cache,
    v_cache=v_cache,
    seqlens=seqlens,
    block_tables=block_tables,
    sinks=sinks,
    softmax_scale=0.125,
    impl="torch_native",
)
torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
```
