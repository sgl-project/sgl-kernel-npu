# lightning_indexer_prefill

## Hardware support

arch22 is split because the two members of that family were measured
separately. A3 means the Ascend910_93 series, A2 the Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `torch_npu` | Supported | Supported | Unsupported |
| `torch_native` | Supported | Supported | Unverified |

`torch_npu` and `auto` are declared for arch22 only, because the grouped launch
geometry was measured there. Correctness was validated on Ascend910_9362 (A3,
CANN 9.0) and on Ascend910B3 (A2, CANN 9.1.0, torch 2.10.0, torch_npu
2.10.0.post4). arch35 is unsupported: `impl="auto"` raises
`UnsupportedGenerationError` there, and the caller keeps its own fallback.

## Signature

```python
import torch
from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

def lightning_indexer_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    sequence_length: int,
    group4: bool = True,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

`auto` selects `torch_npu`; omitting impl has the same effect. There is no runtime
backend fallback.

## Description

Select the ordered Top-512 compressed keys for every query of one ratio-4
causal prefill request. Scores use unit head weights:

```text
score[i, j] = sum_h relu(q[i, h] . k[j])
```

Row `i` is logical position `sequence_length - rows + i`, so
`floor((sequence_length - rows + i + 1) / 4)` compressed keys are causally
visible to it. Keys beyond that horizon are never selected, and a row whose
horizon is empty returns all -1.

For ratio-4 causal prefill up to four adjacent queries share one visible key
range. With `group4=True` those queries are batched into the sequence axis of
one CANN LightningIndexer launch instead of duplicating one page-table row per
query. Every query still computes its own scores and its own ordered Top-K; no
selection is shared. `group4=False` scores every query in its own batch entry
and is the comparison path. Grouping additionally requires `rows >= 1024` and
`keys >= 512`; below either threshold both settings dispatch the same work.

This operator neither applies arbitrary masks nor supports learned head
weights, other shapes, decode or speculative execution. Call it once per
request; the returned indices are local to `k`.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `q` | `torch.Tensor` | `[rows, 4, 128]` | `torch.bfloat16` | Required | The last `rows` queries of the request, contiguous, on an NPU. |
| `k` | `torch.Tensor` | `[sequence_length // 4, 1, 128]` | `torch.bfloat16` | Required | Compressed keys, contiguous, on q's device. |
| `sequence_length` | `int` | — | — | Required | Keyword-only. Request length in tokens, a host integer in `[0, 2**31)`. Must satisfy `rows <= sequence_length` and `k.shape[0] == sequence_length // 4`. No device value is read back for eligibility or phase selection. |
| `group4` | `bool` | — | — | `True` | Keyword-only. Selects the grouped or the independent launch geometry. Both select the same keys, so this is a comparison switch rather than a tuning knob. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `torch_npu`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, 512]` | `torch.int32` | Newly allocated ordered key indices local to `k`, padded with -1. Row `i` holds `min(512, floor((sequence_length - rows + i + 1) / 4))` valid entries. Inputs are not modified. |

Autograd is not supported: an input with `requires_grad` is rejected. Inputs
outside this contract raise `ValueError` before any kernel launch.

## Reproducibility of ties

Exact tie order is not part of the contract. The two backends accumulate the
128-element dot product and the four-head sum in different orders, so keys
whose scores agree to within a few float32 ulps may be returned in either
order, and a tie straddling the Top-512 cut may put a different key on either
side of it. What is well defined, and what the tests assert, is:

- the multiset of selected scores,
- the number of valid entries per row,
- the non-increasing score order of the returned keys.

Between `group4=True` and `group4=False` the ordered indices *are* bitwise
identical, including ties and padding; that is the invariant four-query
batching has to preserve, and the test suite checks it on all-zero inputs
where every score is tied.

## torch_npu backend

Adapts the request onto
`sgl_kernel_npu.operators.indexer.lightning_indexer(..., impl="ops-transformer")`
(backed by `torch_npu.npu_lightning_indexer`) with `layout_query="BSND"` and
`layout_key="PA_BSND"`. Keys are padded to whole
64-entry pages and addressed through an identity block table. The grouping
phase is `(sequence_length - rows + 1) % 4`, which aligns four-query groups so
that every real query in a group shares one visible key range; leading and
trailing dummy queries are discarded. CANN requires at least one key in its
length metadata, so rows with no visible key are masked back to -1 afterwards.

Four-query batching engages when `group4=True`, `rows >= 1024`, and
`keys >= 512`, reducing per-query page-table duplication and improving long
prefill latency; below either threshold, `group4=True` and `group4=False`
dispatch the same independent-query work. Benchmark details and reproduction
instructions live in
[`benchmark/indexer/README.md`](../../../../../benchmark/indexer/README.md) and
[`benchmark/indexer/bench_lightning_indexer_prefill.py`](../../../../../benchmark/indexer/bench_lightning_indexer_prefill.py).

## torch_native backend

Independent reference; it calls no CANN indexer operator. Scores every query
against every key in float32, masks the causal horizon and takes a stable
descending sort, so ties resolve towards the lower key index. Rows are
processed in chunks because the per-head dot product is proportional to
`rows * 4 * keys`. It is a reference, not a performance backend.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

sequence_length, rows = 4096, 1024
q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(sequence_length // 4, 1, 128, device="npu", dtype=torch.bfloat16)
indices = lightning_indexer_prefill(q, k, sequence_length=sequence_length)
print(indices.shape, indices.dtype)  # torch.Size([1024, 512]) torch.int32
```
