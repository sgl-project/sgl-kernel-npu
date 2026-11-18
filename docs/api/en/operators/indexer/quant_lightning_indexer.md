# Quantized lightning indexer

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | supported | supported |
| `torch_native` | supported | supported |
| `ops_transformers_cann910` | supported | supported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

def quant_lightning_indexer(q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl='auto', **kwargs):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, bundled experimental bindings on A3 with CANN 9.2, and released bindings on other supported CANN 9.2 hardware. Execution failures never trigger another backend.

## Description

Causal TND queries and PA_BBND keys; A3 tests use INT8 with FP16 scales/weights, A5 tests use FP8 with FP32 scales/weights.

`score[j] = Σ_h weights[h] × ReLU(dot(dequant(Q[h]), dequant(K[j])))`. Select logical compressed-key Top-K within the causal prefix. Ties can have different valid orders.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| q | Tensor | required | [T,H,D] TND or [B,S,H,D] BSND; NPU in production |
| k | Tensor | required | Paged keys with shape [pages,page_size,1,D] |
| weights | Tensor | required | [T,H], indexer head weights |
| q_scale | Tensor | required | [T,H], query dequantization scales |
| k_scale | Tensor | required | [pages,page_size,1], key dequantization scales |
| topk | int | required | Requested number of logical compressed keys |
| quant_mode | int | required | 1=FP8, 2=INT8 |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### Keyword arguments

| Name | Type | Contract |
| --- | --- | --- |
| cu_seqlens_q | int32 Tensor | [B+1] cumulative query offsets, required by the reference. |
| seqused_k / cmp_residual_k | int32 Tensor | [B] compressed key lengths / uncompressed residual lengths. |
| block_table | int32 Tensor | [B,max_pages] paged key mapping. |
| metadata | Tensor / ReferenceMetadata / None | Matching metadata implementation only. |
| layout_q / layout_k | str | Reference defaults TND / PA_BBND. |
| mask_mode / cmp_ratio | int | Reference defaults causal 3 / compression ratio 4. |
| return_value | int | Default 0 produces empty values; nonzero score output is unsupported on CANN 9.1. |
| seqused_q / cu_seqlens_k / output_idx_offset / max_seqlen_q | optional vendor arguments | CANN 9.1 accepts query lengths as an alternative to cumulative offsets; non-null key offsets/output offsets are rejected. |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| indices | int32 Tensor | [T,1,topk] logical compressed-key positions, -1 padding. |
| values | FP32 Tensor | Optional selected scores; empty when return_value=0. |

## A3 + CANN 9.2 INT8 modes

The A3 production backend on CANN 9.2 supports explicit `candidate_mode` calls. The example below uses TND queries and PA_BBND keys. Explicit modes require `quant_mode=2`, INT8 Q/K, FP16 scales and head weights, 32 or 64 query heads, one key head, head dimension 128, and `1 <= topk <= 2048`.

| `candidate_mode` | Behavior | Return |
| --- | --- | --- |
| Omitted | Ordinary indexing | `(indices, values)` |
| `1` (source) | Select top-k keys and emit candidate blocks | `(indices, values, candidates)` |
| `2` (consumer) | Select top-k keys within `candidate_topk_index` | `(indices, values, empty_candidates)` |
| `3` (ordinary) | Explicit ordinary indexing | `(indices, values, empty_candidates)` |

Source and consumer require `candidate_topk_blocks=2048` and `candidate_block_size=8`, which are also the defaults. For TND, `candidates` contains INT32 logical block IDs with shape `[T, 1, 2048]`. Each block covers eight compressed-key positions, and `-1` marks padding. Pass the source's third result as `candidate_topk_index` to the consumer. Modes 2 and 3 return an empty INT32 tensor in that position. Modes 1 and 2 require `return_value=0`.

This snippet assumes the tensors and page table are already prepared. Set `layout_and_lengths` with `layout_q="TND"`, `layout_k="PA_BBND"`, `cu_seqlens_q`, `seqused_k`, `mask_mode`, `cmp_ratio`, and `cmp_residual_k` when required for causal compressed keys.

```python
from sgl_kernel_npu.operators.indexer import (
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
)

metadata = quant_lightning_indexer_metadata(
    q.shape[-2], 1, 128, topk, 2, **layout_and_lengths
)
args = (q, k, weights, q_scale, k_scale, topk, 2)
kwargs = dict(layout_and_lengths, block_table=block_table, metadata=metadata)

source_indices, _, candidates = quant_lightning_indexer(
    *args, candidate_mode=1, **kwargs
)
consumer_indices, _, _ = quant_lightning_indexer(
    *args, candidate_mode=2, candidate_topk_index=candidates, **kwargs
)
ordinary_indices, _, _ = quant_lightning_indexer(
    *args, candidate_mode=3, **kwargs
)
```

## cann_ops_transformers_cann920

Uses the bundled experimental ops-transformer binding on A3 and the released CANN 9.2 binding on other supported hardware. The old ABI is registered separately as `CANN 9.1`. An explicit implementation inconsistent with the initialized SDK is rejected before execution.

## ops_transformers_cann910

Source: [ops-transformer experimental](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer). The kernel source is experimental ops-transformer. Its custom_ops Python/C++ bridge may be distributed from cann-recipes-infer; the implementation name records kernel provenance, not the bridge package name. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

q = torch.ones(1, 2, 128, device="npu", dtype=torch.int8)
k = torch.ones(1, 128, 128, device="npu", dtype=torch.int8)
w = torch.ones(1, 2, device="npu", dtype=torch.float16)
qs = torch.ones(1, 2, device="npu", dtype=torch.float16)
ks = torch.ones(1, 128, 1, device="npu", dtype=torch.float16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_k=i32([1]),
          cmp_residual_k=i32([0]), block_table=i32([[0]]),
          layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=4)
indices, values = quant_lightning_indexer(q, k, w, qs, ks, 1, 2, impl="torch_native", **kw)
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

Accuracy tests run the independent eager reference on CPU and compare sorted index sets on CPU. This isolates the reference Top-K and comparison Sort from the NPU execution path; index comparison remains exact. Performance timing uses the original NPU executor.

```bash
python3 -m pytest tests/python/operators/indexer/test_quant_lightning_indexer.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
