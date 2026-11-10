# Quantized indexer metadata

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | supported | supported |
| `torch_native` | supported | supported |
| `ops_transformers_cann910` | supported | supported |

## Signature

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

def quant_lightning_indexer_metadata(num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl='auto', **kwargs):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

Use the same implementation and lengths as quant_lightning_indexer. Metadata is not interchangeable between ABIs.

Prepare indexer scheduling from head geometry, Top-K, quantization mode and per-request lengths. Reference metadata is a parameter descriptor, not a byte-level tiling oracle.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| num_heads_q | int | required | Number of query heads |
| num_heads_k | int | required | Number of key heads; reference supports 1 |
| head_dim | int | required | Head width D |
| topk | int | required | Requested number of logical compressed keys |
| quant_mode | int / str | required | Indexer 1=FP8, 2=INT8; packed KV None selects SDK format |
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
| metadata | Native Tensor or ReferenceMetadata | Opaque native scheduling storage, or reference-only descriptor. Never compare bytes across implementations. |

## cann_ops_transformers_cann920

Calls only the released CANN 9.2 binding. The old ABI is registered separately as `CANN 9.1`. An explicit implementation inconsistent with the initialized SDK is rejected before execution.

## ops_transformers_cann910

Source: [ops-transformer experimental AICPU metadata](https://gitcode.com/cann/ops-transformer/tree/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/quant_lightning_indexer_metadata). The kernel source is experimental ops-transformer. Its custom_ops Python/C++ bridge may be distributed from cann-recipes-infer; the implementation name records kernel provenance, not the bridge package name. The CANN 9.1 ABI conversion stays in this operator module; an initialized CANN 9.2 SDK is rejected.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

q = torch.ones(1, 2, 128, device="npu", dtype=torch.int8)
k = torch.ones(1, 128, 1, 128, device="npu", dtype=torch.int8)
w = torch.ones(1, 2, device="npu", dtype=torch.float16)
qs = torch.ones(1, 2, device="npu", dtype=torch.float16)
ks = torch.ones(1, 128, 1, 1, device="npu", dtype=torch.float16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_k=i32([1]),
          cmp_residual_k=i32([0]), block_table=i32([[0]]),
          layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=4)
metadata = quant_lightning_indexer_metadata(2, 1, 128, 1, 2, impl="torch_native", **kw)
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

Accuracy tests run the independent eager reference on CPU and compare sorted index sets on CPU. This isolates the reference Top-K and comparison Sort from the NPU execution path; index comparison remains exact. Performance timing uses the original NPU executor.

```bash
python3 -m pytest tests/python/operators/indexer/test_quant_lightning_indexer_metadata.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
