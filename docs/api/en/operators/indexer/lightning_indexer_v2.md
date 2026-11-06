# lightning_indexer_v2

Target: CANN **9.2.0**, A3 (arch22, build argument `2201`). This integration does not enable an A5 backend.

Upstream documentation: [lightning_indexer_v2](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/lightning_indexer_v2/README.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

`auto` and `cann_ops_transformers_cann920` select the same bundled backend, require CANN 9.2.0, and do not fall back.

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2

def lightning_indexer_v2(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q='BSND',
    layout_k='BSND',
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks=None,
    candidate_block_size=None,
    impl='auto'
):
    ...
```

## Description

For each query, compute `score[j] = sum_h w[h] * relu(dot(q[h], k[j,0]))` and select TopK by score. Mask 3 is right-aligned causal; compressed keys follow the upstream compressed position domain.

## Parameters and return value

| Parameter | dtype | Shape / constraint |
| --- | --- | --- |
| q | fp16/bf16 | BSND `[B,Q,H,128]` / TND `[Tq,H,128]`; H=1..64 |
| k | same as q | BSND `[B,K,1,128]` / TND `[Tk,1,128]` / PA_BBND `[P,S,1,128]` |
| w | fp32 | q.shape[:-1] |
| cu_seqlens_q / cu_seqlens_k | int32 | `[B+1]`; TND cumulative boundaries; BSND q: `[0, length_0, ...]`; BSND k: None |
| seqused_k / cmp_residual_k | int32 | `[B]`; compressed causal mode requires cmp_residual_k |
| block_table | int32 | `[B,max_pages]`; required with seqused_k for PA_BBND |
| metadata | int32 | `[2048]` or None; ignored by A3 |

Inputs share one device; q/w are contiguous, and only axis 0 of PA_BBND k may be noncontiguous. PA block size is a multiple of 16 in 16..1024. Optional index tensors require int32. A3 does not support seqused_q, output_idx_offset or nondefault max_seqlen_q; the adapter rejects them. mask_mode is 0/3, cmp_ratio is a power of two in 1..128, and return_value is 0/1. Tensor contents must satisfy the pinned upstream contract; the performance path does not copy values to the host for validation.

Build/OpType: lightning_indexer_v2 / LightningIndexerV2; upstream Python: lightning_indexer. The default path calls aclnnLightningIndexerV2; the candidate path calls aclnnLightningIndexerV3. Default topk is 1..2048 or 3072/4096/5120/6144/7168/8192.

candidate_topk_blocks=None returns `(indices, values)`. Explicit -1 returns four outputs with empty candidate outputs. Positive C must be a multiple of 64 in 64..2048 and requires topk<=2048, returning `(indices, values, candidate_block_indices, candidate_block_length)`. Position outputs have shape `[B,Q,1,topk]` or `[Tq,1,topk]`. Indices are int32 with -1 padding; values are FP32 with -inf padding, or empty when return_value=0. Candidate indices use width C, int32, and -1 padding; length is always empty int32.

Each block score is its maximum position score. The last reachable block is pinned with +inf for block TopK. candidate_block_size=None uses 8; candidate mode accepts 2/4/8/16/32/64. Candidate slot order is unspecified.

## cann_ops_transformers_cann920 backend

Build with the ops_transformer profile: vendor, bindings, AICPU resources and LICENSE ship in the wheel. First use may JIT-compile C++ bindings and requires matching CANN/Torch/torch_npu and a compiler. Separate first-use JIT from steady-state measurements. CPU checks do not establish NPU correctness.

## torch_native backend

Source and consumer must use the same block size and compatible KV position domains; candidate reuse is within one forward only. torch_native supports BSND/BSND, cmp_ratio=1, mask=0/3 and optional BSND Q/K lengths. Packed/paged/compressed KV are unsupported by the reference. It independently computes in FP32 without the tested binding; tie selection and slot order can differ.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import (
    lightning_indexer_v2, lightning_indexer_v2_metadata, sparse_lightning_indexer,
)
q = torch.randn(1, 1, 32, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(1, 512, 1, 128, device="npu", dtype=torch.bfloat16)
w = torch.randn(1, 1, 32, device="npu", dtype=torch.float32)
indices, values, blocks, length = lightning_indexer_v2(
    q, k, w, 16, candidate_topk_blocks=64, candidate_block_size=8)
```

Upstream copyright: Huawei Technologies Co., Ltd.; CANN Open Software License Agreement Version 2.0.
