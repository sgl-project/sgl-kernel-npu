# lightning_indexer_v2_metadata

Target: CANN **9.2.0**, A3 (arch22, build argument `2201`). This integration does not enable an A5 backend.

Upstream documentation: [lightning_indexer_v2_metadata](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/lightning_indexer_v2_metadata/README.md).

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` / `auto` | Supported | Unsupported |

## Signature

`auto` and `cann_ops_transformers_cann920` select the same bundled backend, require CANN 9.2.0, and do not fall back.

```python
from sgl_kernel_npu.operators.indexer import lightning_indexer_v2_metadata

def lightning_indexer_v2_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    *,
    impl='auto'
):
    ...
```

## Description

Returns opaque `int32[2048]` scheduling data. The upstream Python name is `lightning_indexer_metadata`, its ACLNN symbol is `aclnnLightningIndexerV2Metadata`, and the build name is `lightning_indexer_v2_metadata` (AICPU). A3 kernels ignore this metadata, so it may be omitted. The dependency is bundled to provide the symbols referenced by the binding.

## Parameters and return value

| Parameter | Type / shape | Default | Constraints |
| --- | --- | --- | --- |
| num_heads_q | int | Required | 1..64 |
| num_heads_k | int | Required | 1 |
| head_dim | int | Required | 128 |
| topk | int | Required | Same supported selection count as the main call |
| cu_seqlens_q, cu_seqlens_k | int32 tensor `[B+1]` | None | TND cumulative boundaries; no BSND key lengths |
| seqused_q, seqused_k, cmp_residual_k | int32 tensor `[B]` | None | Follow the upstream metadata contract; A3 main calls reject seqused_q |
| batch_size | int | None | None uses upstream default 0 |
| max_seqlen_q, max_seqlen_k | int | None | None uses upstream default -1 |
| layout_q | str | None | Upstream default BSND; also TND |
| layout_k | str | None | Upstream default BSND; also TND / PA_BBND |
| mask_mode | int | None | Upstream default 0; also 3 |
| cmp_ratio | int | None | Upstream default 1; powers of two in 1..128 |
| impl | str, keyword-only | auto | auto / cann_ops_transformers_cann920 |

Returns an `int32[2048]` tensor on the current NPU, without modifying inputs.
Lengths, layouts, head counts, compression and mask must match the main call.
There is no torch_native metadata backend because the output is opaque
scheduling data; correctness is checked in combination with the main operator.

## cann_ops_transformers_cann920 backend

Build with the ops_transformer profile: vendor, bindings, AICPU resources and LICENSE ship in the wheel. First use may JIT-compile C++ bindings and requires matching CANN/Torch/torch_npu and a compiler. Separate first-use JIT from steady-state measurements. CPU checks do not establish NPU correctness.

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
metadata = lightning_indexer_v2_metadata(32, 1, 128, 16, batch_size=1,
    max_seqlen_q=1, max_seqlen_k=512, layout_q="BSND", layout_k="BSND")
indices, values = lightning_indexer_v2(q, k, w, 16, metadata=metadata)
```

Upstream copyright: Huawei Technologies Co., Ltd.; CANN Open Software License Agreement Version 2.0.
