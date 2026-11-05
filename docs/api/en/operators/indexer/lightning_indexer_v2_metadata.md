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

`num_heads_q` is 1..64, `num_heads_k=1`, and `head_dim=128`. Length tensors are int32; TND uses cumulative boundaries `[B+1]`, and `seqused_k` has shape `[B]`. Layout, mask and compression must match the main call. None attributes use upstream defaults: batch_size=0, maximum lengths=-1, BSND/BSND, mask=0, cmp_ratio=1. There is no torch_native metadata backend: opaque scheduling data is tested in combination with the main operator.

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
