# quant_lightning_indexer_metadata

## Purpose and support

Builds QLI AICPU scheduling and cross-core merge metadata from lengths, mask, compression ratio, and TopK.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

quant_lightning_indexer_metadata(cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', candidate_topk_blocks=-1, candidate_block_size=-1, impl='auto')
```

## Inputs

Cumulative and used sequence lengths are INT32. Geometry is `num_heads_q=32`, `num_heads_k=1`, `head_dim=128`. Candidate settings must match the main QLI call; regenerate metadata when lengths or static settings change.

## Output and ordering

Returns INT32 `[1024]` metadata for `quant_lightning_indexer(..., metadata=metadata)`.

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer_metadata.py --npu-device 0 -q
```
