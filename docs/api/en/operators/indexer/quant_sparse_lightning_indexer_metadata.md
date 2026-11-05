# quant_sparse_lightning_indexer_metadata

## Purpose and support

Plans QSLI AICPU work from each row’s candidate length and visible sequence range.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_metadata

quant_sparse_lightning_indexer_metadata(candidate_block_length, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, quant_mode, candidate_block_size, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', impl='auto')
```

## Inputs

`candidate_block_length`, cumulative/used Q and K lengths, candidate block size, and quantization mode must match the main QSLI call. QLI metadata cannot substitute for QSLI metadata.

## Output and ordering

Returns INT32 `[1024]` metadata for `quant_sparse_lightning_indexer(..., metadata=metadata)`.

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer_metadata.py --npu-device 0 -q
```
