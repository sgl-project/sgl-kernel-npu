# quant_sparse_lightning_indexer

## Purpose and support

Scores only the candidate key blocks selected by QLI and returns sparse TopK tokens.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer

quant_sparse_lightning_indexer(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, descale_q: torch.Tensor, candidate_block_indices: torch.Tensor, candidate_block_length: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, seqused_q: Optional[torch.Tensor]=None, seqused_k: Optional[torch.Tensor]=None, cmp_residual_k: Optional[torch.Tensor]=None, block_table: Optional[torch.Tensor]=None, output_idx_offset: Optional[torch.Tensor]=None, metadata: Optional[torch.Tensor]=None, *, descale_k: Optional[torch.Tensor]=None, topk: int, candidate_block_size: int, quant_mode: int, max_seqlen_q: int=-1, mask_mode: int=0, cmp_ratio: int=1, layout_q: str='TND', layout_k: str='TND', return_value: bool=False, impl='auto')
```

## Inputs

UINT8 `q[T1,32,64]`, INT32 candidate block indices/lengths, E8M0 key descale, and matching QSLI metadata are required. Use `candidate_block_size=8` and `quant_mode=1`; K may be TND or PA_BBND.

## Output and ordering

Returns `(indices, values)` with INT32 indices and BF16 scores of shape `[T1,1,topk]`. If `return_value=False`, values are empty; unfilled indices are `-1`.

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer.py --npu-device 0 -q
```
