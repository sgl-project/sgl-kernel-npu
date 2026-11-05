# quant_lightning_indexer

## Purpose and support

Scores visible MXFP4 keys and selects TopK tokens, optionally producing candidate blocks for QSLI.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

quant_lightning_indexer(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, descale_q: torch.Tensor, descale_k: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, seqused_q: Optional[torch.Tensor]=None, seqused_k: Optional[torch.Tensor]=None, cmp_residual_k: Optional[torch.Tensor]=None, block_table: Optional[torch.Tensor]=None, output_idx_offset: Optional[torch.Tensor]=None, metadata: Optional[torch.Tensor]=None, *, topk: int, quant_mode: int, max_seqlen_q: int=-1, mask_mode: int=0, cmp_ratio: int=1, layout_q: str='TND', layout_k: str='TND', return_value: bool=False, candidate_topk_blocks: int=-1, candidate_block_size: int=-1, impl='auto')
```

## Inputs

UINT8 `q[T1,32,64]`, packed MXFP4 K, E8M0 query/key descales, and metadata from `quant_lightning_indexer_metadata` are required. Use `quant_mode=1`; K layout may be TND or PA_BBND. Candidate mode requires `candidate_block_size=8`.

## Output and ordering

Returns `(indices, values, candidates, candidate_length)`. Indices are INT32 `[T1,1,topk]`; disabled optional outputs are empty tensors and invalid indices are `-1`.

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer.py --npu-device 0 -q
```
