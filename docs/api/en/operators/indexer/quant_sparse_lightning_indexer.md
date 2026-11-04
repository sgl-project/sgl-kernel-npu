# quant_sparse_lightning_indexer

## Purpose and support

Scores only the candidate key blocks selected by QLI and returns sparse TopK tokens.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. The selected target is CANN 9.2.0 with compatible PyTorch/torch_npu and `cannbot-dsl==0.8.0`; device compatibility is pending validation. Upstream documents CANN 9.2.1. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

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

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/quant_sparse_lightning_indexer/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer.py -q
```

The second command requires Ascend 950; tests skip on other devices.
