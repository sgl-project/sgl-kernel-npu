# quant_lightning_indexer_metadata

## Purpose and support

Builds QLI AICPU scheduling and cross-core merge metadata from lengths, mask, compression ratio, and TopK.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. The selected target is CANN 9.2.0 with compatible PyTorch/torch_npu and `cannbot-dsl==0.8.0`; device compatibility is pending validation. Upstream documents CANN 9.2.1. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

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

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/quant_lightning_indexer_metadata/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer_metadata.py -q
```

The second command requires Ascend 950; tests skip on other devices.
