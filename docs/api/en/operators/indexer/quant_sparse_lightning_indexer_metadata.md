# quant_sparse_lightning_indexer_metadata

## Purpose and support

Plans QSLI AICPU work from each row’s candidate length and visible sequence range.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. Install compatible PyTorch/torch_npu, CANN 9.2.1, and `cannbot-dsl==0.8.0`. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

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

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/quant_sparse_lightning_indexer_metadata/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer_metadata.py -q
```

The second command requires Ascend 950; tests skip on other devices.
