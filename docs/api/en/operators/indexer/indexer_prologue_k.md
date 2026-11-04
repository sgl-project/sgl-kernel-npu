# indexer_prologue_k

## Purpose and support

Projects BF16 latent states, applies RMSNorm and RoPE, quantizes to MXFP4 E2M1, and writes paged K cache.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. Install compatible PyTorch/torch_npu, CANN 9.2.1, and `cannbot-dsl==0.8.0`. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import indexer_prologue_k

indexer_prologue_k(latent: torch.Tensor, wk: torch.Tensor, norm_weight: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor, k_cache: torch.Tensor, k_scale_cache: torch.Tensor | None=None, *, cache_index: torch.Tensor, storage_mode: int, norm_eps: float, combined_block_size: int=-1, impl='auto')
```

## Inputs

BF16 `latent[T,H]`, FRACTAL_NZ `wk[D,H]`, FP32 normalization/RoPE inputs, and INT64 `cache_index[T]` are required. A `-1` cache index skips the write. `storage_mode=0` uses a separate scale cache; mode 1 combines rows according to `combined_block_size`.

## Output and ordering

Writes in place and returns the same `k_cache` tensor.

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/indexer_prologue_k/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_indexer_prologue_k.py -q
```

The second command requires Ascend 950; tests skip on other devices.
