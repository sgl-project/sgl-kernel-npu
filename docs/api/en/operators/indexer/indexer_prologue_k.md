# indexer_prologue_k

## Purpose and support

Projects BF16 latent states, applies RMSNorm and RoPE, quantizes to MXFP4 E2M1, and writes paged K cache.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

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

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_indexer_prologue_k.py --npu-device 0 -q
```
