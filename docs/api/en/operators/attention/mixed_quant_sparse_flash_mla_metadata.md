# mixed_quant_sparse_flash_mla_metadata

## Purpose and support

Plans full-row or FlashDecode split work on AICPU from the original and compressed TopK lengths.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. The selected target is CANN 9.2.0 with compatible PyTorch/torch_npu and `cannbot-dsl==0.8.0`; device compatibility is pending validation. Upstream documents CANN 9.2.1. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, *, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None, num_heads_q, num_heads_kv, head_dim, quant_mode, layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=True, impl='auto')
```

## Inputs

`ori_topk_length` and `cmp_topk_length` describe valid sparse lengths per query/head. The `has_ori_kv` and `has_cmp_kv` flags must match the main operator. This profile uses `num_heads_q=64`, `num_heads_kv=1`, `head_dim=512`, and `quant_mode=1`.

## Output and ordering

Returns INT32 `[1024]` metadata. Produce it on the same NPU stream before calling `mixed_quant_sparse_flash_mla`.

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/mixed_quant_sparse_flash_mla/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla_metadata.py -q
```

The second command requires Ascend 950; tests skip on other devices.
