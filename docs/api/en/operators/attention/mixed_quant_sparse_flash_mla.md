# mixed_quant_sparse_flash_mla

## Purpose and support

Computes sparse MLA over BF16 queries, FP8 original KV, and optional FP4 compressed KV.

This `cannbot_dsl` implementation is registered for `arch35` and targets Ascend 950PR/950DT. `impl="auto"` and `impl="cannbot_dsl"` select the same backend. Install compatible PyTorch/torch_npu, CANN 9.2.1, and `cannbot-dsl==0.8.0`. Set `CANNBOTDSL_NATIVE_BINARY_MODE=off` for source JIT execution; the first metadata call compiles its AICPU kernel if a packaged binary is absent. Ascend910_9362 is unsupported.

## Signature

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

mixed_quant_sparse_flash_mla(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, ori_topk_length=None, cmp_topk_length=None, sinks=None, metadata=None, quant_mode, softmax_scale=None, layout_q='TND', layout_kv='PA_BBND', return_softmax_lse=False, out, lse, impl='auto')
```

## Inputs

BF16 TND `q[T1,64,512]`, UINT8 paged original KV, sparse indices and block tables are required for active KV sides. `quant_mode=1`; metadata must come from `mixed_quant_sparse_flash_mla_metadata` with matching configuration. The caller allocates `out` and `lse`.

## Output and ordering

Writes BF16 results into `out` and FP32 log-sum-exp into `lse`; follows the upstream no-return ABI. With `return_softmax_lse=False`, pass an empty LSE tensor.

## Source and tests

Copied byte for byte from [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) commit `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231`, `samples/attention/mixed_quant_sparse_flash_mla/`. The wheel includes the CANN Open Software License Agreement Version 2.0 and SHA256 provenance in `implementation/cannbot-dsl/`.

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_sources.py tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla.py -q
```

The second command requires Ascend 950; tests skip on other devices.
