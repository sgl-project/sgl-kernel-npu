# mixed_quant_sparse_flash_mla

## Purpose and support

Computes sparse MLA over BF16 queries, FP8 original KV, and optional FP4 compressed KV.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

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

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla.py --npu-device 0 -q
```
