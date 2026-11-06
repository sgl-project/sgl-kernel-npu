# mixed_quant_sparse_flash_mla_metadata

## Purpose and support

Plans full-row or FlashDecode split work on AICPU from the original and compressed TopK lengths.

Supports Ascend 950PR/950DT (`arch35`). `impl="auto"` and `impl="cannbot_dsl"` select the same source JIT backend. See the [build and validation guide](../../../../developer_guide/cannbotdsl.md) for dependencies and validation status.

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

## Device test

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla_metadata.py --npu-device 0 -q
```
