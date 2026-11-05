# mixed_quant_sparse_flash_mla_metadata

## 功能与支持范围

MQSMLA 的 AICPU 调度算子：根据两侧 TopK 长度生成整行或 FlashDecode 分片执行计划。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, *, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None, num_heads_q, num_heads_kv, head_dim, quant_mode, layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=True, impl='auto')
```

## 输入约束

`ori_topk_length` 与 `cmp_topk_length` 是每个 Query 行两侧的有效稀疏长度；`has_ori_kv` 与 `has_cmp_kv` 应与主算子提供的 KV 侧一致。`num_heads_q=64`、`num_heads_kv=1`、`head_dim=512`、`quant_mode=1`。

## 返回值与调用顺序

返回 INT32 `[1024]` metadata；同一 NPU stream 中先生成，再传给 `mixed_quant_sparse_flash_mla(..., metadata=metadata)`。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla_metadata.py --npu-device 0 -q
```
