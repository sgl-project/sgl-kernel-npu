# quant_sparse_lightning_indexer_metadata

## 功能与支持范围

QSLI 的 AICPU 调度算子：按每行候选长度和序列可见范围规划分核与跨核归并。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer_metadata

quant_sparse_lightning_indexer_metadata(candidate_block_length, cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, quant_mode, candidate_block_size, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', impl='auto')
```

## 输入约束

`candidate_block_length` 为每个 Query 的有效候选块数量，`cu_seqlens_q/k`、`seqused_q/k` 描述存储和有效长度；`candidate_block_size=8`、`quant_mode=1` 应与主算子一致。候选长度或静态属性变化后需重建 metadata；不可用 QLI metadata 替代。

## 返回值与调用顺序

返回 INT32 `[1024]` metadata，原样传给 `quant_sparse_lightning_indexer(..., metadata=metadata)`。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer_metadata.py --npu-device 0 -q
```
