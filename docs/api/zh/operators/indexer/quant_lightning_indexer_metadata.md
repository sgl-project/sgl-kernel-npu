# quant_lightning_indexer_metadata

## 功能与支持范围

QLI 的 AICPU 调度算子：由序列长度、mask、压缩比例和 TopK 生成分核与跨核归并计划。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

quant_lightning_indexer_metadata(cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None, cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1, num_heads_q, num_heads_k, head_dim, topk, mask_mode=0, cmp_ratio=1, layout_q='TND', layout_k='TND', candidate_topk_blocks=-1, candidate_block_size=-1, impl='auto')
```

## 输入约束

`cu_seqlens_q/k` 为 INT32 累积长度，`seqused_q/k` 为 INT32 有效长度；`num_heads_q=32`、`num_heads_k=1`、`head_dim=128` 与 QLI 配套。候选块参数必须与 QLI 主算子一致。长度或静态属性变化后需重建 metadata。

## 返回值与调用顺序

返回当前 NPU 上的 INT32 `[1024]` metadata，原样传给 `quant_lightning_indexer(..., metadata=metadata)`；不应手工修改内部布局。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer_metadata.py --npu-device 0 -q
```
