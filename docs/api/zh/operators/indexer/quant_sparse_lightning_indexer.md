# quant_sparse_lightning_indexer

## 功能与支持范围

MXFP4 Sparse Lightning Indexer：只访问 QLI 给出的候选 Key 块，再选出 TopK token。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_sparse_lightning_indexer

quant_sparse_lightning_indexer(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, descale_q: torch.Tensor, candidate_block_indices: torch.Tensor, candidate_block_length: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, seqused_q: Optional[torch.Tensor]=None, seqused_k: Optional[torch.Tensor]=None, cmp_residual_k: Optional[torch.Tensor]=None, block_table: Optional[torch.Tensor]=None, output_idx_offset: Optional[torch.Tensor]=None, metadata: Optional[torch.Tensor]=None, *, descale_k: Optional[torch.Tensor]=None, topk: int, candidate_block_size: int, quant_mode: int, max_seqlen_q: int=-1, mask_mode: int=0, cmp_ratio: int=1, layout_q: str='TND', layout_k: str='TND', return_value: bool=False, impl='auto')
```

## 输入约束

`q` 为 UINT8 `[T1,32,64]`；`candidate_block_indices` 为 INT32 候选块号，`candidate_block_length` 给出每行有效前缀。`descale_k` 为 E8M0 scale，`metadata` 需由同配置的 QSLI metadata 算子生成。`candidate_block_size=8`、`quant_mode=1`；K 可为 `TND` 或 `PA_BBND`。

## 返回值与调用顺序

返回 `(indices, values)`；INT32 索引与 BF16 分数均为 `[T1,1,topk]`。`return_value=False` 时 values 为空 Tensor；不足 TopK 的位置为索引 `-1`、分数 `-inf`。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_sparse_lightning_indexer.py --npu-device 0 -q
```
