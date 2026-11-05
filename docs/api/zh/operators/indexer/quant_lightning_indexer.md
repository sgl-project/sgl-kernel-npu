# quant_lightning_indexer

## 功能与支持范围

MXFP4 Lightning Indexer：遍历可见 Key，按 Query head 加权得分选取 TopK token；可同时输出 QSLI 的候选块。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

quant_lightning_indexer(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, descale_q: torch.Tensor, descale_k: torch.Tensor, cu_seqlens_q: Optional[torch.Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, seqused_q: Optional[torch.Tensor]=None, seqused_k: Optional[torch.Tensor]=None, cmp_residual_k: Optional[torch.Tensor]=None, block_table: Optional[torch.Tensor]=None, output_idx_offset: Optional[torch.Tensor]=None, metadata: Optional[torch.Tensor]=None, *, topk: int, quant_mode: int, max_seqlen_q: int=-1, mask_mode: int=0, cmp_ratio: int=1, layout_q: str='TND', layout_k: str='TND', return_value: bool=False, candidate_topk_blocks: int=-1, candidate_block_size: int=-1, impl='auto')
```

## 输入约束

`q` 为 UINT8 `[T1,32,64]`，`k` 为打包 MXFP4 Key（PA_BBND 或 TND），`descale_q` 与 `descale_k` 为 E8M0 scale。`metadata` 必须由同配置的 `quant_lightning_indexer_metadata` 生成。`quant_mode=1`；`layout_q="TND"`，`layout_k` 支持 `TND`/`PA_BBND`；候选输出启用时 `candidate_block_size=8`。

## 返回值与调用顺序

返回 `(indices, values, candidates, candidate_length)`；`indices` 为 INT32 `[T1,1,topk]`，`values` 为 BF16；关闭分数或候选输出时相应 Tensor 为空。无效索引为 `-1`。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer.py --npu-device 0 -q
```
