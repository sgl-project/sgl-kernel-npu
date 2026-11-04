# quant_lightning_indexer

## 功能与支持范围

MXFP4 Lightning Indexer：遍历可见 Key，按 Query head 加权得分选取 TopK token；可同时输出 QSLI 的候选块。

仅支持 `arch35`（Ascend 950PR/950DT）。Ascend910_9362 不受支持，调用公开接口会在执行前明确报错。`impl="auto"` 与 `impl="cannbot_dsl"` 均选择 CANNBot-DSL 实现；不提供其他代际的自动回退。运行环境须安装匹配的 `torch`/`torch_npu`、CANN 9.2.1 和 `cannbot-dsl==0.8.0`。首次调用会加载或编译 DSL 实现；后续调用复用进程内缓存。

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

## 来源与验证

本实现逐文件迁移自 [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) 提交 `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231` 的 `samples/attention/quant_lightning_indexer/`；完整几何、边界条件和精度阈值见[上游说明](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/attention/quant_lightning_indexer/README.md)。迁移的源文件保留原版权声明，许可证与来源校验值见 wheel 内的 `implementation/cannbot-dsl/LICENSE.cannbot-dsl` 和 `SOURCE.md`。

源码 JIT 模式请设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`；metadata 首次调用需要 CANNBot-DSL 的 AICPU 编译工具链。CPU 环境可运行接入与源码一致性测试：

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```

Ascend 950 与上述软件环境中运行设备精度测试：

```bash
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer.py -q
```
