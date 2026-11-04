# quant_lightning_indexer_metadata

## 功能与支持范围

QLI 的 AICPU 调度算子：由序列长度、mask、压缩比例和 TopK 生成分核与跨核归并计划。

仅支持 `arch35`（Ascend 950PR/950DT）。Ascend910_9362 不受支持，调用公开接口会在执行前明确报错。`impl="auto"` 与 `impl="cannbot_dsl"` 均选择 CANNBot-DSL 实现；不提供其他代际的自动回退。运行环境须安装匹配的 `torch`/`torch_npu`、CANN 9.2.1 和 `cannbot-dsl==0.8.0`。首次调用会加载或编译 DSL 实现；后续调用复用进程内缓存。

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

## 来源与验证

本实现逐文件迁移自 [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) 提交 `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231` 的 `samples/attention/quant_lightning_indexer_metadata/`；完整几何、边界条件和精度阈值见[上游说明](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/attention/quant_lightning_indexer_metadata/README.md)。迁移的源文件保留原版权声明，许可证与来源校验值见 wheel 内的 `implementation/cannbot-dsl/LICENSE.cannbot-dsl` 和 `SOURCE.md`。

源码 JIT 模式请设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`；metadata 首次调用需要 CANNBot-DSL 的 AICPU 编译工具链。CPU 环境可运行接入与源码一致性测试：

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```

Ascend 950 与上述软件环境中运行设备精度测试：

```bash
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_quant_lightning_indexer_metadata.py -q
```
