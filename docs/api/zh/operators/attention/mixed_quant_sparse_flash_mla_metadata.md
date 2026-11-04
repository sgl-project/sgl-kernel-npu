# mixed_quant_sparse_flash_mla_metadata

## 功能与支持范围

MQSMLA 的 AICPU 调度算子：根据两侧 TopK 长度生成整行或 FlashDecode 分片执行计划。

仅支持 `arch35`（Ascend 950PR/950DT）。Ascend910_9362 不受支持，调用公开接口会在执行前明确报错。`impl="auto"` 与 `impl="cannbot_dsl"` 均选择 CANNBot-DSL 实现；不提供其他代际的自动回退。运行环境须安装匹配的 `torch`/`torch_npu`、CANN 9.2.1 和 `cannbot-dsl==0.8.0`。首次调用会加载或编译 DSL 实现；后续调用复用进程内缓存。

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

## 来源与验证

本实现逐文件迁移自 [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) 提交 `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231` 的 `samples/attention/mixed_quant_sparse_flash_mla/`；完整几何、边界条件和精度阈值见[上游说明](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/attention/mixed_quant_sparse_flash_mla/README.md)。迁移的源文件保留原版权声明，许可证与来源校验值见 wheel 内的 `implementation/cannbot-dsl/LICENSE.cannbot-dsl` 和 `SOURCE.md`。

源码 JIT 模式请设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`；metadata 首次调用需要 CANNBot-DSL 的 AICPU 编译工具链。CPU 环境可运行接入与源码一致性测试：

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```

Ascend 950 与上述软件环境中运行设备精度测试：

```bash
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla_metadata.py -q
```
