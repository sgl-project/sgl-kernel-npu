# mixed_quant_sparse_flash_mla

## 功能与支持范围

混合量化稀疏 MLA：对 BF16 Query、FP8 原始 KV 和可选 FP4 压缩 KV 执行稀疏注意力。

仅支持 `arch35`（Ascend 950PR/950DT）。Ascend910_9362 不受支持，调用公开接口会在执行前明确报错。`impl="auto"` 与 `impl="cannbot_dsl"` 均选择 CANNBot-DSL 实现；不提供其他代际的自动回退。运行环境须安装匹配的 `torch`/`torch_npu`、CANN 9.2.0（本次验证目标；上游文档为 9.2.1，兼容性待设备验证） 和 `cannbot-dsl==0.8.0`。首次调用会加载或编译 DSL 实现；后续调用复用进程内缓存。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla

mixed_quant_sparse_flash_mla(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, ori_topk_length=None, cmp_topk_length=None, sinks=None, metadata=None, quant_mode, softmax_scale=None, layout_q='TND', layout_kv='PA_BBND', return_softmax_lse=False, out, lse, impl='auto')
```

## 输入约束

`q` 为 BF16 TND `[T1,64,512]`；`ori_kv` 为 UINT8 PA_BBND `[pages,page_size,1,544]`，`cmp_kv` 可选，格式为 `[pages,page_size,1,320]`。两侧需分别提供稀疏索引、页表与可选有效 TopK 长度。`metadata` 来自 `mixed_quant_sparse_flash_mla_metadata`；`quant_mode=1`。`out` 和 `lse` 由调用方预分配。

## 返回值与调用顺序

返回值遵循上游 ABI，结果写入调用方的 `out`（BF16）与 `lse`（FP32）。`return_softmax_lse=True` 时 `lse` 为 `[1,T1,64]`；否则可传空 Tensor。

## 来源与验证

本实现逐文件迁移自 [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) 提交 `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231` 的 `samples/attention/mixed_quant_sparse_flash_mla/`；完整几何、边界条件和精度阈值见[上游说明](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/attention/mixed_quant_sparse_flash_mla/README.md)。迁移的源文件保留原版权声明，许可证与来源校验值见 wheel 内的 `implementation/cannbot-dsl/LICENSE.cannbot-dsl` 和 `SOURCE.md`。

源码 JIT 模式请设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`；metadata 首次调用需要 CANNBot-DSL 的 AICPU 编译工具链。CPU 环境可运行接入与源码一致性测试：

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```

Ascend 950 与上述软件环境中运行设备精度测试：

```bash
CANNBOTDSL_RUN_SMLA_TESTS=1 CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla.py -q
```
