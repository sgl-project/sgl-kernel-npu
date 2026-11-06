# mixed_quant_sparse_flash_mla

## 功能与支持范围

混合量化稀疏 MLA：对 BF16 Query、FP8 原始 KV 和可选 FP4 压缩 KV 执行稀疏注意力。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

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

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_mixed_quant_sparse_flash_mla.py --npu-device 0 -q
```
