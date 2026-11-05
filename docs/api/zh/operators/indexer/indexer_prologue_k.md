# indexer_prologue_k

## 功能与支持范围

Indexer K 前处理：BF16 投影、RMSNorm、RoPE、MXFP4 E2M1 量化，并写入分页 K cache。

支持 Ascend 950PR/950DT（`arch35`），`impl="auto"` 与 `impl="cannbot_dsl"` 选择同一个源码 JIT 实现。依赖版本、构建和验证状态见[开发指南](../../../../developer_guide/cannbotdsl.zh.md)。

## 函数原型

```python
import torch
from typing import Optional

from sgl_kernel_npu.operators.indexer import indexer_prologue_k

indexer_prologue_k(latent: torch.Tensor, wk: torch.Tensor, norm_weight: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor, k_cache: torch.Tensor, k_scale_cache: torch.Tensor | None=None, *, cache_index: torch.Tensor, storage_mode: int, norm_eps: float, combined_block_size: int=-1, impl='auto')
```

## 输入约束

输入 `latent` 为 BF16 `[T,H]`，`wk` 为 BF16 `[D,H]` 的 FRACTAL_NZ 权重；`norm_weight`、`rope_sin`、`rope_cos` 为 FP32。`cache_index` 为 INT64 `[T]`，值 `-1` 跳过写入。`k_cache` 为 UINT8；`storage_mode=0` 时使用独立 `k_scale_cache`，`storage_mode=1` 时按 `combined_block_size` 合并存储。`D` 是 32 的倍数，RoPE 维度为偶数。

## 返回值与调用顺序

原地写入并返回同一个 `k_cache` Tensor。

## 设备测试

```bash
python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_indexer_prologue_k.py --npu-device 0 -q
```
