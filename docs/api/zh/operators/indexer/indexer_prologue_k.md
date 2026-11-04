# indexer_prologue_k

## 功能与支持范围

Indexer K 前处理：BF16 投影、RMSNorm、RoPE、MXFP4 E2M1 量化，并写入分页 K cache。

仅支持 `arch35`（Ascend 950PR/950DT）。Ascend910_9362 不受支持，调用公开接口会在执行前明确报错。`impl="auto"` 与 `impl="cannbot_dsl"` 均选择 CANNBot-DSL 实现；不提供其他代际的自动回退。运行环境须安装匹配的 `torch`/`torch_npu`、CANN 9.2.1 和 `cannbot-dsl==0.8.0`。首次调用会加载或编译 DSL 实现；后续调用复用进程内缓存。

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

## 来源与验证

本实现逐文件迁移自 [cann/cannbot-dsl](https://gitcode.com/cann/cannbot-dsl) 提交 `71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231` 的 `samples/attention/indexer_prologue_k/`；完整几何、边界条件和精度阈值见[上游说明](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/attention/indexer_prologue_k/README.md)。迁移的源文件保留原版权声明，许可证与来源校验值见 wheel 内的 `implementation/cannbot-dsl/LICENSE.cannbot-dsl` 和 `SOURCE.md`。

源码 JIT 模式请设置 `CANNBOTDSL_NATIVE_BINARY_MODE=off`；metadata 首次调用需要 CANNBot-DSL 的 AICPU 编译工具链。CPU 环境可运行接入与源码一致性测试：

```bash
python -m pytest tests/python/sgl_kernel_npu/test_dsv41_cannbot_dsl_api.py -q
```

Ascend 950 与上述软件环境中运行设备精度测试：

```bash
CANNBOTDSL_NATIVE_BINARY_MODE=off python -m pytest tests/python/sgl_kernel_npu/dsv41_cannbot_dsl/test_indexer_prologue_k.py -q
```
