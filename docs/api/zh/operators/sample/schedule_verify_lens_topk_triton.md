# schedule_verify_lens_topk_triton

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持（已验证 Ascend910_9362） | 支持（已验证 Ascend 950） |
| `torch_native` | 支持（已验证 Ascend910_9362） | 支持（已验证 Ascend 950） |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 接口

`auto` 选择 Triton 实现。

```python
from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

def schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto"):
    ...
```

## 功能

根据各 token 的置信度调度 DSpark 验证长度。请求 `r` 在位置 `p` 的生存概率以 FP32 计算：

$$
S_{r,p} = \prod_{j=0}^{p} \operatorname{float32}(\mathrm{confidence}_{r,j}).
$$

仅考虑前 `min(gamma, max_len)` 个位置，从生存概率不低于 `cfg.survival_eps` 的候选中选取最多 `budget` 个；按概率降序、位置升序、请求下标升序排序。设请求 `r` 选中数量为 `count[r]`，结果为 `clamp(cfg.min_verify_len + count[r], max(cfg.min_verify_len, 1), max_len)`，并转换为 int32。`max_len` 为 `cfg.resolved_max_verify_len()` 的返回值；预算不大于零时不选候选。语义参考 [SGLang 上游实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)。

## 参数与返回值

调用方负责保证输入和配置合法；公开接口直接转发参数，不进行输入校验。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明和约束 |
| --- | --- | --- | --- | --- |
| `confidence` | `torch.Tensor` | `[B, gamma]`，float16、bfloat16 或 float32 | 必填 | `[0, 1]` 内的置信度；Triton 要求 NPU 张量，支持非连续输入。 |
| `budget` | `int` | — | 必填 | 最多选取的候选数；不大于零时不选候选。 |
| `cfg` | 对象 | — | 必填 | DSpark 配置，提供 `min_verify_len`、`survival_eps` 和 `resolved_max_verify_len()`。调用方须保证 `0 <= min_verify_len <= max_len <= gamma + 1` 且 `survival_eps >= 0`。 |
| `impl` | `str` | — | `"auto"` | `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `verify_lens` | `torch.Tensor` | `[B]`，int32 | 与输入同设备的新张量，不修改输入。 |

## triton 后端

使用三个 Triton kernel；非连续输入会先复制为连续张量。两两比较排名的计算量随候选数呈二次增长。不同实现的累乘舍入可能改变极接近数值的排序。

## torch_native 后端

使用 PyTorch 累乘、稳定排序和 scatter-add 作为独立参考实现，支持非连续输入。

## 示例

```python
from dataclasses import dataclass

import torch
import torch_npu

from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton


@dataclass
class Config:
    min_verify_len: int = 1
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self):
        return 4


confidence = torch.ones((2, 3), dtype=torch.float32, device="npu")
kwargs = dict(confidence=confidence, budget=3, cfg=Config())
result = schedule_verify_lens_topk_triton(**kwargs, impl="triton")
reference = schedule_verify_lens_topk_triton(**kwargs, impl="torch_native")
torch.testing.assert_close(result, reference, rtol=0, atol=0)
# result: [3, 2]
```
