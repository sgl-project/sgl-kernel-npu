# schedule_verify_lens_topk_triton

## 硬件支持

`triton` 和 `torch_native` 已注册到 arch22、arch35，并已在 Ascend910_9362 上验证。其他设备仍需验证；注册本身不代表硬件验证通过。`auto` 选择 `triton`。

## 接口

```python
from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto")
```

## 功能

对 `[B, gamma]` 的置信度按行计算 FP32 累乘。仅考虑前 `min(gamma, max_len)` 个位置，从生存概率不低于 `cfg.survival_eps` 的候选中选取最多 `budget` 个；按概率降序、位置升序、请求下标升序排序。每个请求的选中数量加上 `cfg.min_verify_len`，再截断到 `[max(cfg.min_verify_len, 1), max_len]`。返回同设备新分配的 `[B]` int32 张量；预算不大于零时不选候选。支持非连续输入，不修改输入。

`cfg` 为 SGLang DSpark 调度配置或提供 `min_verify_len`、`survival_eps`、`resolved_max_verify_len()` 的兼容对象；`max_len` 为该方法的返回值。调用方负责保证配置合法，置信度为 `[0, 1]` 内的浮点数。语义参考 [SGLang 上游实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)。

| 参数 | 说明 |
| --- | --- |
| `confidence` | NPU 张量，形状 `[B, gamma]`，dtype 为 float16、bfloat16 或 float32。 |
| `budget` | Python 整数，最多选取的候选数。 |
| `cfg` | DSpark 调度配置。 |
| `impl` | `auto`、`triton` 或 `torch_native`。 |

Triton 使用两两比较排名，计算量随候选数量呈二次增长。不同实现的浮点累乘舍入可能改变极接近数值的排序；完全相同的值遵循上述平局规则。
