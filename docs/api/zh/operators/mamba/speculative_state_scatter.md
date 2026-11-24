# speculative_state_scatter

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mamba import speculative_state_scatter


def speculative_state_scatter(
    dst, src, dst_indices, src_indices, step_indices, *, impl="auto"
): ...
```

## 功能描述

跨所有层将每个请求选中的投机卷积状态快照提交到持久缓存。对请求 `r`，将 `src[:, src_indices[r], step_indices[r]]` 原地复制到 `dst[:, dst_indices[r]]`。索引会转换为 INT32，三种索引中只要有一个为负，就跳过该请求，保留原有的目标状态。

## 参数与返回值

所有张量位于同一 NPU。状态尾部可包含一至三个维度，源和目标的状态尾部形状必须相同。当目标切片均可写且互不重叠时，支持非连续状态张量。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dst` | Tensor | 必填 | 可写的持久状态，形状为 `[layers, slots, *state]`。 |
| `src` | Tensor | 必填 | 快照，形状为 `[layers, scratch, steps, *state]`，dtype 与 `dst` 相同。 |
| `dst_indices` | Tensor | 必填 | 整数目标槽索引，形状为 `[requests]`。有效请求的目标槽必须在范围内且互不相同。 |
| `src_indices` | Tensor | 必填 | 整数源槽索引，形状为 `[requests]`。有效请求的源槽必须在范围内。 |
| `step_indices` | Tensor | 必填 | 整数快照步索引，形状为 `[requests]`。有效请求的步索引必须在范围内。 |
| `impl` | str | `"auto"` | 选择实现方式。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 原 `dst` 张量。请求列表为空时同样返回该张量。 |

## triton 后端

- 根据源和目标张量的步幅复制选中的快照。
- 在设备侧屏蔽负索引，NPU 图重放时可以使用更新后的索引内容。

## torch_native 后端

- 用 PyTorch 筛选有效请求，通过索引读取和原地写入复制快照。
- PyTorch 索引筛选的结果大小取决于数据。NPU 图捕获使用 Triton 实现。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import speculative_state_scatter

dst = torch.zeros(2, 4, 3, 128, device="npu", dtype=torch.bfloat16)
src = torch.randn(2, 3, 4, 3, 128, device="npu", dtype=dst.dtype)
dst_indices = torch.tensor([2], device="npu", dtype=torch.int32)
src_indices = torch.tensor([1], device="npu", dtype=torch.int32)
step_indices = torch.tensor([3], device="npu", dtype=torch.int32)
result = speculative_state_scatter(
    dst, src, dst_indices, src_indices, step_indices, impl="triton"
)
assert result is dst
```
