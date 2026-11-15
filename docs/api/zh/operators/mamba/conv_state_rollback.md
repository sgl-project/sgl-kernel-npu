# conv_state_rollback

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mamba import conv_state_rollback


def conv_state_rollback(
    conv_states, state_indices, step_indices, draft_token_num, *, impl="auto"
): ...
```

## 功能描述

对各选定请求跨层将卷积历史右移 draft_token_num-1-step 个位置。只覆盖右移后的后缀，前部保持不变；非连续视图保持原存储和步幅。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `conv_states` | Tensor | `Required` | 形状为 [layers, pool, window, channels] 的卷积状态，原地更新。 |
| `state_indices` | Tensor | `Required` | 整数回滚槽 [requests]，转换为 INT32。 |
| `step_indices` | Tensor | `Required` | 整数接受步索引 [requests]，转换为 INT32。 |
| `draft_token_num` | int | `Required` | 草稿 token 数。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor or None | 原 conv_states 的别名；空选择返回 None。 |

## triton 后端

- 根据 draft_token_num-1-step_indices 计算各请求的回滚位移，在原卷积状态视图上移动历史窗口。
- 使用状态 stride 处理非连续布局；不需要位移的请求保持不变，非空调用返回原状态张量。

## torch_native 后端

- 逐请求计算回滚位移，通过切片和复制移动历史状态。
- 复制前克隆源切片以避免重叠写入覆盖尚未读取的数据，保留目标对象和布局。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import conv_state_rollback

state = torch.randn(2, 4, 6, 128, device="npu", dtype=torch.bfloat16)
slots = torch.tensor([1], device="npu", dtype=torch.int32)
steps = torch.tensor([1], device="npu", dtype=torch.int32)
conv_state_rollback(state, slots, steps, 4, impl="triton")
```
