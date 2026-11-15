# causal_conv1d_fn

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `torch_native`。

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_fn


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=-1,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能描述

对变长拼接或批量补齐的序列执行逐通道因果卷积，支持初始状态、偏置和 SiLU。更新选定缓存槽的最终状态并返回卷积结果；变长输入最后一个查询之后的补齐部分输出为零。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | 形状为 [C,total_tokens] 或 [B,C,T] 的浮点输入。 |
| `weight` | Tensor | `Required` | 形状为 [channels,width] 的卷积权重。 |
| `bias` | Tensor or None | `None` | 可选卷积偏置 [channels]。 |
| `query_start_loc` | Tensor | `None` | INT32/INT64 拼接长度前缀和 [batch+1]，以零开始。 |
| `cache_indices` | Tensor | `None` | 整数 [batch]，每条序列的缓存槽。 |
| `has_initial_state` | Tensor | `None` | 布尔 [batch]，表示各请求是否使用已有历史。 |
| `conv_states` | Tensor | `None` | 形状为 [pool,channels,width-1] 的可写卷积状态。 |
| `activation` | str or None | `'silu'` | `'silu'` 和 `'swish'` 启用 SiLU，`None` 关闭激活。 |
| `pad_slot_id` | int or None | `-1` | 保留的兼容参数。 |
| `impl` | str | `'auto'` | auto 选择 torch_native；可显式选择后端。 |
| `kwargs` | dict | `Required` | 额外兼容参数，当前不使用。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 与 x 同形状、同类型的输出；更新 conv_states。 |

## torch_native 后端

- 将打包的变长序列整理为批量输入，通过分组 conv1d 执行各通道独立的因果卷积，再应用可选 SiLU。
- 根据 `has_initial_state` 选择历史状态，写回各请求的最终卷积窗口，并恢复序列布局。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_fn

x = torch.randn(128, 8, device="npu", dtype=torch.float16)
w = torch.randn(128, 4, device="npu", dtype=x.dtype)
starts = torch.tensor([0, 8], device="npu", dtype=torch.int32)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
flags = torch.tensor([False], device="npu")
state = torch.zeros(1, 128, 3, device="npu", dtype=x.dtype)
y = causal_conv1d_fn(
    x,
    w,
    query_start_loc=starts,
    cache_indices=slots,
    has_initial_state=flags,
    conv_states=state,
    impl="torch_native",
)
```
