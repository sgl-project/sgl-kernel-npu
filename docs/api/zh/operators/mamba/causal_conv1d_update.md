# causal_conv1d_update

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_update


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    intermediate_conv_window=None,
    pad_slot_id=-1,
    metadata=None,
    validate_data=False,
    *,
    impl="auto"
): ...
```

## 功能描述

更新解码或多 token 投机执行的逐通道卷积历史。无接受步数元数据的解码返回新输出；投机路径把输出写入 x，并可在每个 token 后保存历史窗口。状态布局为 [pool,channels,history]。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | 浮点输入，单 token 为 [B,C]，多 token 为 [B,C,T]。 |
| `conv_state` | Tensor | `Required` | 可写浮点状态 [pool,channels,history]，容量至少为 width+steps-2。 |
| `weight` | Tensor | `Required` | 形状为 [channels,width] 的卷积权重。 |
| `bias` | Tensor or None | `None` | 可选卷积偏置 [channels]。 |
| `activation` | str, bool or None | `None` | `'silu'`、`'swish'` 和 `True` 启用 SiLU，`None` 或 `False` 关闭激活。 |
| `cache_seqlens` | Tensor or None | `None` | 保留的 [batch] 元数据；当前路径不实现环形缓存索引。 |
| `conv_state_indices` | Tensor or None | `None` | 整数 [batch] 状态槽映射。 |
| `num_accepted_tokens` | Tensor or None | `None` | 整数 [batch]；投机历史起点为 accepted_count-1。 |
| `intermediate_conv_window` | Tensor or None | `None` | 可写逐步历史 [pool,steps,channels,width-1]。 |
| `pad_slot_id` | int or None | `-1` | 补齐槽哨兵；update 的 Triton 路径跳过匹配请求。 |
| `metadata` | object or None | `None` | 保留参数，当前不使用。 |
| `validate_data` | bool | `False` | 启用 update 原有元数据断言。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | 与 x 同形状、同类型的输出，更新 conv_state 和可选中间窗口；投机路径也写入 x。 |

## triton 后端

- cache_seqlens 和 num_accepted_tokens 都为 None 时，使用 PyTorch 处理普通 decode；带相关元数据时进入 Triton 更新分支。
- Triton 分支从选定历史位置读取窗口，逐 token 卷积并原地写入 x；同时更新状态，并可保存每步中间窗口。

## torch_native 后端

- 普通 decode 将历史窗口与新输入拼接，通过 FP32 分组 conv1d 计算，转回输入 dtype 后应用可选 SiLU。
- 多 token 分支用 PyTorch 按接受位置选取历史并逐步更新输出、状态及可选中间窗口。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_update

x = torch.randn(1, 128, device="npu", dtype=torch.float16)
w = torch.randn(128, 4, device="npu", dtype=x.dtype)
state = torch.zeros(1, 128, 3, device="npu", dtype=x.dtype)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
y = causal_conv1d_update(x, state, w, conv_state_indices=slots, impl="triton")
```
