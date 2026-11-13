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
| `x` | Tensor | `Required` | update 为浮点 [B,C] 或 [B,C,T]；prefill 为 [C,total_tokens] 或 [B,C,T]。 |
| `conv_state` | Tensor | `Required` | 可写浮点状态 [pool,channels,history]，容量至少为 width+steps-2。 |
| `weight` | Tensor | `Required` | 卷积为 [channels,width]；Gemma 为 [H]，与输入同类型。 |
| `bias` | Tensor or None | `None` | 可选卷积偏置 [channels]。 |
| `activation` | str, bool or None | `None` | None 关闭激活，silu/swish 启用 SiLU；update 也接受 bool。 |
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

## triton

普通解码使用 PyTorch，投机执行使用 Triton。

## torch_native

使用输入设备上的 PyTorch 张量运算。

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
