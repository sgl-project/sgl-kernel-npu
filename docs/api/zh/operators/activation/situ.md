# situ

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.activation import situ


def situ(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta=25.0,
    impl="auto"
): ...
```

## 功能描述

对每行前半部分计算 beta*tanh(gate/beta)*sigmoid(gate)，再乘以后半部分；可使用 linear_beta*tanh(up/linear_beta) 限制后半部分。仅写入实际路由的行。可选 INT8 量化采用逐行最大绝对值 scale、1e-30 的 scale 下限、floor(x/scale+0.5) 舍入及 [-128,127] 饱和。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `hidden_states` | Tensor | `Required` | 连续浮点输入 [rows,2*D]。 |
| `group_list` | Tensor | `Required` | 连续 INT32/INT64 专家计数。type 0 需要在逻辑视图之外、存储偏移 numel() 处额外保留总行数。 |
| `group_list_type` | int | `Required` | 1 表示计数，0 表示带末尾总数存储的累积元数据。 |
| `need_quant` | bool | `Required` | 为 true 时返回 INT8 数值及 FP32 逐行 scale。 |
| `beta` | float | `4.0` | 对应算子的 softplus 乘数或 SiTU 饱和参数。 |
| `linear_beta` | float or None | `25.0` | SiTU 的 up 通道边界，None 关闭 tanh 限幅。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor, Tensor or None] | [rows,D] 激活或 INT8 输出及 [rows] FP32 scale，关闭量化时为 None。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import situ

x = torch.randn(4, 3072, device="npu", dtype=torch.bfloat16)
groups = torch.tensor([4], device="npu", dtype=torch.int64)
y, scale = situ(x, groups, 1, need_quant=True, impl="triton")
```
