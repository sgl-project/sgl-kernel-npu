# swiglu_group_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `triton` | 不支持 | 不支持 |
| `tilelang` | 不支持 | 不支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.activation import swiglu_group_quant

def swiglu_group_quant(x, topk_weight=None, group_index=None, dst_type=None, quant_mode=1, group_size=128, round_scale=False, ue8m0_scale=False, output_origin=False, group_list_type=0, clamp_value=0.0, *, impl='auto'): ...
```

`auto` 选择 `ascendc`。

## 功能说明

沿最后一维拆分 gate 和 up，计算 SiLU(gate) × up，可乘逐行路由权重，再按指定模式量化。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | [...,2C] | FP16/BF16 | 必填 | 最后一维依次拼接 gate 和 up，要求连续。模式 1/3 的宽度须为 256 的倍数，模式 2 为 128 的倍数。 |
| `topk_weight` | Tensor | [N] | FP32 | None | 可选，每个 token 的路由权重。 |
| `group_index` | Tensor | [E] | int64 | None | 可选，每个专家的 token 数。只写入计数之和以内的有效输入行。 |
| `dst_type` | torch.dtype | — | — | None | 输出 FP8 格式，默认 E4M3FN，也支持 E5M2。 |
| `quant_mode` | int | — | — | 1 | 1 为分组量化，2 为 MX，3 为逐 token 分块 FP8。 |
| `group_size` | int | — | — | 128 | 兼容参数。模式 1/3 每 128 个值一组，模式 2 每 32 个值一组。 |
| `round_scale` | bool | — | — | False | 在模式 3 中将 scale 向上取整为 2 的幂。 |
| `ue8m0_scale` | bool | — | — | False | 模式 3 将取整的 scale 存为 E8M0，与 round_scale 配合使用。 |
| `output_origin` | bool | — | — | False | 模式 3 写出未量化的激活值。 |
| `group_list_type` | int | — | — | 0 | 0 表示逐专家计数。 |
| `clamp_value` | float | — | — | 0.0 | 非零时约束 gate 上界及 up 的上下界。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `y` | Tensor | FP8，形状 [..., C]。 |
| `scale` | Tensor | 模式 2 为 E8M0，形状 [..., C/64, 2]；模式 1/3 为 [..., C/128]。 |
| `y_origin` | Tensor | 输入类型，形状 [..., C]，仅模式 3 且 output_origin=True 时有效。 |

## ascendc 后端

在 A5 上执行融合激活和量化。

## torch_native 后端

使用 PyTorch 在输入所在设备操作。

## cpu 后端

在 CPU 上计算，返回 CPU 张量。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import swiglu_group_quant

x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
counts = torch.tensor([1, 0, 3], device="npu")
y, scale, _ = swiglu_group_quant(x, group_index=counts, quant_mode=2)
print(y.shape, scale.shape)
```
