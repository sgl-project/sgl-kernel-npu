# swiglu_group_quant

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 不支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.activation import swiglu_group_quant

def swiglu_group_quant(x, topk_weight=None, group_index=None, dst_type=None, quant_mode=1, group_size=128, round_scale=False, ue8m0_scale=False, output_origin=False, group_list_type=0, clamp_value=0.0, *, impl='auto'): ...
```

`auto` 选择 arch35 的 `ascendc`，不在不支持的芯片上自动回退其他实现。

## 功能说明

计算专家 FFN 的 SwiGLU 激活并量化输出，包括 DeepSeek-V4 的 MXFP8 激活路径。输入末维由 gate、up 两部分拼接，先计算 SiLU(gate) × up，可在激活前执行 clamp，并乘逐行路由权重。

量化模式决定 scale 的计算方式：

- 模式 1：FP8 分组量化，每 128 个激活值对应一个 FP32 scale，scale 根据目标 FP8 格式计算。
- 模式 2：MXFP8 量化，每 32 个值对应一个 E8M0 scale，依据舍入到输入 dtype 后的激活指数生成。
- 模式 3：FP8 分块量化，每 128 个值计算一个 scale，最大绝对值下限为 1e-4，scale 基准固定为 448。round_scale 将 scale 向上取整为 2 的幂，ue8m0_scale 可将取整后的 scale 存为 E8M0。

返回量化结果 y、scale 和 y_origin；仅模式 3 开启 output_origin 时写出未量化激活。提供 group_index 时，依据逐专家 token 数限制有效处理行。

## 参数与返回值

N 为 x 除最后一维以外各维度的乘积。使用 NPU 实现时，所有张量放在同一 NPU 上，并使用连续输入。可选张量为 None 或空张量时视为未提供。输入不会被修改。

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | [...,2C] | FP16/BF16 | 必填 | 最后一维依次拼接 gate 和 up，要求连续。模式 1/3 的宽度须为 256 的倍数，模式 2 为 128 的倍数。 |
| `topk_weight` | Tensor | [N] | FP32 | None | 可选，每个 token 的路由权重。 |
| `group_index` | Tensor | [E] | int64 | None | 可选，每个专家的 token 数。计数须非负，只处理前 min(N, sum(counts)) 行，其余输出行的值未定义。 |
| `dst_type` | torch.dtype | — | — | None | 输出 FP8 格式，默认 E4M3FN，也支持 E5M2。 |
| `quant_mode` | int | — | — | 1 | 1 为分组量化，2 为 MX，3 为逐 token 分块 FP8。 |
| `group_size` | int | — | — | 128 | 兼容参数。模式 1/3 每 128 个值一组，模式 2 每 32 个值一组。 |
| `round_scale` | bool | — | — | False | 在模式 3 中将 scale 向上取整为 2 的幂。 |
| `ue8m0_scale` | bool | — | — | False | 模式 3 将取整的 scale 存为 E8M0，与 round_scale 配合使用。 |
| `output_origin` | bool | — | — | False | 模式 3 写出未量化的激活值。 |
| `group_list_type` | int | — | — | 0 | 0 表示逐专家计数。 |
| `clamp_value` | float | — | — | 0.0 | 0 表示关闭；正值在激活前约束 gate 上界及 up 的上下界。 |
| `impl` | str | — | — | 'auto' | 可选 auto、ascendc、torch_native；默认 auto 选择 A5 的 ascendc，不自动回退其他实现。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `y` | Tensor | FP8，形状 [..., C]。 |
| `scale` | Tensor | 模式 2 为 E8M0，形状 [..., C/64, 2]；模式 1/3 为 [..., C/128]，数据类型为 FP32；模式 3 设置 ue8m0_scale=True 时为 E8M0。 |
| `y_origin` | Tensor | 输入类型，形状 [..., C]，仅模式 3 且 output_origin=True 时有效。 |

## ascendc 后端

在 A5 上执行融合激活和量化。

## torch_native 后端

使用 PyTorch 在输入所在设备操作。

## 示例

在已安装 AscendC 实现的 arch35 设备上运行。

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import swiglu_group_quant

x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
counts = torch.tensor([1, 0, 3], device="npu", dtype=torch.int64)
y, scale, _ = swiglu_group_quant(x, group_index=counts, quant_mode=2, impl="ascendc")
print(y.shape, scale.shape)
```
