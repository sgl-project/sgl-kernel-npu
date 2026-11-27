# swiglu_quant

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.activation import swiglu_quant


def swiglu_quant(
    x,
    group_list,
    group_list_type,
    need_quant=True,
    do_limit=False,
    limit=7.0,
    *,
    clamp_before_silu=False,
    impl="auto"
): ...
```

## 功能描述

对 MoE 路由的有效行计算前后两半布局的 SwiGLU，可选逐行 INT8 量化。do_limit 对 SiLU(gate) 做上界截断、对 up 做双边截断，再相乘。量化使用 max(abs(value))/127、floor(value/scale+0.5) 及 [-128,127] 饱和；未路由的行以及关闭量化时的 scale 未定义。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | 连续浮点 gate/up 两半 [rows,2*D]。 |
| `group_list` | Tensor | `Required` | 连续 INT32/INT64 专家计数。type 0 需要在逻辑视图之外、存储偏移 numel() 处额外保留总行数。 |
| `group_list_type` | int | `Required` | 1 表示计数，0 表示带末尾总数存储的累积元数据。 |
| `need_quant` | bool | `True` | 为 true 时返回 INT8 数值及 FP32 逐行 scale。 |
| `do_limit` | bool | `False` | 在乘法前截断 SiLU(gate) 和 up。 |
| `limit` | float | `7.0` | 截断幅度。 |
| `clamp_before_silu` | bool | `False` | 在 SiLU 前对 gate 做上界截断，对 up 做双边截断。优先于 `do_limit`。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | [rows,D] 数值及 [rows] FP32 scale；关闭量化时 scale 未定义。 |

设置 `clamp_before_silu=True` 时，type 0 使用包含当前专家的累计计数，直接读取 `group_list` 逻辑视图的末项。type 1 在设备侧求和。逐行 scale 的下界是 `1e-30`，全零行量化为 INT8 零。输出保留固定容量，只写有效前缀。后续 grouped matmul 应传入相同分组信息以跳过无效行。

## triton 后端

- 按连续行读取输入，前半列为 gate，后半列为 up。融合 SwiGLU、可选限幅及逐行量化，中间计算使用 FP32。
- 只写分组信息指定的有效行，有效行数不得超过输入行数。量化时输出 INT8 数值和逐行 FP32 scale，非量化数值保留输入 dtype。

## torch_native 后端

- 将 gate/up 转为 FP32，通过 sigmoid 或 SiLU、限幅和乘法计算结果，再按需要量化。
- 按逐行最大绝对值计算 scale，使用 floor(value/scale+0.5) 舍入；非量化分支不填充 scale，调用方应忽略该返回项。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_quant

x = torch.randn(4, 3072, device="npu", dtype=torch.bfloat16)
groups = torch.tensor([4], device="npu", dtype=torch.int64)
y, scale = swiglu_quant(x, groups, 1, impl="triton")
```
