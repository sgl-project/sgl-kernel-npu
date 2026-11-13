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
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | [rows,D] 数值及 [rows] FP32 scale；关闭量化时 scale 未定义。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_quant

x = torch.randn(4, 3072, device="npu", dtype=torch.bfloat16)
groups = torch.tensor([4], device="npu", dtype=torch.int64)
y, scale = swiglu_quant(x, groups, 1, impl="triton")
```
