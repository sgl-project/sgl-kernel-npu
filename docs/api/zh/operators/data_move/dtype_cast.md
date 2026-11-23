# dtype_cast

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-math` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-math`。

```python
from sgl_kernel_npu.operators.data_move import dtype_cast


def dtype_cast(input, dtype, input_dtype=None, *, impl="auto"): ...
```

## 功能说明

将值转换为目标 dtype。打包类型的编码方式和输出形状由已安装的 SDK 决定。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | any supported shape 与 输入 dtype | 必传 | 源 NPU Tensor。打包类型须使用 SDK 要求的表示。 |
| `dtype` | torch.dtype | - | 必传 | 目标 dtype。 |
| `input_dtype` | torch.dtype or None | - | `None` | 打包表示的可选逻辑输入 dtype。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-math`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | dtype | 同卡转换后的结果。普通 cast 保持逻辑形状。 |

## ops-math 后端

调用 `torch_npu.npu_dtype_cast`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6168) [CANN ops-math 源码](https://gitcode.com/cann/ops-math/tree/a6032a3fbb02592142169b05a2157bea08b47ac5/math/cast)

## torch_native 后端

使用 PyTorch 转换常规张量 dtype。支持将 FP8 E4M3 权重转换为 E2M1，每个输出字节打包两个 FP4 值。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import dtype_cast

x = torch.randn((4, 32), dtype=torch.float32, device="npu")
y = dtype_cast(x, torch.float16, impl="ops-math")
```
