# format_cast

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.data_move import format_cast


def format_cast(
    input, acl_format, *, customize_dtype=None, input_dtype=None, impl="auto"
): ...
```

## 功能说明

转换 NPU Tensor 的物理存储格式。普通 dtype 的格式转换保持逻辑形状和值。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | any supported shape 与 输入 dtype | 必传 | 待转换 Tensor。 |
| `acl_format` | int or Tensor | - | 必传 | 目标 ACL 格式编号（例如 2 表示 ND），也可传入提供格式的 Tensor。 |
| `customize_dtype` | torch.dtype or None | - | `None` | 目标打包表示的可选逻辑 dtype。 |
| `input_dtype` | torch.dtype or None | - | `None` | 输入打包表示的可选逻辑 dtype。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | target storage format | 输入 NPU 上转换后的结果。 |

## torch_npu 后端

调用 `torch_npu.npu_format_cast`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.data_move import format_cast

x = torch.randn((16, 32), dtype=torch.float16, device="npu")
y = format_cast(x, 2, impl="torch_npu")
```
