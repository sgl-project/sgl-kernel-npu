# swiglu

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.activation import swiglu


def swiglu(input, dim=-1, *, impl="auto"): ...
```

## 功能说明

将打包的 gate 和 up 投影切成等宽的 $g$ 和 $u$，计算 $y=g\,\sigma(g)\odot u$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32 | 必传 | ND 输入，在偶数长度的切分轴上前半段为 gate，后半段为 up。 |
| `dim` | int | - | `-1` | 切分轴。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | 输入 dtype | 沿 dim 的长度减半，其余维度保持输入形状。 |

## ops-nn 后端

调用 `torch_npu.npu_swiglu`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L7011) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/activation/swi_glu)

## torch_native 后端

使用 PyTorch 以 FP32 计算 SiLU 和乘积，再转换回输入 dtype。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu

x = torch.randn((16, 4096), dtype=torch.float16, device="npu")
y = swiglu(x, impl="ops-nn")
```
