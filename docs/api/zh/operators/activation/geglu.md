# geglu

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.activation import geglu


def geglu(input, dim=-1, approximate=1, activate_left=False, *, impl="auto"): ...
```

## 功能说明

将输入分成等宽的两半。设激活的一半为 $g$，另一半为 $u$，返回 $(\mathrm{GELU}(g)\odot u,\mathrm{GELU}(g))$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32 | 必传 | ND 输入，切分轴长度须为偶数。 |
| `dim` | int | - | `-1` | 切分轴。 |
| `approximate` | int | - | `1` | 0 选择 erf 公式，1 选择 tanh 近似。 |
| `activate_left` | bool | - | `False` | 为 True 时激活前半部分，否则激活后半部分。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `gated_output`, `gelu_output` | tuple[Tensor, Tensor] | FP16/BF16/FP32 | 乘积和 GELU 中间结果，沿 dim 的长度均为输入的一半。 |

## ops-nn 后端

调用 `torch_npu.npu_geglu`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6371) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/activation/ge_glu_v2)

## torch_native 后端

使用 PyTorch 切分，以 FP32 计算 GELU 和乘积，再将两个结果转换回输入 dtype。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import geglu

x = torch.randn((16, 6144), dtype=torch.float16, device="npu")
y, gate = geglu(x, activate_left=True, impl="ops-nn")
```
