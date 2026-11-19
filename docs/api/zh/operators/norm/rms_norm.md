# rms_norm

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.norm import rms_norm


def rms_norm(input, gamma, epsilon=1e-06, *, impl="auto"): ...
```

## 功能说明

在 gamma 对应的末尾维度上计算 $r=1/\sqrt{\mathrm{mean}(x^2)+\epsilon}$，返回 $(x\odot r\odot\gamma,r)$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [B, ..., H], FP16/BF16/FP32 | 必传 | 浮点 ND 激活。 |
| `gamma` | Tensor | trailing dimensions of input, 输入 dtype | 必传 | 归一化权重。 |
| `epsilon` | float | - | `1e-06` | 平方根内的非负稳定项。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `output`, `rstd` | tuple[Tensor, Tensor] | 输入 dtype, FP32 | 输出形状与输入一致。rstd 的归约轴保留为长度 1。 |

## torch_npu 后端

调用 `torch_npu.npu_rms_norm`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/op_plugin/config/op_plugin_functions.yaml#L6860)

## torch_native 后端

以 FP32 计算均方根统计量和乘积，再转换输出 dtype。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import rms_norm

x = torch.randn((16, 7168), dtype=torch.float32, device="npu")
gamma = torch.ones((7168,), dtype=x.dtype, device=x.device)
y, rstd = rms_norm(x, gamma, impl="torch_npu")
```
