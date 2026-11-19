# add_rms_norm

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.norm import add_rms_norm


def add_rms_norm(x1, x2, gamma, epsilon=1e-06, *, impl="auto"): ...
```

## 功能说明

先计算残差和 $z=x_1+x_2$，再用 $r=1/\sqrt{\mathrm{mean}(z^2)+\epsilon}$ 归一化，返回 $(z\odot r\odot\gamma,r,z)$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | same ND shape, FP16/BF16/FP32 | 必传 | 同卡同 dtype 的激活和残差。 |
| `gamma` | Tensor | trailing dimensions of x1, 输入 dtype | 必传 | 归一化权重。 |
| `epsilon` | float | - | `1e-06` | 平方根内的非负稳定项。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `output`, `rstd`, `residual` | tuple[Tensor, Tensor, Tensor] | 输入 dtype, FP32, 输入 dtype | 输出和残差和的形状与 x1 一致。rstd 保留长度为 1 的归约轴。 |

## torch_npu 后端

调用 `torch_npu.npu_add_rms_norm`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md)

## torch_native 后端

以 FP32 执行残差相加和归一化，并分别将归一化输出和残差和转换回输入 dtype。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import add_rms_norm

x = torch.randn((16, 7168), dtype=torch.float32, device="npu")
residual = torch.randn_like(x)
gamma = torch.ones((7168,), dtype=x.dtype, device=x.device)
y, rstd, updated_residual = add_rms_norm(x, residual, gamma, impl="torch_npu")
```
