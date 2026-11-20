# weight_quant_batchmatmul

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul


def weight_quant_batchmatmul(
    x, weight, antiquant_scale, *args, impl="auto", **kwargs
): ...
```

## 功能说明

按通道或分组尺度与偏移反量化权重，再与浮点激活相乘。设反量化后的权重为 $W'$，输出再量化前的结果为 $Y=XW'+b$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [..., K], FP16/BF16 | 必传 | NPU 浮点激活。 |
| `weight` | Tensor | [K, N] 逻辑ly, INT8 或 SDK 打包 dtype | 必传 | 量化权重。打包、物理格式和步幅限制取决于模式。 |
| `antiquant_scale` | Tensor | [N] 或 [ceil(K/group_size), N] for ordinary scales | 必传 | 与权重量化方案相配的反量化尺度。 |
| `*args / **kwargs` | SDK arguments | - | SDK 默认值 | 支持厂商参数顺序内的 antiquant_offset、quant_scale、quant_offset、bias、antiquant_group_size、inner_precise 和 weight_dtype。组大小默认 0。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | [..., N], dtype from SDK output configuration | 矩阵乘结果。dtype 取决于输入和可选输出量化。 |

## ops-nn 后端

调用 `torch_npu.npu_weight_quant_batchmatmul`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/matmul/weight_quant_batch_matmul_v2)

## torch_native 后端

使用通道 scale 和可选偏移反量化 INT8 权重，计算 A16W8 矩阵乘法，输出浮点结果。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul

x = torch.randn((8192, 320), dtype=torch.float16, device="npu")
weight = torch.randint(-8, 8, (320, 256), dtype=torch.int8, device=x.device)
s = torch.full((1, 256), 0.125, dtype=x.dtype, device=x.device)
y = weight_quant_batchmatmul(x, weight, s, impl="ops-nn")
```
