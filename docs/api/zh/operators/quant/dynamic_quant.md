# dynamic_quant

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.quant import dynamic_quant


def dynamic_quant(input, *, impl="auto", **kwargs): ...
```

## 功能说明

对称 INT8 模式逐 token 计算 $s=\max|x|/127$，按 SDK 的舍入和范围量化 $x/s$。可在计算最大值前应用平滑系数。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [..., H], FP16/BF16/FP32 as supported | 必传 | H 为特征宽度的 ND 浮点激活。 |
| `dst_type` | torch.dtype | - | `None` | 可选关键字。标准逐 token 模式的目标为 INT8。 |
| `smooth_scales`, `group_index` | Tensor or None | [H] 或 [groups, H], 与 [groups] | `None` | 可选平滑系数和专家组边界。 |
| `quant_mode` | str | - | `"pertoken"` | 可选 SDK 量化粒度，默认为 pertoken。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `values`, `scales` | tuple[Tensor, Tensor] | destination dtype, FP32 | 普通逐 token 模式返回输入形状的 INT8 值和每行一个尺度。 |

## ops-nn 后端

调用 `torch_npu.npu_dynamic_quant`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/quant/dynamic_quant)

## torch_native 后端

计算逐 token 的 INT8 或 FP8 量化值和 FP32 scale，支持可选的分专家平滑系数。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_quant

x = torch.randn((1, 4096), dtype=torch.float16, device="npu")
values, scales = dynamic_quant(x, dst_type=torch.int8, impl="ops-nn")
```
