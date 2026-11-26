# dynamic_block_quant

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.quant import dynamic_block_quant


def dynamic_block_quant(x, *, impl="auto", **kwargs): ...
```

## 功能说明

沿输入最后两个轴分块量化。各块按最大绝对值计算尺度，可用 min_scale 限制，再做量化和舍入。DSA 的 FP8 配置用于 arch35。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x` | Tensor | rank 2 或 3, FP16/BF16/FP32 as supported | 必传 | ND 激活。 |
| `row_block_size`, `col_block_size` | int | - | `1` / `128` | 可选分块尺寸，SDK 默认分别为 1 和 128。 |
| `dst_type` | torch.dtype | - | `torch.int8` | 可选量化 dtype。FP8 需要兼容的 arch35 SDK。 |
| `min_scale`, `round_mode` | float, str | - | `0.0` / `"rint"` | 可选尺度下限和舍入方式，常用 rint。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `values`, `scales` | tuple[Tensor, Tensor] | input shape, FP32 per block | 量化后的输入及对应的分块尺度。 |

## ops-nn 后端

调用 `torch_npu.npu_dynamic_block_quant`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_block_quant.md) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/quant/dynamic_block_quant)

## torch_native 后端

在默认 scale 限制及 RINT 舍入下，计算分块 INT8 或 FP8 量化值与 scale。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_block_quant

x = torch.randn((16, 256), dtype=torch.float16, device="npu")
y, s = dynamic_block_quant(x, row_block_size=1, col_block_size=128, impl="ops-nn")
```
