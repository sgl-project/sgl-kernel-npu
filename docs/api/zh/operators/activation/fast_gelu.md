# fast_gelu

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.activation import fast_gelu


def fast_gelu(input, *, impl="auto"): ...
```

## 功能说明

逐元素计算快速 GELU 近似，公式为 $y=x\,\sigma(1.702x)$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | 任意形状, FP16/BF16/FP32 | 必传 | ND 输入。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | 与输入一致 | 输出形状和 dtype 与输入一致。 |

## torch_npu 后端

调用 `torch_npu.npu_fast_gelu`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fast_gelu.md)

## torch_native 后端

使用 FP32 计算 sigmoid 和乘积，再转换回输入 dtype。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fast_gelu

x = torch.randn((16, 3072), dtype=torch.float16, device="npu")
y = fast_gelu(x, impl="torch_npu")
```
