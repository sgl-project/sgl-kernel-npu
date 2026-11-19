# dynamic_dual_level_mx_quant

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant


def dynamic_dual_level_mx_quant(input, *, impl="auto", **kwargs): ...
```

## 功能说明

对输入末轴做 FP4 量化。每 512 元素组对应一个 FP32 一级尺度，每 32 元素子组对应 E8M0 二级尺度。解码值乘以两级尺度得到近似原值。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | [..., K], FP16/BF16 | 必传 | ND 浮点激活。打包输出要求末轴为偶数。 |
| `smooth_scale` | Tensor or None | [K] | `None` | 可选输入特征平滑系数。 |
| `round_mode` | str | - | `"rint"` | 可选舍入方式，SDK 默认 rint。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `values` | Tensor | [..., K/2], 打包 FP4 | 每字节含两个 FP4 值。 |
| `level0_scale` | Tensor | [..., ceil(K/512)], FP32 | 每 512 元素组的尺度。 |
| `level1_scale` | Tensor | [..., ceil(K/64), 2], E8M0 bytes | 每 32 元素子组的尺度，每两个配成一对。 |

## torch_npu 后端

调用 `torch_npu.npu_dynamic_dual_level_mx_quant`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_dual_level_mx_quant.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

x = torch.randn((1, 512), dtype=torch.bfloat16, device="npu")
y, s0, s1 = dynamic_dual_level_mx_quant(x, impl="torch_npu")
```
