# dynamic_mx_quant

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.quant import dynamic_mx_quant


def dynamic_mx_quant(input, *, impl="auto", **kwargs): ...
```

## 功能说明

沿量化轴为每块计算共享的二进制尺度 $s=2^e$，再将 $x/s$ 转换成 FP8 或打包 FP4。指数选择和舍入由尺度算法和目标 dtype 决定。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | FP16/BF16/FP32, ND | 必传 | ND 浮点激活。目标为 FP4 时末轴须为偶数。 |
| `axis`, `block_size` | int | - | `-1` / `32` | 可选量化轴和分块宽度，默认为 -1 和 32。 |
| `dst_type` | torch.dtype or SDK dtype | - | `torch_npu.float4_e2m1fn_x2` | 可选目标类型，例如 torch.float8_e4m3fn 或 torch_npu.float4_e2m1fn_x2。 |
| `round_mode`, `scale_alg` | str, int | - | `"rint"` / `0` | 可选舍入和尺度算法，默认 rint 和 0。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `values` | Tensor | FP8 或 打包 FP4 bytes | FP8 保持输入形状。FP4 在末轴按两个值一个字节打包。 |
| `scales` | Tensor | E8M0, stored as bytes | 尺度在新增末轴中两两打包，末轴大小为 2。量化轴含 ceil(ceil(size / block_size) / 2) 个组对。 |

## torch_npu 后端

调用 `torch_npu.npu_dynamic_mx_quant`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.quant import dynamic_mx_quant

x = torch.randn((16, 256), dtype=torch.bfloat16, device="npu")
y, s = dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, impl="torch_npu")
```
