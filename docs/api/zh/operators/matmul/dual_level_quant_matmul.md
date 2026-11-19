# dual_level_quant_matmul

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul


def dual_level_quant_matmul(
    x1,
    x2,
    x1_level0_scale,
    x2_level0_scale,
    x1_level1_scale,
    x2_level1_scale,
    *,
    impl="auto",
    **kwargs
): ...
```

## 功能说明

分别应用输入的两级尺度后对 FP4 矩阵执行乘法。右矩阵以转置布局传入，计算 $Y=A_{\mathrm{dequant}}B_{\mathrm{dequant}}^\top+b$。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, K/2], [N, K/2], 打包 FP4 | 必传 | 左矩阵为 ND，右矩阵为转置的 FRACTAL_NZ。 |
| `x1_level0_scale`, `x2_level0_scale` | Tensor | [M, ceil(K/512)], [ceil(K/512), N], FP32 | 必传 | 每 512 元素组的 FP32 一级尺度。 |
| `x1_level1_scale`, `x2_level1_scale` | Tensor | [M, ceil(K/64), 2], [N, ceil(K/64), 2], E8M0 bytes | 必传 | 每 32 元素组的二级尺度，按 SDK 格式成对打包。 |
| `bias`, `output_dtype` | Tensor or None, torch.dtype | [N], FP32 bias | `None` / `torch.bfloat16` | 可选关键字。结果 dtype 为 FP16 或 BF16。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | [M, N], output_dtype | 输入 NPU 上的矩阵乘结果。 |

## torch_npu 后端

调用 `torch_npu.npu_dual_level_quant_matmul`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dual_level_quant_matmul.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul

m, n, k = 256, 2048, 1024
a = torch.full((m, k // 2), 0x22, dtype=torch.int8, device="npu")
b_nd = torch.full((n, k // 2), 0x22, dtype=torch.int8, device="npu")
b_nz = torch_npu.npu_format_cast(b_nd, 29, customize_dtype=b_nd.dtype)
a0 = torch.ones((m, k // 512), dtype=torch.float32, device=a.device)
b0 = torch.ones((k // 512, n), dtype=torch.float32, device=a.device)
a1 = torch.full((m, k // 64, 2), 127, dtype=torch.uint8, device=a.device)
b1 = torch.full((n, k // 64, 2), 127, dtype=torch.uint8, device=a.device)
y = dual_level_quant_matmul(
    a, b_nz, a0, b0, a1, b1, output_dtype=torch.bfloat16, impl="torch_npu"
)
```
