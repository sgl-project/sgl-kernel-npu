# quant_matmul

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ops-nn` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `ops-nn`。

```python
from sgl_kernel_npu.operators.matmul import quant_matmul


def quant_matmul(x1, x2, scale, *, impl="auto", **kwargs): ...
```

## 功能说明

执行量化矩阵乘。INT8 逐 token 与逐通道配置中，转换结果 dtype 前的核心计算为 $Y_{mn}=(\sum_k A_{mk}B_{kn})\,s_m\,s_n$。其他 SDK 模式可使用不同偏移和分组。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, K], [K, N] for INT8 | 必传 | 归约维度一致的量化矩阵。打包和批量形式取决于 SDK。 |
| `scale` | Tensor | [N], FP32 在示例中 | 必传 | 右矩阵通道或分组尺度，表示方式取决于量化模式。 |
| `pertoken_scale` | Tensor or None | [M], FP32 在示例中 | `None` | 可选逐 token 尺度。 |
| `output_dtype`, `group_sizes` | torch.dtype, list[int] | - | `None` | 可选结果 dtype 和 SDK 分组维度。 |
| `impl` | str | - | `"auto"` | `auto`, `ops-nn`, `torch_native`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | [M, N] 在示例中 | 指定 dtype 的量化矩阵乘结果。 |

## ops-nn 后端

调用 `torch_npu.npu_quant_matmul`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md) [CANN ops-nn 源码](https://gitcode.com/cann/ops-nn/tree/f4e626de9a175c45b022f1413607bf7d4142103f/matmul/quant_batch_matmul_v4)

## torch_native 后端

对 INT8 矩阵计算累加结果，应用浮点通道 scale，支持可选的 token scale 和 bias。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import quant_matmul

a = torch.randint(-3, 4, (16, 128), dtype=torch.int8, device="npu")
b = torch.randint(-3, 4, (128, 64), dtype=torch.int8, device=a.device)
channel_scale = torch.full((64,), 0.25, dtype=torch.float32, device=a.device)
token_scale = torch.full((16,), 0.125, dtype=torch.float32, device=a.device)
y = quant_matmul(
    a,
    b,
    channel_scale,
    pertoken_scale=token_scale,
    output_dtype=torch.bfloat16,
    impl="ops-nn",
)
```
