# transpose_quant_batchmatmul

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 不支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul


def transpose_quant_batchmatmul(x1, x2, dtype, *, impl="auto", **kwargs): ...
```

## 功能说明

按调用方指定顺序置换轴，对每个 batch 独立执行带尺度的量化矩阵乘。置换为 (1,0,2)、(0,1,2)、(1,0,2) 时，输出按 token 优先排列为 [M,B,N]。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `x1`, `x2` | Tensor | [M, B, K], [B, K, N] 在示例中, FP8 | 必传 | 量化输入。按指定置换后，归约维和批维须一致。 |
| `dtype` | torch.dtype | - | 必传 | 输出 dtype，例如 torch.float16。 |
| `x1_scale`, `x2_scale` | Tensor or None | [M], [N], FP32 在逐行和逐列配置下 | `None` | 可选输入尺度。分组模式要求不同的形状。 |
| `perm_x1`, `perm_x2`, `perm_y` | tuple[int, int, int] | - | `None` | 左右矩阵和结果的可选轴置换。 |
| `group_sizes` | list[int] or None | - | `None` | 可选 SDK 量化分组。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | 其他受支持的 TorchNPU 关键字参数。具体默认值和约束见厂商接口。 |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | [M, B, N] 在示例中, dtype | 经输出置换后的批量矩阵乘结果。 |

## torch_npu 后端

调用 `torch_npu.npu_transpose_quant_batchmatmul`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_transpose_quant_batchmatmul.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul

m, batch, k, n = 32, 32, 512, 128


def fp8(shape):
    return torch.randint(-4, 5, shape, dtype=torch.int8).to(torch.float8_e4m3fn).npu()


a = fp8((m, batch, k))
b = fp8((batch, k, n))
s1 = torch.ones((m,), dtype=torch.float32, device=a.device)
s2 = torch.ones((n,), dtype=torch.float32, device=a.device)
y = transpose_quant_batchmatmul(
    a,
    b,
    torch.float16,
    x1_scale=s1,
    x2_scale=s2,
    perm_x1=(1, 0, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    impl="torch_npu",
)
```
