# l2norm_fwd

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import l2norm_fwd

def l2norm_fwd(x, eps=1e-6, output_dtype=None, *, impl="auto"):
    ...
```

`auto` 和 `triton` 当前指向同一入口，不会在执行失败后自动切换实现。

## 功能说明

沿最后一维进行 L2 归一化，保留输入 shape：

$$
Y_{\ldots,j}=\frac{X_{\ldots,j}}{\sqrt{\sum_{k=0}^{D-1}X_{\ldots,k}^{2}+\varepsilon}}
$$

`eps` 加在平方和之后、平方根之内。运算不将平方和除以维度长度。计算使用 FP32 中间值，结果转换为指定的输出 dtype；未指定时使用输入 dtype。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 至少一维，`D > 0`。Triton 要求连续的 NPU tensor；前面的维度会通过 `view` 合并为行。 |
| `eps` | `float` | — | — | `1e-6` | 加入平方和的稳定项，建议有限且非负；零输入应使用正数以避免零分母。 |
| `output_dtype` | `torch.dtype` 或 `None` | — | `torch.float32`、`torch.float16`、`torch.bfloat16` 或 — | `None` | 为 None 时使用 x.dtype，否则按指定 dtype 分配输出。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `x.shape` | `output_dtype`，默认 `x.dtype` | 新建输出，与输入位于同一设备，不修改输入。 |

公共入口不添加连续化；两个实现都需要输入可按 `view(-1, D)` 整理。Triton 存在特征宽度限制：`D` 不能超过 `65536 // x.element_size()`，超过时抛出 `RuntimeError`。此限制按输入 dtype 计算。

## triton 后端

- `D <= 512` 使用 `l2norm_fwd_kernel_opt`；`D > 512` 使用 `l2norm_fwd_kernel`。
- 使用 FP32 中间值及按特征宽度选择的 tile、grid 和运行期参数配置；两个内部 kernel 共用一个公共入口。

## torch_native 后端

- 输入转 FP32 后执行 `values * torch.rsqrt(values.square().sum(-1, keepdim=True) + eps)`。
- 写入目标 dtype 的输出，再恢复原 shape。

## 调用示例

示例需要已安装包含本接口的 wheel 及 NPU 运行依赖。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import l2norm_fwd

torch.npu.set_device(0)
x = torch.randn(2, 3, 128, device="npu", dtype=torch.bfloat16)

y_default = l2norm_fwd(x)  # 保持 BF16 输出
y_triton = l2norm_fwd(x, output_dtype=torch.float32, impl="triton")
y_native = l2norm_fwd(x, output_dtype=torch.float32, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([2, 3, 128]), torch.float32
```
