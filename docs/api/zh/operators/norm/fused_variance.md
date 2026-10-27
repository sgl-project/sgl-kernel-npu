# fused_variance

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 已注册，待验证 | 已注册，待验证 |
| `torch_native` | 已注册，待验证 | 已注册，待验证 |

注册范围覆盖上述代际；真实 NPU 的编译、精度及性能验证尚未完成。

## 函数原型

```python
from sgl_kernel_npu.operators.norm import fused_variance

def fused_variance(x, *, impl="auto"):
    ...
```

`auto` 和 `triton` 当前指向同一入口，不会在执行失败后自动切换实现。

## 功能说明

对三维输入的最后一维计算均方，保留该维度：

$$
Y_{b,l,0}=\frac{1}{C}\sum_{j=0}^{C-1}X_{b,l,j}^{2}
$$

虽然接口名包含 variance，计算时并不减去输入均值。该接口只计算本地均方，不包含 RMSNorm 的缩放、epsilon 或跨卡通信。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | `C > 0`；Triton 要求连续的 NPU tensor，且 `C` 为 2 的幂。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, 1]` | `x.dtype` | 新建输出，与输入位于同一设备，不修改输入。 |

公共入口保留原调用逻辑，不增加输入校验或连续化；调用方应满足相应后端的输入约束。

## triton 后端

- 使用原 `fused_variance_kernel` 和 autotune 配置；按连续布局读取输入。
- hidden 维度通过 `tl.arange(0, C)` 展开，因此需要 2 的幂宽度。

## torch_native 后端

- 当前实现为 `x.square().float().mean(dim=-1, keepdim=True).to(x.dtype)`。
- 先按输入 dtype 计算平方，再转 FP32 求均值，最后转换回输入 dtype。
- 保留均方的 shape 和 dtype 约定；不同后端归约结果按精度标准比较。

## 调用示例

示例需要已安装包含本接口的 wheel 及 NPU 运行依赖。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_variance

torch.npu.set_device(0)
x = torch.randn(2, 17, 128, device="npu", dtype=torch.float32)

y_default = fused_variance(x)
y_triton = fused_variance(x, impl="triton")
y_native = fused_variance(x, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([2, 17, 1]), torch.float32
```
