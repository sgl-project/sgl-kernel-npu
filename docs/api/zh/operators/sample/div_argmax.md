# div_argmax

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

默认 `impl="auto"` 与 `triton` 注册到同一入口。注册成功不代表所有芯片均已完成验证。

## 函数原型

```python
from sgl_kernel_npu.operators.sample import div_argmax

def div_argmax(x, divisor, *, divisor_min=None, ignore_nan=False, impl="auto"):
    ...
```

## 功能说明

返回每行 `(x / divisor)` 最大值的索引，其中除法逐元素执行。计算规则：

- `divisor_min` 非 None 时，先对 `divisor` 执行 `clamp_min(divisor_min)` 再相除；`divisor` 中的 NaN 元素保持 NaN。
- 最大值相同时取最小列索引。
- `ignore_nan` 为 `True` 时，NaN 商视为 `-inf`、永不获胜；整行没有高于 `-inf` 的值时返回索引 0。
- `ignore_nan` 为 `False` 时，NaN 商的处理遵循 Ascend `tl.max` 的默认 NaN 行为，不保证与 `torch.argmax` 一致（全 NaN tile 的索引停留在行哨兵 `N` 上），不建议依赖该场景的结果。

## 参数与返回值

| 参数 | 类型 | Shape / dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, V]`，`torch.float32` | 必填 | NPU 输入，`V >= 1`，列步长为 1；行可有额外步长。 |
| `divisor` | `torch.Tensor` | `[B, V]`，`torch.float32` | 必填 | 与 `x` 同 shape、同设备的除数，列步长为 1。 |
| `divisor_min` | `float` | — | `None` | 除数下限，先 clamp 再除法。 |
| `ignore_nan` | `bool` | — | `False` | NaN 商是否强制视为 `-inf`。 |
| `impl` | `str` | — | `"auto"` | 可选 `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape / dtype | 描述 |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`，`torch.int64` | 在输入设备上新建的索引张量；`B` 可为 0。 |

## triton 后端

两段式归约：partial kernel 将每行按块拆分（行数不超过 256 时块长 4096，否则 8192），并行计算各块最大值与索引；final kernel 归约各块候选。Ascend Triton 不支持二元 `tl.reduce`（会 lowering 为 region `tt.reduce`），因此采用一元 `tl.max`/`tl.min` 实现，NaN 商永不获胜。要求输入为二维 FP32 张量且列步长为 1。

## torch_native 后端

先执行 `divisor.clamp_min(divisor_min)`（`None` 时跳过），再逐元素相除；`ignore_nan` 为 `True` 时将 NaN 商替换为 `-inf`，最后调用 `torch.argmax(dim=-1)`。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import div_argmax

x = torch.tensor([[1.0, 6.0, 3.0], [4.0, 2.0, 1.0]], device="npu")
divisor = torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]], device="npu")
indices = div_argmax(x, divisor, divisor_min=1e-38, ignore_nan=True)
# x / divisor 为 [1.0, 3.0, 3.0] 与 [2.0, 2.0, 1.0]，indices 为 [1, 0]
```
