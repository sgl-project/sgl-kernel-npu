# elementwise_add

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |

## 函数原型

默认选项为 `auto`，当前与 `ascendc` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Ascend C 算子。

```python
from sgl_kernel_npu.operators.elementwise import elementwise_add

def elementwise_add(x, y, *, impl="auto"):
    ...
```

## 功能说明

对两个张量逐元素相加，返回新分配的张量：

$$
Y_{i_0,\dots,i_{n-1}} = X_{i_0,\dots,i_{n-1}} + Z_{i_0,\dots,i_{n-1}}
$$

Ascend C 实现会先把输入展平成一维视图，因此输入维度不影响 Kernel 执行路径。Host 侧会把元素个数向上对齐到 `blockDim * tileLength`（零填充），Kernel 只处理完整 tile，最后裁掉填充部分并恢复输入形状。这样 Kernel 无需处理尾块，`DataCopy` 也始终满足 32 B 对齐。

fp16 输入按 fp16 累加（向量 `Add` 使用输入 dtype）。可能溢出 fp16 的输入请改用 `impl="torch_native"`（或 fp32 张量），后者遵循 torch 的常规类型提升规则。

该算子同时是教学样例，完整开发与验证流程见
[`docs/developer_guide/ascendc_operator_sample.md`](../../../../developer_guide/ascendc_operator_sample.md)。

## 参数与返回值

公共接口不做广播和隐式类型转换，由 Ascend C 入口完成校验。

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | 任意维度 | `torch.float16`、`torch.float32` | 必选 | 第一个加数，需为连续张量且位于 NPU 上。 |
| `y` | `torch.Tensor` | 与 `x` 相同 | 与 `x` 相同 | 必选 | 第二个加数，需连续，且与 `x` 形状、数据类型、设备一致。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字参数；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 当前选择 Ascend C 实现。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 `x` 相同 | 与 `x` 相同 | 新分配张量，不修改输入。 |

## 后端说明

### ascendc

位于 `csrc/elementwise_add` 的向量单元（AI Core）实现，支持任意维度、任意长度的连续 fp16/fp32 张量，包括非对齐长度。空输入直接返回空张量且不启动 Kernel。非连续、形状不一致、数据类型不一致以及 bfloat16 输入会在下发设备任务前抛出 `RuntimeError`。

### torch_native

使用常规 torch 调度的 `x + y`，作为精度对照实现；其支持的 dtype 与广播规则取决于安装的 torch 版本，因此不受 Ascend C 输入约束限制。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401  (注册 NPU 设备)
from sgl_kernel_npu.operators.elementwise import elementwise_add

x = torch.randn((1, 4096), dtype=torch.float16, device="npu")
y = torch.randn((1, 4096), dtype=torch.float16, device="npu")

out = elementwise_add(x, y)                       # auto -> ascendc
ref = elementwise_add(x, y, impl="torch_native")  # 对照实现
torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
```
