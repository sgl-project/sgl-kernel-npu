# elu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持 | 不支持 |
| `torch_native` | 支持 | 支持 |

Ascend C Kernel 仅声明 arch22，因此 arch35 上 `ascendc` 不可用。`auto` 是 `ascendc`
的别名，不算独立实现；Ascend C 实现不可用时不会自动回退。
请安装与设备架构匹配、且通过 `--ops elu` 选择了本算子的 wheel，构建方式见
[构建安装说明](../../../../../python/sgl_kernel_npu/README.md)。

## 函数原型

```python
from sgl_kernel_npu.operators.examples import elu

def elu(x, alpha=1.0, *, impl="auto"):
    ...
```

不传 `impl` 或传入 `"auto"` 均选择 `ascendc`。arch35 上未声明 Ascend C 实现，
需要显式传入 `impl="torch_native"`。

## 功能说明

逐元素计算 Exponential Linear Unit，其中 $\alpha$ 为负半区系数：

$$
\mathrm{elu}(x)=
\begin{cases}
x, & x > 0 \\
\alpha\left(e^{x}-1\right), & x \le 0
\end{cases}
$$

Ascend C 实现不做逐元素分支，而是利用 $\max(x,0)=x-\min(x,0)$ 将分段函数改写为

$$
\mathrm{elu}(x)=\alpha\left(e^{\min(x,0)}-1\right)+\left(x-\min(x,0)\right)
$$

因此每个 tile 上只执行一段无分支的向量指令。`Exp` 的输入始终满足
$\min(x,0)\le 0$，结果落在 $(0,1]$，fp16 不会溢出。`torch_native` 参考实现
按分段定义在 float32 下用 `torch.expm1` 直接计算（避免 $e^{x}-1$ 在零附近的
精度损失），再转回输入 dtype。

数值说明：fp16 Kernel 直接在 fp16 下计算 $e^{\min(x,0)}-1$，因此当
$x\to0^-$ 时*相对*误差会增大（$|x|\le0.01$ 时约 2-3%）；该区间内的*绝对*
误差仅约 2.3e-5，可被常用容差的 `atol` 项吸收，标准 `1e-3` 仍然成立。若需要
该窄区间内的相对精度，请使用 fp32。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | 任意 shape，至少一维且至少一个元素 | `torch.float16`、`torch.float32` | 必填 | 位于 NPU、连续的 ND 格式张量。算子按展平后的一维元素序列处理，不要求 rank 或对齐。 |
| `alpha` | `float` | — | — | `1.0` | 负半区系数，必须为有限值。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"ascendc"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | 与 `x` 相同 | 与 `x` 相同 | 同一 NPU 上新分配的连续 ND 张量，输入保持不变。 |

公开接口在分派前检查 dtype、设备、连续性、存储格式、非空输入和 `alpha`。
不支持的输入会报错，不会被静默转换。

## ascendc 后端

- 在 `csrc/examples/elu/` 中声明为 AICORE 算子，`SUPPORTED_ARCHS 2201`，通过
  `torch.ops.npu.elu` 调用。
- Host tiling 依据 UB 容量和 dtype 计算 `tileLength`（512 元素向量对齐，上限 4096），
  每个 tile 至少使用一个 AIV core，并将元素数平均分给参与核。最后一个短 tile 由
  Kernel 用 `DataCopyPad` 处理，因此任意元素数都无需 Host 侧补齐。
- Kernel 采用双缓冲 CopyIn / Compute / CopyOut 流水，配合两个单缓冲 scratch
  张量，每个 tile 执行一段无分支的向量指令序列完成 ELU。

## torch_native 后端

仅使用 PyTorch 算子：在 float32 下按分段定义用布尔掩码计算，再转回输入 dtype。
支持任意 shape（包括非连续输入），作为独立的参考实现。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.examples import elu

x = torch.randn(16, 7168, device="npu", dtype=torch.float16)

print(list_impls(elu))  # ['ascendc', 'auto', 'torch_native']
y_default = elu(x)  # auto -> ascendc
y_ascendc = elu(x, alpha=1.0, impl="ascendc")
y_native = elu(x, alpha=1.0, impl="torch_native")

torch.testing.assert_close(y_default, y_ascendc, rtol=0, atol=0)
torch.testing.assert_close(y_ascendc, y_native, rtol=1e-3, atol=1e-3)
```
