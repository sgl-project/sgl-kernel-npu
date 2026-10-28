# elu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | 支持（已在 Ascend910 + CANN 9.1.0 实测） | 待验证（`3510` 仅是构建声明，无实测数据） |
| `torch_native` | 支持 | 待验证 |

未声明 `triton` 实现：传入 `impl="triton"` 会抛 `ImplementationNotFoundError`。

## 函数原型

默认选项为 `auto`，当前与 `ascendc` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均调用 Ascend C 算子；`auto` 不会在原生算子缺失时回退到其他实现。

```python
from sgl_kernel_npu.operators.activation import elu

def elu(x, alpha=1.0, *, impl="auto"):
    ...
```

## 功能说明

对输入逐元素计算 ELU（Exponential Linear Unit），$\alpha$ 为系数：

$$
\mathrm{elu}(x)=
\begin{cases}
x, & x > 0 \\
\alpha\left(e^{x}-1\right), & x \le 0
\end{cases}
$$

Ascend C 实现不按元素分支，而是利用 $\max(x,0)=x-\min(x,0)$ 把分段函数改写为单一表达式：

$$
\mathrm{elu}(x)=\alpha\left(e^{\min(x,0)}-1\right)+\left(x-\min(x,0)\right)
$$

由于 `Exp` 只作用于 $\min(x,0)\le 0$，其结果始终位于 $(0,1]$，因此 fp16 不会溢出。`torch_native` 参考实现先在 float32 上计算 `torch.expm1`（避免 $e^{x}-1$ 在零点附近的精度损失），再转换回输入 dtype。

精度说明：fp16 下 kernel 直接以 fp16 计算 $e^{\min(x,0)}-1$，在 $x\to0^-$ 附近存在**灾难性抵消**——例如 $x=-0.001$ 时*相对*误差约 2.3%。但该处*绝对*误差仅约 2.3e-5，落在常用容差的 `atol` 范围内，`1e-3` 标准仍然满足。若这一狭窄区间的**相对**精度重要，请使用 fp32。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | 任意，元素数 $\ge 0$ | `torch.float16`、`torch.float32` | 必填 | 位于 NPU 上的连续 tensor；kernel 按展平后的一维元素序列处理，不要求特定维度或对齐。 |
| `alpha` | `float` | — | — | `1.0` | 负半轴系数，要求为有限值。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 当前选择 Ascend C 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 `x` 相同 | 与 `x` 相同 | 新建输出，与 `x` 同设备，不修改输入。 |

附加说明：Ascend C 实现要求 `x` 连续且至少一维，非 32 字节对齐或长度不能整除 `blockDim × tileLength` 的输入会在 host 侧零填充，计算后再裁剪回原长度与形状；LLM 中常见的对齐激活张量直接走零开销快路径。不支持的数据类型在 kernel 启动前即报错。

## ascendc 后端

- 使用 Ascend C `op_host` / `op_kernel` 实现，通过 `EXEC_KERNEL_CMD` 直接启动 kernel，无 workspace、无 GE tiling。
- host 侧根据 UB 容量与 dtype 计算 `tileLength`（512 元素对齐，上限 4096），并按 tile 数量与 AIV 核数确定 `blockDim`。
- 支持任意维度的连续输入：展平 → 按需零填充 → 计算 → 裁剪并恢复原形状。

## torch_native 后端

- 使用 PyTorch 运算在 float32 上计算，再转换回输入 dtype。
- 支持任意形状与非连续输入，作为精度对照的独立参考实现。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.activation import elu

x = torch.randn(16, 7168, device="npu", dtype=torch.float16)

print(list_impls(elu))  # ['ascendc', 'auto', 'torch_native']
y_default = elu(x)  # 默认 auto，使用 Ascend C
y_ascendc = elu(x, alpha=1.0, impl="ascendc")
y_native = elu(x, alpha=1.0, impl="torch_native")

torch.testing.assert_close(y_default, y_ascendc, rtol=0, atol=0)
torch.testing.assert_close(y_ascendc, y_native, rtol=1e-3, atol=1e-3)
```
