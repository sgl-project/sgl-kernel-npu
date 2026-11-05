# situ_and_mul

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

arch22 在 A3 上验证，A2 未单独验证。arch35 在 A5 上验证。
当前提供 Triton 和 PyTorch 两种实现。

## 函数原型

```python
from sgl_kernel_npu.operators.activation import situ_and_mul

def situ_and_mul(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: Optional[int] = None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` 与 `triton` 注册到同一入口，默认使用 Triton，后端按需延迟导入。实现不可用时会报错，不自动切换后端。调用方须满足下述输入约束，算子不全面校验不支持的输入。

## 功能说明

对输入的 gate/up 两半执行 Kimi-K3 使用的 SiTU 激活并相乘。本 API 计算浮点激活，不执行量化。[Kimi-K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/config.json)的 gate/up 界限为 4/25。[SGLang KimiK3MLP](https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/kimi_k3.py#L290-L344) 在打包的 gate/up 投影与 down 投影之间执行这一激活。

对于形状为 `[..., 2*d]` 的输入，沿最后一维拆分 gate 和 up：

$$
g = x[..., :d], \qquad u = x[..., d:].
$$

$$
a = \beta\tanh(g / \beta)\,\sigma(g), \qquad
v = \begin{cases}
L\tanh(u/L), & L = \mathrm{linear\_beta} \ne \mathrm{None},\\
u, & \mathrm{linear\_beta} = \mathrm{None},
\end{cases}
\qquad y = a v.
$$

所有中间计算使用 float32，结果转换为 `x.dtype`。

前导维度展平为 `N` 个 token 行。未传入 `group_list` 时处理全部行。`group_list_type=1` 时有效行数为 `sum(group_list)`；`group_list_type=0` 时为 `group_list[-1]`。累积计数每个 expert 对应一项，不要求额外的前导零。只写入有效前缀，剩余输出行为值未定义的 padding，不得读取或用于比较。

逐 expert 计数须非负；累积计数须非负且单调不减；总数须满足 `0 <= total <= N`。这些设备端数值约束由调用方保证，Triton 路径不检查，以避免主机同步。参与计算的浮点输入按有限值使用。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., 2*d]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | strided Tensor，至少一维且 `d > 0`；前导维允许为空。前半为 gate，后半为 up。 |
| `group_list` | `torch.Tensor` 或 `None` | `[E]` | `torch.int32`、`torch.int64` | `None` | 可选的连续、非空逐 expert 向量，与 x 同设备。None 表示全部 token 行有效。 |
| `group_list_type` | `int` 或 `None` | — | — | `None` | `0` 为累积计数，`1` 为逐 expert 计数；传入 group_list 时必填，否则忽略。 |
| `beta` | `int` 或 `float` | — | — | `4.0` | 有限正 Python 标量，排除 bool；gate 饱和界限。 |
| `linear_beta` | `int`、`float` 或 `None` | — | — | `25.0` | 有限正 Python 标量，排除 bool；up 饱和界限。None 表示不对 up 做饱和变换。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

公开 operator 在分发前执行一次旧版三项校验：输入最后一维为偶数、传入 group 时 group_list_type 属于 {0, 1}、group dtype 为 int32/int64，另拒绝空 group_list。两个实现共享这些校验，空输入也执行。输入 dtype/device/layout、group 为连续一维向量、标量界限为有限正数等约束仍由调用方保证，不显式检查。后端不重复执行这些公共校验；Triton 另检查下文说明的行容量上限。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., d]` | x.dtype | 与 x 同设备的新建连续 Tensor，不修改输入。分组调用仅定义展平后的有效前缀。 |

Triton 要求展平后的输入容量满足 `N <= 2**31 - C`，其中 `C` 为当前 NPU 的向量核数量。超过上限时，wrapper 在分配输出或启动 kernel 前报错，为原有 int32 行调度的向上取整分块预留空间。分组总数仍须满足 `0 <= total <= N`。PyTorch 参考实现不使用这一调度方式。

空 token 输入返回形状正确的空输出，不启动 kernel；分组总数为零时不写入任何输出值。

## triton 后端

- 实现位于 `implementation/triton/activation/situ_and_mul.py`，kernel 名为 `_situ_and_mul_kernel`；非空输入须位于 NPU。
- 保留现有向量核行分配、H 维分块、带 mask 的特征尾部、FP32 运算以及 `BLOCK_H` 为 `{1024, 2048, 4096, 8192}` 的 multibuffer autotune。行偏移使用 int64，支持较大的展平输入偏移。
- 将非连续输入转换为紧密排列的行；连续输入无需复制。group 元数据须已连续，非连续输入的复制成本计入性能测量。
- 累积计数读取最后一项 `E - 1`；计数归约范围向上取整到二次幂，并对 expert 尾部使用 mask，覆盖非二次幂 expert 数量。分组总数保留在设备端。

## torch_native 后端

- 独立使用 PyTorch 运算，不调用 Triton；支持 strided 输入，也可在 CPU 上单独验证。
- 使用 float32 计算 tanh、sigmoid 与乘法，再转回 x.dtype。分组调用将总数同步到主机以截取有效前缀，这一参考路径不适合图捕获。
- 后端数学近似可能不同，应使用适合 dtype 的容差，不保证逐位相等。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import situ_and_mul

x = torch.randn(17, 6144, device="npu", dtype=torch.bfloat16)
output = situ_and_mul(x)
reference = situ_and_mul(x, impl="torch_native")
torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

# Routed MoE: only the first 12 flattened token rows are valid.
counts = torch.tensor([0, 3, 5, 4], device="npu", dtype=torch.int64)
routed = situ_and_mul(x, counts, 1, impl="triton")
routed_ref = situ_and_mul(x, counts, 1, impl="torch_native")
torch.testing.assert_close(routed[:12], routed_ref[:12], rtol=2e-2, atol=2e-2)
```
