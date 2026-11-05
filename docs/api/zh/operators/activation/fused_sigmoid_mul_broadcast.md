# fused_sigmoid_mul_broadcast

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

省略 `impl` 或传入 `"auto"` 时，选择 `"triton"`。

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    ...
```

## 功能说明

对输入的每一行应用一个门控值。输入 $X$ 的形状为 $[N, D]$，门控 $g$ 的形状为
$[N]$ 或 $[N, 1]$，其中 $g_n$ 表示第 $n$ 行的门控值：

$$
Y_{n,j} = X_{n,j} \cdot \sigma(g_n), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

门控值为 0 时，对应行的值减半。两个后端都将输入转为 FP32 计算，最后转回 `x.dtype`。

在 SGLang 中，`sglang/srt/models/qwen2_moe.py` 的
`Qwen2MoeSparseMoeBlock._forward_shared_experts` 会在 NPU 上为单独运行的共享专家做门控。
Qwen2-MoE、Qwen3-Next、Qwen3.5-MoE、Qwen4Exp 和 InternS2Preview 都可以通过这个模块走到该分支。
模型传入的共享专家输出形状为 `[tokens, hidden_size]`，门控值形状为 `[tokens, 1]`。
SGLang 目前仍从旧路径导入这个算子。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[N, D]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | NPU 输入；要求 `N > 0` 且 `D > 0`。支持连续和非连续的 strided 张量。 |
| `gate` | `torch.Tensor` | `[N]` 或 `[N, 1]` | 与 `x` 相同 | 必填 | 每行一个门控 logit，与 `x` 位于同一块 NPU。支持连续和非连续的 strided 张量。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 选择 Triton。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, D]` | `x.dtype` | 在 `x.device` 上新分配，不修改输入。Triton 返回连续张量；torch_native 的输出布局由 PyTorch 决定。 |

输入维数或门控形状不兼容时抛出 `ValueError`。数据类型、设备、布局和正维度要求由调用方保证，
不在每次调用时显式校验。其他数据类型不属于文档声明的输入范围。

## triton 后端

- 使用 FP32 中间值计算逐行 sigmoid 门控。
- 计算前将非连续输入转为连续张量，输出为连续张量。

## torch_native 后端

- 使用独立的 PyTorch 运算，以 FP32 中间值计算。
- 支持非连续输入，输出布局由 PyTorch 运算决定。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

x = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device="npu")
gate = torch.zeros(2, device=x.device, dtype=x.dtype)
output = fused_sigmoid_mul_broadcast(x, gate)  # auto 选择 Triton
reference = fused_sigmoid_mul_broadcast(x, gate, impl="torch_native")
torch.testing.assert_close(output, x / 2)
torch.testing.assert_close(output, reference)
```
