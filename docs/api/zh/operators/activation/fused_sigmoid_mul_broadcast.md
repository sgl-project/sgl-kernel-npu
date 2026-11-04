# fused_sigmoid_mul_broadcast

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 已验证 | 已验证 |
| `torch_native` | 已验证 | 已验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

def fused_sigmoid_mul_broadcast(x, gate, *, impl="auto"):
    ...
```

## 功能说明

算子从 `gate` 中取出 `x` 每行对应的值，计算 sigmoid，再用结果乘以该行的每个元素。
例如，门控值为 0 时，这一行的值会减半。两个后端都用 FP32 计算，最后转回 `x.dtype`。

在 SGLang 中，`sglang/srt/models/qwen2_moe.py` 的
`Qwen2MoeSparseMoeBlock._forward_shared_experts` 会在 NPU 上为单独运行的共享专家做门控。
Qwen2-MoE、Qwen3-Next、Qwen3.5-MoE、Qwen4Exp 和 InternS2Preview 都可以通过这个模块走到该分支。
模型传入的共享专家输出形状为 `[tokens, hidden_size]`，门控值形状为 `[tokens, 1]`。
SGLang 目前仍从旧路径导入这个算子。

## 参数与返回值

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `x` | `[N, D]` | NPU 张量，类型为 `torch.float32`、`torch.float16` 或 `torch.bfloat16`。请使用 `N > 0` 且 `D > 0` 的输入。 |
| `gate` | `[N]` 或 `[N, 1]` | 每行一个值，与 `x` 位于同一块 NPU。 |
| `impl` | — | 仅限关键字。默认值 `"auto"` 选择 `"triton"`。可用 `"torch_native"` 选择 PyTorch 实现。 |

返回一个新的 `[N, D]` 张量，数据类型和设备与 `x` 一致。两个后端都支持非连续输入。
Triton 返回连续张量，PyTorch 实现的输出布局由 PyTorch 决定。

公共 `_check_inputs` 仅保留旧版输入维数和 `gate` 形状校验，不兼容的形状会抛出
`ValueError`。两个后端共享这些校验，内部不重复执行。上述 dtype 和设备要求由调用方保证，
不在每次调用时显式检查。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

x = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device="npu")
gate = torch.zeros(2, device=x.device)
output = fused_sigmoid_mul_broadcast(x, gate)
torch.testing.assert_close(output, x / 2)
```
