# fused_sigmoid_mul

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

省略 `impl` 或传入 `"auto"` 时选择 `"triton"`，不会自动回退到其他后端。

```python
from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

def fused_sigmoid_mul(x, gate, *, impl="auto"):
    ...
```

## 功能说明

逐元素计算 `x * sigmoid(gate)`。例如，`gate` 为 0 时，输出就是 `x` 的一半。
Sigmoid 和乘法都用 FP32 计算，最后将结果转回 `x.dtype`。

在 SGLang 中，Qwen3.5 稠密版和 MoE 版的全注意力层在输出投影前使用这一算子。
启用输出门控且走常规 NPU 路径时，注意力输出和 `gate` 以相同形状传入算子。
[查看 SGLang 调用位置](https://github.com/sgl-project/sglang/blob/53a24c448ecd6c26bf6461b80ffb8d55397d57cf/python/sglang/srt/models/qwen3_5.py#L1576-L1587)

Intern-S2-Preview、Intern-S2-Mobius 和 MiniCPM-V 4.6 也复用了这条注意力路径。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | 任意形状，包括标量和空张量 | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | NPU 上的输入，要求 `torch.strided` 布局；支持连续和非连续视图。 |
| `gate` | `torch.Tensor` | 与 `x` 相同 | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | Sigmoid 的输入，与 `x` 位于同一 NPU，要求 `torch.strided` 布局。dtype 可与 `x` 不同，支持非连续视图。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 选择 Triton。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 `x` 相同 | `x.dtype` | 在 `x.device` 上新分配，不修改输入。空输入返回空张量。Triton 返回连续张量；torch_native 的输出布局由 PyTorch 决定。 |

输入形状不一致时抛出 `ValueError`。数据类型、布局和设备要求由调用方保证，
不在每次调用时显式校验。

## triton 后端

- 将两输入转为连续张量后，逐元素计算 sigmoid 门控。
- 两输入分别转为 FP32 计算，结果转回 `x.dtype`；支持混合 dtype 输入。
- 提供推理所需的前向计算；当前 NPU 应与输入所在的 NPU 一致。

## torch_native 后端

- 使用独立的 PyTorch 运算，两输入分别转为 FP32 计算，结果转回 `x.dtype`。
- 支持非连续输入，输出布局由 PyTorch 运算决定。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

x = torch.tensor([2.0, -4.0], device="npu", dtype=torch.bfloat16)
gate = torch.zeros_like(x)
y = fused_sigmoid_mul(x, gate)  # auto 选择 Triton；[1.0, -2.0]
reference = fused_sigmoid_mul(x, gate, impl="torch_native")
torch.testing.assert_close(y, reference)
```
