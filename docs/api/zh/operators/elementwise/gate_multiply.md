# gate_multiply

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

融合 kernel 及其阈值在 arch22（Ascend910_9362 / 910C）上测得。Triton 后端本身可移植，但在其他 NPU 上的收益未验证；在阈值以下以及 arch35 上，它与 `torch_native` 执行相同的 PyTorch 表达式。

## 函数原型

```python
from sgl_kernel_npu.operators.elementwise import gate_multiply

def gate_multiply(y, gate, *, impl="auto"):
    ...
```

`auto` 与 `triton` 注册到同一入口，默认使用 Triton，后端按需延迟导入。输入不支持或实现不可用时会报错，不自动切换后端。

## 功能说明

GDN 的 sigmoid-after-normalization 路径会把 BF16 归一化激活提升为 FP32，乘以已存的 FP32 sigmoid 门控，再把 FP32 乘积转回 BF16。本算子仅用于推理，完成该乘法与类型转换：

$$
\mathrm{out} = \mathrm{cast}_{\mathrm{bf16}}\big(\mathrm{cast}_{\mathrm{fp32}}(y)\odot \mathrm{gate}\big).
$$

调用前由调用方用已有 sigmoid 计算 `gate`。`y` 上已完成的 BF16 归一化舍入予以保留；本算子不融合归一化、不近似 sigmoid，也不替代 swish-gated layernorm kernel，不改变模型 dispatch。不提供反向。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | 任意 | `torch.bfloat16` | 必填 | 位于 NPU 的 BF16 归一化激活。 |
| `gate` | `torch.Tensor` | 与 `y` 相同 | `torch.float32` | 必填 | FP32 sigmoid 门控；须与 `y` 同 shape、同 NPU 设备。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

两个输入须位于同一 NPU、不得要求梯度、shape 必须一致。违反约束时在任何 kernel 启动前抛出 `ValueError`。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | 与 `y` 相同 | `torch.bfloat16` | 位于 `y` 设备上的新建结果。不修改输入。 |

## triton 后端

- 保留原 kernel 的计算方式：以 BF16 加载 `y`，提升为 FP32，乘以 FP32 的 `gate`，再以 BF16 存储。
- 连续且元素数不小于 16,777,216 的输入使用融合 NPU kernel；更小或非连续输入保留 PyTorch 表达式。该阈值在 910C 上测得，并非对所有 NPU 都有收益。
- 向量核数量来自共享的设备属性辅助函数。运行时元素数不参与 kernel specialization，因此在图捕获前需对 shape 与 kernel 预热。
- 融合结果与 PyTorch 表达式逐位相等：有限值完全一致，NaN 位置重合。

## torch_native 后端

- 独立使用 PyTorch 运算，不调用 Triton：`(y.float() * gate).to(y.dtype)`。
- 作为精度参考。由于 Triton 后端精确复现该表达式，两个后端以零容差比较（有限值逐位相等，NaN 位置单独比较）。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.elementwise import gate_multiply

# 保留已有的归一化与 sigmoid 实现。
y = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)  # 归一化后的 BF16
z = torch.randn(4096, 4096, device="npu")
gate = torch.sigmoid(z.float())  # FP32 门控

out = gate_multiply(y, gate)
ref = gate_multiply(y, gate, impl="torch_native")
assert torch.equal(out, ref)
```
