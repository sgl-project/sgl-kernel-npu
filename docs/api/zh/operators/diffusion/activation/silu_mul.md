# silu_mul

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.diffusion import silu_mul

def silu_mul(a, b, *, impl="auto"):
    ...
```

## 功能说明

逐元素计算 `silu(a) * b`，复现 eager 链路的两次 BF16 舍入：

$$
s = \operatorname{round}_{bf16}\left(a \cdot \sigma(a)\right),\qquad
y = \operatorname{round}_{bf16}\left(s \cdot b\right)
$$

其中 $\sigma$ 为 sigmoid。输出与 `torch.nn.functional.silu(a) * b` 逐位相同。Triton 后端把两次舍入放进同一个 kernel，中间结果不写回全局内存。该算子起源于 Qwen-Image 2.1 的 SwiGLU FFN，但对任何需要与 eager 逐位一致的 BF16 silu 乘门控都适用。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `a` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | 必填 | 门控输入；NPU 上的连续 tensor。 |
| `b` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | 必填 | 与 `a` 同 shape、同设备、连续。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[*S]` | `torch.bfloat16` | 新分配的 tensor，与 `a` 同设备；不修改输入。 |

## triton 后端

- 每个 program 连续处理多个 4096 元素的块，尾块由单独的带 mask kernel 处理。
- 要求同 shape、同设备、连续的 BF16 输入，且 `0 < numel < 2**31`；不满足时在启动前抛出 `ValueError`。
- 两次 BF16 舍入分别由显式 `vcast` 和最终 store 完成。

## torch_native 后端

- 直接计算 `torch.nn.functional.silu(a) * b`，是独立参考实现，也是精度对比的基准。

## 调用示例

```python
import torch

from sgl_kernel_npu.operators.diffusion import silu_mul

a = torch.randn((1, 4096, 12288), dtype=torch.bfloat16, device="npu")
b = torch.randn_like(a)
out = silu_mul(a, b)
ref = torch.nn.functional.silu(a) * b
assert torch.equal(out, ref)
```
