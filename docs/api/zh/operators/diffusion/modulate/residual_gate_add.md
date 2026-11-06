# residual_gate_add

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.diffusion import residual_gate_add

def residual_gate_add(residual, update, gate, *, impl="auto"):
    ...
```

## 功能说明

计算 `residual + update * gate`，复现 eager 链路的两次 BF16 舍入：

$$
p = \operatorname{round}_{bf16}\left(update \odot gate\right),\qquad
y = \operatorname{round}_{bf16}\left(residual + p\right)
$$

输出与先乘后加的两次 eager BF16 运算逐位相同。`gate` 可以是与 `residual` 同 shape 的逐元素门控，也可以是单独一行（`[1, C]` 或 `[C]`）的逐通道门控；后者对应扩散模型中每步的 modulation。该算子起源于 Qwen-Image 2.1 的 DiT block 残差更新。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `residual` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | 必填 | 累加项；NPU 上的连续 tensor，至少二维。 |
| `update` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | 必填 | 与 `residual` 同 shape、同设备、连续。 |
| `gate` | `torch.Tensor` | `[..., C]` 或 `[C]` | `torch.bfloat16` | 必填 | 与 `residual` 同 shape，或一行广播门控。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., C]` | `torch.bfloat16` | 新分配的 tensor，与 `residual` 同设备；不修改输入。 |

## triton 后端

- 以 `[ROWS, C]` 为一块，`ROWS` 由 UB 预算和行数整除关系共同截断；每个 program 连续处理多块。
- 要求最后一维为 2 的幂、每块至少 16 行且行数能被块行数整除；不满足时在启动前抛出 `ValueError`。
- 乘积累积在 FP32，先显式舍入到 BF16，再做 FP32 加法并由最终 store 舍入。

## torch_native 后端

- 直接计算 `residual + update * gate`，是独立参考实现，也是精度对比的基准。

## 调用示例

```python
import torch

from sgl_kernel_npu.operators.diffusion import residual_gate_add

residual = torch.randn((1, 4096, 4096), dtype=torch.bfloat16, device="npu")
update = torch.randn_like(residual)
gate = torch.randn((4096,), dtype=torch.bfloat16, device="npu")
out = residual_gate_add(residual, update, gate)
ref = residual + update * gate
assert torch.equal(out, ref)
```
