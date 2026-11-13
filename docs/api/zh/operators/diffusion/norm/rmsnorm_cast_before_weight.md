# rmsnorm_cast_before_weight

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

def rmsnorm_cast_before_weight(x, weight, eps, *, impl="auto"):
    ...
```

## 功能说明

对最后一维做 RMSNorm，并复现 `RMSNorm(cast_x_before_out_mul=True)` 的舍入顺序：

$$
v_i = \frac{1}{D}\sum_{j=0}^{D-1} x_{ij}^2
$$

$$
n_{ij} = \operatorname{round}_{dtype}\left(x_{ij} \cdot \frac{1}{\sqrt{v_i+\varepsilon}}\right),\qquad
y_{ij} = \operatorname{round}_{dtype}\left(n_{ij} \cdot w_j\right)
$$

即归一化结果先转回输入 dtype，再与权重相乘并舍入一次。这与先乘权重再转换的写法不同，是本算子与 `rmsnorm_bias` 等接口的关键差异。该算子起源于 Qwen-Image 2.1 的 Q/K 归一化。

Triton 后端用 `tl.sum` 做行归约；在已验证的 NPU 架构上它与 aten `mean` 的归约树一致，因此可以逐位复现。其余架构需要重新验证（见芯片支持表）。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., D]` | `torch.bfloat16`、`torch.float16` | 必填 | 输入；NPU 上连续，最后一维 `D` 必须为 2 的幂。 |
| `weight` | `torch.Tensor` | `[D]` | 与 `x` 相同 | 必填 | 逐通道权重；与 `x` 同设备、连续。 |
| `eps` | `float` | — | — | 必填 | 加到均方值上的稳定项。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., D]` | 与 `x` 相同 | 新分配的 tensor，与 `x` 同设备；不修改输入。 |

## triton 后端

- 每个 program 处理 `[ROWS, D]` 块，`ROWS` 由 UB 预算和行数整除关系截断，归约在块内完成。
- 要求 `D` 为 2 的幂、每块至少 16 行且行数能被块行数整除；不满足时在启动前抛出 `ValueError`。
- 归约使用 `tl.sum`，归一化值经显式 cast 后再乘权重，最终由 store 舍入。

## torch_native 后端

- 按 `x.float() → square().mean(-1) → rsqrt → cast → 乘权重` 逐步计算，是独立参考实现，也是精度对比的基准。

## 调用示例

```python
import torch

from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

x = torch.randn((1, 4096, 32, 128), dtype=torch.bfloat16, device="npu")
weight = torch.randn((128,), dtype=torch.bfloat16, device="npu")
out = rmsnorm_cast_before_weight(x, weight, 1e-6)

values = x.float()
variance = values.square().mean(dim=-1, keepdim=True)
ref = weight * (values * torch.rsqrt(variance + 1e-6)).to(x.dtype)
assert torch.equal(out, ref)
```
