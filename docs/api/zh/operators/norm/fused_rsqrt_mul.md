# fused_rsqrt_mul

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul

def fused_rsqrt_mul(x, variance, weight, eps=1e-6, *, impl="auto"):
    ...
```

## 功能说明

使用已同步的方差完成张量并行 RMSNorm 的缩放与加权，是 `fused_variance` 的第二阶段。设输入为 $X \in \mathbb{R}^{B\times L\times C}$，方差为 $V \in \mathbb{R}^{B\times L\times 1}$，权重为 $w \in \mathbb{R}^{C}$：

$$
Y_{blj} = X_{blj}\cdot\frac{1}{\sqrt{V_{bl}+\varepsilon}}\cdot w_{j}
$$

`variance` 必须是跨 TP rank 求平均（`all_reduce(AVG)`）之后的值；通常直接传入 `fused_variance` 的输出。输出与输入同 dtype，不修改输入。

## 参数与返回值

下表描述推荐调用约定和当前实现要求。公开 operator 直接透传参数，不执行输入检查。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`（该路径实际仅使用 fp32） | 必填 | 待归一化输入。Triton 要求 NPU 上的连续三维 tensor，`C` 无 2 的幂限制。 |
| `variance` | `torch.Tensor` | `[B, L, 1]` 或 `[B*L]` | 与 `x` 相同 | 必填 | 已跨 rank 平均的方差；与 `x` 位于同一设备。Triton 按行连续读取。 |
| `weight` | `torch.Tensor` | `[C]` | 与 `x` 相同 | 必填 | 每个通道的乘法权重；与 `x` 位于同一设备。Triton 要求连续。 |
| `eps` | `float` | — | — | `1e-6` | 加到方差上的稳定项；建议为有限正数。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 当前选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, C]` | 同 `x` | 新建的归一化输出，与 `x` 位于同一设备；不修改输入。 |

附加说明：输入按有限值使用，NaN 的跨后端行为未定义。公开接口不检查输入，也不自动切换实现；包初始化仍要求 `torch_npu` 和原生动态库。

## triton 后端

- 使用 CANN Triton kernel，按 `(num_tokens, C)` 二维分块，grid 为 vector core 数量。
- 按 `num_tokens` 与 `hidden_size` autotune `(block_l, block_c)`，行列尾部均通过 mask 处理，因此 `C` 不要求是 2 的幂。
- 每个 program 以 `kernel_num` 为步长循环处理任务，输出通过 `torch.empty_like(x)` 分配。
- 计算跟随输入 dtype；生产路径传入 float32，因此与参考实现一致。

## torch_native 后端

- 使用 PyTorch 广播运算 `x * torch.rsqrt(variance + eps) * weight`，保持输入 dtype，支持非连续输入。
- 一维 `[B*L]` 方差会先按 `[B, L, 1]` 解释，与 Triton 按行读取的布局一致。
- `rsqrt` 实现与归约顺序可能与 Triton 不同，float32 下差异通常在默认容差内。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul, fused_variance

x = torch.randn(1, 4096, 512, device="npu", dtype=torch.float32)
weight = torch.randn(512, device="npu", dtype=torch.float32)

variance = fused_variance(x)
# 真实 Tensor Parallel 场景在这里对 variance 做 all_reduce(AVG)。
print(list_impls(fused_rsqrt_mul))  # ['auto', 'torch_native', 'triton']
y_default = fused_rsqrt_mul(x, variance, weight, 1e-6)  # 默认 auto
y_triton = fused_rsqrt_mul(x, variance, weight, 1e-6, impl="triton")
y_native = fused_rsqrt_mul(x, variance, weight, 1e-6, impl="torch_native")
torch.testing.assert_close(y_default, y_triton, rtol=0, atol=0)
torch.testing.assert_close(y_triton, y_native, rtol=1e-5, atol=1e-5)
```
