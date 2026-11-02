# fused_variance

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

默认选项为 `auto`，当前与 `triton` 注册到同一个入口函数。不传 `impl` 或显式传入 `"auto"` 时均使用 Triton。

```python
from sgl_kernel_npu.operators.norm import fused_variance

def fused_variance(x, *, impl="auto"):
    ...
```

## 功能说明

计算每个 token 在最后一维上的均方值，是张量并行 RMSNorm 的第一阶段。设输入为 $X \in \mathbb{R}^{B\times L\times C}$：

$$
v_{bl} = \frac{1}{C}\sum_{j=0}^{C-1}X_{blj}^{2}
$$

输出形状为 `[B, L, 1]`，与输入同 dtype。该结果是**局部**方差：Tensor Parallel 场景下每个 rank 只持有 `C` 个通道，调用方必须先对输出做 `all_reduce(AVG)`，再交给 `fused_rsqrt_mul` 完成归一化。单独调用本算子只得到局部方差，不构成完整的 RMSNorm。

## 参数与返回值

下表描述推荐调用约定和当前实现要求。公开 operator 直接透传参数，不执行输入检查。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, C]` | `torch.float32`（该路径实际仅使用 fp32） | 必填 | 待求均方的输入；按 `B > 0, L > 0, C > 0` 调用。Triton 要求 NPU 上的连续三维 tensor，且 `C` 为 2 的幂。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。`auto` 当前选择 Triton 实现。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `variance` | `torch.Tensor` | `[B, L, 1]` | 同 `x` | 新建的局部均方值，与 `x` 位于同一设备；不修改输入。 |

附加说明：输入按有限值使用，NaN 的跨后端行为未定义。公开接口不检查输入，也不自动切换实现；包初始化仍要求 `torch_npu` 和原生动态库。

## triton 后端

- 使用 CANN Triton kernel，将 `B*L` 个 token 展平后按行归约，grid 为 vector core 数量。
- 按 `num_tokens` autotune `block_l ∈ {96, 64, 32, 16, 8, 4, 2, 1}`。
- 归约维使用 `tl.arange(0, C)`，因此 `C` 必须是 2 的幂；要求连续三维输入。
- 累加跟随输入 dtype；生产路径传入 float32，因此与参考实现一致。

## torch_native 后端

- 使用 PyTorch 运算 `x.square().sum(dim=-1, keepdim=True) / C`，保持输入 dtype，支持任意前导维度和非连续输入。
- 归约顺序可能与 Triton 不同，float32 下差异通常在默认容差内。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import fused_rsqrt_mul, fused_variance

x = torch.randn(1, 4096, 512, device="npu", dtype=torch.float32)
weight = torch.randn(512, device="npu", dtype=torch.float32)

print(list_impls(fused_variance))  # ['auto', 'torch_native', 'triton']
v_default = fused_variance(x)  # 默认 auto
v_triton = fused_variance(x, impl="triton")
v_native = fused_variance(x, impl="torch_native")
torch.testing.assert_close(v_default, v_triton, rtol=0, atol=0)
torch.testing.assert_close(v_triton, v_native, rtol=1e-5, atol=1e-5)

# 真实 Tensor Parallel 场景在这里对 variance 做 all_reduce(AVG)。
y = fused_rsqrt_mul(x, v_triton, weight, 1e-6)
```
