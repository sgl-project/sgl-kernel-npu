# attn_res_mix_fused

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

def attn_res_mix_fused(
    prefix_sum,
    bank,
    num_valid_blocks,
    combined_weight,
    variance_epsilon,
    *,
    impl="auto",
):
    ...
```

`auto` 与 `triton` 注册到同一入口，默认使用 Triton，后端按需延迟导入。输入不支持或实现不可用时会报错，不自动切换后端。

## 功能说明

计算 Kimi-K3 attention residual 的归一化打分、行间 softmax 和加权混合，返回最终 RMSNorm 之前的混合值。算子只读取已有快照，不写入快照，也不执行残差相加。

设 token 数为 $N$，隐藏维度为 $H$，有效快照数为 $B$。对每个 token $t$，将 `bank[t, :B, :]` 与 `prefix_sum[t, :]` 依次组成 $B+1$ 行：

$$
v_{t,r,h}=\begin{cases}
\mathrm{bank}_{t,r,h}, & 0\le r<B,\\
\mathrm{prefix\_sum}_{t,h}, & r=B.
\end{cases}
$$

使用 $w=\mathrm{combined\_weight}$、$\varepsilon=\mathrm{variance\_epsilon}$ 计算：

$$
q_{t,r}=\left(\frac{1}{H}\sum_{h=0}^{H-1}v_{t,r,h}^{2}+\varepsilon\right)^{-1/2},
\qquad s_{t,r}=\sum_{h=0}^{H-1}(v_{t,r,h}q_{t,r})w_h.
$$

$$
p_{t,r}=\frac{\exp(s_{t,r}-\max_j s_{t,j})}{\sum_{j=0}^{B}\exp(s_{t,j}-\max_k s_{t,k})},
\qquad y_{t,h}=\sum_{r=0}^{B}p_{t,r}v_{t,r,h}.
$$

加权求和使用原始行值，而不是归一化后的行值；输出不再做 RMSNorm。所有中间计算均使用 float32，最终转换为 `prefix_sum.dtype`。

在 SGLang 调用场景中，`combined_weight` 是 score RMSNorm 权重与 score projection 权重的逐元素乘积，通常为连续 float32 向量；该权重需由调用方预先计算。bank 容量可大于有效快照数，`bank[:, B:, :]` 不参与计算，允许包含未初始化值或 NaN。参与计算的输入按有限值使用。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `prefix_sum` | `torch.Tensor` | `[N, H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 当前残差前缀；`N >= 0`，`H > 0`。 |
| `bank` | `torch.Tensor` | `[N, C, H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 快照 bank；`C >= 0`，与 prefix_sum 同设备，dtype 可以不同。只读取前 `num_valid_blocks` 行。 |
| `num_valid_blocks` | `int` | — | — | 必填 | Python 整数，排除 bool；必须满足 `0 <= num_valid_blocks <= C`，Triton 实现对越界值抛出 `ValueError`。 |
| `combined_weight` | `torch.Tensor` | `[H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 合并后的打分权重；与 prefix_sum 同设备，dtype 可以不同。 |
| `variance_epsilon` | `int` 或 `float` | — | — | 必填 | Python 实数，排除 bool；必须为有限正数，例如 `1e-6`。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

三个 Tensor 均要求 `torch.strided` 布局。Triton 在计算前校验元信息，空输入也执行校验；公开 operator 负责分发。`torch_native` 假定输入合法，作为数学参考实现，不显式校验元信息。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, H]` | prefix_sum.dtype | 与 prefix_sum 同设备；非空时新建输出，不修改任何输入。 |

当 `N == 0` 时，返回 `prefix_sum` 本身，不启动 kernel。当 `N > 0` 且 `num_valid_blocks == 0` 时，只有 prefix 行参与混合；对有限输入，输出值与 prefix_sum 相同，但仍为新 Tensor。

## triton 后端

- 实现位于 `implementation/triton/hyper_conn/attn_residual_mix.py`，kernel 名为 `_attn_res_mix_fused_kernel`。
- 非空输入须位于 NPU。prefix_sum 和 bank 的最后一维 stride 均须为 1，combined_weight 的 stride 须为 1。token stride 与 bank 行 stride 可不同，支持满足上述约束的非连续视图。
- 隐藏维度按 `BLOCK_H = next_power_of_2(H)` 展开，读取和写入均使用 mask，填充通道按 0 处理，均方值仍除以真实 `H`。`H` 不再要求为二次幂，包括 `7168` 在内的隐藏维尾部可通过掩码处理。
- 将打分、softmax 与混合融合在一个持久化向量核 kernel 中。`N` 和 `B` 保持动态；`NB` 为 `B + 1` 向上取整到 2 的幂。无需分配打分矩阵。

## torch_native 后端

- 独立使用 PyTorch 运算，不调用 Triton。支持一般合法 `H` 和非连续 stride，也可在 CPU 上单独验证数学行为。
- 仅拼接有效 bank 行和 prefix 行，使用 float32 计算均方值、打分、softmax 与加权混合，再转回 prefix_sum.dtype。
- 归约顺序和融合乘加可能造成数值差异，两后端不保证逐位相等，应按 dtype 设置容差。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.hyper_conn import attn_res_mix_fused

prefix_sum = torch.randn(17, 1024, device="npu", dtype=torch.float32)
bank = torch.randn(17, 8, 1024, device="npu", dtype=torch.float32)
combined_weight = torch.randn(1024, device="npu", dtype=torch.float32) * 0.01

output = attn_res_mix_fused(prefix_sum, bank, 3, combined_weight, 1e-6)
reference = attn_res_mix_fused(
    prefix_sum, bank, 3, combined_weight, 1e-6, impl="torch_native"
)
torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
```
