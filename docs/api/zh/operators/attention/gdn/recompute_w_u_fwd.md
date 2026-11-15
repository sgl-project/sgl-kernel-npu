# recompute_w_u_fwd

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

def recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens=None, *, impl="auto"):
    ...
```

## 功能说明

对每条序列、每个 chunk 和每个 value head 计算 Gated DeltaNet 的 WY 表示：

$$
W_c=A_c\operatorname{diag}(\beta_c\odot\exp(G_c))K_c^{\uparrow},
\qquad
U_c=A_c\operatorname{diag}(\beta_c)V_c.
$$

$A_c$ 是 `solve_tril` 输出的逆矩阵，不是求逆前的严格下三角矩阵。
$G_c$ 是输入 `g_cumsum`，即 chunk 内已经累加好的对数衰减；本算子不再累加。
$\beta_c$ 是写入权重，$\odot$ 和 $\exp$ 均逐元素计算。
$K_c^{\uparrow}$ 按 head 分组复用 key：value head $h$ 使用
key head $\lfloor h/(H/H_g)\rfloor$。不需要调用方提前复制 key。

输入序列按宽度 C 分块，尾块只返回有效行；提供 `cu_seqlens` 时在每个序列边界重新分块。
该步骤在 GDN 分块算法中位于三角求逆之后、chunk 状态更新之前。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `k` | `torch.Tensor` | `[B,T,Hg,K]` | FP16/BF16/FP32 | 必填 | Key；非空。Triton 要求 NPU 上的连续张量。 |
| `v` | `torch.Tensor` | `[B,T,H,V]` | FP16/BF16/FP32 | 必填 | Value；非空，H 必须是 Hg 的整数倍。Triton 要求连续。 |
| `beta` | `torch.Tensor` | `[B,T,H]` | FP16/BF16/FP32 | 必填 | 每个 token、value head 的写入权重。 |
| `g_cumsum` | `torch.Tensor` | `[B,T,H]` | FP16/BF16/FP32 | 必填 | Chunk 内累计对数衰减，通常使用 FP32。 |
| `A` | `torch.Tensor` | `[B,T,H,C]` | FP16/BF16/FP32 | 必填 | 从 `solve_tril` 得到的分块逆矩阵；非空。Triton 要求连续且 C 为 16、32、64；torch_native 支持任意正整数 C。 |
| `cu_seqlens` | `torch.Tensor` 或 `None` | `[N+1]` | int32/int64 | `None` | 序列边界，起点 0、终点 T、严格递增；packed 输入要求 B=1。Triton 要求连续且与 k 同设备；torch_native 也接受 CPU 或非连续边界。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：`auto`、`triton`、`torch_native`；`auto` 默认使用 Triton。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `w` | `torch.Tensor` | `[B,T,H,K]` | k.dtype | 与 k 同设备，新分配的张量。 |
| `u` | `torch.Tensor` | `[B,T,H,V]` | v.dtype | 与 v 同设备，新分配的张量；返回顺序为 `(w, u)`。 |

`k`、`v`、`beta`、`g_cumsum`、`A` 必须位于同一设备，且不被修改。
调用方应保证数值有限，并避免 `exp(g_cumsum)` 溢出。
Triton 不逐元素检查序列边界；torch_native 会检查边界起点、终点和递增顺序。
Triton 会缓存分块索引，使用后不要原地修改 `cu_seqlens`。

## triton 后端

先将 beta 和累计衰减转换为连续的 `[B,H,T]` 布局，再按 chunk 启动 kernel。
内部使用 FP32 计算，沿 K、V 按 128 元素分片，支持非对齐特征维度和序列尾块。
支持等长多 batch、packed 变长序列，以及不同 key/value head 数的分组映射。
输出分别转换为 k、v 的 dtype。

## torch_native 后端

使用 PyTorch 的指数、逐元素乘法和矩阵乘法独立计算。
内部使用 FP32；支持非连续张量和任意正整数块大小。
变长边界转换为 Python 列表后检查；边界位于 NPU 上时，`tolist()` 会同步设备。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import recompute_w_u_fwd

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.bfloat16)
v = torch.randn(1, 64, 4, 128, device="npu", dtype=torch.bfloat16)
beta = torch.rand(1, 64, 4, device="npu")
g_cumsum = -torch.rand(1, 64, 4, device="npu").cumsum(1)
# 单位矩阵是 solve_tril 的一种合法输出，每个 head 独立使用。
A = torch.eye(64, device="npu", dtype=torch.bfloat16)[None, :, None, :]
A = A.expand(1, 64, 4, 64).contiguous()
w, u = recompute_w_u_fwd(k, v, beta, g_cumsum, A)
```
