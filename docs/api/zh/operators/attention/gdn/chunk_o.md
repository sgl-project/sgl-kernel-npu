# chunk_fwd_o

## 芯片支持

| 实现 | ARCH22 | ARCH35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64, *, impl="auto"):
    ...
```

`auto` 默认选择 `triton`。

## 功能说明

在 Gated DeltaNet 分块状态更新之后重建输出。每个 chunk 的输出由 Query 与该块进入状态
的乘积，以及块内具有因果掩码的注意力对变换后 V 的乘积共同组成：

$$
O_c = s\left[\operatorname{diag}(e^{G_c})Q_cH_c
       +\operatorname{tril}\left((Q_cK_c^T)\odot E_c\right)V_c\right],
\quad E_{ij}=\begin{cases}e^{G_i-G_j}&G_i\le G_j\\0&\text{其他情况}.\end{cases}
$$

`g` 是已经完成块内累积的对数衰减，不会再次累积；`h` 是每个块开始时的状态，
不能只传初始状态或最终状态；`v` 是状态更新阶段产生的值残差，不一定是模型原始 V 投影。
每 `H/Hg` 个 value head 共用一个 Query/Key head。块内注意力矩阵在与 V 相乘之前
转换为 v.dtype，输出也使用 v.dtype。

变长序列在各边界重新划分 chunk，进入状态按序列、块的顺序存储。等长 B>1 时，
门控沿用展平 `[B,H,T]` 存储上的 `b*T + head*T + token` 地址；需要独立的逐序列
门控时，使用带 `cu_seqlens` 的 B=1 打包输入。

## 参数与返回值

| 参数 | 类型／形状 | dtype／默认值 | 说明与约束 |
| --- | --- | --- | --- |
| `q`、`k` | Tensor `[B,T,Hg,K]` | FP16/BF16/FP32；必填 | 连续的 NPU Query/Key，形状、dtype 和设备一致。 |
| `v` | Tensor `[B,T,H,V]` | q.dtype；必填 | 连续的变换后 V，H 为 Hg 的整数倍。 |
| `h` | Tensor `[B,ceil(T/C),H,K,V]` | q.dtype；必填 | 连续的进入状态；打包输入为 `[1,total_chunks,H,K,V]`。 |
| `g` | Tensor `[B,T,H]` | 浮点，通常 FP32；`None` | 与 q 同设备的块内累积对数衰减。实际必须传 Tensor，None 会抛出 AttributeError。 |
| `scale` | float 或 `None` | `None` | 输出缩放系数，默认 `K**-0.5`。 |
| `cu_seqlens` | Tensor `[N+1]` 或 `None` | int32/int64；`None` | B=1 打包输入的边界，从 0 开始，到 T 结束，各序列长度为正。 |
| `chunk_size` | int | `64` | 块宽 C，Triton 使用正的 2 次幂。 |
| `impl` | str | `"auto"` | `auto`、`triton`、`torch_native`。 |

输入不被修改。调用者需提供有限数值及足够的进入状态。Triton 会缓存打包元数据，
不要原地修改 cu_seqlens。

| 返回值 | 类型／形状 | dtype／设备 | 说明 |
| --- | --- | --- | --- |
| 输出 | Tensor `[B,T,H,V]` | v.dtype／v.device | 新分配的重建输出。 |

## triton 实现

支持等长和打包输入、分组 Query/Key head，以及特征和序列尾部。
两种输入形式均保留调用者指定的 chunk 边界。

## torch_native 实现

用 PyTorch FP32 矩阵乘法分别计算进入状态项和块内因果项，保留块内注意力矩阵转换到
v.dtype 的阶段。cu_seqlens 在 NPU 上时，读取边界会同步。该算子不执行状态更新。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

q = torch.randn(1, 65, 2, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(1, 65, 4, 128, device="npu", dtype=q.dtype)
h = torch.zeros(1, 2, 4, 128, 128, device="npu", dtype=q.dtype)
g = torch.zeros(1, 65, 4, device="npu", dtype=torch.float32)
o = chunk_fwd_o(q, k, v, h, g=g, impl="triton")
```
