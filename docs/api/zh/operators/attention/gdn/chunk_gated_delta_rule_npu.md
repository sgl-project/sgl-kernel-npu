# chunk_gated_delta_rule_npu

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

省略 `impl` 或传入 `"auto"` 时选择 `"triton"`，不会自动回退到其他后端。
默认使用分解式 Triton 路径。设置 `GDN_USE_MEGA_GDN=1` 时选择 mega 路径，
要求 ARCH22 且已安装原生 `mega_chunk_gdn` 算子。

```python
from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

def chunk_gated_delta_rule_npu(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=True, cu_seqlens=None, head_first=False,
    use_qk_l2norm_in_kernel=False, *, impl="auto",
):
    ...
```

## 功能说明

计算 gated delta rule 的前向结果，返回注意力输出、可选的最终递推状态，
以及进入每个分块前的状态。每 `H/Hg` 个 value head 共享一个 query/key head。
`g` 是逐 token 的对数衰减，`beta` 控制状态更新的权重。

默认路径按 64 个 token 分块。可选 Q/K L2 归一化在注意力计算前执行
`x / sqrt(sum(x*x) + 1e-6)`。`scale=None` 使用 `K**-0.5`，显式传入 0 时保留零缩放。
Packed 输入中的各序列独立维护状态。

本接口对应 SGLang 线性注意力路径使用的分解式 GDN 接口，
与融合 CANN 接口 `chunk_gated_delta_rule` 分别注册。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- | --- |
| `q`, `k` | `torch.Tensor` | `[B,T,Hg,K]` | `torch.float16`、`torch.bfloat16` | 必填 | 同一 NPU 上的 query 和 key，dtype 一致。 |
| `v` | `torch.Tensor` | `[B,T,H,V]` | 与 `q` 相同 | 必填 | 同一 NPU 上的 value；`H` 必须是 `Hg` 的整数倍。 |
| `g` | `torch.Tensor` | `[B,T,H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 输入 NPU 上的逐 token 对数衰减，尚未累加。 |
| `beta` | `torch.Tensor` | `[B,T,H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | 输入 NPU 上的状态更新权重。 |
| `scale` | `float` 或 `None` | — | — | `None` | 输出缩放系数；`None` 对应 `K**-0.5`。 |
| `initial_state` | `torch.Tensor` 或 `None` | `[N,H,K,V]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | `None` | 输入 NPU 上的初始状态；`None` 表示从零状态开始。 |
| `output_final_state` | `bool` | — | — | `True` | 是否返回最终状态。 |
| `cu_seqlens` | `torch.Tensor` 或 `None` | `[N+1]` | `torch.int32`、`torch.int64` | `None` | 输入 NPU 上的 packed 序列边界：首项为 0、末项为 `T`，序列长度为正；`None` 表示固定长度输入。 |
| `head_first` | `bool` | — | — | `False` | 仅支持 `False`；传入 `True` 时抛出 `DeprecationWarning`。 |
| `use_qk_l2norm_in_kernel` | `bool` | — | — | `False` | 在注意力计算前沿最后一维归一化 Q/K。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `o` | `torch.Tensor` | `[B,T,H,V]` | `q.dtype` | `q.device` 上的注意力输出，不修改输入。 |
| `final_state` | `torch.Tensor` 或 `None` | `[N,H,K,V]` | `torch.float32` | `q.device` 上的最终状态；`output_final_state=False` 时返回 `None`。 |
| `h` | `torch.Tensor` | `[1,NT,H,K,V]` | `q.dtype` | `q.device` 上进入各分块前的状态，按序列、分块顺序排列。 |

使用固定 `B=1` 或 packed `B=1` 输入，要求 `K,V <= 128`。`N` 为序列数，
未传入 `cu_seqlens` 时为 1。`NT` 为所有序列的分块总数，
即各序列 `ceil(sequence_length / chunk_size)` 之和。
Packed 输入的初始状态数必须满足 `N=len(cu_seqlens)-1`。
不要原地修改已缓存的序列边界。形状、数据类型和设备要求由调用方保证，
接口仅对其中部分约束进行显式校验。

## triton 后端

- 将 Tensor 输入转为连续张量，并切换到输入所在的 NPU 设备上下文。
- 以 64 个 token 分块，组合分块累加、scaled KKT、三角求逆、WY 重算、状态更新和输出计算。
- 设置 `GDN_USE_MEGA_GDN=1` 时使用 128-token 分块的原生
  [mega_chunk_gdn](mega_chunk_gdn.md) 路径，并遵循它的形状和 dtype 约束。
  该路径仅支持 ARCH22，在 ARCH35 上不会自动回退。
- 提供推理所需的前向计算，保留既有中间 dtype 和类型转换。

## torch_native 后端

- 在 CPU 上以 FP64 独立计算逐 token 递推，再恢复输出 dtype 和设备，支持相同的归一化、缩放和状态选项。
- 设置 `GDN_USE_MEGA_GDN=1` 时使用独立 mega 分块代数参考，包括其 FP16 转换和 128-token 分块状态。
- 为精度检查提供参考实现。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import chunk_gated_delta_rule_npu

q = torch.randn(1, 65, 4, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(1, 65, 8, 128, device="npu", dtype=q.dtype)
g = -torch.rand(1, 65, 8, device="npu", dtype=torch.float32)
beta = torch.rand(1, 65, 8, device="npu", dtype=q.dtype)
cu_seqlens = torch.tensor([0, 31, 65], device="npu", dtype=torch.int32)
o, final_state, h = chunk_gated_delta_rule_npu(
    q, k, v, g, beta, cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=True,
)
```
