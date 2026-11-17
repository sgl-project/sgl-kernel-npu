# solve_tril

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import solve_tril

def solve_tril(A, cu_seqlens=None, output_dtype=torch.float, *, impl="auto"):
    ...
```

## 功能说明

对每个序列 chunk、每个 head，计算单位下三角矩阵的逆：

$$
R_c=(I_C+A_c)^{-1},\qquad (I_C+A_c)R_c=I_C.
$$

其中 $A_c$ 是严格下三角的 $C\times C$ 矩阵，$I_C$ 为单位矩阵。
输入把连续的矩阵行存为 `[B, T, H, C]`；最后一维是 chunk 宽度，
不是 key 特征维度。最后不足 C 行的 chunk 补零后求逆，仅返回有效行。
提供 `cu_seqlens` 时，每个序列边界重新开始分块。

这是 Gated DeltaNet 分块线性 attention 的三角求逆步骤，位于
`sgl_kernel_npu.fla.chunk` 的 `chunk_scaled_dot_kkt_fwd` 与
`recompute_w_u_fwd` 之间。SGLang NPU 的 `TritonGDNKernel.extend`
通过该 chunk 流程执行 GDN prefill/extend，例如 Qwen3-Next。
在一个 chunk 内，$i>j$ 时 $A_{ij}=\beta_i\exp(G_i-G_j)k_i^\top k_j$，
$G$ 是累计的对数衰减。该逆矩阵用于消除 Delta Rule 更新之间的依赖。
独立的 recurrent decode 路径不需要此求逆。
`GDN_USE_MEGA_GDN=1` 会绕过 chunk 流程中的这组独立 kernel。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `A` | `torch.Tensor` | `[B, T, H, C]` | FP16/BF16/FP32 | 必填 | 非空，`C>0`；每个 chunk 严格下三角、数值有限。Triton 要求 NPU 上的连续输入，且 `C` 为 16、32、64；torch_native 支持其他正整数 `C` 和非连续输入。 |
| `cu_seqlens` | `torch.Tensor` 或 `None` | `[N+1]` | int32/int64 | `None` | 变长输入要求 `B=1`。Triton 要求连续且与 A 同设备；torch_native 也接受 CPU 上和非连续的边界张量。起点 0、终点 T，严格递增，不支持空序列。None 表示 B 个等长序列。 |
| `output_dtype` | `torch.dtype` | — | FP16/BF16/FP32 | `torch.float32` | FP32 计算后的输出转换类型。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：`auto`、`triton`、`torch_native`；`auto` 默认使用 Triton。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `R` | `torch.Tensor` | 与 A 相同 | `output_dtype` | 与 A 同设备的新张量，不修改输入。 |

使用时请注意：

1. 两个后端都检查 shape、dtype 等属性；只有 Triton 要求输入连续。
2. 严格下三角、有限值由调用方保证，代码不逐元素检查。Triton 也要求调用方保证序列边界正确；torch_native 会检查起点为 0、终点为 T，且边界严格递增。
3. Triton 会缓存分块索引，原地修改 `cu_seqlens` 可能导致复用错误的索引；torch_native 不使用该缓存。
4. 病态矩阵可能导致较大的数值误差。

## triton 后端

先用 FP32 求 16×16 对角块的逆，
再通过分块矩阵乘合并为 32×32 或 64×64 的逆。
输出转换采用最近偶数舍入。要求 NPU 上的连续输入。

## torch_native 后端

独立使用 FP32 `torch.linalg.solve_triangular`，以单位矩阵为右端项，
再转换到 `output_dtype`。各序列分别处理，尾块独立补齐。
支持任意正整数块大小和非连续输入，变长边界也可以位于 CPU 上或不连续。
空输入仍不支持；输入和输出类型为 FP16、BF16 或 FP32，内部统一使用 FP32 计算。
将边界转换为 Python 列表后检查起点、终点和递增顺序；边界位于 NPU 上时，`tolist()` 会同步设备。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import solve_tril

A = (0.01 * torch.randn(1, 2, 64, 64, device="npu")).tril(-1)
# 两个 chunk、一个 head：[B, chunks, rows, columns] -> [B, T, H, C]。
A = A.reshape(1, 128, 1, 64).contiguous()
R = solve_tril(A, output_dtype=torch.float32)
R_native = solve_tril(A, impl="torch_native")
```
