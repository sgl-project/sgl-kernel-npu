# grouped_gemma_rmsnorm

## 芯片支持

arch22 拆成两列，因为该家族的两个成员验证程度不同。A3 指 Ascend910_93 系列，
A2 指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `ascendc` | 支持 | 待验证 | 未实现 |
| `torch_native` | 支持 | 待验证 | 待验证 |
| `triton` | 未实现 | 未实现 | 未实现 |
| `tilelang` | 未实现 | 未实现 | 未实现 |

`ascendc` 实现注册到 arch22（同时覆盖 A2 与 A3）；`cmake/legacy_kernels.cmake`
中的 `SGL_KERNEL_ENABLE_A3_ONLY_OPS` 会为 Ascend950 系列之外的所有 SoC 构建其
kernel，并在运行时通过 `GetCoreNumAiv()` 查询 Vector Core 数量。A2 标为"待验证"
是因为实测仅在 A3 上完成；Ascend950 构建不包含该内部 helper，访问
`torch.ops.npu.hyper_conn_grouped_norm` 会抛出 `AttributeError`。硬件验证环境为
Ascend910_9382、CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0。arch35 未实现。

## 函数原型

默认选项为 `auto`，与 `ascendc` 注册到同一入口。不传 `impl` 或传 `"auto"` 时选择
Ascend C kernel。

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: Optional[int] = None,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## 功能说明

对固定宽度的分组执行 Gemma 风格分组 RMS 归一化，并按 `1 + weight` 缩放。分组是
`x` 按行优先顺序切出的连续 `group_size` 段，因此形状为 `[rows, branches * width]`
（或 `[rows, branches, width]`）的 hyper-connection 残差在 `group_size = width`
时，会对每行的每个分支分别归一化。

记 `W = group_size`（`group_size` 为 `None` 时取 `x.shape[-1]`）、
`L = weight.numel()`，`g` 为按行优先顺序在整个张量上计数的分组下标：

$$
\mathrm{inv}_g=\left(\frac{1}{W}\sum_{j=0}^{W-1}x_{g,j}^{2}+\varepsilon\right)^{-1/2},
\qquad
y_{g,j}=\mathrm{bf16}\left(x_{g,j}\,\mathrm{inv}_g\left(1+w_{(gW \bmod L)+j}\right)\right)
$$

权重偏移以 `L / W` 个分组为周期轮转，算子要求分组总数是该周期的整数倍，从而确保
不会有分组被错误的权重段缩放。每个分支各持一段权重时，各分支使用自身对应的切片；
权重长度恰为一个分组宽度时，所有分组共享同一切片。平方和与缩放均在 float32 中累
加，输出做一次 BF16 取整。

本算子不含 bias、不含量化、不做残差相加。与常规 RMSNorm 的区别在于权重以单位偏置
`1 + weight` 的形式参与计算，且归一化粒度是固定宽度分组而非仅限整行。

## 参数与返回值

公开 operator 在选择实现之前校验下表契约，因此两个后端拒绝同样的输入。非法输入在
任何 kernel 下发之前抛出 `ValueError`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., H]` | `torch.bfloat16` | 必填 | 至少二维的输入；连续，位于 NPU 设备。`H` 必须是有效分组宽度的整数倍。行数为 0 时返回空结果。 |
| `weight` | `torch.Tensor` | `[L]` | `torch.bfloat16` | 必填 | 与 x 同设备的连续一维权重。`L` 必须是有效分组宽度 `width` 的正整数倍，且 `x.numel() / width` 必须是 `L / width` 的整数倍。 |
| `group_size` | `int` 或 `None` | — | — | `None` | 分组宽度；`[8, 4096]` 内 8 的倍数。`None` 表示取 `H`，即最后一维每段一个分组。`bool` 被拒绝。 |
| `eps` | `int` 或 `float` | — | — | `1e-6` | 加到均方值上的有限正数。`bool` 被拒绝。 |
| `out` | `torch.Tensor` 或 `None` | 与 x 相同或 — | `torch.bfloat16` 或 — | `None` | 可选的预分配连续输出缓冲区，须位于 x 所在设备、与 x 同 shape、`requires_grad=False`，且与 `x` 或 `weight` 无内存重叠；为 `None` 时新建张量。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 选择 Ascend C。 |

不支持自动求导：任何带 `requires_grad=True` 的输入或预分配 `out` 均会被拒绝。若
提供 `out`，则其必须与 `x` 的 shape、dtype（`torch.bfloat16`）及设备一致，保持连
续，且不得与 `x` 或 `weight` 存在别名或内存重叠。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 x 相同 | `torch.bfloat16` | 位于 x 所在设备上的归一化输出。提供 `out` 时原地写入并返回 `out`，否则返回新建的连续张量。不修改任何输入。 |

## ascendc 后端

- 调用内部 native helper `torch.ops.npu.hyper_conn_grouped_norm`，源码见
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md)。
- 一个 block 以跨步循环处理完整分组。权重被所有分组共享时只在循环外加载一次，否
  则每个分组重新加载一次，因此相同 shape 下按分支切分的权重会慢于共享权重。
- 功能上支持任意 `[*leading, cols]`（`ndim >= 2`）形状；block 数来自按分组宽度和
  `x.size(0)`（`x` 的首维）查表得到的实测值，并裁剪到分组数与设备运行时 Vector
  Core 数（`GetCoreNumAiv()`）之内。由于查表假设首维即 token 数，传入带前导
  batch 维的形状（如 `[1, S, D]`）在数值上完全正确，但选出的 block 数会少于最优
  值；若需完整吞吐，请展平为 `[S, D]` 或 `[B * S, D]`。65 到 1022 行之间由 128
  行表项插值得到，未逐点实测。
- 中间计算全程在 `float32` 中进行，仅在输出时取整一次为 `bfloat16`。平方和经
  `ReduceSum` 归约后乘以预计算的 `1.0f / width` 倒数，并通过 `Sqrt` + `Div`（而
  非 `torch.rsqrt`）求逆均方根，因此归一化因子在输出取整前可能与参考实现相差约 1
  个 `float32` ulp。

## torch_native 后端

- 独立的 PyTorch 实现，不调用任何自定义 native 算子。
- 先 reshape 为 `[-1, L / width, width]`，使每个分组无需 gather 即可对上自己的权
  重切片，再按上面的公式用 float32 计算。
- 会额外分配一份 `x` 的 float32 副本，因此 prefill 规模的输入临时内存约为输入的
  两倍。它是参考实现，不是性能路径。
- 归约顺序与逆均方根计算方式与 kernel 不同，不保证逐位相等；精度测试采用收紧的
  `BF16_TIGHT` 容差（`rtol = 8e-3`，1–2 个 BF16 ulp）。实测覆盖整个测试矩阵时，
  两个后端至少 99.999% 的元素完全相同，最大差异为输出的一个 BF16 ulp。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

rows, branches, width = 32, 4, 2048
x = torch.randn(rows, branches * width, device="npu", dtype=torch.bfloat16)
# 每个分支各持一段权重：切片偏移在一行内轮转。
weight = torch.randn(branches * width, device="npu", dtype=torch.bfloat16) * 0.02

print(list_impls(grouped_gemma_rmsnorm))  # ['ascendc', 'auto', 'torch_native']
out = grouped_gemma_rmsnorm(x, weight, width, 1e-6)  # 默认 auto 选择 Ascend C
out_native = grouped_gemma_rmsnorm(x, weight, width, 1e-6, impl="torch_native")
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
