# hc_combine

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
`torch.ops.npu.hyper_conn_branch_combine` 会抛出 `AttributeError`。硬件验证环境
为 Ascend910_9382、CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0。arch35 未实现。

## 函数原型

默认选项为 `auto`，与 `ascendc` 注册到同一入口。不传 `impl` 或传 `"auto"` 时选择
Ascend C 流水线。

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## 功能说明

从 `normed_residual` 投影逐分支注入门控，并将缩放后的子层输出 `block_output` 加
回到 `residual` 的各个分支上。记 $C = \mathrm{hc\_count}$、
$H = \mathrm{hidden\_size}$、$\hat{r} = \mathrm{normed\_residual}$、
$\sigma(z)=1/(1+e^{-z})$：

$$
z = \hat{r}\,W_{\text{inject}}^{\top},
\qquad
g_{m,c} = 2\,\sigma\!\left(\frac{z_{m,c}}{C}\right)
$$

$$
\mathrm{out}_{m,c,j} = \mathrm{bf16}\!\left(r_{m,c,j} + b_{m,j}\,g_{m,c}\right)
$$

其中 $r_{m,c,j} = \mathrm{residual}_{m,\,cH+j}$，
$b_{m,j} = \mathrm{block\_output}_{m,j}$，$\mathrm{out}_{m,c,j}$ 写入
`output[m, c * H + j]`。门控缩放、sigmoid 求值、广播相乘与残差相加等中间计算均
在 `float32` 中进行，仅在最终输出时取整一次为 `bfloat16`，与上游 SGLang CUDA
kernel 契约一致。sigmoid 双侧尾部饱和到 0 和 2，不会产生 NaN。

与接收预计算 `post` 及 $4\times 4$ `comb` 稠密混合矩阵且限定隐藏维度为
`4096`/`7168` 的 `hc_post`、`mhc_post` 不同，`hc_combine` 在算子内部完成注入投
影与 `2 * sigmoid` 门控链计算，残差流保持单位系数无跨分支稠密混合，并支持
`[16, 4096]` 内的单分支隐藏维度。

## 参数与返回值

公开 operator 在选择实现之前校验下表契约，因此两个后端拒绝同样的输入。非法输入在
任何 kernel 下发之前抛出 `ValueError`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `block_output` | `torch.Tensor` | `[rows, hidden_size]` | `torch.bfloat16` | 必填 | 子层输出；NPU 上的连续二维张量。 |
| `residual` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | 必填 | 多分支残差；与 block_output 同设备的连续二维张量。`rows == 0` 时返回空结果。 |
| `normed_residual` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | 必填 | 注入投影的归一化残差输入；与 block_output 同设备的连续二维张量。 |
| `inject_weight` | `torch.Tensor` | `[hc_count, hc_count * hidden_size]` | `torch.bfloat16` | 必填 | 注入投影权重；与 block_output 同设备的连续二维张量。 |
| `hc_count` | `int` | — | — | 必填 | hyper-connection 分支数；`[1, 8]` 内的 2 的幂。`bool` 被拒绝。 |
| `hidden_size` | `int` | — | — | 必填 | 单分支隐藏维度；`[16, 4096]` 内 16 的倍数。`bool` 被拒绝。 |
| `out` | `torch.Tensor` 或 `None` | `[rows, hc_count * hidden_size]` 或 — | `torch.bfloat16` 或 — | `None` | 可选的预分配连续输出缓冲区，须位于 residual 所在设备、与 residual 同 shape、`requires_grad=False`，且与任何输入张量无内存重叠；为 `None` 时新建张量。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 选择 Ascend C。 |

不支持自动求导：任何带 `requires_grad=True` 的输入或预分配 `out` 均会被拒绝。若
提供 `out`，则其必须与 `residual` 的 shape（`[rows, hc_count * hidden_size]`）、
dtype（`torch.bfloat16`）及设备一致，保持连续，且不得与 `block_output`、
`residual`、`normed_residual` 或 `inject_weight` 存在别名或内存重叠。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | 位于 residual 所在设备上的合并输出。提供 `out` 时原地写入并返回 `out`，否则返回新建的连续张量。不修改任何输入（不原地改写 `residual`）。 |

## ascendc 后端

- 通过 `torch.nn.functional.linear`（`aclnn`）投影出 `[rows, hc_count]` 注入
  logit，再由内部 Ascend C helper `torch.ops.npu.hyper_conn_branch_combine` 融合
  门控缩放、sigmoid、广播相乘与残差相加（源码见
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md)），中间计算全程
  保持 `float32`，仅在输出时取整一次为 `bfloat16`。请通过本公开算子调用，不要直
  接调用内部 helper 绑定。
- 每个 kernel 任务处理一个 `(row, column tile)`：每行的 `hc_count` 个门控在 8
  lane float32 向量缓冲上计算（同一核连续处理同行分块时直接复用），全部分支的残差
  分块通过单次跨步 MTE2/MTE3 搬运完成。
- block 数与列分块取自按 `hidden_size` 与 `rows` 查表的实测值（裁剪到任务数与通
  过 `GetCoreNumAiv()` 查询的运行时 Vector Core 数之内）。65 到 1022 行之间由
  128 行表项插值得到。

## torch_native 后端

- 独立的 PyTorch 实现，不调用任何自定义 native 算子。
- 注入投影调用 `torch.nn.functional.linear`，后续门控缩放、`1 / (1 + exp(-x))`
  sigmoid、广播相乘与残差相加等中间计算均在 `float32` 中进行，仅在输出时取整一次
  为 `bfloat16`。
- 会分配与完整残差等大的 float32 临时张量，prefill 规模下内存开销较大。它是参考
  实现，不是性能路径。
- 两个后端的超越函数实现不同，不保证逐位相等；精度测试采用收紧的 `BF16_TIGHT` 容
  差（`rtol = 8e-3`，1–2 个 BF16 ulp）。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.hyper_conn import hc_combine

rows, hc_count, hidden_size = 32, 4, 2048
total = hc_count * hidden_size
block_output = torch.randn(
    rows, hidden_size, device="npu", dtype=torch.bfloat16
)
residual = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
normed_residual = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
inject_weight = (
    torch.randn(hc_count, total, device="npu", dtype=torch.bfloat16) * 0.1
)

print(list_impls(hc_combine))  # ['ascendc', 'auto', 'torch_native']
out = hc_combine(
    block_output, residual, normed_residual, inject_weight, hc_count, hidden_size
)
out_native = hc_combine(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
    impl="torch_native",
)
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
