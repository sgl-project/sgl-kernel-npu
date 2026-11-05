# hc_mix

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
是因为实测仅在 A3 上完成；Ascend950 构建不包含这些内部 helper，访问
`torch.ops.npu.hyper_conn_scaled_silu` 或 `torch.ops.npu.hyper_conn_branch_mix`
会抛出 `AttributeError`。硬件验证环境为 Ascend910_9382、CANN 9.1.0、torch 2.10.0、
torch_npu 2.10.0。arch35 未实现。

## 函数原型

默认选项为 `auto`，与 `ascendc` 注册到同一入口。不传 `impl` 或传 `"auto"` 时选择
Ascend C 流水线。

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(
    normed,
    down_weight,
    up_weight,
    hc_count,
    hidden_size,
    *,
    impl="auto",
):
    ...
```

## 功能说明

对 hyper-connection 归一化残差执行低秩门控投影与分支加权均值。给定形状为
`[rows, hc_count * hidden_size]` 的 `normed`、形状为
`[low_rank, hc_count * hidden_size]` 的 `down_weight` 以及形状为
`[hc_count * hidden_size, low_rank]` 的 `up_weight`，记 $C = \mathrm{hc\_count}$、
$H = \mathrm{hidden\_size}$、$\sigma(z)=1/(1+e^{-z})$：

$$
t = \frac{x\,W_{\text{down}}^{\top}}{C},
\qquad
a = \mathrm{bf16}\!\left(t\,\sigma(t)\right),
\qquad
g = a\,W_{\text{up}}^{\top}
$$

$$
\mathrm{out}_{m,j} = \mathrm{bf16}\!\left(\frac{1}{C}\sum_{c=0}^{C-1} \sigma(g_{m,c,j})\,x_{m,c,j}\right)
$$

其中 $x = \mathrm{normed}$，$x_{m,c,j} = \mathrm{normed}_{m,\,cH+j}$，
$g_{m,c,j} = g_{m,\,cH+j}$，$\mathrm{out}_{m,j}$ 写入 `output[m, j]`。缩放 SiLU
与分支混合两个阶段内部的中间计算均在 `float32` 中进行，仅在写入全局内存的输出边界
（上投影前的 `a` 与最终输出 `out`）取整为 `bfloat16`，与上游 SGLang kernel 契约一
致。

sigmoid 双侧尾部均饱和而不会产生 NaN：`exp(-z)` 上溢到无穷大时门控为 0，下溢到 0
时门控为 1。

## 参数与返回值

公开 operator 在选择实现之前校验下表契约，因此两个后端拒绝同样的输入。非法输入在
任何 kernel 下发之前抛出 `ValueError`。

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `normed` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | 必填 | 归一化后的残差；NPU 上的连续二维张量。`rows == 0` 时返回形状为 `[0, hidden_size]` 的空输出。 |
| `down_weight` | `torch.Tensor` | `[low_rank, hc_count * hidden_size]` | `torch.bfloat16` | 必填 | 下投影权重；与 normed 同设备的连续二维张量。`low_rank` 必须是 `[16, 4096]` 内 16 的倍数。 |
| `up_weight` | `torch.Tensor` | `[hc_count * hidden_size, low_rank]` | `torch.bfloat16` | 必填 | 上投影权重；与 normed 同设备的连续二维张量。同时支持常规 ND 排布与预转好的 `FRACTAL_NZ`（`torch_npu.npu_format_cast(w, 29)`）。 |
| `hc_count` | `int` | — | — | 必填 | hyper-connection 分支数；`[1, 8]` 内的 2 的幂。`bool` 被拒绝。 |
| `hidden_size` | `int` | — | — | 必填 | 单分支隐藏维度；`[16, 4096]` 内 16 的倍数。`bool` 被拒绝。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字；可选 `"auto"`、`"ascendc"`、`"torch_native"`。`auto` 选择 Ascend C。 |

不支持自动求导：带 `requires_grad` 的输入会被拒绝。

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, hidden_size]` | `torch.bfloat16` | 在 normed 所在设备上新建的连续输出，不修改任何输入。 |

## ascendc 后端

- 下投影与上投影通过 `torch.nn.functional.linear`（`aclnn`）执行，两个向量阶段
  通过内部 Ascend C helper `torch.ops.npu.hyper_conn_scaled_silu` 和
  `torch.ops.npu.hyper_conn_branch_mix` 执行（源码见
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md)）；两个向量
  kernel 的中间计算均保持 `float32`，仅在将各自输出（上投影前的 `a` 与最终输出
  `out`）写入全局内存时取整一次为 `bfloat16`。请通过本公开算子调用，不要直接调用
  内部 helper 绑定。
- 调用方可在加载权重时开启 `torch.npu.config.allow_internal_format = True` 并通过
  `torch_npu.npu_format_cast(up_weight, 29)` 将 `up_weight` 一次性转为
  `FRACTAL_NZ` 并替换原权重：稳态无额外显存开销，精度与 `ND` 在常规 BF16 GEMM 舍
  入内一致（CANN 9.1.0 下 decode 常用分桶逐位相等），在 decode 小 batch 下可显著
  缩短上投影耗时。
- `hyper_conn_branch_mix` 中每个任务处理一个 `(row, column tile)`，通过单次跨步
  拷贝同时读入全部分支；block 数和列分块取自按 `hidden_size` 与 `rows` 查表的实
  测值（并裁剪到任务数与通过 `GetCoreNumAiv()` 查询的运行时 Vector Core 数之内）。
  65 到 1022 行之间由 128 行表项插值得到。

## torch_native 后端

- 独立的 PyTorch 实现，不调用任何自定义 native 算子。
- 两次投影调用 `torch.nn.functional.linear`，缩放 SiLU 与分支混合阶段的中间计算
  在 `float32` 中采用 `1 / (1 + exp(-x))` sigmoid 形式执行，仅在全局内存输出边界
  （上投影前的 `a` 与最终输出 `out`）取整为 `bfloat16`。
- 会分配与 `[rows, hc_count, hidden_size]` 等大的 float32 临时张量，prefill 规模
  下内存开销较大。它是参考实现，不是性能路径。
- 超越函数在两个后端上的实现不同，不保证逐位相等；精度测试采用收紧的
  `BF16_TIGHT` 容差（`rtol = 8e-3`，1–2 个 BF16 ulp）。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.hyper_conn import hc_mix

rows, hc_count, hidden_size, low_rank = 32, 4, 2048, 320
total = hc_count * hidden_size
normed = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
down_weight = (
    torch.randn(low_rank, total, device="npu", dtype=torch.bfloat16) * 0.05
)
up_weight = (
    torch.randn(total, low_rank, device="npu", dtype=torch.bfloat16) * 0.2
)
# 可选：加载权重时一次性转为 FRACTAL_NZ 以加速 decode GEMM
torch.npu.config.allow_internal_format = True
up_weight_nz = torch_npu.npu_format_cast(up_weight.contiguous(), 29)

print(list_impls(hc_mix))  # ['ascendc', 'auto', 'torch_native']
out = hc_mix(normed, down_weight, up_weight_nz, hc_count, hidden_size)
out_native = hc_mix(
    normed, down_weight, up_weight, hc_count, hidden_size, impl="torch_native"
)
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
