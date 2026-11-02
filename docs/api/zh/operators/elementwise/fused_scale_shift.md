# fused_scale_shift

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.elementwise import fused_scale_shift

def fused_scale_shift(x, scale, shift, scale_constant=1.0, *, impl="auto"):
    ...
```

默认的 `impl="auto"` 选择 Triton 后端。

## 功能说明

对输入 $X \in \mathbb{R}^{B\times L\times H}$ 应用标量或按通道缩放，再加标量、
按通道或逐元素偏移。scale 和 shift 在使用前展平，以元素数量决定广播方式。
记 $N=BLH$、$n_s=\mathrm{scale.numel()}$、
$n_t=\mathrm{shift.numel()}$：

$$
S_h=\begin{cases}s_0,&n_s=1,\\s_h,&n_s=H,\end{cases}
\qquad
T_{blh}=\begin{cases}
\mathrm{reshape}(t,[B,L,H])_{blh},&n_t=N,\\
t_0,&n_t=1\ne N,\\
t_h,&n_t=H\ne N.
\end{cases}
$$

$$
c=\begin{cases}\mathrm{scale\_constant},&n_t=N,\\1,&n_t\ne N,\end{cases}
\qquad Y_{blh}=X_{blh}(c+S_h)+T_{blh}.
$$

只有逐元素 shift 分支使用 `scale_constant`；广播 shift 分支始终使用 `1 + scale`，
即使传入 `scale_constant=0` 也不改变该分支。
元素数量重合时优先选择逐元素分支。例如 `B * L == 1` 时，按通道 shift 也属于
完整输入的逐元素 shift。Triton 的该分支要求 scale 恰好有 `H` 个值，只有 `H=1`
时才能使用单元素 scale。

输出 dtype 为 `x.dtype`，算子不修改输入。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, H]` | float32、float16 或 bfloat16 | 必填 | 非空三维输入，各维度必须为正数。Triton 要求 NPU 上的连续存储。 |
| `scale` | `torch.Tensor` | 任意 shape，元素数为 `1` 或 `H` | float32、float16 或 bfloat16 | 必填 | 标量或按通道缩放，与 x 同设备。Triton 要求连续；shift.numel() 等于 x.numel() 时必须有 `H` 个元素。 |
| `shift` | `torch.Tensor` | 任意 shape，元素数为 `1`、`H` 或 `B*L*H` | float32、float16 或 bfloat16 | 必填 | 标量、按通道或逐元素偏移，与 x 同设备。Triton 要求连续。逐元素值按展平顺序还原为 x.shape。 |
| `scale_constant` | Python `int` 或 `float` | — | — | `1.0` | 有限标量，不接受 bool；仅逐元素 shift 分支使用。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, H]` | x.dtype | 与 x 同设备的新建连续 tensor，不与任何输入共享存储。 |

三个 tensor 的 dtype 可以不同。输入须满足上述约束；Triton 后端对非法输入
抛出 `AssertionError`。不支持元素数超过 `H` 的按 batch 或按帧 scale。

## triton 后端

使用 `impl="triton"` 或 `impl="auto"` 在 NPU 上执行。输入须为连续的 NPU tensor，
并满足上述 shape 和 dtype 约束。

## torch_native 后端

使用 `impl="torch_native"` 进行精度对照或问题排查，并遵循上述输入约束。
精度对照使用有限输入；不同后端可能存在浮点舍入差异，不保证逐位一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.elementwise import fused_scale_shift

x = torch.randn(2, 3, 129, device="npu", dtype=torch.float32) * 0.25
scale = torch.randn(1, 1, 129, device="npu", dtype=torch.float32) * 0.25
shift = torch.randn_like(x) * 0.25

# 逐元素 shift：scale_constant=0 时计算 x * scale + shift。
output = fused_scale_shift(x, scale, shift, scale_constant=0)
reference = fused_scale_shift(x, scale, shift, scale_constant=0, impl="torch_native")
torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)

# 广播 shift：计算 x * (1 + scale) + channel_shift。
channel_shift = torch.zeros_like(scale)
broadcast_output = fused_scale_shift(x, scale, channel_shift)
```

## 测试

公开接口测试包含 46 个默认用例和 266 个可选 `extra_checks` 用例。
精度测试命令为：

```bash
pytest tests/python/operators/elementwise/test_scale_shift.py --enable-perf=false
pytest tests/python/operators/elementwise/test_scale_shift.py -m extra_checks --enable-perf=false
```
