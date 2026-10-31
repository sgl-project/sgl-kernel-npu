# fused_scale_shift

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 待验证 | 待验证 |
| `torch_native` | 待验证 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import fused_scale_shift

def fused_scale_shift(x, scale, shift, scale_constant=1.0, *, impl="auto"):
    ...
```

`auto` 与 `triton` 注册到同一个入口，共享后端延迟导入缓存。默认选择 Triton，
不自动回退。两个实现均声明支持全部已知代际；芯片表表示实际硬件验证状态，
本次迁移尚未完成硬件验证。

## 功能说明

对输入 $X \in \mathbb{R}^{B\times L\times H}$ 应用标量或按通道缩放，再加标量、
按通道或逐元素偏移。scale 和 shift 在使用前展平，以元素数量决定广播方式，
不依赖原始 shape。记 $N=BLH$、$n_s=\mathrm{scale.numel()}$、
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

保留原有两个 kernel 的行为：只有逐元素 shift 分支使用 `scale_constant`；广播
shift 分支始终使用 `1 + scale`，即使传入 `scale_constant=0` 也不改变该分支。
元素数量重合时优先选择逐元素分支。例如 `B * L == 1` 时，按通道 shift 也属于
完整输入的逐元素 shift。Triton 的该分支要求 scale 恰好有 `H` 个值，只有 `H=1`
时才能使用单元素 scale。

公式描述数学运算，浮点中间结果遵循下述 dtype 规则，最终输出转换为 `x.dtype`。
算子不修改输入。

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

三个 tensor 的 dtype 可以不同。精度比较使用有限输入。公开入口只负责分发，
Triton wrapper 在查询启动参数前检查类型、维度、元素数、布局、设备和连续性，
不支持的输入在启动 kernel 前抛出 `AssertionError`。不支持元素数超过 `H` 的按
batch 或按帧 scale。

## triton 后端

- 位于 `implementation/triton/norm/scale_shift.py`，保留原 kernel 函数体及调优配置。
- `fused_scale_shift_kernel` 处理广播 shift，行块为 128/112、通道块为 128，scale
  和 shift 独立支持标量或按通道值。`fused_scale_shift_kernel_2` 处理逐元素 shift，
  行块为 96/80/64、通道块为 128，每个通道读取一个 scale 值。
- 两者均使用持续分配任务的向量核 grid，以 mask 处理行和通道尾块；wrapper
  不额外填充或转换布局。
- 常数加法使用 scale 的 dtype。广播 kernel 中按通道 shift 转为 float32，标量
  shift 保持原 dtype；逐元素 kernel 始终将 shift 转为 float32。乘法使用 Triton
  的类型提升规则，float16 与 bfloat16 混合时优先使用 float16。
- 旧脚本 `sgl_kernel_npu.norm.scale_shift` 保持原样。重构后的实现及其输入检查
  通过公开接口 `sgl_kernel_npu.operators.norm.fused_scale_shift` 使用。

## torch_native 后端

- 使用独立的 PyTorch 运算，不导入 Triton；展平调制参数，并保持相同的元素数
  分支优先级。
- 常数加法保持 scale.dtype，匹配 shift 的 float32 转换，显式处理 Triton 的
  float16/bfloat16 类型提升规则。
- 假定输入合法，不显式检查元信息；还支持非连续输入，以及逐元素 shift 配合
  标量 scale，但这两种组合不属于 Triton 的支持范围。
- 分步运算与 kernel 的乘加融合可能造成舍入差异，不保证逐位一致。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_scale_shift

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
