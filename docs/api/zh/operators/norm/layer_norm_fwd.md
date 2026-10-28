# layer_norm_fwd

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import layer_norm_fwd

def layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
    *,
    impl="auto",
):
    ...
```

`auto` 和 `triton` 当前指向同一入口，不会在执行失败后自动切换实现。

## 功能说明

对二维输入逐行、逐组执行 LayerNorm 或 RMSNorm，并支持仿射变换及 SiLU 门控。设输入为 `[M, N]`，组宽 `H = group_size`，组数 `G = N // H`；未指定组宽时 `H = N`。

存在 `z` 且 `norm_before_gate=False` 时，先令 `u = x * SiLU(z)`；其余情况令 `u = x`。每组按以下规则计算：

| 模式 | 统计量与归一化 |
| --- | --- |
| LayerNorm | `mean = mean(u)`，`rstd = 1 / sqrt(mean((u - mean) ** 2) + eps)`，`normalized = (u - mean) * rstd` |
| RMSNorm | `rstd = 1 / sqrt(mean(u ** 2) + eps)`，`normalized = u * rstd`，不返回 mean tensor |

随后乘 `weight`、加可选 `bias`。存在 `z` 且 `norm_before_gate=True` 时，在仿射之后乘 `SiLU(z)`。SiLU 为 `z * sigmoid(z)`。

`activation` 是保留的兼容参数，当前不参与实现分支；传入不同字符串不会改变门控函数。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[M, N]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | `N > 0`，末维 stride 必须为 1；Triton 输入位于 NPU。允许行间有额外间隔。 |
| `weight` | `torch.Tensor` | `[N]` | 浮点 | 必填 | 与 x 同设备、末维 stride 为 1 的逐通道乘法权重。 |
| `bias` | `torch.Tensor` 或 `None` | `[N]` 或 — | 浮点或 — | 必填，可传 `None` | 可选逐通道偏置；存在时与 x 同设备，末维 stride 为 1。 |
| `eps` | `float` | — | — | 必填 | 加到每组方差或均方的稳定项；建议有限正数。 |
| `z` | `torch.Tensor` 或 `None` | `[M, N]` 或 — | 浮点或 — | `None` | 门控输入；存在时与 x 同设备，末维 stride 为 1。此入口要求二维同 shape 的 z。 |
| `out` | `torch.Tensor` 或 `None` | `[M, N]` 或 — | 浮点或 — | `None` | 可复用输出缓冲区；与 x 同设备，末维 stride 为 1，建议与 x dtype 相同。提供时写入并返回该对象，否则新建输出。 |
| `group_size` | `int` 或 `None` | — | — | `None` | 正整数，必须整除 N；None 表示 N。组宽不能超过 `65536 // x.element_size()`。 |
| `norm_before_gate` | `bool` | — | — | `True` | True：归一化、仿射后门控；False：对门控后的输入归一化。 |
| `is_rms_norm` | `bool` | — | — | `False` | False 使用 LayerNorm，True 使用 RMSNorm。 |
| `activation` | `str` 或 `None` | — | — | `None` | 兼容参数，当前被忽略；门控始终为 SiLU。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

返回 `(out, mean, rstd)`：

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[M, N]` | 默认 `x.dtype`；提供缓冲区时为 `out.dtype` | 归一化及可选门控结果；与 x 同设备。 |
| `mean` | `torch.Tensor` 或 `None` | `[G * M]` 或 — | `torch.float32` 或 — | LayerNorm 的逐组均值；RMSNorm 返回 None。 |
| `rstd` | `torch.Tensor` | `[G * M]` | `torch.float32` | 每组的倒数标准差或倒数均方根。 |

统计量按 group-major 展平：第 g 组、第 m 行的值位于 `g * M + m`，可用 `stats.view(G, M)` 查看。公共函数执行 shape、stride、分组及组宽检查，不自动连续化，也不增加 dtype 或设备检查。调用方仍需保证设备和 dtype 合适；未提供 out 时，x、weight、bias、z 均只读。

## triton 后端

- 使用 `_layer_norm_fwd_1pass_kernel_npu`，采用 `BLOCK_M=48`。
- 用输入、输出及 gate 的实际行 stride 寻址，末维保持连续。
- 输入、权重、偏置和 gate 按 FP32 计算，结果写入输出 dtype，统计量保留 FP32。

## torch_native 后端

- 在本接口内准备 `[M, G, H]` 分组视图和统计量视图，使用 PyTorch 完成相同的归一化、仿射及门控。
- 支持输出缓冲区复用，返回与 Triton 相同的统计量布局。

## 调用示例

示例需要已安装包含本接口的 wheel 及 NPU 运行依赖。比较不同实现时使用独立输出，避免复用同一缓冲区覆盖结果。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import layer_norm_fwd

torch.npu.set_device(0)
x = torch.randn(4, 128, device="npu", dtype=torch.float32)
weight = torch.ones(128, device="npu", dtype=x.dtype)
z = torch.randn_like(x)
buffer = torch.empty_like(x)

# 分组 RMSNorm，仿射后进行 SiLU 门控，复用输出。
y, mean, rstd = layer_norm_fwd(
    x, weight, None, 1e-5,
    z=z, out=buffer, group_size=32,
    norm_before_gate=True, is_rms_norm=True,
    impl="triton",
)
assert y is buffer
assert mean is None
print(rstd.shape)  # torch.Size([16])，4 组 * 4 行

reference, _, _ = layer_norm_fwd(
    x, weight, None, 1e-5,
    z=z, group_size=32, is_rms_norm=True,
    impl="torch_native",
)

# 默认 LayerNorm，保留 mean 和 rstd，默认 auto 对应 Triton。
ln_output, ln_mean, ln_rstd = layer_norm_fwd(x, weight, None, 1e-5)
```
