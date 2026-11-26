# grouped_gemma_rmsnorm

## 硬件支持

由于 arch22 系列成员验证方式不同，此处分开列出。A3 指 Ascend910_93 系列，A2 指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton`（`auto`） | 支持 | 未验证 | 支持 |
| `ascendc` | 支持 | 未验证 | 不支持 |
| `torch_native` | 支持 | 未验证 | 支持 |

`auto` 在 `arch22` 和 `arch35` 上均默认选择 `triton`（固定形状 `x[R, 10240]`、
`weight[10240]`、`group_size=2560`）。显式指定 `impl="ascendc"` 时调用 arch22
CANN/ACLNN 算子 `hyper_conn_grouped_norm`（`SUPPORTED_ARCHS 2201`，覆盖
`ascend910b` 与 `ascend910_93`）；在 arch35 上由注册表直接抛出
`UnsupportedGenerationError`。A2 因无硬件 CI 覆盖标注为“未验证”。

## 接口

省略 `impl` 或传入 `"auto"` 时默认选择 `triton`，显式传入 `impl="ascendc"` 选择 CANN/ACLNN Ascend C 实现。

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

沿最后一维按固定组宽 `group_size` 做独立 RMSNorm，并在 float32 下乘以 `1 + weight`，最终写回时舍入为 BF16：

```text
inv    = rsqrt(mean(x_g^2) + eps)
out[g] = bf16((x_g * inv) * (1 + weight[(g * group_size) % weight.numel()]))
```

`triton`（`auto`）面向 Qwen3.8-Flash-Next 形状 `x[R, 10240]`、`weight[10240]`、`group_size=2560`。
`ascendc` 支持通用组宽（8 的倍数，最大 4096）、共享或分 branch 循环权重、2-D/3-D 连续输入以及零拷贝预分配 `out`。

## 参数与返回值

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., N]`，BF16 | 必填 | NPU 连续张量（`ndim >= 2`），`N` 需整除 `group_size`。 |
| `weight` | `torch.Tensor` | `[W]`，BF16 | 必填 | 与 `x` 同设备的 1-D 连续张量，`W` 为 `group_size` 正整数倍且周期整除总组数。 |
| `group_size` | `int` 或 `None` | — | `None` | `[8, 4096]` 内 8 的倍数（`None` 取 `x.shape[-1]`）；`triton` 固定为 `2560`。 |
| `eps` | `float` | — | `1e-6` | 正有限 float32 稳定项。 |
| `out` | `torch.Tensor` 或 `None` | `x.shape`，BF16 | `None` | 可选连续预分配输出，不可与 `x` 或 `weight` 别名或重叠。 |
| `impl` | `str` | — | `"auto"` | 仅关键字：`auto`、`triton`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `x.shape` | `torch.bfloat16` | 归一化结果；提供 `out` 时原位写入并返回 `out`。 |

## 后端实现

- **`triton`（`auto`）**：arch22 与 arch35 默认后端，支持 NPUGraph 捕获与回放。
- **`ascendc`**：显式 arch22 CANN/ACLNN AIV 后端（`torch.ops.npu.hyper_conn_grouped_norm`），支持通用组宽与零拷贝 `out`，支持 NPUGraph。
- **`torch_native`**：独立 CPU FP64 精度参考实现。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=torch.bfloat16)
y_triton = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6)
y_ascendc = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="ascendc")
```
