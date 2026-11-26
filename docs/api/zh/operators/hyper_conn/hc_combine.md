# hc_combine

## 硬件支持

由于 arch22 系列成员验证方式不同，此处分开列出。A3 指 Ascend910_93 系列，A2 指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton`（`auto`） | 支持 | 未验证 | 支持 |
| `ascendc` | 支持 | 未验证 | 不支持 |
| `torch_native` | 支持 | 未验证 | 支持 |

`auto` 在 `arch22` 和 `arch35` 上均默认选择 `triton`（`hc=4, hs=2560`）。
显式指定 `impl="ascendc"` 时采用 BF16 `F.linear` 投影加 arch22 CANN/ACLNN AIV
算子 `hyper_conn_branch_combine`（`SUPPORTED_ARCHS 2201`）；在 arch35 上由注册表直接抛出
`UnsupportedGenerationError`。

## 接口

省略 `impl` 或传入 `"auto"` 默认选择 `triton`，显式传入 `impl="ascendc"` 选择 Ascend C 实现。

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(
    block: torch.Tensor,
    residual: torch.Tensor,
    normed: torch.Tensor,
    weight: torch.Tensor,
    hc: int,
    hs: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## 功能说明

根据 `normed` 投影各分支注入门控，将门控后的 `block` 广播叠加到 `residual` 的每个分支：

```text
gate = 2 * sigmoid((normed @ weight.T) / hc)
out[r, c * hs + d] = residual[r, c * hs + d] + gate[r, c] * block[r, d]
```

- **`triton`（`auto`）**：中间投影保持 FP32，仅最终输出舍入为 BF16（固定 `hc=4, hs=2560`）。
- **`ascendc`**：`F.linear` 投影以 BF16 执行，`torch.ops.npu.hyper_conn_branch_combine` 内部以 FP32 计算并在写回 `out` 时舍入为 BF16。支持 `[1, 8]` 内 2 的幂次 `hc`、`[16, 4096]` 内 16 倍数 `hs` 以及零拷贝预分配 `out`。

## 参数与返回值

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `block` | `torch.Tensor` | `[R, hs]`，BF16 | 必填 | NPU 连续输入。 |
| `residual` | `torch.Tensor` | `[R, hc * hs]`，BF16 | 必填 | 同设备连续残差。 |
| `normed` | `torch.Tensor` | `[R, hc * hs]`，BF16 | 必填 | 同设备连续归一化残差。 |
| `weight` | `torch.Tensor` | `[hc, hc * hs]`，BF16 | 必填 | 同设备连续注入权重。 |
| `hc` | `int` | — | 必填 | `[1, 8]` 内 2 的幂（`triton` 固定为 `4`）。 |
| `hs` | `int` | — | 必填 | `[16, 4096]` 内 16 的倍数（`triton` 固定为 `2560`）。 |
| `out` | `torch.Tensor` 或 `None` | `[R, hc * hs]`，BF16 | `None` | 可选连续预分配输出，不可与任何输入别名或重叠。 |
| `impl` | `str` | — | `"auto"` | 仅关键字：`auto`、`triton`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, hc * hs]` | `torch.bfloat16` | 合并结果；提供 `out` 时原位写入并返回 `out`。 |

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(32, 2560, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(4, 10240, device="npu", dtype=torch.bfloat16)
y_triton = hc_combine(block, residual, residual, weight, 4, 2560)
y_ascendc = hc_combine(block, residual, residual, weight, 4, 2560, impl="ascendc")
```
