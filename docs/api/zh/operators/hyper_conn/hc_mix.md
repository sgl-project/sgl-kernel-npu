# hc_mix

## 硬件支持

由于 arch22 系列成员验证方式不同，此处分开列出。A3 指 Ascend910_93 系列，A2 指 Ascend910B 系列。

| 实现 | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton`（`auto`） | 支持 | 未验证 | 支持 |
| `ascendc` | 支持 | 未验证 | 不支持 |
| `torch_native` | 支持 | 未验证 | 支持 |

`auto` 在 `arch22` 和 `arch35` 上均默认选择 `triton`（`hc=4, hs=2560, low_rank=320`）。
显式指定 `impl="ascendc"` 时调用 arch22 CANN/ACLNN AIV 算子
`hyper_conn_scaled_silu` 与 `hyper_conn_branch_mix`（`SUPPORTED_ARCHS 2201`）配合
BF16 `F.linear` 投影；在 arch35 上由注册表直接抛出 `UnsupportedGenerationError`。

## 接口

省略 `impl` 或传入 `"auto"` 默认选择 `triton`，显式传入 `impl="ascendc"` 选择 Ascend C 实现。

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    ...
```

## 功能说明

对 `[rows, hc * hs]` 归一化残差执行低秩门控与多分支加权平均：

```text
t   = (x @ down.T) / hc
h   = t * sigmoid(t)
g   = sigmoid(h @ up.T)
out = sum_c (g[:, c, :] * x[:, c, :]) / hc
```

- **`triton`（`auto`）**：中间投影与激活保持 FP32，仅最终输出舍入为 BF16（固定 `hc=4, hs=2560, low_rank=320`）。
- **`ascendc`**：`F.linear` 投影以 BF16 执行（`up` 支持 ND 或预转换的 `FRACTAL_NZ`），各 AIV 向量阶段内部以 FP32 计算并在阶段输出边界舍入为 BF16。支持 `[1, 8]` 内 2 的幂次 `hc` 以及 `[16, 4096]` 内 16 倍数的 `hs` 与 `low_rank`。

## 参数与返回值

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[R, hc * hs]`，BF16 | 必填 | NPU 连续输入。 |
| `down` | `torch.Tensor` | `[low_rank, hc * hs]`，BF16 | 必填 | 同设备连续权重；`low_rank` 为 `[16, 4096]` 内 16 的倍数。 |
| `up` | `torch.Tensor` | `[hc * hs, low_rank]`，BF16 | 必填 | 同设备连续权重；`ascendc` 另支持预转换 `FRACTAL_NZ`。 |
| `hc` | `int` | — | 必填 | `[1, 8]` 内 2 的幂（`triton` 固定为 `4`）。 |
| `hs` | `int` | — | 必填 | `[16, 4096]` 内 16 的倍数（`triton` 固定为 `2560`）。 |
| `impl` | `str` | — | `"auto"` | 仅关键字：`auto`、`triton`、`ascendc`、`torch_native`。 |

| 返回值 | 类型 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, hs]` | `torch.bfloat16` | 新分配的同设备连续输出。 |

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_mix

x = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
down = torch.zeros(320, 10240, device="npu", dtype=torch.bfloat16)
up = torch.zeros(10240, 320, device="npu", dtype=torch.bfloat16)
y_triton = hc_mix(x, down, up, 4, 2560)
y_ascendc = hc_mix(x, down, up, 4, 2560, impl="ascendc")
```
