# hc_mix

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    ...
```

## 说明

HC 门控混合：连续 BF16 x[R,10240]、down[320,10240]、up[10240,320]，hc=4、hs=2560。使用 FP32 投影和 SiLU(x@down.T/4)，再对四个分支做 sigmoid 门控平均，输出 BF16[R,2560]。SiLU 复用公共 fused_sigmoid_mul。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

令 `h = SiLU(x @ down.T / 4)`、`g = sigmoid(h @ up.T)`，将 x 和 g 变为 `[R,4,2560]`，输出 `sum(g*x, dim=1)/4`。中间值为 FP32，仅输出舍入到 BF16。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [R,10240], BF16 | required | 连续 NPU 输入。 |
| `down` | Tensor | [320,10240], BF16 | required | x 同设备连续权重。 |
| `up` | Tensor | [10240,320], BF16 | required | x 同设备连续权重。 |
| `hc / hs` | int | — | required | 4 / 2560。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,2560] | BF16 | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_mix

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
down = torch.zeros(320, 10240, device="npu", dtype=x.dtype)
up = torch.zeros(10240, 320, device="npu", dtype=x.dtype)
y = hc_mix(x, down, up, 4, 2560, impl="auto")
```
