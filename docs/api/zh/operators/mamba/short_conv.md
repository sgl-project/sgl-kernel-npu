# short_conv

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.mamba import short_conv

def short_conv(conv_input, weight, dilation, *, single_token=False, impl="auto"):
    ...
```

## 说明

PLE 深度卷积：输入为 [B,10240,9+W]，W=1..4，权重为 [10240,1,4]，两者使用相同 BF16/FP16 类型。dilation 必须为 3。输出连续 [B,W,10240]；single_token=True 只允许 W=1，输出 [B,10240]。状态更新和 SiLU 由框架负责。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

`out[b,t,c] = sum_j conv_input[b,c,t+3*j] * weight[c,0,j]`，j=0..3。乘积在 FP32 中累加，最后转换为输入类型。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `conv_input` | Tensor | [B,10240,9+W], BF16/FP16 | required | 连续 NPU 张量；W=1..4。 |
| `weight` | Tensor | [10240,1,4], conv_input.dtype | required | 同设备连续张量。 |
| `dilation` | int | — | required | 必须为 3。 |
| `single_token` | bool | — | False | 仅 W=1；去掉 token 维。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [B,W,10240] or [B,10240] | conv_input.dtype | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mamba import short_conv

x = torch.randn(1, 10240, 13, device="npu", dtype=torch.bfloat16)
w = torch.randn(10240, 1, 4, device="npu", dtype=x.dtype)
y = short_conv(x, w, 3, impl="auto")
```
