# hc_combine

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    ...
```

## 说明

HC 残差合并：连续 BF16 block[R,2560]、residual/normed[R,10240]、weight[4,10240]，hc=4、hs=2560。门控为 2*sigmoid(normed@weight.T/4)，逐分支乘以 block 后加到 residual；输出连续 BF16[R,10240]。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `block` | Tensor | [R,2560], BF16 | required | 连续 NPU 输入。 |
| `residual / normed` | Tensor | [R,10240], BF16 | required | block 同设备连续输入。 |
| `weight` | Tensor | [4,10240], BF16 | required | block 同设备连续权重。 |
| `hc / hs` | int | — | required | 4 / 2560。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,10240] | BF16 | 输入设备；新分配的连续张量。 |

## triton

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(1, 2560, device="npu", dtype=residual.dtype)
weight = torch.zeros(4, 10240, device="npu", dtype=residual.dtype)
y = hc_combine(block, residual, residual, weight, 4, 2560, impl="auto")
```
