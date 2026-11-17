# chunk_scaled_dot_kkt

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_scaled_dot_kkt

def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

块内计算 $A_{ij}=\beta_i(k_i^\mathsf{T}k_j)\exp(g_i-g_j)$，仅保留 $i>j$ 且门控差不大于零的元素，其余置零。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `k` | torch.Tensor | `[B,T,Hg,K]; float16/bfloat16` | 必填 | 分组键头；H 必须能被 Hg 整除。 |
| `beta` | torch.Tensor | `[B,T,H]; k.dtype` | 必填 | 逐 token 更新权重。 |
| `g_cumsum` | torch.Tensor | `[B,T,H]; float32` | `None` | 块内累计对数门控；当前后端要求提供。 |
| `cu_seqlens` | torch.Tensor 或 None | `[N+1]; int32/int64` | `None` | 累计序列偏移；变长输入须展平且 B=1。 |
| `chunk_size` | int | `—` | `64` | 正的二次幂块长；GDN 使用 64。 |
| `output_dtype` | torch.dtype 或 None | `—` | `torch.float32` | 输出类型；限制见后端说明。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `A` | torch.Tensor 或 None | `[B,T,H,chunk_size]; output_dtype` | 严格下三角分块乘积，填充列置零。 |

## triton 后端

使用 FP32 累加并转为 output_dtype；必须提供 g_cumsum。输入须连续，块内累计门控应单调不增。

## torch_native 后端

通过独立 PyTorch 运算生成参考结果；主机元数据传输可能触发同步，仅用于精度对比。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_scaled_dot_kkt

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float16)
b = torch.rand(1, 64, 4, device="npu", dtype=torch.float16)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
y = chunk_scaled_dot_kkt(k, b, g, impl="triton")
```
