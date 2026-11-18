# chunk_local_cumsum

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

def chunk_local_cumsum(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float32, *, impl='auto'):
    ...
```

默认 `auto` 与 `triton` 注册到同一入口；不自动回退到参考实现。

## 功能说明

在每个序列块内独立计算 $o_t=s\sum_{j\le t}g_j$；反向模式累加 $j\ge t$，scale=None 时不缩放。

## 参数与返回值

Triton 张量输入应位于同一 NPU 并采用连续布局；输出位于输入设备。

| 参数 | 类型 | 形状 / 数据类型 | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `g` | torch.Tensor | `[B,T,H] 或 [B,H,T]; float16/bfloat16/float32` | 必填 | 对数域标量门控；累加算子也接受 FP16/BF16。 |
| `chunk_size` | int | `—` | 必填 | 正的二次幂块长；GDN 使用 64。 |
| `reverse` | bool | `—` | `False` | 在块内沿时间维反向累加。 |
| `scale` | float 或 None | `—` | `None` | 累加后的可选乘数。 |
| `cu_seqlens` | torch.Tensor 或 None | `[N+1]; int32/int64` | `None` | 累计序列偏移；变长输入须展平且 B=1。 |
| `head_first` | bool | `—` | `False` | 将 g 解释为 [B,H,T]。 |
| `output_dtype` | torch.dtype 或 None | `—` | `torch.float32` | 输出类型；限制见后端说明。 |
| `impl` | str | `—` | `'auto'` | 仅限关键字：auto、triton 或 torch_native。 |

| 返回值 | 类型 | 形状 / 数据类型 | 描述 |
| --- | --- | --- | --- |
| `output` | torch.Tensor 或 None | `g.shape; output_dtype or g.dtype` | 新建的块内累加结果。 |

## triton 后端

使用 FP32 累加，写入前转为输入 dtype。输出类型必须等于 g.dtype：低精度输入须传 None 或 g.dtype；默认 FP32 仅匹配 FP32 输入。

公开适配层拒绝 Triton 的 head_first=True 与 cu_seqlens 组合。令 BLOCK_T 为 floor(2**18/(H*chunk_size)) 向上取二次幂的结果，要求商为正且 BLOCK_T 不小于 chunk_size；头维优先布局还要求 T 能被 BLOCK_T 整除，不支持时间轴尾块。上述 Triton 限制不适用于 torch_native；各维度均须非空。

## torch_native 后端

通过独立 PyTorch 运算生成参考结果；主机元数据传输可能触发同步，仅用于精度对比。

## 调用示例

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

g = torch.randn(1, 128, 8, device="npu", dtype=torch.float16)
y = chunk_local_cumsum(g, 64, output_dtype=None, impl="triton")
```
