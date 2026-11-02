# chunk_local_cumsum

## 硬件支持

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 未验证 | 未验证 |
| `torch_native` | 未验证 | 未验证 |

## Signature

```python
from sgl_kernel_npu.operators.attention import chunk_local_cumsum
def chunk_local_cumsum(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float32, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

按序列分块沿时间维正向或反向累加，可选缩放。累加使用 FP32 中间值，写入前转换为输入 dtype。Triton 输出 dtype 必须与输入 g.dtype 相同：默认 output_dtype=torch.float32 适用于 FP32 输入；FP16/BF16 输入必须传 output_dtype=None 或显式传 g.dtype。不同 dtype 会在公开接口处抛出 ValueError。

## 参数与返回值

g: [B,T,H], or [B,H,T] when head_first; cu_seqlens: [N+1], batch must be one when provided; chunk_size: positive power of two. Returns a new tensor with the input shape.

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import chunk_local_cumsum

g = torch.randn(1, 128, 8, device="npu", dtype=torch.float16)
y = chunk_local_cumsum(g, 64, output_dtype=None, impl="triton")
```
