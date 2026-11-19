# mrope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Unsupported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.rope import mrope


def mrope(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default",
    impl="auto"
): ...
```

## Description

Looks up rotary coefficients by position and rotates the rotary prefix of each query and key head. Multimodal positions select separate cache rows for the axis sections before the standard RoPE operation $x\odot\cos+\mathrm{rotate}(x)\odot\sin$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `positions` | Tensor | [T] or [axes, T], INT32/INT64 | Required | Single-axis or multimodal position IDs within the cache. |
| `query`, `key` | Tensor | [T, H * head_size], FP16/BF16 | Required | Packed query/key heads. The two head counts can differ. |
| `cos_sin_cache` | Tensor | [max_seq_len, rotary_dim], floating | Required | Trig cache. In the default layout each row stores cosine frequencies followed by sine frequencies. |
| `head_size` | int | - | Required | Width of each packed query/key head. |
| `mrope_section` | list[int] or None | - | `None` | Splits half the rotary frequencies among multimodal axes. |
| `rotary_mode` | str | - | `'half'` | Rotary pairing mode. half is the default. |
| `cache_mode` | str | - | `'default'` | Trig cache layout. Vendor support for other layouts depends on the installed SDK. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`, `torch_native`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `query_out`, `key_out` | tuple[Tensor, Tensor] | corresponding input shape and dtype | Rotated query and key on the input device. |

## torch_npu backend

Calls `torch_npu.npu_mrope`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_mrope.md)

## torch_native backend

Implements segmented cache lookup for cache_mode='default' and half/interleaved rotary pairing.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.rope import mrope

pos = torch.arange(7, dtype=torch.int64, device="npu")
q = torch.randn((7, 4 * 128), dtype=torch.float16, device="npu")
k = torch.randn((7, 128), dtype=q.dtype, device=q.device)
theta = torch.zeros((9, 64), dtype=q.dtype, device=q.device)
cache = torch.cat((theta.cos(), theta.sin()), dim=-1)
q_out, k_out = mrope(pos, q, k, cache, 128, impl="torch_npu")
```
