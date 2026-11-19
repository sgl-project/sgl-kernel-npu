# kv_rmsnorm_rope_cache

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache


def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    *,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-05,
    cache_mode="Norm",
    is_output_kv=False,
    impl="auto"
): ...
```

## Description

Splits KV into latent and rotary components. It applies $g\odot x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$ to the latent component, rotates the remaining coordinates, then writes both streams into the specified slots.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `kv` | Tensor | [B, 1, S, D + Dr], FP16/BF16 | Required | Input with latent width D followed by rotary width Dr. |
| `gamma` | Tensor | [D], input dtype in example | Required | Latent normalization weight. |
| `cos`, `sin` | Tensor | [B, 1, S, Dr], input dtype in example | Required | Rotary coefficients. |
| `index` | Tensor | [B] for one token per batch, INT64 in example | Required | Physical slots for PA_BNSD. Interpretation depends on cache_mode. |
| `k_cache`, `ckv_cache` | Tensor | [P, L, 1, Dr] and [P, L, 1, D] for PA_BNSD | Required | Rotary and latent cache buffers, updated in place. |
| `k_rope_scale`, `c_kv_scale` | Tensor or None | SDK quantization scale shapes | `None` | Optional cache quantization factors. |
| `k_rope_offset`, `c_kv_offset` | Tensor or None | SDK quantization offset shapes | `None` | Optional cache quantization offsets. |
| `v` | Tensor or None | SDK-specific | `None` | Optional SDK-specific value tensor. Check availability in the installed version. |
| `epsilon` | float | - | `1e-05` | Stability constant for RMS normalization. |
| `cache_mode` | str | - | `'Norm'` | Vendor storage layout and slot interpretation. |
| `is_output_kv` | bool | - | `False` | Requests intermediate rotated and normalized KV tensors when true. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `k_cache`, `ckv_cache`, `rotated`, `normalized` | tuple of four tensors | cache shapes and intermediate shapes | First two results are the updated rotary and latent caches. The last two are valid when is_output_kv=True. |

## torch_npu backend

Calls `torch_npu.npu_kv_rmsnorm_rope_cache`. Shape, dtype and layout combinations follow the installed SDK. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_kv_rmsnorm_rope_cache.md)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import kv_rmsnorm_rope_cache

kv = torch.randn((2, 1, 1, 576), dtype=torch.float16, device="npu")
gamma = torch.ones((512,), dtype=kv.dtype, device=kv.device)
cos = torch.ones((2, 1, 1, 64), dtype=kv.dtype, device=kv.device)
sin = torch.zeros_like(cos)
slots = torch.tensor([19, 0], dtype=torch.int64, device=kv.device)
rotary_cache = torch.zeros((2, 16, 1, 64), dtype=kv.dtype, device=kv.device)
latent_cache = torch.zeros((2, 16, 1, 512), dtype=kv.dtype, device=kv.device)
results = kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    slots,
    rotary_cache,
    latent_cache,
    cache_mode="PA_BNSD",
    is_output_kv=True,
    impl="torch_npu",
)
```
