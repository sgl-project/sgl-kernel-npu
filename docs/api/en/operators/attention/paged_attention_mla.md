# paged_attention_mla

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |

## Signature

`auto` selects `torch_npu` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import paged_attention_mla


def paged_attention_mla(
    query,
    key_cache,
    num_kv_heads,
    num_heads,
    scale_value,
    block_table,
    context_lens,
    mla_vheadsize,
    out,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Decodes against the paged MLA cache. The complete latent-plus-rotary key participates in $\mathrm{softmax}(sQK^\top)$. The leading mla_vheadsize latent values of that cache supply $V$.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [T, H, D], FP16/BF16 | Required | One row per query token on NPU. |
| `key_cache` | Tensor | [P, L, Hkv, Dk], input dtype | Required | Paged latent and RoPE cache. Dk includes both components. |
| `num_heads`, `num_kv_heads` | int | - | Required | Query and KV head counts. H must be divisible by Hkv. |
| `scale_value` | float | - | Required | Attention logit scale, commonly Dk**-0.5. |
| `block_table` | Tensor | [B, max_pages], INT32, NPU | Required | Maps logical pages of each request to physical pages. |
| `context_lens` | Tensor | [B], INT32 CPU in the example | Required | Number of visible cached tokens per request. |
| `out` | Tensor | [T, H, Dv], same dtype and NPU as query | Required | Caller-owned output buffer, written in place. |
| `mla_vheadsize` | int | - | Required | Width of the latent value prefix of each cache row and the output head. |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `result` | None | - | The output is written into out. |

## torch_npu backend

Calls `torch_npu._npu_paged_attention_mla`. Shape, dtype and layout combinations follow the installed SDK. Requires the SDK's private ATB paged MLA binding. The example uses a 512-value latent prefix and a 64-value rotary tail. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_atb_paged_attention_mla.py)

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_attention_mla

heads, dk, dv = 4, 576, 512
q = torch.randn((4, heads, dk), dtype=torch.float16, device="npu")
cache = torch.randn((64, 16, 1, dk), dtype=q.dtype, device=q.device)
table = torch.arange(4, dtype=torch.int32, device=q.device).reshape(4, 1)
lengths = torch.full((4,), 4, dtype=torch.int32, device="cpu")
out = torch.empty((4, heads, dv), dtype=q.dtype, device=q.device)
paged_attention_mla(
    q, cache, 1, heads, dk**-0.5, table, lengths, dv, out, impl="torch_npu"
)
```
