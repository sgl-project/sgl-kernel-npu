# mla_prolog_v3

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import mla_prolog_v3


def mla_prolog_v3(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Runs the V3 Multi-head Latent Attention prolog. It projects token activations into query and KV paths, applies RMS normalization and RoPE, then writes the compressed latent and rotary keys into the requested cache slots.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `token_x` | Tensor | [B, S, He] or SDK packed variant | Required | Input token activations. The example uses BF16. |
| `weight_dq` | Tensor | [He, Hcq] | Required | Query down-projection matrix. The example uses BF16 FRACTAL_NZ. |
| `weight_uq_qr` | Tensor | [Hcq, H * (D + Dr)] | Required | Query up-projection matrix, including rotary width. The example uses BF16 FRACTAL_NZ. |
| `weight_uk` | Tensor | [H, D, Hckv] | Required | Per-head latent query projection, BF16 ND in the example. |
| `weight_dkv_kr` | Tensor | [He, Hckv + Dr] | Required | Combined KV down-projection and rotary-key matrix, BF16 FRACTAL_NZ in the example. |
| `rmsnorm_gamma_cq`, `rmsnorm_gamma_ckv` | Tensor | [Hcq], [Hckv], BF16 in the example | Required | Learned normalization weights for the two latent paths. |
| `rope_sin`, `rope_cos` | Tensor | [B, S, Dr], BF16 in the example | Required | Sine and cosine factors. |
| `kv_cache`, `kr_cache` | Tensor | [P, L, 1, Hckv] and [P, L, 1, Dr] for PA_BSND | Required | Caller-owned latent and rotary key caches, updated in place. |
| `cache_index`, `cache_mode` | Tensor or None, str | [B, S], INT64 for the example | `None` / `"PA_BSND"` | Optional cache slots and SDK layout. The example uses PA_BSND. |
| `rmsnorm_epsilon_cq`, `rmsnorm_epsilon_ckv` | float | - | `1e-05` | Optional normalization epsilons. The SDK defaults are both 1e-5. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `query`, `query_rope`, `dequant_scale_q_nope`, `query_norm`, `dequant_scale_q_norm` | tuple of five tensors | Shapes and dtypes follow projection and quantization modes | Projected query, rotated query, dequantization scale, optional normalized query and its scale. The two caches are mutated separately. |

## ops-transformer backend

Calls `torch_npu.npu_mla_prolog_v3`. Shape, dtype and layout combinations follow the installed SDK. The V3 result tuple and cache writes follow the installed TorchNPU ABI. Pass additional quantization attributes through the SDK keyword arguments. The example uses the A3 BF16 PA_BSND configuration. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/en/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/mla_prolog_v3)

## torch_native backend

Evaluates unquantized BF16 projections, normalization, RoPE, and updates separate paged KV and rotary caches. Weights are read in logical matrix order.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import mla_prolog_v3

b, seq, h, hidden, cq, ckv, d, dr = 8, 2, 32, 7168, 1536, 512, 128, 64


def bf16(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, device="npu")


def nz(*shape):
    return torch_npu.npu_format_cast(bf16(*shape), 29)


x = bf16(b, seq, hidden)
sin = torch.zeros_like(bf16(b, seq, dr))
cos = torch.ones_like(sin)
kv_cache = bf16(64, 128, 1, ckv)
kr_cache = bf16(64, 128, 1, dr)
slots = torch.arange(b * seq, dtype=torch.int64, device=x.device).reshape(b, seq)
out = mla_prolog_v3(
    x,
    nz(hidden, cq),
    nz(cq, h * (d + dr)),
    bf16(h, d, ckv),
    nz(hidden, ckv + dr),
    bf16(cq),
    bf16(ckv),
    sin,
    cos,
    kv_cache,
    kr_cache,
    cache_index=slots,
    cache_mode="PA_BSND",
    impl="ops-transformer",
)
```
