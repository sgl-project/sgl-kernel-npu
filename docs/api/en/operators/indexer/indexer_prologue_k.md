# indexer_prologue_k

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | Unsupported | Supported |
| torch_native | Unsupported | Supported |

## Signature

`from sgl_kernel_npu.operators.indexer import indexer_prologue_k`

```python
import torch

def indexer_prologue_k(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
    impl="auto"
): ...
```

## Behavior

Projects BF16 latent states, applies RMSNorm and RoPE, quantizes to MXFP4 E2M1, and writes paged K cache.

BF16 `latent[T,H]`, FRACTAL_NZ `wk[D,H]`, FP32 normalization/RoPE inputs, and INT64 `cache_index[T]` are required. A `-1` cache index skips the write. `storage_mode=0` uses a separate scale cache; mode 1 combines rows according to `combined_block_size`.

Semantics (BF16 rounding at quantized compute boundaries):

```text
p = BF16(latent @ wk.T); k = RoPE(BF16(p / sqrt(mean(p**2) + norm_eps) * norm_weight)); cache = MXFP4(k)
```

Each group of 32 values shares a power-of-two scale `s = 2**ceil(log2(max(amax, 6 * 2**-126) / 6))`. Magnitudes are mapped to E2M1 levels `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with no zero point; midpoint ties select the smaller magnitude and values saturate at 6. The first value occupies the low nibble and the second the high nibble. The E8M0 scale byte stores `log2(s) + 127`.

## Parameters

T is query rows, B requests, P physical pages and S slots per page.

| Parameter | Default | Type, shape and meaning |
| --- | --- | --- |
| `latent` | Required | BF16 [T,H], compressor latent before RoPE. |
| `wk` | Required | BF16 [D,H], projection weight; FRACTAL_NZ on the DSL path. |
| `norm_weight` | Required | FP32 [D], RMSNorm weight. |
| `rope_sin` | Required | FP32 [T,Dr], sine coefficients for the last Dr features. |
| `rope_cos` | Required | FP32 [T,Dr], cosine coefficients for the last Dr features. |
| `k_cache` | Required | UINT8 writable cache: mode 0 [P,S,1,D/2]; mode 1 [P,S/G,1,G*(D/2+D/32)]. |
| `k_scale_cache` | `None` | Optional UINT8 [P,S,1,D/32], separate E8M0 scales for mode 0. |
| `cache_index` | Required | INT64 [T], flattened cache slots; -1 skips a row. |
| `storage_mode` | Required | int: 0 for separate data/scales, 1 for combined storage. |
| `norm_eps` | Required | float, RMSNorm epsilon. |
| `combined_block_size` | `-1` | int G, rows per combined record; mode 1 requires a positive divisor of S. |
| `impl` | `'auto'` | str, auto selects cannbot_dsl; torch_native is an independent compute reference. |

## Returns

| Value | Meaning |
| --- | --- |
| k_cache | The same updated tensor; separate scale cache is also updated. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and a compatible CANNBot-DSL runtime installed in the environment. This operator does not enforce a runtime version. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

## torch_native

Independent CPU Torch accuracy reference; results are copied back to the input device, preserving in-place semantics. Does not require DSL. Not intended for performance or NPU graph capture.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import indexer_prologue_k

x = torch.randn(1, 512, device="npu", dtype=torch.bfloat16)
w = torch_npu.npu_format_cast(
    torch.randn(128, 512, device="npu", dtype=torch.bfloat16),
    torch_npu.Format.FRACTAL_NZ,
)
cache = torch.zeros(1, 64, 1, 64, device="npu", dtype=torch.uint8)
scale = torch.zeros(1, 64, 1, 4, device="npu", dtype=torch.uint8)
indexer_prologue_k(
    x,
    w,
    torch.ones(128, device="npu"),
    torch.zeros(1, 64, device="npu"),
    torch.ones(1, 64, device="npu"),
    cache,
    scale,
    cache_index=torch.tensor([0], device="npu", dtype=torch.int64),
    storage_mode=0,
    norm_eps=1e-6,
)
```

## Device test

```bash
python -m pytest tests/python/operators/indexer/test_indexer_prologue_k.py --npu-device=0 --enable-perf=false -v -s
```
