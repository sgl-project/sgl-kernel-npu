# hc_mix

## Hardware support

arch22 is split because the two members of that family are validated
differently. A3 means Ascend910_93 series, A2 means Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton` (`auto`) | Supported | Unverified | Supported |
| `ascendc` | Supported | Unverified | Unsupported |
| `torch_native` | Supported | Unverified | Supported |

`auto` selects `triton` on both `arch22` and `arch35` (`hc=4, hs=2560,
low_rank=320`). Explicit `impl="ascendc"` uses the CANN/ACLNN AIV operators
`hyper_conn_scaled_silu` and `hyper_conn_branch_mix` (`SUPPORTED_ARCHS 2201`)
with BF16 `F.linear` projections; arch35 rejects `impl="ascendc"` via
`UnsupportedGenerationError` in the registry.

## Signature

Omitting `impl` or passing `"auto"` selects `triton`. Select the Ascend C path
explicitly with `impl="ascendc"`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    ...
```

## Description

Compute the low-rank sigmoid-gated branch average over a packed `[rows, hc * hs]`
normalized residual:

```text
t   = (x @ down.T) / hc
h   = t * sigmoid(t)
g   = sigmoid(h @ up.T)
out = sum_c (g[:, c, :] * x[:, c, :]) / hc
```

- **`triton` (`auto`)**: keeps intermediate projections and activations in FP32
  and rounds only the final output to BF16 (`hc=4, hs=2560, low_rank=320`).
- **`ascendc`**: runs `F.linear` projections in BF16 (accepting both ND and
  pre-converted `FRACTAL_NZ` `up` weights) and evaluates each AIV vector stage
  (`hyper_conn_scaled_silu`, `hyper_conn_branch_mix`) in FP32, rounding to BF16
  at stage output boundaries. Supports power-of-two `hc` in `[1, 8]` and
  multiples of 16 up to 4096 for `hs` and `low_rank`.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[R, hc * hs]`, BF16 | Required | Contiguous NPU input. |
| `down` | `torch.Tensor` | `[low_rank, hc * hs]`, BF16 | Required | Contiguous on `x.device`; `low_rank` multiple of 16 in `[16, 4096]`. |
| `up` | `torch.Tensor` | `[hc * hs, low_rank]`, BF16 | Required | Contiguous on `x.device`; `ascendc` also accepts pre-converted `FRACTAL_NZ`. |
| `hc` | `int` | — | Required | Power of two in `[1, 8]` (`triton` requires `4`). |
| `hs` | `int` | — | Required | Multiple of 16 in `[16, 4096]` (`triton` requires `2560`). |
| `impl` | `str` | — | `"auto"` | Keyword-only: `"auto"`, `"triton"`, `"ascendc"`, or `"torch_native"`. |

| Return | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, hs]` | `torch.bfloat16` | Newly allocated contiguous output on `x.device`. |

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_mix

x = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
down = torch.zeros(320, 10240, device="npu", dtype=torch.bfloat16)
up = torch.zeros(10240, 320, device="npu", dtype=torch.bfloat16)
y_triton = hc_mix(x, down, up, 4, 2560)
y_ascendc = hc_mix(x, down, up, 4, 2560, impl="ascendc")
```
