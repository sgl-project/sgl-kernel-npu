# markov_sample_step_triton

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

The default `impl="auto"` selects the same implementation as `"triton"`.
Registration does not establish hardware validation; the table records the
currently verified status.

## Signature

```python
from sgl_kernel_npu.operators.sample import markov_sample_step_triton

def markov_sample_step_triton(
    *, base_logits, bias, temperatures, key, offset, step, step_dtype,
    vocab_start=0, gather=None, gather_logits=None, logits_out=None,
    has_sampled_rows=None, exp_noise=None, impl="auto"
):
    ...
```

## Description

Adds `bias` to the valid columns of `base_logits`, rounds the resulting logits
to `step_dtype`, and returns one global vocabulary index per row. A zero
temperature selects the maximum logit; a nonzero temperature selects the maximum
of `logit / temperature - log(noise)`, where noise is an exponential sample.
Padding columns beyond the width of `bias` are excluded from selection. With
Philox noise, `key`, `offset`, `step`, row, and global vocabulary index determine
the noise stream. Alternatively, `exp_noise` supplies precomputed noise.

## Parameters and return value

Use NPU tensors on the same device. The public operator forwards arguments to
the selected backend; the table describes the implementation requirements.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `base_logits` | `torch.Tensor` | `[B, P]`, floating point | Required | Padded local logits, with `B > 0`, `P > 0` and unit column stride. |
| `bias` | `torch.Tensor` | `[B, V]`, floating point | Required | Bias for the first `V` columns, where `0 <= V <= P`; unit column stride. |
| `temperatures` | `torch.Tensor` | `[B]`, `torch.float32` | Required | Per-row temperature; contiguous. Zero selects greedy decoding. Use nonnegative finite values. |
| `key` | `int` | - | Required | Philox key; ignored when `exp_noise` is supplied. |
| `offset` | `torch.Tensor` | One element, `torch.int64` | Required | Philox offset on the input device; still required when using `exp_noise`. |
| `step` | `int` | - | Required | Philox step; ignored when `exp_noise` is supplied. |
| `step_dtype` | `torch.dtype` | - | Required | Rounding dtype: `torch.float32`, `torch.float16`, or `torch.bfloat16`. |
| `vocab_start` | `int` | - | `0` | Global start index of the local shard; must be divisible by 4. |
| `gather` | callable or `None` | - | `None` | Exchanges per-rank candidate tensors for sharded selection. It returns candidates stacked along the batch dimension; candidate layout is backend-dependent. |
| `gather_logits` | callable or `None` | - | `None` | Gathers local logits along the vocabulary dimension; required when both `gather` and `logits_out` are provided. |
| `logits_out` | `torch.Tensor` or `None` | `[B, P]` locally, `[B, V_global]` when gathered; `step_dtype` | `None` | Optional output buffer mutated in place with rounded logits; unit column stride. |
| `has_sampled_rows` | `bool` or `None` | - | `None` | Set to `False` only when every row is greedy; skips noise generation. |
| `exp_noise` | `torch.Tensor` or `None` | Same shape as local logits, or gathered `logits_out`; `torch.float32` | `None` | Positive precomputed exponential samples on the input device, with unit column stride. The caller owns the RNG sequence. |
| `impl` | `str` | - | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `tokens` | `torch.Tensor` | `[B]`, `torch.int64` | Newly allocated global token IDs on the input device. An empty local valid shard contributes the sentinel `2147483647`. |

For distributed selection without `logits_out`, `gather` exchanges each
backend's candidate buffer. When retaining gathered logits, `gather_logits`
returns the concatenated logits, and `exp_noise`, if supplied, has the shape
of `logits_out`. The caller must ensure callbacks and shard layouts are
compatible with the selected backend.

## triton backend

Uses Triton kernels for per-block selection and final reduction. With
`exp_noise=None`, it generates Philox noise inside the kernel. Supplying
precomputed noise bypasses that generation; refreshing noise is the caller's
responsibility, including during graph replay.

## torch_native backend

Uses Torch operations and the same Philox counter protocol or supplied noise.
This implementation can be used as a reference for the Triton backend.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import markov_sample_step_triton

base = torch.randn(2, 1024, device="npu", dtype=torch.float32)
bias = torch.randn_like(base)
temperatures = torch.tensor([0.0, 1.0], device="npu", dtype=torch.float32)
offset = torch.zeros(1, device="npu", dtype=torch.int64)
tokens = markov_sample_step_triton(base_logits=base, bias=bias, temperatures=temperatures,
                       key=1234, offset=offset, step=0, step_dtype=torch.float32)
```
