# markov_greedy

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

The default `impl="auto"` selects the same implementation as `"triton"`.
Registration does not establish hardware validation; the table records the
currently verified status.

## Signature

```python
from sgl_kernel_npu.operators.sample import markov_greedy

def markov_greedy(*, base_logits, prev_embeds, w2_weight, impl="auto"):
    ...
```

## Description

Projects `prev_embeds` with `w2_weight`, adds the projection to `base_logits`,
and returns the maximum index for each row. The computation follows
`(base_logits + torch.nn.functional.linear(prev_embeds, w2_weight)).argmax(-1)`,
including the projection and addition dtype rounding. Ties select the lowest
vocabulary index, and the first NaN takes precedence over finite values.

## Parameters and return value

All tensor arguments must be on the same NPU. Use compatible floating-point
dtypes for the linear projection; FP32, FP16 and BF16 are covered by the
implementation.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `base_logits` | `torch.Tensor` | `[B, V]`, floating point | Required | Base logits; `V > 0` and unit column stride. |
| `prev_embeds` | `torch.Tensor` | `[B, H]`, floating point | Required | Previous token embeddings. |
| `w2_weight` | `torch.Tensor` | `[V, H]`, floating point | Required | Linear projection weight; unit column stride. |
| `impl` | `str` | - | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`, `torch.int64` | Newly allocated vocabulary indices on the input device. `B` may be zero. |

## triton backend

Uses the Torch linear projection and a Triton add-and-reduce path. For batch
sizes from 16 through 31, the current implementation uses Torch addition and
argmax after the projection.

## torch_native backend

Uses Torch linear projection, addition and argmax.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import markov_greedy

base = torch.randn(2, 1024, device="npu", dtype=torch.float16)
embeds = torch.randn(2, 128, device="npu", dtype=torch.float16)
weight = torch.randn(1024, 128, device="npu", dtype=torch.float16)
tokens = markov_greedy(base_logits=base, prev_embeds=embeds, w2_weight=weight)
```
