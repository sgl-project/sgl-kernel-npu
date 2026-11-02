# schedule_verify_lens_topk_triton

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported (Ascend910_9362 tested) | Supported (Ascend 950 tested) |
| `torch_native` | Supported (Ascend910_9362 tested) | Supported (Ascend 950 tested) |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

`auto` selects the Triton implementation.

```python
from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

def schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto"):
    ...
```

## Description

Schedules DSpark verify lengths from per-token confidence. For request `r` and position `p`, the survival probability is computed in FP32 as:

$$
S_{r,p} = \prod_{j=0}^{p} \operatorname{float32}(\mathrm{confidence}_{r,j}).
$$

For the first `min(gamma, max_len)` positions, select up to `budget` candidates with survival probability at least `cfg.survival_eps`, ordered by descending probability, ascending position, then ascending request index. Let `count[r]` be the number selected for request `r`. The result is `clamp(cfg.min_verify_len + count[r], max(cfg.min_verify_len, 1), max_len)`, cast to int32. `max_len` is `cfg.resolved_max_verify_len()`. A nonpositive budget selects no candidates. The behavior follows the [SGLang scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py).

## Parameters and return value

The caller is responsible for valid inputs and configuration; the public interface forwards arguments without validation.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `confidence` | `torch.Tensor` | `[B, gamma]`, float16, bfloat16, or float32 | Required | Confidence values in `[0, 1]`. Triton requires an NPU tensor; noncontiguous inputs are accepted. |
| `budget` | `int` | — | Required | Maximum number of candidates; nonpositive values select none. |
| `cfg` | object | — | Required | DSpark configuration exposing `min_verify_len`, `survival_eps`, and `resolved_max_verify_len()`. The caller must ensure `0 <= min_verify_len <= max_len <= gamma + 1` and `survival_eps >= 0`. |
| `impl` | `str` | — | `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `verify_lens` | `torch.Tensor` | `[B]`, int32 | Newly allocated tensor on the input device. Inputs are not modified. |

## triton backend

Uses three Triton kernels and accepts noncontiguous inputs by making a contiguous copy. Pairwise ranking has quadratic work in the number of candidates. Cumulative-product rounding can change the order of nearly equal values across implementations.

## torch_native backend

Uses PyTorch cumulative product, stable sorting, and scatter-add as an independent reference. It accepts noncontiguous inputs.

## Example

```python
from dataclasses import dataclass

import torch
import torch_npu

from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton


@dataclass
class Config:
    min_verify_len: int = 1
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self):
        return 4


confidence = torch.ones((2, 3), dtype=torch.float32, device="npu")
kwargs = dict(confidence=confidence, budget=3, cfg=Config())
result = schedule_verify_lens_topk_triton(**kwargs, impl="triton")
reference = schedule_verify_lens_topk_triton(**kwargs, impl="torch_native")
torch.testing.assert_close(result, reference, rtol=0, atol=0)
# result: [3, 2]
```
