# schedule_verify_lens_topk_triton

## Hardware support

The `triton` and `torch_native` implementations are registered for arch22 and arch35. Both were tested on Ascend910_9362; other devices require validation. Registration alone is not evidence of hardware support. `auto` selects `triton`.

## Signature

```python
from sgl_kernel_npu.operators.sample import schedule_verify_lens_topk_triton

schedule_verify_lens_topk_triton(*, confidence, budget, cfg, impl="auto")
```

## Description

Computes FP32 cumulative products of the `[B, gamma]` confidence tensor. Among the first `min(gamma, max_len)` positions, selects up to `budget` candidates with survival probability at least `cfg.survival_eps`. Candidates are ordered by descending probability, then ascending position, then ascending request index. For each request, adds its selected count to `cfg.min_verify_len`, clamps to `[max(cfg.min_verify_len, 1), max_len]` and returns a newly allocated `[B]` int32 tensor on the input device. A nonpositive budget selects no candidates. Inputs may be noncontiguous and are not modified.

`cfg` is the SGLang DSpark schedule configuration or a compatible object exposing `min_verify_len`, `survival_eps`, and `resolved_max_verify_len()`. `max_len` is the result of that method. Confidence must be floating point and in `[0, 1]`; the caller is responsible for validating the configuration. The implementation follows the [SGLang scheduler](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py).

| Parameter | Description |
| --- | --- |
| `confidence` | NPU tensor of shape `[B, gamma]`, float16, bfloat16 or float32. |
| `budget` | Python integer, maximum number of candidates to select. |
| `cfg` | DSpark scheduling configuration. |
| `impl` | `auto`, `triton`, or `torch_native`. |

The Triton implementation uses pairwise ranking and therefore has quadratic work in the number of candidates. Floating-point cumulative-product rounding can change the ordering of nearly equal candidates across implementations; exact ties follow the stated tie-break rule.
