# chain_speculative_sampling

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Registered; refactor validation pending | Registered; refactor validation pending |
| `torch_native` | Registered; refactor validation pending | Registered; refactor validation pending |

`auto` selects `triton` without fallback. Registration is a support declaration,
not evidence of device correctness or performance. This migration preserves the
existing Triton computation and launch settings; NPU accuracy and performance
comparisons against the pre-migration implementation remain required.

## Signature

```python
from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

def chain_speculative_sampling(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single=1.0,
    threshold_acc=1.0,
    deterministic=True,
    *,
    impl="auto",
):
    ...
```

## Description

Verifies a **linear** draft chain in non-greedy speculative decoding, after the
target model has produced verification probabilities. SGLang's NPU EAGLE/NEXTN
rejection-sampling path and dense DSpark acceptance path use this operation.
Logit processing, probability normalization, random-number generation and KV
cache updates are performed by the caller.

Let `B` be the actual verification request count, `D` the candidate width including
the root at column 0, `R` the draft probability row count, and `V` the full vocabulary
width. For draft token `x = candidates[b, j]`, `j = 1, ..., D-1`, accept while
`uniform_samples[b, j-1] * draft_probs[b, j-1, x] < target_probs[b, j-1, x]`.
Stop at the first rejection. If `a` drafts were accepted, sample a final token
using row `a`:

- All drafts accepted: weights `w = target_probs[b, a]`.
- A draft rejected: weights `w = max(target_probs[b, a] - draft_probs[b, a], 0)`.

Choose the first token whose cumulative weight is strictly greater than
`uniform_samples_for_final_sampling[b] * sum(w)`. The implementation does not
normalize the residual. The existing end-of-CDF fallback is retained: Triton
uses the last positive entry in the selected vocabulary block, or `V-1` if that
block has no positive entries; the PyTorch reference searches the entire row.

## Parameters and return value

All participating tensors must be contiguous and on the same NPU device.
Inputs are read-only except for the three explicitly marked outputs. `B > 0`
requires `D >= 1` and `R >= max(D-1, 1)`. Use a nonempty vocabulary and valid
token/slot indices; the wrapper does not check index ranges or probability values.
The original spelling `retrive` is retained for keyword compatibility.

| Parameter | Shape / dtype | Meaning |
| --- | --- | --- |
| `predicts` | One-dimensional, usually `[B*D]`, int32 | Output: accepted drafts and the final token, written at slots selected by `retrive_index`. |
| `accept_index` | `[B,D]`, int32 | Output: root slot followed by accepted slots. Valid prefix length is `accept_token_num[b] + 1`. |
| `accept_token_num` | `[B]`, int32 | Output: number of accepted drafts, excluding the final sampled token. SGLang adds one to obtain the decoding advance. |
| `candidates` | `[B,D]`, int64 | Root and draft token IDs. The root token value is not sampled or verified. |
| `retrive_index` | `[B,D]`, int64 | Valid, non-conflicting global output slots; each slot must fit `predicts` and the int32 output indices. |
| `retrive_next_token`, `retrive_next_sibling` | Ignored | Compatibility arguments; do not enable tree traversal. |
| `uniform_samples` | `[B,D]`, float32 | Caller-supplied acceptance samples in `[0,1)`. The last column is unused. |
| `uniform_samples_for_final_sampling` | `[B]`, float32 | Caller-supplied final sampling values in `[0,1)`. |
| `target_probs` | `[B,D,V]`, float32 | Target probabilities after the caller's sampling transforms. |
| `draft_probs` | `[B,R,V]`, float32 | Draft probabilities on the same vocabulary. Extra rows are allowed. |
| `threshold_single`, `threshold_acc` | float, default `1.0` | Ignored compatibility arguments; do not alter acceptance. |
| `deterministic` | bool, default `True` | Ignored compatibility argument; this operator draws no random numbers. |
| `impl` | str, default `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

Returns `None`. The caller allocates and initializes all three output buffers;
unwritten slots keep their previous values. Output buffers must not overlap inputs or one another; disjoint output
views into a shared allocation are allowed. For `B == 0`, Triton returns after checking candidate and target
ranks, before the other parameter checks.

## Implementations

`triton` keeps three sequential kernels: accept a prefix, sum residual weights in
blocks of 2048 vocabulary entries, and sample the final token. Temporary metadata
has shape `[B,3]` and dtype int64; block sums have shape `[B,ceil(V/2048)]` and dtype
float32. The existing shape, dtype, contiguity and same-device checks are retained.

`torch_native` reuses the original repository test's `chain_rejection_reference`
calculation with the public signature. It does not duplicate Triton's input
validation. It uses Python scalar floats for acceptance and whole-row PyTorch
`sum`/`cumsum`, while Triton uses FP32 arithmetic and block reductions. Near
decision boundaries, roundoff can change a discrete token selection. Integer
output comparisons remain exact; a disagreement requires investigation.

The original implementation under `sgl_kernel_npu.sample`, its package exports,
and its original tests remain unchanged. Existing imports of
`chain_speculative_sampling_triton` continue to execute that original copy.
This new architecture has its own implementation copy; new integrations should
use `operators`.

## Test coverage

The unified suite uses independent output buffers, compares all three complete
int32 outputs with the shared accuracy helper, then invokes the shared performance
helper. It preserves both original fixtures and covers these deployment-derived
scenarios in FP32 with model-specific pytest IDs:

| Model | `(D,R,V)` | `B` |
| --- | --- | --- |
| GLM-4.7-Flash | `(4,3,154880)` | 1, 2, 4, 8, 16, 32 |
| Qwen3.5-9B | `(4,3,248320)` | 1, 2, 4, 8, 16 |
| DeepSeek-V4-Flash | `(6,5,129280)`<br>`(7,6,129280)` | 1, 2, 4, 8<br>1, 2, 4, 6, 8, 10, 12 |

Use the standard test CLI described in [the test framework guide](../../../../../tests/python/README.md).
