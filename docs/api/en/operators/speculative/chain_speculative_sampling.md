# chain_speculative_sampling

## Hardware support

| Implementation | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` option selects the same entry point as `triton`.

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

Verifies a linear draft chain during non-greedy speculative decoding. For each
draft token `x = candidates[b,j]`, starting at `j=1`, acceptance requires
`uniform_samples[b,j-1] * draft_probs[b,j-1,x] < target_probs[b,j-1,x]`.
Verification stops at the first rejection.

After accepting `a` drafts, sample one final token from row `a`. The sampling
weights are `target_probs[b,a]` when all drafts are accepted, or
`max(target_probs[b,a] - draft_probs[b,a], 0)` after a rejection. Select the first
token whose cumulative weight exceeds
`uniform_samples_for_final_sampling[b] * sum(weights)`.

## Parameters and return value

`B` is the request count, `D` the candidate width including the root, `R` the
draft-probability row count, and `V` the vocabulary size. Use contiguous tensors
on the same NPU, with `D >= 1`, `R >= max(D-1,1)`, and a nonempty vocabulary.
The caller supplies valid token IDs, global output slots, probabilities and
random samples.

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

Returns `None`. The caller allocates and initializes the three output buffers.
Unwritten slots retain their initial values. Outputs must have disjoint write
regions and must not overlap inputs. For `B == 0`, Triton returns after checking
the candidate and target tensor ranks.

## triton backend

- Uses three kernels for prefix acceptance, residual sums over 2048-token blocks,
  and final sampling. The wrapper checks shape, dtype, device and contiguity.
- At the end of the cumulative distribution, selects the last positive entry in
  the chosen block, or `V-1` if that block has no positive entries.

## torch_native backend

- Reuses `chain_rejection_reference` from the existing repository test, adapted to
  the public signature and in-place output buffers.
- Uses Python scalar acceptance checks and whole-row PyTorch `sum`/`cumsum`.
  The cumulative-distribution fallback searches the entire row. Floating-point
  rounding near acceptance or sampling boundaries can change a token choice.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import chain_speculative_sampling

# B=1, D=2, R=1, V=4; acceptance is exactly at the rejection boundary.
candidates = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
slots = torch.arange(2, dtype=torch.int64, device="npu").reshape(1, 2)
links = torch.full_like(candidates, -1)
target = torch.full((1, 2, 4), 0.25, dtype=torch.float32, device="npu")
draft = torch.tensor([[[0.25, 0.5, 0.25, 0.0]]], device="npu")
coins = torch.tensor([[0.5, 0.125]], device="npu")
final_coin = torch.tensor([0.37], device="npu")

for impl in ("triton", "torch_native"):
    predicts = torch.full((2,), -1, dtype=torch.int32, device="npu")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(1, dtype=torch.int32, device="npu")
    result = chain_speculative_sampling(
        predicts, accept_index, accept_token_num,
        candidates, slots, links, links, coins, final_coin, target, draft,
        impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## Tests

The [operator tests](../../../../../tests/python/operators/speculative/test_chain_speculative_sampling.py)
cover full acceptance, early rejection, model shapes, legacy cases and boundaries.
They compare independent output buffers with exact integer checks.
See the [test framework guide](../../../../../tests/python/README.md) for accuracy
and performance options.
