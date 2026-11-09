# tree_speculative_sampling_target_only

## Hardware support

| Implementation | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` option selects the same entry point as `triton`.

```python
from sgl_kernel_npu.operators.speculative import tree_speculative_sampling_target_only

def tree_speculative_sampling_target_only(
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

Selects an accepted path through a candidate tree using target probabilities,
then samples one final token. Starting at root node 0, visit each node's children
in sibling-link order. Accumulate their target probabilities into `prob_acc` and
accept a candidate with probability `p` when either condition holds.

- `coin <= prob_acc / max(float(threshold_acc), 1e-9)`.
- `p >= threshold_single`.

On acceptance, record the node's slot and continue with its probability row and
random sample. On rejection, record its probability in `draft_probs` and try the
next sibling. The accepted path contains the root and at most `S-1` drafts.

At the final row `r`, sample from
`max(target_probs[b,r] - draft_probs[b,r], 0)`. Select the first token whose
cumulative weight exceeds `uniform_samples_for_final_sampling[b] * sum(weights)`.

## Parameters and return value

`B` is the request count, `D` the node count including the root, `S` the accepted
path capacity, and `V` the vocabulary size. Use contiguous tensors on the same NPU,
with `D >= 1`, `1 <= S <= D`, and a nonempty vocabulary. The caller supplies an
acyclic tree, valid token IDs and output slots, probabilities and random samples.

| Parameter | Shape / dtype | Meaning |
| --- | --- | --- |
| `predicts` | One-dimensional, usually `[B*D]`, int32 | Mutable output for accepted drafts and the final token, written at global slots from `retrive_index`. |
| `accept_index` | `[B,S]`, int32 | Mutable output for the root slot and accepted node slots. The valid prefix has length `accept_token_num[b]+1`. |
| `accept_token_num` | `[B]`, int32 | Mutable accepted-draft count in `[0,S-1]`. The final sampled token is counted separately. |
| `candidates` | `[B,D]`, int64 | Root and candidate token IDs. Traversal reads the child nodes' token IDs. |
| `retrive_index` | `[B,D]`, int64 | Global output slot for each local node. Slots must fit `predicts` and the int32 output indices. |
| `retrive_next_token` | `[B,D]`, int64 | Local index of the first child, or `-1` when there is no child. |
| `retrive_next_sibling` | `[B,D]`, int64 | Local index of the next sibling, or `-1` at the end of the sibling chain. |
| `uniform_samples` | `[B,D]`, float32 | Caller-supplied samples in `[0,1)`, indexed by the current accepted node. |
| `uniform_samples_for_final_sampling` | `[B]`, float32 | Caller-supplied final sampling values in `[0,1)`. |
| `target_probs` | `[B,D,V]`, float32 | Read-only target probabilities. Each node provides the row for its next verification or final sample. |
| `draft_probs` | `[B,D,V]`, float32 | Fourth mutable output, used as rejection-probability scratch. Every call clears it before recording rejected siblings' target probabilities. |
| `threshold_single` | float, default `1.0` | Accept directly when a token's target probability reaches this threshold. Valid range is `[0,1]`. |
| `threshold_acc` | float, default `1.0` | Accumulated-probability acceptance threshold in `[0,1]`. Computation uses `max(float(threshold_acc),1e-9)`. |
| `deterministic` | bool, default `True` | Retained parameter whose value is currently ignored. The caller supplies all random samples. |
| `impl` | str, default `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

Returns `None`. The caller allocates and initializes the three integer outputs;
unwritten slots retain their initial values. `draft_probs` is cleared on every
nonempty call and may be allocated with `empty_like(target_probs)`. All four
mutable buffers must have disjoint write regions and must not overlap inputs.
Triton rejects `draft_probs` sharing the target tensor's `data_ptr()`.
For `B == 0`, Triton returns after rank checks. The native backend clears the
empty scratch before returning.

## triton backend

- Clears scratch, then launches three kernels for tree acceptance, residual sums
  over 2048-token blocks, and final sampling. The wrapper checks shape, dtype,
  device and contiguity.
- At the end of the cumulative distribution, selects the last positive entry in
  the chosen block, or `V-1` if that block has no positive entries.

## torch_native backend

- Reuses `target_only_tree_reference` from the existing repository test, with the
  public signature and `draft_probs` as rejection scratch.
- Uses Python scalar tree traversal. For final sampling, computes the cumulative
  residual weights in FP64 and uses the last cumulative value as the total
  weight. This reduces rounding differences near sampling boundaries. The
  fallback searches the entire row.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import tree_speculative_sampling_target_only

# B=2, D=S=4, V=32; a linear candidate tree.
candidates = torch.tensor([[0, 3, 4, 5], [0, 7, 8, 9]], device="npu")
slots = torch.arange(8, device="npu").reshape(2, 4)
children = torch.tensor([[1, 2, 3, -1], [1, 2, 3, -1]], device="npu")
siblings = torch.full_like(children, -1)
target = torch.full((2, 4, 32), 1.0 / 32, device="npu")
for b in range(2):
    for j in range(1, 4):
        target[b, j - 1] *= 0.2
        target[b, j - 1, candidates[b, j]] += 0.8
        target[b, j - 1] /= target[b, j - 1].sum()
coins = torch.tensor([[0.1, 0.9, 0.2, 0.0], [0.7, 0.2, 0.95, 0.0]], device="npu")
final_coin = torch.tensor([0.25, 0.75], device="npu")

for impl in ("triton", "torch_native"):
    predicts = torch.full((8,), -1, dtype=torch.int32, device="npu")
    accept_index = torch.full((2, 4), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(2, dtype=torch.int32, device="npu")
    scratch = torch.empty_like(target)
    result = tree_speculative_sampling_target_only(
        predicts, accept_index, accept_token_num,
        candidates, slots, children, siblings, coins, final_coin, target, scratch,
        impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## Tests

The [operator tests](../../../../../tests/python/operators/speculative/test_tree_speculative_sampling_target_only.py)
cover branching trees, linear chains, thresholds and model shapes. Each backend
uses independent outputs and scratch, all compared with zero tolerance.
See the [test framework guide](../../../../../tests/python/README.md) for accuracy
and performance options.
