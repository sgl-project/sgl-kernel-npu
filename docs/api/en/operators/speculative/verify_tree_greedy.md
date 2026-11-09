# verify_tree_greedy

## Hardware support

| Implementation | arch 2201 | arch 3510 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` option selects the same entry point as `triton`.

```python
from sgl_kernel_npu.operators.speculative import verify_tree_greedy

def verify_tree_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    *,
    impl="auto",
):
    ...
```

## Description

Verifies a draft prefix against the target model's next-token predictions in
greedy speculative decoding. Starting at column `j=1`, accept while
`candidates[b,j] == target_predict[b,j-1]` and stop at the first mismatch.

For `a` accepted drafts, write the root and accepted slots to
`accept_index[b,0:a+1]` and set `accept_token_num[b] = a`. With canonical slots,
write `predicts[b*D+j] = target_predict[b,j]` for `0 <= j <= a`; the last token
is the final target prediction. The caller computes `target_predict` using
argmax before calling this operator.

## Parameters and return value

`B` is the request count and `D` the candidate width including the root. The
common backend contract uses a linear chain, contiguous tensors on one NPU,
`B >= 1`, `D >= 1`, `accept_index[B,D]`, and canonical slots
`retrive_index[b,j] = b*D+j`. Provide chain links for the native backend.
Token IDs and slots must fit the int32 outputs.

| Parameter | Shape / dtype | Meaning |
| --- | --- | --- |
| `predicts` | One-dimensional, length at least `B*D`, int32 | Caller-allocated mutable token output. |
| `accept_index` | `[B,D]`, int32 | Mutable root/accepted slots; unwritten positions retain their initial values. |
| `accept_token_num` | `[B]`, int32 | Accepted-draft count, excluding the final target token. |
| `candidates` | `[B,D]`, int32 or int64 | Root and draft IDs; the root value itself is not compared. |
| `retrive_index` | `[B,D]`, integer | Canonical mapping `b*D+j`. |
| `retrive_next_token` | `[B,D]`, integer | First-child local indices for the chain; ignored by Triton, used by native. |
| `retrive_next_sibling` | `[B,D]`, integer | `-1` for a chain; ignored by Triton, used by native. |
| `target_predict` | `[B,D]`, int32 or int64 | Target model's next-token prediction at each node. |
| `impl` | str, default `"auto"` | `"auto"`, `"triton"`, or `"torch_native"`. |

Returns `None`. The caller allocates and initializes the three output buffers;
unwritten positions retain their initial values. Output write regions must be
disjoint and must not overlap inputs.

## triton backend

- Uses one program per request, masked power-of-two loads and sequential prefix
  comparison. Child/sibling links are ignored.
- Requires linear chains and an accepted-index row stride of `D`. Branching trees
  and `accept_index` widths smaller than `D` are unsupported. The wrapper forwards
  arguments without validation.

## torch_native backend

- Reuses the existing `speculative.py::verify_tree_greedy_native` helper through
  its in-place traversal path, adapting the return value to `None`.
- Traverses child/sibling links and also supports branching trees. Backend
  accuracy comparisons use the common linear-chain contract.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.speculative import verify_tree_greedy

# B=2, D=2; the original int32 chain fixture.
candidates = torch.tensor([[10375, 28], [223, 15098]], dtype=torch.int32, device="npu")
target = torch.tensor([[28, 334], [15098, 18]], dtype=torch.int32, device="npu")
slots = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
children = torch.tensor([[1, -1], [1, -1]], dtype=torch.int32, device="npu")
siblings = torch.full_like(children, -1)

for impl in ("triton", "torch_native"):
    predicts = torch.zeros(4, dtype=torch.int32, device="npu")
    accept_index = torch.full((2, 2), -1, dtype=torch.int32, device="npu")
    accept_token_num = torch.zeros(2, dtype=torch.int32, device="npu")
    result = verify_tree_greedy(
        predicts, accept_index, accept_token_num,
        candidates, slots, children, siblings, target, impl=impl,
    )
    assert result is None
    print(impl, predicts, accept_index, accept_token_num)
```

## Tests

The [operator tests](../../../../../tests/python/operators/speculative/test_verify_tree_greedy.py)
cover full acceptance, early mismatch, model shapes, root-only inputs, padded
widths and int32/int64 inputs. They compare independent integer outputs exactly.
A separate branching-tree regression checks `torch_native` against fixed outputs.
See the [test framework guide](../../../../../tests/python/README.md) for accuracy
and performance options.
