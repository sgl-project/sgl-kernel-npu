# verify_tree_greedy

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Registered; device validation pending | Registered; device validation pending |
| `torch_native` | Registered; device validation pending | Registered; device validation pending |

`auto` selects `triton`, with no shape-dependent fallback. Registration does not
establish NPU correctness or performance.

## Signature

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

## Behavior and actual support

The SGLang greedy path takes argmax over target logits before calling this operator.
It then verifies the longest matching draft prefix and writes one final target
token. Probability tensors, random samples and vocabulary size are not operator
inputs. Model execution, argmax and KV-state updates belong to the caller.

**The migrated Triton kernel verifies a linear chain.** It compares columns in
order without reading child/sibling links. Its physical accepted-index row stride
is the candidate width `D`. The common backend comparison contract uses contiguous
`[B,D]` chains, `accept_index[B,D]`, and canonical global slots
`retrive_index[b,j] = b*D+j`.

Arbitrary branching trees and index buffers narrowed to `S<D` are unsupported by
this Triton kernel. The original `B=2,D=6,S=4` branching-tree test executes only
the original native implementation. It is retained, with an additional public
`torch_native` regression; Triton is not launched on that data. The original
wrapper has no validation, and this migration preserves that behavior. An
incorrect accepted-index row stride can cause out-of-bounds writes, so callers
must satisfy the documented contract.

Write `c_j=candidates[b,j]` and `t_j=target_predict[b,j]`. Starting at `j=1`,
accept while `c_j == t_(j-1)` and stop at the first mismatch. If `a` drafts were
accepted:

- `accept_token_num[b] = a`, in `[0,D-1]`.
- Write the root and accepted slots to `accept_index[b,0:a+1]`.
- For canonical slots, write `predicts[b*D+j] = t_j` for `0 <= j <= a`.
  The last value is the final target token; SGLang includes it in decoding advance.
- Leave other output positions unchanged and return `None`.

## Parameters

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

Use contiguous tensors on one NPU with `B>=1`, `D>=1`. The original regression
uses int32 inputs; model cases use int64 inputs matching SGLang's index buffers
and argmax IDs. NPU execution of both input types remains pending in this round.
Output write ranges must not overlap inputs or other outputs. Tokens and slots
must fit the int32 outputs. Empty batches and arbitrary strided layouts are not
declared supported.

## Implementations and provenance

`triton` copies the complete kernel and wrapper from this repository's original
`sample/verify_tree_greedy.py`. Each request has one program. Candidate, slot and
target vectors are padded to a power-of-two width for masked loading, followed
by sequential prefix verification. There are no probability scratch buffers,
reductions or random draws. The recorded introduction is commit
`96dfd27fbc15bf893f96d8d7837d818086d23d04` (#165). The direct copied source is
baseline `26ec66efe01f8acef43db55da679107702876b23`, including the #473 CANN API
adaptation.

`torch_native` copies this repository's `speculative.py::verify_tree_greedy_native`
as a private helper, preserving its computation and return behavior. A new
eight-argument wrapper invokes its default `topk=-1` in-place traversal path
and discards the returned three tensors to match the public `None` return.
The original `topk=1,D=2` fast path allocates replacement outputs and is therefore
not selected by the public wrapper. This does not change the candidate tree's
topk: neither the public signature nor the original Triton signature takes topk.

Native follows child/sibling links and can express the original branching-tree
fixture. It is an independent reference within the common linear contract;
its broader reference behavior does not extend Triton's supported inputs. Neither
backend calls the other. Original operators, `speculative.py` and tests remain
unchanged.

## Tests and validation status

Default tests include the original int32 chain and eleven model configurations
with three acceptance paths each, giving 34 cases:

| Scenario | `D=S` | Token-ID range `V` | `B` |
| --- | --- | --- | --- |
| GLM-4.7-Flash, NEXTN greedy, TP2, topk1 | 4 | 154880 | 1, 2, 4, 8, 16, 32 |
| Qwen3.5-9B, NEXTN greedy, TP2, topk1 | 4 | 248320 | 1, 2, 4, 8, 16 |

`V` supplies legal token IDs, not an input tensor dimension. Shapes reuse the
confirmed model/deployment coverage. Candidate IDs, match positions and final
tokens are synthetic inputs designed for this suite. The source NPU CI recipes
do not enable NEXTN, so these combined configurations are not production traces.
Test comments record the fixed SGLang recipes and official model configs.
Default performance coverage selects each model's smallest and largest B;
the shared CLI can override that choice.

`extra_checks` adds six root-only/padded-width/dtype cases and one explicitly
`torch_native`-only regression with the original branching-tree expected values,
giving 41 cases total. The latter always checks that specific reference backend,
independently of the DUT option, and does not collect performance.
The shared CLI accuracy-skip option also disables this fixed-output regression.

The two backends receive separate int32 output buffers. Tests compare complete
outputs using the shared exact integer standard and check return values and
read-only inputs. CPU reference replay checks only native
migration and fixtures; it does not execute Triton. Direct old/new NPU comparison,
independent-reference accuracy and same-device performance remain required.

Sources: [original Triton](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/python/sgl_kernel_npu/sgl_kernel_npu/sample/verify_tree_greedy.py),
[original native](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/python/sgl_kernel_npu/sgl_kernel_npu/speculative.py#L135),
[SGLang greedy call](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/python/sglang/srt/speculative/eagle_utils.py#L733-L745),
[NPU dispatch](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/python/sglang/kernels/ops/speculative/spec_tree.py#L523-L536).
Use the standard CLI in [the test framework guide](../../../../../tests/python/README.md).
