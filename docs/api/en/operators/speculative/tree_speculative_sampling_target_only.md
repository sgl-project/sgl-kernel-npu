# tree_speculative_sampling_target_only

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Registered, refactor validation pending | Registered, refactor validation pending |
| `torch_native` | Registered, refactor validation pending | Registered, refactor validation pending |

`auto` selects `triton`. Implementation lookup and execution errors propagate to
the caller. These registrations declare support. NPU accuracy and performance
comparisons against the original implementation remain required on each device.

## Signature

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

After the target model produces verification probabilities, this operation
selects an accepted path through a candidate tree and samples one final token.
The caller supplies the tree, target probabilities and random samples, then uses
the outputs to advance decoding.

Let `B` be the request count, `D` the candidate node count including the root,
`S` the width of the accepted-index buffer, and `V` the vocabulary width. The
root has local node index 0. `retrive_next_token` points to the first child and
`retrive_next_sibling` to the next sibling. A value of `-1` ends that link.

Initially the current probability row is `r=0` and the coin is
`uniform_samples[b,0]`. At each depth, visit children in sibling-link order.
For a candidate token `x`, read `p=target_probs[b,r,x]` and add it to the current
depth's `prob_acc`. Accept the node when either condition holds.

- `coin <= prob_acc / max(float(threshold_acc), 1e-9)`.
- `p >= threshold_single`.

On acceptance, write the token to the previous accepted node's global output
slot, record the new node's slot, switch the probability row to this node, and
load its entry from `uniform_samples`. On rejection, write the token's target
probability into `draft_probs[b,r,x]` and try its sibling. End the path when no
child is accepted. The index buffer holds the root and at most `S-1` accepted
draft nodes.

At the final probability row `r`, compute
`w=max(target_probs[b,r]-draft_probs[b,r],0)`. Select the first token whose
cumulative weight is strictly greater than
`uniform_samples_for_final_sampling[b] * sum(w)` and write it to the last
accepted node's output slot. At the end of the CDF, Triton uses the last positive
entry in the selected vocabulary block, or `V-1` when the block has no positive
entries. The PyTorch reference searches the entire row for its last positive
entry and uses `V-1` when the entire row has zero weight.

## Parameters and return value

Triton executes on contiguous tensors on one NPU. Nonempty batches require
`D >= 1` and `1 <= S <= D`. The caller must provide a nonempty vocabulary, a valid
acyclic tree, valid token IDs and valid global output slots. The original wrapper
checks shape, dtype, device and contiguity. Index bounds, tree validity and
probability values are the caller's responsibility. The spelling `retrive` is
retained for existing keyword calls.

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

Returns `None`. The caller allocates and initializes the three integer output
buffers. Unwritten positions keep their previous values. `draft_probs` may be
allocated using `empty_like(target_probs)` because the call overwrites its entire
contents. Its storage must be separate from the target probabilities. The
original wrapper rejects equal `data_ptr()` values. The four mutable buffers
must be separate from read-only inputs, and their write regions must be disjoint.

For `B == 0`, Triton returns after checking candidate and target tensor ranks,
skipping other checks and scratch clearing. The PyTorch reference first reads
the candidate shape and `accept_index.shape[1]`, converts the accumulated
threshold, and clears scratch before entering the empty request loop. Empty
batches satisfying the documented shapes return `None`, with empty scratch.

## Implementations and provenance

After clearing scratch, `triton` launches three kernels in sequence for tree
acceptance, residual sums over blocks of 2048 vocabulary entries, and final
sampling. Temporary `metadata` has shape `[B,2]` and dtype int64, storing the final
probability row and output slot. `block_sums` has shape `[B,ceil(V/2048)]` and dtype
float32. The block count is padded to a power of two for the final sampler. The
first and last launches use `B` programs. The middle launch uses
`min(B*ceil(V/2048), vector-core count)` programs, each striding over vocabulary
blocks. Performance measurements must include wrapper scratch clearing and all
three kernels.

The new Triton computation and complete wrapper are copied from this repository's
original `sample` implementation, with the device-property utility import moved
to the implementation layer. Their recorded source is the merge commit for
#623. The original and new directories each retain an independent implementation
copy. [Original Triton implementation](https://github.com/sgl-project/sgl-kernel-npu/blob/15ea25766eb0f9a30951d7d127af8600998272ea/python/sgl_kernel_npu/sgl_kernel_npu/sample/tree_speculative_sampling_target_only.py)

`torch_native` copies the computation from `target_only_tree_reference` in this
repository's original test. That reference already handles general child and
sibling traversal, thresholds and scratch writes. The adapted signature matches
the public interface, aliases the reference variable `rejected_probs` to
`draft_probs`, and ignores `deterministic`. The reference's four-tensor return is
adapted to return `None` after the same in-place writes. [Original PyTorch tree reference](https://github.com/sgl-project/sgl-kernel-npu/blob/15ea25766eb0f9a30951d7d127af8600998272ea/tests/python/sgl_kernel_npu/test_tree_speculative_sampling_target_only.py#L10-L77)

Explicit input validation resides in the Triton wrapper. The PyTorch backend
directly executes the existing reference calculation. Inputs must satisfy the
interface contract, and error timing differs between backends. It accumulates probabilities and evaluates acceptance using Python
floats, then uses whole-row PyTorch `sum` and `cumsum`. Triton uses FP32
accumulation and block reductions. Existing rounding differences near acceptance
or CDF boundaries can change a token choice. Exact integer comparisons must be
followed by investigation when a disagreement occurs.

The original implementation and exports under `sgl_kernel_npu.sample`, together
with the original test file, remain intact. Existing imports execute that original
copy. New integrations select the added implementation through
`sgl_kernel_npu.operators.speculative`.

## Test coverage

The baseline regression set reuses all 24 original fixtures, covering three
general-tree threshold cases, 18 linear-tree cases and three linear-tree threshold
cases. Each backend receives its own integer outputs and scratch. The shared
accuracy helper compares all four mutable buffers. Integer comparisons are exact,
and scratch follows the original general-tree test's zero-tolerance comparison.
Read-only inputs are also checked before and after execution.

Eleven additional FP32 cases cover these model shapes. They use NEXTN, TP2 and
topk=1 linear candidate trees with rejection sampling disabled and both acceptance
thresholds set to their default `1.0`. 

| Scenario | `(D,S,V)` | `B` |
| --- | --- | --- |
| GLM-4.7-Flash, NEXTN, TP2, topk=1 | `(4,4,154880)` | 1, 2, 4, 8, 16, 32 |
| Qwen3.5-9B, NEXTN, TP2, topk=1 | `(4,4,248320)` | 1, 2, 4, 8, 16 |

Batch choices refer to the [GLM NPU CI](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/test/registered/npu/accuracy/glm4_7_flash/test_npu_glm4_7_flash_1p_gsm8k.py#L21-L52)
and [Qwen NPU CI](https://github.com/sgl-project/sglang/blob/77c3bcbe2c28e1a6d1ad4121ee57231582d39bdc/test/registered/npu/accuracy/qwen3_5_9b/test_npu_qwen3_5_9b_bf16_1p_gsm8k.py#L24-L48).
Qwen batches 1, 2, 4 and 8 are supplementary coverage points. Neither NPU CI
configuration itself enables speculative decoding. D/S follow SGLang's NEXTN
configuration and path-capacity convention; V follows official model configs.
The test source records these references. These combined regression configurations
are not recorded target-only end-to-end calls.

The model inputs reuse the original linear-tree test's seed formula, probability
construction and stable final-sampling coin generator, rather than production
probability data. The original branching-tree cases are retained, giving 35 tree
cases in total. The default performance subset contains each model's smallest
and largest batch, four shapes in total; the shared CLI can override this choice.

With fixed inputs, each repeated call clears scratch again and writes the same
slots. Preallocated output buffers can therefore be reused for the shared
performance measurement. Direct old/new NPU comparisons, comparisons of each
against the independent reference, and performance acceptance remain required on
the target device. Use the standard CLI in [the test framework guide](../../../../../tests/python/README.md).
