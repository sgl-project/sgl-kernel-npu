# build_tree

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

The default `auto` uses the same entry point as `ascendc`.

```python
from sgl_kernel_npu.operators.speculative import build_tree

def build_tree(
    parent_list, selected_index, verified_seq_len, tree_mask, positions,
    retrive_index, retrive_next_token, retrive_next_sibling,
    topk, depth, draft_token_num, tree_mask_mode, *, impl="auto",
) -> None:
    ...
```

## Description

Builds a speculative decoding verification tree from parent candidates and selected
nodes. Writes the attention mask, absolute positions, global node indices, and
child/sibling links in place. Node 0 is the root; each node may attend to itself
and its ancestors.

Let `B` be batch size, `D = draft_token_num`, `k = topk`, and `L_b` the verified
prefix length of request `b`. Node `i` has position `L_b + node_depth` and global
index `b*D+i`. Child links follow increasing node indices.

## Parameters and return value

| Parameter | Shape / type | Description |
| --- | --- | --- |
| `parent_list` | int64 `[B,k*(depth-1)+1]` | Parent candidate table. |
| `selected_index` | int64 `[B,D-1]` | Selected candidate indices, excluding root; unique and containing every selected node's ancestors. |
| `verified_seq_len` | int64 `[B]` | Verified prefix length for each request. |
| `tree_mask` | Flat bool tensor | Mutable mask; capacity depends on `tree_mask_mode`. |
| `positions` | int64 `[B*D]` or contiguous `[B,D]` | Mutable absolute positions. |
| `retrive_index` | int64 `[B,D]` | Mutable global node indices. |
| `retrive_next_token` | int64 `[B,D]` | Mutable first-child index, or `-1` if absent. |
| `retrive_next_sibling` | int64 `[B,D]` | Mutable next-sibling index, or `-1` if absent. |
| `topk` | Required `int` | Candidate count per step. |
| `depth` | Required `int` | Candidate tree depth. |
| `draft_token_num` | Required `int` | Draft node count, including the root. |
| `tree_mask_mode` | Required `int` | `0` selects FULL_MASK; see modes below. |
| `impl` | `str`, default `"auto"` | Keyword-only; `auto`, `ascendc` or `torch_native`. |

Returns `None`. All five output buffers retain their storage; read-only inputs
are unchanged. All tensors must share a device and use contiguous buffers with
sufficient capacity. Output buffers must not overlap.

## Mask modes

- `0` (FULL_MASK): concatenates each request's `[D,L_b+D]` mask, totaling
  `D*sum(L_b)+B*D*D` elements. Only the last `D` columns of each row are written;
  the caller initializes verified-prefix columns, which remain unchanged.
- `1` (QLEN_ONLY): a flat `[B,D,D]` mask. AscendC retains a legacy row-address
  defect that can overwrite incorrect rows or exceed the buffer; use FULL_MASK.
  `torch_native` uses correct per-node row addresses.
- `2` (QLEN_ONLY_BITPACKING): not implemented; raises an error.

## ascendc backend

- Requires all tensors on the same NPU and reads dense physical storage;
  general strided semantics are unsupported.
- Writes the five caller-owned outputs in place without contiguous copies
  or additional copyback.

## torch_native backend

- Constructs the tree and masks using independent PyTorch operations.
- Preserves output shapes, dtypes and the in-place interface.
