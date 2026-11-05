# build_tree

```python
from sgl_kernel_npu.operators.speculative import build_tree

build_tree(parent_list, selected_index, verified_seq_len, tree_mask, positions,
           retrive_index, retrive_next_token, retrive_next_sibling,
           topk, depth, draft_token_num, tree_mask_mode, *, impl="auto") -> None
```

Build a verification tree and write five caller-owned buffers in place.
`auto` selects `ascendc` without fallback. `torch_native` is an independent
PyTorch implementation. The old `torch.ops.npu.build_tree_kernel_efficient`
entry and old implementation are unchanged.

## Inputs and side effects

Let `B` be batch size, `D=draft_token_num`, `k=topk`, and `L_b` the verified
prefix length of request `b`. All tensors use the same device. The CANN route
requires an NPU. The intended layout is dense ND, with these shapes:

| Parameter | dtype and shape | Meaning |
|---|---|---|
| `parent_list` | int64 `[B, k*(depth-1)+1]` | Parent candidate table |
| `selected_index` | int64 `[B, D-1]` | Selected candidate indices, excluding root |
| `verified_seq_len` | int64 `[B]` | Verified prefix lengths |
| `tree_mask` | bool, flat | Mutable mask buffer; size below |
| `positions` | int64 `[B*D]` or dense `[B,D]` | Mutable absolute positions |
| `retrive_index` | int64 `[B,D]` | Mutable flattened node indices |
| `retrive_next_token` | int64 `[B,D]` | Mutable first child, or -1 |
| `retrive_next_sibling` | int64 `[B,D]` | Mutable next sibling, or -1 |

The return value is `None`; all five supplied output buffers retain their
storage. Read-only inputs remain unchanged. No contiguous temporary or
copyback is introduced: the low-level device implementation retains the old
dense, raw-pointer interpretation rather than providing strided semantics.
Output buffers must be distinct and large enough for the requested layout.
The reference is defined for valid, ancestor-closed trees with unique selected
IDs. Malformed trees and insufficient buffers have no defined legacy result.
The legacy uint16 narrowing of `D` and mask length remains in the Kernel; this
refactor does not establish a new validated maximum shape.

## Computation and mask modes

Node 0 is the root. For node `i>0`, let `q=selected_index[b,i-1]//k`.
Its parent is the root when `q=0`; otherwise find `parent_list[b,q]` in
`selected_index[b]`, then add 1 to get the parent node index. The outputs are
`positions[b,i]=L_b+tree_depth(i)` and `retrive_index[b,i]=b*D+i`.
Children and sibling lists visit child indices in increasing order.
The mathematical draft mask is true exactly for the node itself and ancestors.

* `tree_mask_mode=0` (`FULL_MASK`): flat concatenation of `[D,L_b+D]` masks,
  totaling `D*sum(L_b)+B*D*D` entries. Only the last `D` columns of each row
  are written. The caller initializes verified-prefix columns as required.
* `tree_mask_mode=1` (`QLEN_ONLY`): intended flat `[B,D,D]`, `B*D*D` entries.
  **Known retained defect:** old and new Ascend C use `b*D*D+b*D` for every
  node row, instead of `b*D*D+i*D`. Repeated writes leave many rows unchanged
  and can exceed the buffer for some batch sizes. This refactor preserves that
  computation by explicit decision; it does not claim mathematical correctness.
  `torch_native` uses correct per-node rows, so a comparison exposes the defect.
* `tree_mask_mode=2` (`QLEN_ONLY_BITPACKING`): unsupported and raises an error.

There is no automatic mode selection or backend fallback. Other integer modes
retain the legacy non-FULL branch; they are not additional validated modes.
Empty batches had no defined legacy execution; the new CANN tiler rejects a
zero block count instead of dividing by zero.

## Validation status

| Backend | Source declaration | Device validation in this change |
|---|---|---|
| `ascendc`, `auto` | ARCH22 / 2201, ARCH35 / 3510 | FULL_MASK passed in the reported test environment; architecture matrix pending |
| `torch_native` | Same registry generations; standalone PyTorch computation | Independent reference, not evidence of device correctness |

The formal public-interface test reuses the old exact fixture: B=2, k=4,
depth=4, D=8, L=[5,10]. All five bool/integer outputs use the shared exact
accuracy check, with isolated buffers and unchanged-input checks. This is an
existing test fixture, not a claimed production workload.

The same test file supports separate-installation replay: `--impl legacy
--save PATH` captures the old result; `--impl ascendc --compare-legacy PATH`
checks new/old first, then the independent mathematical reference. Add
`--mode qlen` to both calls for the opt-in known-defect diagnostic; its failing
mathematical assertion is not skipped or relaxed. Default pytest covers
FULL_MASK new/reference, not a completed legacy replay. Additional production
cases, graph capture and validation across architectures remain separate work.

## Performance

The original fixture is selected in `perf_shape`; no new dimensions are used.
After accuracy passes, the shared helper profiles five warmup and five active
calls. Repeated FULL_MASK calls overwrite the tree columns and preserve prefix
columns, so the same output buffers are reused without reset/copy kernels.

From the repository root, using the Python environment containing the new wheel:

```bash
SGL_TEST_ROOT="$(pwd -P)"
python3 -m pytest "$SGL_TEST_ROOT/tests/python/operators/speculative/test_build_tree.py" \
  --impl ascendc --impl-ref torch_native --skip-acc-check=false \
  --enable-perf=true --csv-path "$SGL_TEST_ROOT/test-results/build_tree.csv" \
  --prof-dir "$SGL_TEST_ROOT/test-results/profilings" -sv -ra
```

`--enable-perf=false` disables profiling. Skipping accuracy records
`accuracy_checked=False`. CSV times are mean NPU device-task times, not host
end-to-end latency. Actual profiling and comparison with the legacy
implementation remain pending; the reported FULL_MASK pass is an accuracy result.
