# lp_post

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.moe import lp_post


def lp_post(
    log2phy_prob,
    x,
    t1,
    phy_single,
    phy_replicated,
    log2phy,
    *,
    impl="auto"
): ...
```

## Description

Turns the LPLB interior-point solution into the per-logical-expert replica probability table. The physical probability of every slot is assembled from two disjoint sources and then gathered back through the logical-to-physical table:

```text
phy_prob[:]                  = 0
phy_prob[phy_replicated]     = max(x, 0)
phy_prob[phy_single]         = t1
log2phy_prob[i, j]           = phy_prob[log2phy[i, j] < 0 ? P - 1 : log2phy[i, j]]
```

`P = num_single + num_red_phy + 1` reserves one always-zero sink slot, so the `-1` padding entries of `log2phy` read zero without a branch. Every row of `log2phy_prob` therefore sums to the fraction of the logical expert's traffic that stays replicable after the LP solve.

## Parameters and return value

All tensors must be on the same NPU device and contiguous.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `log2phy_prob` | Tensor | `Required` | Writable float32 [num_logical,max_copies] output table. |
| `x` | Tensor | `Required` | Float32 [num_red_phy] replica ratios, usually `solution[:num_red_phy]` of `solve_ipm`. Negative values clamp to 0. |
| `t1` | Tensor | `Required` | Float32 [num_single] single-copy shares produced by `lp_prep`. |
| `phy_single` | Tensor | `Required` | Int32/int64 [num_single] physical slot of each single-copy expert. |
| `phy_replicated` | Tensor | `Required` | Int32/int64 [num_red_phy] physical slot of each replica. |
| `log2phy` | Tensor | `Required` | Int32/int64 [num_logical,max_copies] logical-to-physical table; `-1` marks an unused replica. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

`phy_single` and `phy_replicated` must be disjoint and every entry must lie in `[0, P - 1]`.

| Return | Type | Description |
| --- | --- | --- |
| log2phy_prob | Tensor | The same buffer, overwritten with the replica probabilities. |

## triton backend

- Two launches. The first builds the physical table with a compare-sum inverse instead of a scatter: every slot sums the source lanes that point at it. This avoids the per-lane scalar store path on Ascend and is bit-identical to a sequential scatter because the two sources are disjoint.
- The zero initialisation is folded into the first launch by full coverage of `[0, P)`, so no separate memset kernel is needed.
- The second launch preloads the whole table into UB once per program and resolves `log2phy_prob` with the hardware gather path; `-1` entries wrap to the sink slot.

## torch_native backend

- Builds `phy_prob` with `zeros` plus two index-assignments, clamps `x` with `clamp(min=0)`, and gathers the output table with `phy_prob[log2phy]` after replacing `-1` by the sink slot.
- Independent of the kernel, so it serves as the accuracy reference.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import lp_post

num_logical, max_copies = 3, 2
log2phy = torch.tensor([[0, 1], [2, -1], [3, -1]], dtype=torch.int32, device="npu")
log2phy_prob = torch.zeros(num_logical, max_copies, device="npu")
x = torch.tensor([0.4, 0.6], device="npu")  # replica ratios from solve_ipm
t1 = torch.tensor([0.2], device="npu")  # single-copy shares from lp_prep
phy_single = torch.tensor([3], dtype=torch.int32, device="npu")
phy_replicated = torch.tensor([0, 2], dtype=torch.int32, device="npu")

lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy, impl="triton")
```
