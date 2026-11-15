# dispatch_probability

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.moe import dispatch_probability


def dispatch_probability(
    topk_ids, log2phy_prob, log2phy_map, random_vals=None, *, impl="auto"
): ...
```

## Description

Samples one physical expert per routing slot of every token, following the replica probabilities produced by `lp_post`:

```text
p = log2phy_prob[topk_ids]                     # (..., max_copies)
p = where(p.sum(-1) > 0, p, (log2phy_map[topk_ids] >= 0))
physical = log2phy_map[topk_ids, multinomial(p)]
```

A row whose probabilities are all zero falls back to a uniform choice over the valid copies (`log2phy_map != -1`). Sampling is done by inverse CDF on a single uniform value per slot, so the result is a deterministic function of `random_vals` and the tables.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `topk_ids` | Tensor | `Required` | Int32/int64 logical expert ids of any shape; non-contiguous inputs are materialised. |
| `log2phy_prob` | Tensor | `Required` | Contiguous float32 [num_logical,max_copies] replica probabilities. |
| `log2phy_map` | Tensor | `Required` | Contiguous int32/int64 [num_logical,max_copies] physical ids; `-1` marks an unused replica. Shape and device must match `log2phy_prob`. |
| `random_vals` | Tensor \| None | `None` | Optional contiguous float32 [numel(topk_ids)] uniform samples in `[0, 1)`. Fresh values are drawn when omitted. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| physical | Tensor | Sampled physical ids with `topk_ids`' shape and dtype. |

## triton backend

- Each program loads the whole `[num_logical, max_copies]` table into UB once (transposed so the copy axis is the reduction axis) and then strides over token blocks of 64 entries.
- Per slot the program computes the CDF row with `tl.cumsum`, turns the uniform sample into a threshold `r * total`, counts the CDF entries at or below it, and gathers the physical id. Counting instead of a search avoids a serial loop over copies.
- Logical and copy counts are padded with masked power-of-two tiles, so no power-of-two restriction is imposed on the caller; the chosen copy index is clamped to `max_copies - 1`.

## torch_native backend

- Builds `topk_probs` with a table gather, applies the zero-row fallback mask, samples with inverse CDF on the same `random_vals`, and gathers `log2phy_map`.
- Independent of the kernel and deterministic for a given `random_vals`, so it serves as the accuracy reference.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.moe import dispatch_probability

log2phy_prob = torch.tensor([[0.4, 0.6], [0.0, 0.0], [1.0, 0.0]], device="npu")
log2phy_map = torch.tensor([[0, 1], [2, -1], [3, -1]], dtype=torch.int32, device="npu")
topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="npu")
random_vals = torch.tensor([0.1, 0.5, 0.9], device="npu")

physical = dispatch_probability(
    topk_ids, log2phy_prob, log2phy_map, random_vals, impl="triton"
)
```
