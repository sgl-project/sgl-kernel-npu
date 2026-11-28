# clear_mamba_slots

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Supported |
| `torch_native` | Unverified | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

The default option is `auto`, currently registered to the same entry point as `triton`. Omitting `impl` or passing `"auto"` selects Triton.

```python
from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

def clear_mamba_slots(
    buf,
    indices,
    *,
    impl="auto",
):
    ...
```

## Description

Zeros selected dim-1 slots in place across all layers of a Mamba/GDN state pool tensor `buf` of shape `[num_layers, pool_size, ...]`:

$$
\text{buf}[\ell, \text{indices}[i], :] \leftarrow 0, \quad \forall \ell \in [0, L), \; \forall i \in [0, N)
$$

The Triton kernel schedules persistent 1D AIV blocks (`BLOCK = 8192`) across `(indices, layers, feature_blocks)` and computes element offsets in 64-bit integer arithmetic (`tl.int64`) to prevent `int32` wraparound on multi-gigabyte SSM state pools (where total elements can exceed $2^{31} - 1$).

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `buf` | `torch.Tensor` | `[L, P, ...]` | `torch.bfloat16`, `torch.float16`, `torch.float32` | Required | Writable NPU state pool tensor (`ndim >= 2`). Triton requires contiguous per-slot slices (`buf[0, 0].is_contiguous()`). Modified in place. |
| `indices` | `torch.Tensor` | `[N]` | `torch.int32`, `torch.int64` | Required | 1D tensor of slot indices in `[0, P)` on the same device as `buf`. Triton requires contiguous storage. Empty tensors are a no-op. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"`, or `"torch_native"`. `auto` selects Triton. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `None` | `NoneType` | — | — | Modifies `buf` in place and returns `None`. |

## triton backend

- Launches up to 56 persistent AIV programs (`BLOCK = 8192`) to saturate HBM write bandwidth in a single kernel launch.
- Uses `tl.int64` for `layer_stride`, `slot_stride`, and block offsets to safely address pools exceeding $2^{31}$ elements.
- Requires contiguous `indices` and contiguous per-slot feature slices `buf[0, 0]`.

## torch_native backend

- Calls `buf.index_fill_(1, indices.to(torch.int64), 0)` directly on the input device.
- Serves as the exact bitwise reference implementation.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.mem_cache import clear_mamba_slots

buf = torch.randn(48, 32, 48, 128, 128, device="npu", dtype=torch.bfloat16)
indices = torch.tensor([1, 5, 12], device="npu", dtype=torch.int64)

print(list_impls(clear_mamba_slots))  # ['auto', 'torch_native', 'triton']
buf_ref = buf.clone()
clear_mamba_slots(buf, indices)  # Default auto selects Triton
clear_mamba_slots(buf_ref, indices, impl="torch_native")
torch.testing.assert_close(buf, buf_ref, rtol=0, atol=0)
```
