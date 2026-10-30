# build_pooled_page_table

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Supported | Unverified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention import build_pooled_page_table

def build_pooled_page_table(
    page_table_64: torch.Tensor,
    pool_size: int,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` selects `torch_native`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

For `K=pool_size` and input `T[...,M]`:

$$O_{\ldots,p}=T_{\ldots,pK},\qquad 0\le p<\lceil M/K\rceil.$$

Leading dimensions are preserved and `M=0` leaves the last dimension empty, unlike the minimum one-column padding of `kpool_indexer_prepare_decode_input`. Inputs are read only; output dtype is preserved. `pool_size` is a positive divisor of 128; the input column count need not divide 128. The `_64` suffix does not impose a physical page size on this interface.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `page_table_64` | `torch.Tensor` | `[..., NUM_BLOCKS]` | The backend does not restrict it; use an integer page-ID dtype | Required | Token-page table. The last dimension is sampled. |
| `pool_size` | `int` | — | — | Required | Number of token pages represented by one pool page. Must be positive and divide the backend constant `BLOCK_SIZE_K = 128`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., ceil(NUM_BLOCKS / pool_size)]` | Same as `page_table_64` | Newly allocated result; the input is not modified. Consumers must impose any separate contiguity requirement they have. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range.

## torch_native backend

Uses torch.arange on the input device and advanced indexing. Input need not be contiguous but must have a last dimension. Only auto and torch_native are registered; triton is not implemented.

Importing the public registry requires NPU architecture detection, including for `torch_native`. Support is limited to accepted A3 cases; architecture registration does not establish coverage of every chip, shape or dtype.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import build_pooled_page_table
page_table = torch.arange(32, device='npu', dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table(page_table, pool_size=4, impl='torch_native')
print(pooled.shape)
```
