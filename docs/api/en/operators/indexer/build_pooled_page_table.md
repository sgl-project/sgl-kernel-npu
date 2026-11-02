# build_pooled_page_table

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Supported | Verified |

## Signature

```python
import torch
from sgl_kernel_npu.operators.indexer import build_pooled_page_table

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

Build a pooled page table by taking the first page ID from each group of token pages. The grouping interval is controlled by `pool_size`.

For example, with `pool_size=4`, a row `[10, 11, 12, 13, 20, 21]` becomes `[10, 20]`. Leading dimensions and dtype are preserved; the input is not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `page_table_64` | `torch.Tensor` | `[..., NUM_BLOCKS]` | The backend does not restrict it; use an integer page-ID dtype | Required | Token-page table. The last dimension is sampled. |
| `pool_size` | `int` | — | — | Required | Number of token pages represented by one pool page. Must be positive and divide the backend constant `BLOCK_SIZE_K = 128`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., ceil(NUM_BLOCKS / pool_size)]` | Same as `page_table_64` | Newly allocated result; the input is not modified. Consumers must impose any separate contiguity requirement they have. |

Input and output reside on the same NPU; the input is not modified.

## torch_native backend

Uses torch.arange on the input device and advanced indexing. Input need not be contiguous but must have a last dimension. Only auto and torch_native are registered; triton is not implemented.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import build_pooled_page_table
page_table = torch.arange(32, device='npu', dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table(page_table, pool_size=4, impl='torch_native')
print(pooled.shape)
```
