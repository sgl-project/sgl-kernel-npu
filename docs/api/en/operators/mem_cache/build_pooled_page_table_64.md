# build_pooled_page_table_64

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Unverified | Unverified |

## Signature

`auto` is an alias for `torch_native`; omitting `impl` and passing `impl="auto"` have the same effect.

```python
from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

def build_pooled_page_table_64(
    page_table_64,
    pool_size,
    *,
    impl="auto",
):
    ...
```

## Description

Selects the first token-page entry of every logical pool group. For an input whose last dimension has `NUM_BLOCKS` entries, the returned last dimension contains indices `0, pool_size, 2 * pool_size, ...`. This is equivalent to advanced indexing with `torch.arange(0, NUM_BLOCKS, pool_size)` on the input device.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `page_table_64` | `torch.Tensor` | `[..., NUM_BLOCKS]` | The backend does not restrict it; use an integer page-ID dtype | Required | Token-page table. The last dimension is sampled. |
| `pool_size` | `int` | — | — | Required | Number of token pages represented by one pool page. Must be positive and divide the backend constant `BLOCK_SIZE_K = 128`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., ceil(NUM_BLOCKS / pool_size)]` | Same as `page_table_64` | Newly allocated result; the input is not modified. Consumers must impose any separate contiguity requirement they have. |

## torch_native backend

- Uses `torch.arange` and tensor indexing. The divisibility requirement is enforced by an assertion.
- `auto` is only an alias, not dynamic implementation selection or fallback. Importing the public registry performs ACL architecture detection and requires the NPU runtime.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

page_table = torch.arange(32, device="npu", dtype=torch.int32).reshape(2, 16)
pooled = build_pooled_page_table_64(page_table, pool_size=4)

print(pooled.shape)  # torch.Size([2, 4])
print(list_impls(build_pooled_page_table_64))  # ['auto', 'torch_native']
```
