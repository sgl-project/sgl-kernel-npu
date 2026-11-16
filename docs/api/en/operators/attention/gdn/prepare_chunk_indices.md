# prepare_chunk_indices

## Hardware support

| Implementation | ARCH22 | ARCH35 |
| --- | --- | --- |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import prepare_chunk_indices

def prepare_chunk_indices(cu_seqlens, chunk_size, *, impl="auto"):
    ...
```

`auto` selects `torch_native`.

## Description

Build the chunk coordinates for a packed GDN batch. A sequence of length
`L` contributes `ceil(L / chunk_size)` rows. Each row stores the sequence
index among nonempty sequences and the zero-based chunk index within that
sequence. Empty sequences contribute no rows. In Ascend GDN prefill, the
number of rows determines the size of the entering chunk-state buffer.

## Parameters and return value

| Parameter | Type / shape | Dtype / default | Description and constraints |
| --- | --- | --- | --- |
| `cu_seqlens` | Tensor `[B+1]` | int32/int64, required | Nondecreasing cumulative token boundaries starting at 0, with `B >= 1`. |
| `chunk_size` | int | Required | Positive maximum token count per chunk. |
| `impl` | str | `"auto"` | `auto` or `torch_native`. |

| Return | Type / shape | Dtype / device | Description |
| --- | --- | --- | --- |
| Chunk indices | Tensor `[total_chunks, 2]` | Same as `cu_seqlens` | Sequence and local chunk coordinates in packed order. |

## torch_native backend

Uses PyTorch to construct the integer coordinates. Chunk counts are read on
the host, which can synchronize when the input is on the NPU. Results are
cached by input tensor identity and chunk size. Keep reused input tensors
and their returned metadata unchanged.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention.gdn import prepare_chunk_indices

cu_seqlens = torch.tensor([0, 70, 100], dtype=torch.int32, device="npu")
chunk_indices = prepare_chunk_indices(cu_seqlens, 64, impl="torch_native")
# [[0, 0], [0, 1], [1, 0]]
```
