# sparse_flash_mla_metadata

The CANN 9.2 native contract below is also exposed through the cross-SDK public API.
The preserved mainline CANN 9.2 binding returns 2048 int32 entries; CANN 9.1 retains 1024 entries. Query `get_sparse_flash_mla_metadata_size()` before allocating graph buffers.

Upstream documentation: [Sparse Flash MLA Torch API](https://gitcode.com/cann/ops-transformer/blob/e7da18105f08b06e01ef0fff78f439098d483746/attention/sparse_flash_mla/docs/torchapi_sparse_flash_mla.md).


## CANN 9.1 compatibility

`cann_ops_transformers_cann920` selects the bundled CANN 9.2 implementation.
On CANN 9.1, window-only inputs select `ops_transformers_cann910`; explicit
original-KV sparse inputs select `ascendc` compute and CPU metadata internally
through metadata `impl="auto"`. No separate `host` implementation name is
registered by the metadata adapter. Those explicit inputs cannot also include
compressed KV. CPU metadata requires the
existing CPU int32 `cu_seqlens_q_cpu` and `seqused_ori_kv_cpu` mirrors.
All production bindings are bundled in the wheel; no standalone operator wheel
is required. A5 uses the separate mixed-quant sparse APIs.

The migrated `torch_native` metadata API returns a reference descriptor, not
native tiling bytes. Paired migrated tests build that descriptor independently.
The upstream compute reference also accepts opaque native metadata and covers
SWA/CSA/HCA and BSND inputs; native metadata byte layouts are SDK-specific.

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `cann_ops_transformers_cann920` | Supported | Unsupported |
| `auto` (SDK/input selection) | Supported | Unsupported |
| `ops_transformers_cann910` (CANN 9.1 window inputs) | Supported | Unsupported |
| `torch_native` (reference descriptor) | Supported | Supported |

## Signature

On arch22, `auto` selects the matching CANN 9.1 or 9.2 implementation. The expanded arguments below describe the CANN 9.2 binding; omitted arguments retain the selected binding's defaults.

```python
from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

def sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=None,
    cmp_topk=None,
    cmp_ratio=None,
    ori_mask_mode=None,
    cmp_mask_mode=None,
    ori_win_left=None,
    ori_win_right=None,
    layout_q=None,
    layout_kv=None,
    has_ori_kv=None,
    has_cmp_kv=None,
    impl="auto",
):
    ...
```

## Description

Builds the AI Core task-splitting metadata consumed by
[sparse_flash_mla](sparse_flash_mla.md). The usual flow is: prepare `q`, `ori_kv`,
the optional `cmp_kv`, sequence lengths, sparse indices and block tables; call this
operator to obtain `metadata`; pass `metadata` to the main operator.

Both calls must receive the same scheduling arguments with identical values,
otherwise behaviour is undefined. In particular this operator defaults `cmp_ratio`,
`ori_mask_mode`, `cmp_mask_mode`, `ori_win_left` and `ori_win_right` to `None` while
the main operator defaults them to `1`, `0`, `0`, `-1` and `-1`, so pass these five
arguments explicitly instead of relying on defaults.

Returns an opaque `int32[2048]` tensor whose contents are implementation defined and
must not be interpreted by callers.

## Parameters and return value

| Parameter | Type / shape / dtype | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `num_heads_q` | `int` | Required | Query head count; on Atlas A2/A3 one of 1, 2, 4, 8, 16, 32, 64, 128. |
| `num_heads_kv` | `int` | Required | KV head count; only 1. |
| `head_dim` | `int` | Required | Head width; only 512. |
| `cu_seqlens_q`, `cu_seqlens_ori_kv`, `cu_seqlens_cmp_kv` | Optional Tensor `[B+1]`; int32 | `None` | Packed TND boundaries starting at zero and non-decreasing. |
| `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv` | Optional Tensor `[B]`; int32 | `None` | Valid length per request. |
| `cmp_residual_kv` | Tensor `[B]`; int32 | `None` | Compression remainder, each value below `cmp_ratio`; required with `cmp_mask_mode=3` and `cmp_ratio!=1`. |
| `ori_topk_length` | Optional Tensor; int32 | `None` | Valid length of `ori_sparse_indices`. |
| `cmp_topk_length` | Optional Tensor; int32; `[T,N2]` (TND) or `[B,S,N2]` | `None` | Valid length of `cmp_sparse_indices`. |
| `batch_size` | `int` | `None` | Batch size for the BSND layout. |
| `max_seqlen_q`, `max_seqlen_ori_kv`, `max_seqlen_cmp_kv` | `int` | `None` | Maximum valid length of each input. |
| `ori_topk` | `int` | `None` | Sparse token count on the original side; only 0 on Atlas A2/A3. |
| `cmp_topk` | `int` | `None` | Compressed TopK length; required for CSA and equal to the last dimension of `cmp_sparse_indices`. |
| `cmp_ratio` | `int` | `None` | Compression ratio: 1 for SWA, 1/2/4 for CSA, 128 for HCA. |
| `ori_mask_mode` | `int` | `None` | `0` unmasked, `3` causal, `4` sliding window. |
| `cmp_mask_mode` | `int` | `None` | `0` unmasked, `3` causal. |
| `ori_win_left`, `ori_win_right` | `int` | `None` | Sliding-window bounds; Atlas A3 requires 127 and 0. |
| `layout_q` | `str` | `None` | `BSND` or `TND`. |
| `layout_kv` | `str` | `None` | `BSND`, `TND`, or `PA_BBND`. |
| `has_ori_kv`, `has_cmp_kv` | `bool` | `None` | Must match whether the main operator receives `ori_kv` and `cmp_kv`. |
| `impl` | `str`, keyword only | `"auto"` | `auto`, `ops_transformers_cann910`, `cann_ops_transformers_cann920`, or `torch_native`. CANN 9.1 explicit sparse inputs require `auto`. |

Returns `metadata`: Tensor, `int32`, shape `[2048]`, on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

boundaries = torch.tensor([0, 16], dtype=torch.int32, device="npu")
seqused_ori = torch.tensor([128], dtype=torch.int32, device="npu")

metadata = sparse_flash_mla_metadata(
    8, 1, 512,
    cu_seqlens_q=boundaries,
    seqused_ori_kv=seqused_ori,
    max_seqlen_q=16,
    max_seqlen_ori_kv=128,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)
print(metadata.shape, metadata.dtype)  # [2048], int32
```
