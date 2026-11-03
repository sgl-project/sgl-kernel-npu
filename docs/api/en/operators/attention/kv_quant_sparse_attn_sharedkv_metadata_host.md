# kv_quant_sparse_attn_sharedkv_metadata_host

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `host` / `auto` | Unsupported | Supported; public-API changes await revalidation |
| `torch_native` | Unsupported | Single-request decode reference; NPU validation pending |

## Signature

`auto` selects `host`.

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_attn_sharedkv_metadata_host

def kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q,
    num_heads_kv,
    head_dim,
    layout_q,
    layout_kv,
    *,
    cu_seqlens_q=None,
    seqused_kv=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    has_ori_kv=True,
    has_cmp_kv=True,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    impl="auto",
):
    ...
```

## Description

Builds the core-scheduling table for DeepSeek-V4 quantized shared-KV attention. The CPU estimates work from query/KV lengths, the window and compression mode, assigns tasks and copies the table to the current NPU. It does not compute attention.

The fixed `int32[1024]` format contains 36 cube slots of 9 integers, followed by 72 vector slots of 8 integers and reserved space. The 128-head mode duplicates each logical cube assignment into adjacent physical core slots. Cross-core Flash Decode splitting is disabled.

## Parameters and return value

Length tensors must be contiguous CPU int32 vectors; NPU lengths are rejected. The output uses the current NPU. Attributes must match the attention call.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `num_heads_q` | int | `—` | — | `Required` | 64 or 128 query heads. |
| `num_heads_kv` | int | `—` | — | `Required` | One shared KV head. |
| `head_dim` | int | `—` | — | `Required` | Must be 512. |
| `layout_q` | str | `—` | — | `Required` | BSND or TND. |
| `layout_kv` | str | `—` | — | `Required` | Must be PA_ND. |
| `cu_seqlens_q` | Tensor or None | `[B+1]` | CPU int32 | `None` | Contiguous cumulative query lengths, starting at zero; required for TND. |
| `seqused_kv` | Tensor or None | `[B]` | CPU int32 | `None` | Contiguous nonnegative valid KV lengths; otherwise use max_seqlen_kv. |
| `batch_size` | int | `—` | — | `0` | 0 infers B from lengths; otherwise use the explicit batch size. |
| `max_seqlen_q` | int | `—` | — | `0` | Required positive query length for BSND; maximum query length for TND. |
| `max_seqlen_kv` | int | `—` | — | `0` | Maximum KV length; required when seqused_kv is absent. |
| `cmp_topk` | int | `—` | — | `0` | Compressed schedule selection count: 0, 512 or 1024. |
| `cmp_ratio` | int | `—` | — | `1` | Use 4 or 128 when has_cmp_kv=True. |
| `ori_mask_mode` | int | `—` | — | `4` | Original-KV sliding-window mode. |
| `cmp_mask_mode` | int | `—` | — | `3` | Compressed-KV mask mode. |
| `ori_win_left` | int | `—` | — | `127` | Nonnegative int32 left window extent; match the attention call. |
| `ori_win_right` | int | `—` | — | `0` | Nonnegative int32 right window extent; match the attention call. |
| `has_ori_kv` | bool | `—` | — | `True` | Whether original KV is present. |
| `has_cmp_kv` | bool | `—` | — | `True` | Whether compressed KV is present; explicitly pass False for original-KV-only attention. |
| `kv_quant_mode` | int | `—` | — | `1` | Only 1 is supported. |
| `tile_size` | int | `—` | — | `64` | Only 64 is supported. |
| `rope_head_dim` | int | `—` | — | `64` | Only 64 is supported. |
| `impl` | str | `—` | — | `"auto"` | auto, host or torch_native; auto selects host. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `metadata` | Tensor | `[1024]` | NPU int32 | Newly allocated schedule; inputs are not modified. |

## host backend

Uses the C++ CPU scheduler for multiple requests, mixed lengths, prefill and compressed modes, then initiates a copy to the NPU. Each call allocates an output. Execute it outside NPU Graph capture and copy refreshed results into a stable buffer before replay. NPU task profiling excludes CPU scheduling time; host latency also needs wall-clock timing and synchronization.

## torch_native backend

Independently constructs a one-request, one-query decode schedule. Requires `max_seqlen_q=1`, positive KV length, the default 127/0 window and original KV. Supports 64/128 heads; compressed modes support 64 heads only. TND requires CPU `cu_seqlens_q=[0,1]`. Other modes raise `NotImplementedError`, without falling back to the host implementation under test.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

kwargs = dict(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="TND", layout_kv="PA_ND",
    cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cpu"),
    seqused_kv=torch.tensor([128], dtype=torch.int32, device="cpu"),
    batch_size=1, max_seqlen_q=1, max_seqlen_kv=128,
    has_ori_kv=True, has_cmp_kv=False,
)
metadata = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs, impl="host")
reference = kv_quant_sparse_attn_sharedkv_metadata_host(**kwargs, impl="torch_native")
torch.testing.assert_close(metadata, reference, rtol=0, atol=0)
# Before graph replay: stable_metadata.copy_(metadata).
# Generate metadata outside graph capture.
```
