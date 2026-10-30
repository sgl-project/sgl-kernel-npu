# kv_quant_sparse_attn_sharedkv

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` / `auto` | Unsupported | Supported; public-API changes await revalidation |
| `torch_native` | Unsupported | Restricted reference; NPU validation pending |

## Signature

`auto` selects the same native entry point as `ascendc`.

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_attn_sharedkv

def kv_quant_sparse_attn_sharedkv(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    kv_quant_mode=1,
    tile_size=64,
    rope_head_dim=64,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_ND",
    return_softmax_lse=False,
    impl="auto",
):
    ...
```

## Description

Computes shared-KV attention for DeepSeek-V4, including the DSpArk draft path, by dequantizing paged KV, computing QK scores, normalizing and accumulating values. All query heads share one KV head. Sparse indices are supplied by the caller; the operator does not select top-k entries.

$$ z_{h,j} = \mathrm{softmax\_scale}\; q_h^T k_j,\qquad o_h = \frac{\sum_{j\in\mathcal{S}} e^{z_{h,j}} v_j}{e^{s_h}+\sum_{j\in\mathcal{S}} e^{z_{h,j}}} $$

Here $s_h$ is the sink logit and $\mathcal{S}$ is determined by the window, page map or sparse indices. The decoded shared vector is used as both K and V. Query position is aligned to `kv_length - query_length + query_index`.

Each 640-byte row packs `[64 BF16 RoPE values (128 bytes), 448 FP8 NoPE values, 7 E8M0 scales, padding]`. Each 64-value NoPE group is multiplied by `2**(scale_byte-127)`, rounded to BF16 and reordered as `[NoPE, RoPE]`. A plain vector of 640 FP8 values is invalid. Outputs are BF16; reduction order may introduce numerical differences.

## Parameters and return value

All supplied main-operator tensors must be contiguous and on the same NPU as `q`. A `None` default does not make a tensor optional when the native implementation requires it.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `q` | Tensor | `[B,S,H,512] / [T,H,512]` | BF16 | `Required` | Contiguous NPU query; H=64 or 128; BSND or TND. |
| `ori_kv` | Tensor or None | `[blocks,page_size,1,640]` | FP8 E4M3FN | `None` | Required packed original cache; page_size is a multiple of 16 and <=1024. |
| `cmp_kv` | Tensor or None | `[blocks,page_size,1,640]` | FP8 E4M3FN | `None` | Optional compressed cache; requires cmp_block_table and cmp_ratio=4 or 128. |
| `ori_sparse_indices` | Tensor or None | `[T,1,K]` | int32 | `None` | TND physical slot IDs; K=512 or 1024, valid prefix terminated by -1. Requires cmp_kv=None. |
| `cmp_sparse_indices` | Tensor or None | `[T,1,K]` | int32 | `None` | Optional compressed-cache indices; requires cmp_kv. |
| `ori_block_table` | Tensor or None | `[B,ceil(max_KV/page_size)]` | int32 | `None` | Required logical-to-physical page map, including when physical sparse indices are provided. |
| `cmp_block_table` | Tensor or None | `[B,compressed_pages]` | int32 | `None` | Required when cmp_kv is provided. |
| `cu_seqlens_q` | Tensor or None | `[B+1]` | int32 | `None` | Cumulative query lengths starting at zero; required for TND. |
| `cu_seqlens_ori_kv` | Tensor or None | `—` | — | `None` | Unsupported; leave None. |
| `cu_seqlens_cmp_kv` | Tensor or None | `—` | — | `None` | Unsupported; leave None. |
| `seqused_q` | Tensor or None | `[B]` | int32 | `None` | Optional valid query lengths. |
| `seqused_kv` | Tensor or None | `[B]` | int32 | `None` | Required valid original-KV lengths. |
| `sinks` | Tensor or None | `[H]` | float32 | `None` | Required per-head softmax sink logits; their value contribution is zero. |
| `metadata` | Tensor or None | `[1024]` | int32 | `None` | Required task table produced by the matching A5 metadata operator. |
| `kv_quant_mode` | int | `—` | — | `1` | Only 1 is supported. |
| `tile_size` | int | `—` | — | `64` | NoPE quantization group size; only 64 is supported. |
| `rope_head_dim` | int | `—` | — | `64` | BF16 RoPE width; only 64 is supported. |
| `softmax_scale` | float | `—` | — | `1.0` | Scale applied to QK logits; callers typically pass 1/sqrt(512). |
| `cmp_ratio` | int | `—` | — | `1` | 1 for original KV; 4 or 128 for compressed-KV modes. |
| `ori_mask_mode` | int | `—` | — | `4` | Only sliding-window mode 4 is supported. |
| `cmp_mask_mode` | int | `—` | — | `3` | Only mode 3 is supported. |
| `ori_win_left` | int | `—` | — | `127` | Nonnegative int32 left window extent. |
| `ori_win_right` | int | `—` | — | `0` | Nonnegative int32 right window extent. |
| `layout_q` | str | `—` | — | `"BSND"` | BSND or TND. |
| `layout_kv` | str | `—` | — | `"PA_ND"` | Only PA_ND is supported. |
| `return_softmax_lse` | bool | `—` | — | `False` | True is unsupported; the second output is an empty placeholder. |
| `impl` | str | `—` | — | `"auto"` | auto, ascendc or torch_native; auto selects ascendc. |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `attention_out` | Tensor | Same as q | BF16 | Newly allocated output; inputs are not modified. |
| `softmax_lse` | Tensor | `[0]` | float32 | Empty placeholder on the same device. |

## ascendc backend

Launches the native AscendC kernel on Ascend 950 for BSND/TND, paged windows and restricted sparse/compressed paths. Requires a wheel built with A5 kernels. Metadata must match the attention window, lengths and compression configuration.

The attention call can be captured in an NPU Graph. Generate host metadata outside capture and copy it into the graph's stable NPU buffer before replay.

## torch_native backend

Independently unpacks KV and computes attention using PyTorch on the CPU, then copies the result to the query device. Supports only BF16 BSND uncompressed paged windows. TND, explicit sparse indices, compressed KV and capture are unsupported. This is an accuracy reference, not a device-kernel performance baseline.

## Example

```python
import math

import torch
import torch_npu

from sgl_kernel_npu.operators.attention import (
    kv_quant_sparse_attn_sharedkv,
    kv_quant_sparse_attn_sharedkv_metadata_host,
)

lengths_cpu = torch.tensor([128], dtype=torch.int32, device="cpu")
metadata = kv_quant_sparse_attn_sharedkv_metadata_host(
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    layout_q="BSND", layout_kv="PA_ND",
    seqused_kv=lengths_cpu, batch_size=1,
    max_seqlen_q=1, max_seqlen_kv=128,
    has_cmp_kv=False, impl="host",
)
# Valid packed zero KV: RoPE/NoPE values are zero; E8M0 scale 127 is 1.0.
rows = torch.zeros((1, 128, 1, 640), dtype=torch.uint8, device="cpu")
rows[..., 576:583] = 127
kv = rows.view(torch.float8_e4m3fn).to("npu")
q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device="npu")
output, lse = kv_quant_sparse_attn_sharedkv(
    q, ori_kv=kv,
    ori_block_table=torch.zeros((1, 1), dtype=torch.int32, device="npu"),
    seqused_kv=lengths_cpu.to("npu"),
    sinks=torch.zeros(64, dtype=torch.float32, device="npu"),
    metadata=metadata, softmax_scale=1 / math.sqrt(512), impl="ascendc",
)
assert output.shape == q.shape and lse.numel() == 0
```
