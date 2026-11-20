# kv_quant_sparse_flash_attention_torch_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ops-transformer` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `ops-transformer` wherever it is registered.

```python
from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention_torch_npu


def kv_quant_sparse_flash_attention_torch_npu(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Computes scaled attention over positions chosen by sparse_indices, using the SDK's quantized key/value encoding. Attention is evaluated after dequantization over the selected set only. In MLA mode the query includes a rotary tail, while the returned value dimension is the latent width.

## Parameters and return value

Tensor arguments use the same NPU device unless the table specifies host metadata. The examples exercise one SDK-supported configuration.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `query` | Tensor | [B, Sq, Hq, D + Dr], BF16 in the example | Required | Concatenated latent and rotary query for MLA. |
| `key`, `value` | Tensor | paged [P, L, Hkv, ...], INT8 in the example | Required | Separate K and V arguments. The example stores the INT8 key, BF16 rotary bytes and FP32 scales together in key. |
| `sparse_indices` | Tensor | [B, Sq, Hkv, K], INT32 | Required | Selected logical KV positions for each query and KV head. |
| `scale_value` | float | - | Required | Factor applied to the attention logits. |
| `key_quant_mode`, `value_quant_mode` | int | - | Required | Vendor encoding modes, both 2 for the INT8 example. |
| `block_table`, `actual_seq_lengths_query`, `actual_seq_lengths_kv` | Tensor or None | [B, max_pages], [B], [B], INT32 | `None` | Optional paged metadata. Required by the example. |
| `layout_query`, `layout_kv`, `attention_mode` | str, str, int | - | `"BSND"` / `0` | Optional SDK settings. Example uses BSND, PA_BSND and MLA mode 2. |
| `rope_head_dim`, `tile_size`, `quant_scale_repo_mode` | int | - | `64` / `128` / `1` | Optional rotary width, quantization tile width and scale storage mode. Values must match how key is encoded. |
| `impl` | str | - | `"auto"` | `auto`, `ops-transformer`, `torch_native`. |
| `**kwargs` | dict | - | - | Additional supported TorchNPU keyword arguments. The linked vendor contract specifies their defaults and constraints. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `out` | Tensor | [B, Sq, Hq, D] with BF16 in the example | Attention output on the input NPU. For MLA, D excludes the rotary tail. |

## ops-transformer backend

Calls `torch_npu.npu_kv_quant_sparse_flash_attention`. Shape, dtype and layout combinations follow the installed SDK. Uses the vendor signature with separate key and value parameters, including the requested quantization modes. Call the packed FP8 repository API only with its own cache layout and signature. [TorchNPU reference](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/test/test_custom_ops/test_npu_kv_quant_sparse_flash_attention.py) [CANN ops-transformer source](https://gitcode.com/cann/ops-transformer/tree/7326dcc4f1e2dde5d2a802458877f927570267d1/attention/kv_quant_sparse_flash_attention)

## torch_native backend

Decodes packed per-tile INT8 or E4M3 KV records with scales stored in the same record and computes sparse MLA for TND or BSND queries and PA_BSND caches.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention_torch_npu

b, pages, page, h, d, dr, tile = 4, 128, 256, 128, 512, 64, 128
q = torch.randn((b, 1, h, d + dr), dtype=torch.bfloat16, device="npu")
value = torch.zeros((pages, page, 1, d), dtype=torch.int8, device=q.device)
key_rope = torch.zeros((pages, page, 1, dr), dtype=q.dtype, device=q.device)
scales = torch.ones((pages, page, 1, d // tile), dtype=torch.float32, device=q.device)
key = torch.cat((value, key_rope.view(torch.int8), scales.view(torch.int8)), dim=-1)
positions = torch.arange(2048, dtype=torch.int32, device=q.device).view(1, 1, 1, -1)
positions = positions.expand(b, 1, 1, -1).contiguous()
table = torch.arange(pages, dtype=torch.int32, device=q.device).reshape(b, -1)
q_lens = torch.ones((b,), dtype=torch.int32, device=q.device)
kv_lens = torch.full((b,), 4096, dtype=torch.int32, device=q.device)
out = kv_quant_sparse_flash_attention_torch_npu(
    q,
    key,
    value,
    positions,
    (d + dr) ** -0.5,
    2,
    2,
    block_table=table,
    actual_seq_lengths_query=q_lens,
    actual_seq_lengths_kv=kv_lens,
    layout_query="BSND",
    layout_kv="PA_BSND",
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile,
    rope_head_dim=dr,
    impl="ops-transformer",
)
```
