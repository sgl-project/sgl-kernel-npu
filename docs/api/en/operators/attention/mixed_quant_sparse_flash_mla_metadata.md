# Mixed-quant attention metadata

## Hardware support

| Implementation | ARCH22 / A3 | ARCH35 / A5 |
| --- | --- | --- |
| `cann_ops_transformer` | unsupported | declared; validate SDK kernel |
| `torch_native` | unsupported | declared; validate SDK kernel |

## Signature

```python
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

def mixed_quant_sparse_flash_mla_metadata(*, impl='auto', **kwargs):
    ...
```

`auto` selects the initialized SDK production path: original bindings on CANN 9.1, released bindings on CANN 9.2. Execution failures never trigger another backend.

## Description

A5 only; use the same packing and implementation as mixed_quant_sparse_flash_mla.

Prepare scheduling for mixed-quant attention. Explicit sparse visibility is translated to the vendor window convention inside the adapter.

## Parameters

| Name | Type | Default | Contract |
| --- | --- | --- | --- |
| impl | str | 'auto' | Registered implementation; auto selects the SDK path; never falls back to torch_native |

### Keyword arguments

| Name | Type | Contract |
| --- | --- | --- |
| ori_kv / cmp_kv | Tensor / None | Original/optional compressed paged KV [pages,page_size,1,D]; BF16 or SDK-selected packed bytes. |
| ori_block_table / cmp_block_table | int32 Tensor / None | [B,max_pages] logical-to-physical page tables. |
| cu_seqlens_q / seqused_q | int32 Tensor / None | [B+1] cumulative query offsets / [B] query lengths. |
| seqused_ori_kv / seqused_cmp_kv / cmp_residual_kv | int32 Tensor / None | [B] original/compressed/residual KV lengths; configure consistently with cmp_ratio. |
| ori_sparse_indices / cmp_sparse_indices | int32 Tensor / None | Logical Top-K positions; -1 padding. |
| ori_topk_length / cmp_topk_length | int32 Tensor / None | Per-row valid Top-K lengths; legacy support is restricted. |
| sinks | Tensor / None | Optional per-head softmax sink logits. |
| metadata | Tensor / ReferenceMetadata / None | Build with matching metadata implementation; native bytes are not reference descriptors. |
| softmax_scale / cmp_ratio | float / int | Attention scale / compression ratio; reference defaults 1.0 / 1. |
| ori_mask_mode / cmp_mask_mode | int | Reference supports original explicit 0 or SWA 4; compressed 0 or causal 3. |
| ori_win_left / ori_win_right | int | Original visibility window; reference defaults 127 / 0. |
| layout_q / layout_kv | str | Reference defaults TND / PA_BBND; BSND queries also supported. |
| return_softmax_lse | bool | Default False; unused LSE is an empty FP32 tensor. |
| quant_mode / rope_head_dim | int / int | Mixed-quant mode 1 and 64 RoPE features in the migrated layout. |
| num_heads_q / num_heads_k / head_dim / batch_size / max_seqlen_q | int | Native scheduling geometry; use the same values as paired compute. |
| has_ori_kv / has_cmp_kv | bool | Whether original/compressed streams participate. |
| cu_seqlens_q_cpu / seqused_ori_kv_cpu | CPU int32 Tensor / None | Existing host mirrors required by legacy explicit sparse metadata; no device-to-host copy is introduced. |

## Return value

| Name | Type | Contract |
| --- | --- | --- |
| metadata | Native Tensor or ReferenceMetadata | Opaque native scheduling storage, or reference-only descriptor. Never compare bytes across implementations. |

## cann_ops_transformer

Retains the existing registration name; the compatibility wrapper selects original or released bindings by SDK. Metadata/compute pairing and A5 KV reader/writer format checks run before allocation or execution.

## torch_native

Independent eager PyTorch reference for correctness; never an automatic production or graph-capture fallback. Metadata descriptors are not native tiling bytes. Modes outside the documented reference domain are rejected.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

q = torch.randn(1, 1, 512, device="npu", dtype=torch.bfloat16)
i32 = lambda values: torch.tensor(values, device="npu", dtype=torch.int32)
kw = dict(cu_seqlens_q=i32([0, 1]), seqused_ori_kv=i32([1]),
          ori_block_table=i32([[0]]), layout_q="TND", layout_kv="PA_BBND",
          ori_mask_mode=4, cmp_mask_mode=3, cmp_ratio=1)
from sgl_kernel_npu.operators.kvcacheio import get_npu_kv_quant_path, kv_compress_epilog
path = get_npu_kv_quant_path()  # A5 only.
cache = torch.empty(1, 128, 1, path.packed_dim(448, 64), device="npu", dtype=torch.uint8)
kv_compress_epilog(cache, q[:, 0], i32([0]), impl="auto")
kw.update(quant_mode=1, rope_head_dim=64)
metadata = mixed_quant_sparse_flash_mla_metadata(impl="torch_native", **kw)
```

Build/install the current wheel and initialize the SDK first. The example demonstrates the API; it does not establish NPU accuracy, performance or model inference.

```bash
python3 -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla_metadata.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

[Upstream operators](https://github.com/Ascend/ops-transformer)
