# mixed_quant_sparse_flash_mla

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| cannbot_dsl | Unsupported | Supported |
| torch_native | Unsupported | Supported |

## Signature

`from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla`

```python
def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
    impl="auto"
): ...
```

## Behavior

Computes sparse MLA over BF16 queries, FP8 original KV, and optional FP4 compressed KV.

BF16 TND `q[T1,64,512]`, UINT8 paged original KV, sparse indices and block tables are required for active KV sides. `quant_mode=1`; metadata must come from `mixed_quant_sparse_flash_mla_metadata` with matching configuration. The caller allocates `out` and `lse`.

Semantics (BF16 rounding at quantized compute boundaries):

```text
z_i = softmax_scale * dot(q, dequant(k_i)); out = sum_i exp(z_i) * dequant(v_i) / (exp(sink) + sum_i exp(z_i)); lse = log(exp(sink) + sum_i exp(z_i))
```

Original KV stores E4M3FN values with one BF16 scale per 32 features. Compressed KV stores signed E2M1 values with one BF16 scale per 16 features, low nibble first. Both use `dequant(value, scale) = value * scale` without a zero point and round the dequantized values to BF16. The PV weights are rounded to BF16 before the value product. The sink contributes to the softmax denominator and LSE, with zero value contribution.

## Parameters

T is query rows, B requests, P physical pages and S slots per page.

| Parameter | Default | Type, shape and meaning |
| --- | --- | --- |
| `q` | Required | BF16 [T,64,512], attention queries. |
| `ori_kv` | `None` | UINT8 [P,S,1,544], FP8 KV bytes plus BF16 scales; required by cannbot_dsl. |
| `cmp_kv` | `None` | Optional UINT8 [P,S,1,320], packed FP4 KV bytes plus BF16 group scales. |
| `ori_sparse_indices` | `None` | INT32 [T,1,Kori], required original logical token IDs; -1 is padding. |
| `cmp_sparse_indices` | `None` | Optional INT32 [T,1,Kcmp], compressed logical token IDs; -1 is padding. |
| `ori_block_table` | `None` | Required INT32 [B,max_ori_blocks], original KV page map. |
| `cmp_block_table` | `None` | Optional INT32 [B,max_cmp_blocks], compressed KV page map. |
| `cu_seqlens_q` | `None` | INT32 [B+1], required by cannbot_dsl; cumulative boundaries ending at T. |
| `seqused_q` | `None` | Optional INT32 [B]; when supplied, must equal cu_seqlens_q spans (no padded rows). |
| `seqused_ori_kv` | `None` | Optional INT32 [B], original KV lengths; retained for API compatibility. |
| `seqused_cmp_kv` | `None` | Optional INT32 [B], compressed KV lengths; retained for API compatibility. |
| `ori_topk_length` | `None` | Optional INT32 [T,1], valid sparse prefix; None uses all original index columns. |
| `cmp_topk_length` | `None` | Optional INT32 [T,1], valid sparse prefix; None uses all compressed index columns. Omit when cmp_kv is absent. |
| `sinks` | `None` | FP32 [64], attention-sink logits (zero value contribution). |
| `metadata` | `None` | INT32 [1024] device schedule required by cannbot_dsl; torch_native ignores it. |
| `quant_mode` | Required | int, use 1 for the supported quantized profile. |
| `softmax_scale` | `None` | Optional float, defaults to 1/sqrt(512). |
| `layout_q` | `'TND'` | str, use TND (flattened query tokens). |
| `layout_kv` | `'PA_BBND'` | str, use PA_BBND packed paged KV. |
| `return_softmax_lse` | `False` | bool, write lse when True. |
| `out` | Required | Writable BF16 [T,64,512] output tensor. |
| `lse` | Required | Writable FP32 [1,T,64] when return_softmax_lse=True; otherwise pass an empty FP32 [0] tensor. |
| `impl` | `'auto'` | str, auto selects cannbot_dsl; torch_native is an independent compute reference. |

## Returns

| Value | Meaning |
| --- | --- |
| None | Writes out and, when requested, lse in place. |

## cannbot_dsl

`auto` selects this backend. Requires A5 and a compatible CANNBot-DSL runtime installed in the environment. This operator does not enforce a runtime version. Inputs share a device. First use triggers JIT compilation; warm actual shapes before graph capture.

## torch_native

Independent CPU Torch accuracy reference; results are copied back to the input device, preserving in-place semantics. Does not require DSL. Not intended for performance or NPU graph capture.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.attention import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

q = torch.randn(1, 64, 512, device="npu", dtype=torch.bfloat16)
kv = torch.zeros(1, 128, 1, 544, device="npu", dtype=torch.uint8)
kv[..., 512:] = torch.ones(1, 128, 1, 16, device="npu", dtype=torch.bfloat16).view(
    torch.uint8
)
cu = torch.tensor([0, 1], device="npu", dtype=torch.int32)
length = torch.tensor([[128]], device="npu", dtype=torch.int32)
metadata = mixed_quant_sparse_flash_mla_metadata(
    length,
    torch.zeros_like(length),
    cu_seqlens_q=cu,
    num_heads_q=64,
    num_heads_kv=1,
    head_dim=512,
    quant_mode=1,
    has_cmp_kv=False,
)
out = torch.empty_like(q)
lse = torch.empty(1, 1, 64, device="npu")
mixed_quant_sparse_flash_mla(
    q,
    ori_kv=kv,
    ori_sparse_indices=torch.arange(128, device="npu", dtype=torch.int32).reshape(
        1, 1, -1
    ),
    ori_block_table=torch.zeros(1, 1, device="npu", dtype=torch.int32),
    cu_seqlens_q=cu,
    ori_topk_length=length,
    sinks=torch.zeros(64, device="npu"),
    metadata=metadata,
    quant_mode=1,
    out=out,
    lse=lse,
    return_softmax_lse=True,
)
```

## Device test

```bash
python -m pytest tests/python/operators/attention/test_mixed_quant_sparse_flash_mla.py --npu-device=0 --enable-perf=false -v -s
```
