# paged_mtp_attention_decode

## Hardware support

| Implementation | A3 (arch22) | A5 (arch35) |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

def paged_mtp_attention_decode(
    *,
    query,
    k_cache,
    v_cache,
    seq_lens,
    cu_q_lens,
    block_tables,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    gamma,
    softmax_scale,
    max_kv_len,
    causal=False,
    impl="auto",
):
    ...
```

All arguments are keyword-only. `auto` selects `triton` without fallback.

## Description

Paged grouped-query attention for static DSpark draft blocks. Each of B requests
has exactly G=`gamma` query rows; Hq=`num_q_heads`, Hkv=`num_kv_heads`,
D=`head_dim`, and Hq/Hkv must be integral. Q uses TND layout; K/V use FIA's
ND paged layout.

For request b, query i, head h, and KV position j, let R=Hq/Hkv,
P=page_size, g=floor(h/R), p=block_tables[b, floor(j/P)], and r=j mod P.
K and V are the D values starting at g*D in the corresponding cache[p, r].

$$
S_{bhij}=\mathrm{softmax\_scale}\sum_d Q_{bG+i,h,d}K_{bj,g,d},\qquad
O_{bG+i,h,d}=\sum_{j\in A_{bi}}\mathrm{softmax}_{A_{bi}}(S_{bhi})_j V_{bj,g,d}.
$$

For Lb=`seq_lens[b]`, noncausal visibility is `0 <= j < Lb`. Causal visibility
additionally requires `j <= Lb - G + i`: queries are right aligned to the valid
KV end, which includes the current query block. An empty visibility set produces
exact zeros, including padded requests with Lb=0.

`max_kv_len` only controls work partitioning; valid reads always use `seq_lens`.
Unused cache tails are ignored. Ragged query blocks and sliding windows are
unsupported. Inputs are not modified. Floating-point reduction order and
intermediate rounding can differ between implementations.

## Parameters and return value

All tensor inputs must share a device. Triton requires NPU storage. The reference
accepts CPU or NPU tensors, but importing the package/registry still requires
the configured native runtime, torch_npu, ACL and an available NPU.

| Parameter | Type | Shape / default | Description and constraints |
| --- | --- | --- | --- |
| `query` | Tensor | `[B*G, Hq, D]`, required | float16 or bfloat16; TND layout; arbitrary strides supported. Valid values must be finite. |
| `k_cache` | Tensor | `[pages, P, Hkv*D]`, required | Same dtype as query; ND layout with final stride 1. Outer strides may be noncontiguous. Valid KV values must be finite. |
| `v_cache` | Tensor | Same shape as k_cache, required | Same dtype and layout constraints as k_cache. |
| `seq_lens` | Tensor | `[B]`, required | Contiguous int32; valid KV end per request. Nonnegative, within the request's page-table capacity. Zero denotes padding. |
| `cu_q_lens` | Tensor | `[B+1]`, required | Contiguous int32; values must equal `arange(B+1)*G`. |
| `block_tables` | Tensor | `[B, max_pages]`, required | int32 or int64; arbitrary strides supported. Referenced entries must index physical cache pages. Entries outside the valid KV range may be invalid, including all entries for empty requests. |
| `page_size` | int | Required | Power of two, at least 16; equals the cache's second dimension. |
| `num_q_heads` | int | Required | Positive Hq, divisible by Hkv. |
| `num_kv_heads` | int | Required | Positive Hkv. |
| `head_dim` | int | Required | D in `[16, 256]`; equals query.shape[-1]. |
| `gamma` | int | Required | Positive fixed query width G. |
| `softmax_scale` | float | Required | Finite score multiplier; normally `D**-0.5`. |
| `max_kv_len` | int | Required | Nonnegative scheduling bound, normally the allocated per-request KV capacity. Does not mask or truncate valid KV values. |
| `causal` | bool | `False` | Enables the right-aligned causal mask. |
| `impl` | str | `"auto"` | `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype / device | Description |
| --- | --- | --- | --- | --- |
| `output` | Tensor | `[B*G, Hq, D]` | Same as query | Newly allocated result; B=0 returns an empty tensor. Output strides are not part of the API contract. |

The public API checks metadata before dispatch. Tensor-value constraints on
lengths, offsets, page indices and finite data are caller responsibilities;
the Triton path does not copy lengths to the CPU to validate them.

## triton backend

Supports FP16/BF16 paged attention on NPU and NPU Graph capture. Lengths,
page tables and Q/K/V are read from device buffers on each replay without
extracting lengths on the host.

## torch_native backend

An independent correctness reference with FP32 intermediates and output cast
to query.dtype. It reads lengths and offsets on the host and does not support
NPU Graph capture.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention import paged_mtp_attention_decode

device = "npu:0"
args = dict(
    query=torch.randn(7, 16, 64, dtype=torch.bfloat16, device=device),
    k_cache=torch.randn(2, 128, 256, dtype=torch.bfloat16, device=device),
    v_cache=torch.randn(2, 128, 256, dtype=torch.bfloat16, device=device),
    seq_lens=torch.tensor([129], dtype=torch.int32, device=device),
    cu_q_lens=torch.tensor([0, 7], dtype=torch.int32, device=device),
    block_tables=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    page_size=128,
    num_q_heads=16,
    num_kv_heads=4,
    head_dim=64,
    gamma=7,
    softmax_scale=64**-0.5,
    max_kv_len=256,
    causal=True,
)
output = paged_mtp_attention_decode(**args)
reference = paged_mtp_attention_decode(**args, impl="torch_native")
torch.testing.assert_close(output, reference, atol=0.004, rtol=0.03)
```
