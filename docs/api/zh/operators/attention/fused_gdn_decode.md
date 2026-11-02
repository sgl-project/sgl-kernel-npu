# fused_gdn_decode

## 硬件支持

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 未验证 | 未验证 |
| `torch_native` | 未验证 | 未验证 |

## Signature

```python
from sgl_kernel_npu.operators.attention import fused_gdn_decode
def fused_gdn_decode(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source, initial_state_indices, scale=None, use_qk_l2norm_in_kernel=False, cu_seqlens=None, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

执行单 token sigmoid 门控 delta-rule 更新，并原地写入选中的连续状态槽。

## 参数与返回值

q,k: [B,T,H,K]; v: [B,T,HV,V]; a,b: [N,HV]; physical state slot: [HV,V,K]. T=1 without cu_seqlens; flattened varlen uses one token per sequence. A non-contiguous state is copied by the legacy Triton wrapper. Returns [B,T,HV,V].

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import fused_gdn_decode

q = torch.randn(1, 1, 2, 64, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(1, 1, 4, 64, device="npu", dtype=torch.float16)
a = torch.randn(1, 4, device="npu", dtype=torch.float16)
b = torch.randn_like(a)
A = torch.randn(4, device="npu")
bias = torch.randn(4, device="npu")
state = torch.zeros(1, 4, 64, 64, device="npu")
idx = torch.tensor([0], device="npu", dtype=torch.int32)
y = fused_gdn_decode(A, a, bias, 1.0, 20.0, q, k, v, b, state, idx, impl="triton")
```
