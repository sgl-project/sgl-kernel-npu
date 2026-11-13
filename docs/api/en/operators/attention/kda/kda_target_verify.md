# kda_target_verify

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.attention.kda import kda_target_verify


def kda_target_verify(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    scale=None,
    gates_are_preactivated=None,
    impl="auto"
): ...
```

## Description

Runs fixed-width KDA speculative verification: normalize Q/K, decay the recurrent state per channel, apply the gated delta update, and project each updated state onto Q. Initial states are read-only. Per-step snapshots are written to selected scratch slots; negative source indices select zero state and negative destination indices skip snapshots. Raw or preactivated gates and grouped query/key heads are supported.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | Floating [heads] (or equivalent singleton-expanded shape) decay weights. |
| `dt_bias` | Tensor | `Required` | GDN: [heads]; KDA: [Hk,K] gate bias. |
| `q` | Tensor | `Required` | KDA: [1,tokens,Hq,K]; FIA: FP16/BF16 [tokens,QH,D]. |
| `k` | Tensor | `Required` | Floating [1,tokens,Hk,K] keys. |
| `v` | Tensor | `Required` | Floating [1,tokens,Hv,V] values; Hv divisible by Hq and Hk. |
| `a` | Tensor | `Required` | GDN: [tokens,heads]. KDA: [tokens,Hk,K] with optional leading singleton. |
| `b` | Tensor | `Required` | Floating [tokens,Hv], optionally [1,tokens,Hv] for KDA. |
| `initial_state_source` | Tensor | `Required` | Floating [pool,Hv,V,K] read-only initial states; strided V/K supported. |
| `initial_state_indices` | Tensor | `Required` | INT32/INT64 [batch] source slots; negative means zero state. |
| `intermediate_states_buffer` | Tensor | `Required` | Writable FP32/BF16 [scratch,cache_steps,Hv,V,K] snapshots, not aliasing initial state. |
| `intermediate_state_indices` | Tensor | `Required` | INT32/INT64 [batch] destination slots; negative skips writes. |
| `cache_steps` | int | `Required` | Positive steps per request; total tokens must be divisible by it. |
| `scale` | float or None | `None` | QK scaling; KDA defaults to key_dim**-0.5 and requires positive values. |
| `gates_are_preactivated` | bool or None | `None` | True: a is log-decay and b is sigmoid output. None infers true when both have leading singleton dimensions. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | [1,tokens,Hv,V], same dtype/device as v; writes snapshot buffer. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.kda import kda_target_verify

q = torch.randn(1, 4, 2, 128, device="npu", dtype=torch.bfloat16)
k, v = torch.randn_like(q), torch.randn_like(q)
a = -torch.rand(1, 4, 2, 128, device="npu")
b = torch.rand(1, 4, 2, device="npu")
initial = torch.zeros(1, 2, 128, 128, device="npu")
snapshots = torch.empty(1, 4, 2, 128, 128, device="npu")
slot = torch.tensor([0], device="npu", dtype=torch.int32)
y = kda_target_verify(
    A_log=torch.zeros(2, device="npu"),
    dt_bias=torch.zeros(2, 128, device="npu"),
    q=q,
    k=k,
    v=v,
    a=a,
    b=b,
    initial_state_source=initial,
    initial_state_indices=slot,
    intermediate_states_buffer=snapshots,
    intermediate_state_indices=slot,
    cache_steps=4,
    gates_are_preactivated=True,
    impl="triton",
)
```
