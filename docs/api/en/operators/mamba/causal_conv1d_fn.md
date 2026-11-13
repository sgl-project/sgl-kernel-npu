# causal_conv1d_fn

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `torch_native`.

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_fn


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=-1,
    *,
    impl="auto",
    **kwargs
): ...
```

## Description

Applies depthwise causal convolution to packed variable-length or padded batched sequences, with optional initial state, bias and SiLU. It updates selected final-state cache slots and returns the convolution result; packed input padding after the last query is zero-filled.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | Floating [B,C] or [B,C,T] for update; [C,total_tokens] or [B,C,T] for prefill. |
| `weight` | Tensor | `Required` | Convolution: [channels,width]. Gemma: [H], same dtype as input. |
| `bias` | Tensor or None | `None` | Optional [channels] convolution bias. |
| `query_start_loc` | Tensor | `None` | INT32/INT64 [batch+1] cumulative packed lengths beginning at zero. |
| `cache_indices` | Tensor | `None` | Integer [batch] cache slot for each sequence. |
| `has_initial_state` | Tensor | `None` | Boolean [batch] indicating which requests use existing history. |
| `conv_states` | Tensor | `None` | Prefill: [pool,channels,width-1]; rollback: [layers,pool,window,channels]. |
| `activation` | str, bool or None | `'silu'` | None disables activation; silu/swish enables SiLU. Update also accepts bool. |
| `pad_slot_id` | int or None | `-1` | Padding slot sentinel; update Triton path skips matching requests. |
| `impl` | str | `'auto'` | auto selects torch_native; explicit backend selection. |
| `kwargs` | dict | `Required` | Additional compatibility arguments, currently unused. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | Output with the same shape/dtype as x; updates conv_states. |

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_fn

x = torch.randn(128, 8, device="npu", dtype=torch.float16)
w = torch.randn(128, 4, device="npu", dtype=x.dtype)
starts = torch.tensor([0, 8], device="npu", dtype=torch.int32)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
flags = torch.tensor([False], device="npu")
state = torch.zeros(1, 128, 3, device="npu", dtype=x.dtype)
y = causal_conv1d_fn(
    x,
    w,
    query_start_loc=starts,
    cache_indices=slots,
    has_initial_state=flags,
    conv_states=state,
    impl="torch_native",
)
```
