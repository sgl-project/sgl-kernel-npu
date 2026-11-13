# causal_conv1d_update

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_update


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    intermediate_conv_window=None,
    pad_slot_id=-1,
    metadata=None,
    validate_data=False,
    *,
    impl="auto"
): ...
```

## Description

Updates depthwise convolution history for decode or speculative multi-token execution. Decode without acceptance metadata returns a new output. The speculative path writes the output into x and can also save a history window after each token. The state layout is [pool,channels,history].

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `x` | Tensor | `Required` | Floating [B,C] or [B,C,T] for update; [C,total_tokens] or [B,C,T] for prefill. |
| `conv_state` | Tensor | `Required` | Floating [pool,channels,history] writable state, enough capacity for width+steps-2. |
| `weight` | Tensor | `Required` | Convolution: [channels,width]. Gemma: [H], same dtype as input. |
| `bias` | Tensor or None | `None` | Optional [channels] convolution bias. |
| `activation` | str, bool or None | `None` | None disables activation; silu/swish enables SiLU. Update also accepts bool. |
| `cache_seqlens` | Tensor or None | `None` | Reserved [batch] metadata; this path does not implement circular-cache indexing. |
| `conv_state_indices` | Tensor or None | `None` | Integer [batch] state slot mapping. |
| `num_accepted_tokens` | Tensor or None | `None` | Integer [batch]; speculative history starts at accepted_count-1. |
| `intermediate_conv_window` | Tensor or None | `None` | Writable [pool,steps,channels,width-1] per-step history. |
| `pad_slot_id` | int or None | `-1` | Padding slot sentinel; update Triton path skips matching requests. |
| `metadata` | object or None | `None` | Reserved and unused. |
| `validate_data` | bool | `False` | Enable the existing update metadata assertions. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | Output with x shape/dtype; updates conv_state and optional intermediate windows, and writes x on the speculative path. |

## triton

Uses PyTorch for ordinary decode and Triton for speculative execution.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_update

x = torch.randn(1, 128, device="npu", dtype=torch.float16)
w = torch.randn(128, 4, device="npu", dtype=x.dtype)
state = torch.zeros(1, 128, 3, device="npu", dtype=x.dtype)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
y = causal_conv1d_update(x, state, w, conv_state_indices=slots, impl="triton")
```
