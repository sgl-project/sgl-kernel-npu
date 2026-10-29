# causal_conv1d_target_verify

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

```python
from typing import Optional, Union
import torch
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify

def causal_conv1d_target_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    validate_data: bool = False,
    *,
    impl="auto",
) -> torch.Tensor:
    ...
```

`auto` selects `triton`; omitting impl has the same effect. There is no runtime backend fallback.

## Description

For width K, history h is ordered oldest to newest and comes from committed state or ancestor candidates:

$$u_{t,d}=b_d+\sum_{j=0}^{K-2}w_{d,j}h_{t,d,j}+w_{d,K-1}x_{t,d},\qquad y_{t,d}=\mathrm{act}(u_{t,d}).$$

Without bias, b=0; act is identity or SiLU (`u*sigmoid(u)`). Computation uses FP32 and casts back to x.dtype. Scratch stores the subsequent K-1 history values including the current **raw input**, preserving extra window columns. Linear mode overwrites x and returns an alias. Tree mode preserves x, returns a new tensor, and derives/mutates parents from child/sibling links. Both modes mutate scratch and leave committed conv_state unchanged; accepted state is committed separately by the caller.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, D]` or `[B, D, S]` | Floating point; output uses `x.dtype`, and the wrapper does not enumerate a supported-dtype set | Required | Candidate tokens. Rank 2 is treated as `S = 1`; `B` and `S` must be non-empty. |
| `conv_state` | `torch.Tensor` | `[cache_lines, D, state_len >= K - 1]` | Floating point; history uses the tensor's element dtype | Required | Committed convolution state; read only. |
| `weight` | `torch.Tensor` | `[D, K]` or `[K, D]` | Floating point; loaded into FP32 computation | Required | Per-channel convolution weight; `K` must be in `2..4`. |
| `bias` | `torch.Tensor` or `None` | `[D]` or — | Floating point; loaded into FP32 computation | `None` | Optional per-channel bias. |
| `activation` | `bool`, `str`, or `None` | — | — | `None` | `True` maps to `"silu"`; `False` maps to `None`. Strings may be `"silu"` or `"swish"`. |
| `conv_state_indices` | `torch.Tensor` | One-dimensional, `[>= B]` | Integer; no explicit dtype check | `None`, but required | Selects the committed state row for each batch item. An entry equal to `pad_slot_id` makes that candidate inactive. |
| `intermediate_conv_window` | `torch.Tensor` | `[scratch_size, steps >= S, D, window >= K - 1]` | Floating point; no explicit supported-dtype list | `None`, but required | Scratch windows updated in place for active candidates. |
| `intermediate_state_indices` | `torch.Tensor` | One-dimensional, `[>= B]` | Integer; no explicit dtype check | `None`, but required | Selects the scratch row for each batch item. |
| `retrieve_next_token` | `torch.Tensor` or `None` | `[rows >= B, steps >= S]` or — | Integer tree mapping | `None` | Enables tree mode when provided; the three `retrieve_*` arguments must then all be provided. |
| `retrieve_next_sibling` | `torch.Tensor` or `None` | `[rows >= B, steps >= S]` or — | Integer tree mapping | `None` | Next-sibling mapping for tree mode. |
| `retrieve_parent_token` | `torch.Tensor` or `None` | `[rows >= B, steps >= S]` or — | Integer tree mapping | `None` | Receives the derived parent mapping in tree mode; updated in place for active candidates. |
| `pad_slot_id` | `int` | — | — | `-1` | Sentinel used to identify inactive state slots. |
| `validate_data` | `bool` | — | — | `False` | Enables additional stride checks; it is not a numerical validation mode. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `auto`, `triton`, `torch_native` |

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | Same as x | `x.dtype` | Aliases x in linear mode; new tensor in tree mode. |

All tensors must be on the same NPU. Callers supply valid indices, sufficient storage, and initialized history, and prevent conflicting state writes; wrappers do not validate every value range.

## triton backend

Supports widths 2..4, both weight layouts, and strided views. If both weight axes match the channel count, [D,K] takes precedence. validate_data=True requires channel stride=1 for x/state, width stride=1 for [D,K] weight, or channel stride=1 for [K,D] weight. The three state/scratch parameters are required despite their None defaults.

## torch_native backend

An independent CPU numerical reference: detaches inputs, copies them to CPU, evaluates FP32 loops, then copies results and scratch/parent to the original device. Preserves linear aliasing and tree input preservation. Its latency is not a fast NPU PyTorch baseline.

Importing the public registry requires NPU architecture detection, including for `torch_native`. Support is limited to accepted A3 cases; architecture registration does not establish coverage of every chip, shape or dtype.

## Example

Requires sgl-kernel-npu, torch_npu and an NPU runtime. The example explicitly selects `torch_native`. `triton` and `auto` are also available; isolate mutable state when comparing backends.

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify
b, d, s, k = (1, 4, 2, 3)
x = torch.randn(b, d, s, device='npu', dtype=torch.bfloat16)
conv_state = torch.randn(1, d, k - 1, device='npu', dtype=torch.bfloat16)
weight = torch.randn(d, k, device='npu', dtype=torch.bfloat16)
windows = torch.zeros(1, s, d, k - 1, device='npu', dtype=torch.bfloat16)
out = causal_conv1d_target_verify(
    x,
    conv_state,
    weight,
    activation='silu',
    conv_state_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    intermediate_conv_window=windows,
    intermediate_state_indices=torch.tensor([0], device='npu', dtype=torch.int64),
    impl='torch_native',
)
```
