# causal_conv1d_target_verify_npu

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

def causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    intermediate_conv_window=None,
    intermediate_state_indices=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    retrieve_parent_token=None,
    pad_slot_id=-1,
    validate_data=False,
    *,
    impl="auto",
):
    ...
```

## Description

Computes the per-channel causal convolution for speculative target verification, with an optional bias and SiLU activation. Each active candidate's post-convolution window is written to scratch at `intermediate_state_indices`. The accepted window is committed later by the caller, so the committed `conv_state` remains read only.

A linear chain consumes tokens in order. Tree verification instead follows `retrieve_next_token` and `retrieve_next_sibling`, and writes the derived parent mapping to `retrieve_parent_token`.

## Parameters

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
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"` or `"triton"`. |

Use mutually compatible floating-point tensors on the same NPU. The implementation does not comprehensively validate dtype combinations, and this page does not claim support for untested combinations.

## Return value and in-place state

Returns a tensor with the same logical shape, dtype, and device as `x`; a rank-2 input produces a rank-2 output.

- **Linear mode** (`retrieve_next_token is None`): the returned tensor aliases `x`, whose values are overwritten as tokens are consumed. `retrieve_next_sibling` and `retrieve_parent_token` must also be `None`.
- **Tree mode** (all three `retrieve_*` tensors provided): returns a newly allocated `empty_like(x)`, leaves `x` unchanged, and updates `retrieve_parent_token`.
- `intermediate_conv_window` is updated in both modes. `conv_state`, `weight`, optional `bias`, `conv_state_indices`, `intermediate_state_indices`, `retrieve_next_token`, and `retrieve_next_sibling` are read only; in particular, `conv_state` is never modified by this operator.

## triton backend

- Requires non-`None` `conv_state_indices`, `intermediate_conv_window`, and `intermediate_state_indices`. It supports kernel widths 2 through 4 and both weight layouts.
- With `validate_data=True`, it additionally requires channel-contiguous `x` and `conv_state` (`stride(1) == 1`), width-contiguous `[D, K]` weight, or channel-contiguous `[K, D]` weight. The general wrapper still expects valid index ranges and compatible layouts.
- `auto` aliases `triton`, without dynamic selection or fallback. Registry import still performs ACL architecture detection.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mamba import causal_conv1d_target_verify_npu

b, d, s, k = 1, 4, 2, 3
x = torch.randn(b, d, s, device="npu", dtype=torch.bfloat16)
conv_state = torch.randn(1, d, k - 1, device="npu", dtype=torch.bfloat16)
weight = torch.randn(d, k, device="npu", dtype=torch.bfloat16)
windows = torch.zeros(1, s, d, k - 1, device="npu", dtype=torch.bfloat16)

out = causal_conv1d_target_verify_npu(
    x,
    conv_state,
    weight,
    activation="silu",
    conv_state_indices=torch.tensor([0], device="npu", dtype=torch.int64),
    intermediate_conv_window=windows,
    intermediate_state_indices=torch.tensor([0], device="npu", dtype=torch.int64),
)
# Linear mode: out aliases x, and windows contains the scratch checkpoints.
```
