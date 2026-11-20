# dynamic_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Supported | Supported |
| `torch_native` | Supported | Supported |

The registry exposes the implementations on both generations. The installed
`torch_npu` and CANN versions determine the vendor options available on each device.

## Signature

```python
from sgl_kernel_npu.operators.activation import dynamic_quant

def dynamic_quant(*args, impl="auto", **kwargs):
    ...
```

`auto` and `torch_npu` call `torch_npu.npu_dynamic_quant`. `torch_native`
selects the PyTorch reference. The registry dispatches the requested implementation
without falling back to a different implementation.

## Description

The `torch_npu` implementation forwards positional and keyword arguments to
`torch_npu.npu_dynamic_quant` and returns its result unchanged. The vendor
operator supports dynamic quantization modes including per-token quantization.
Supported arguments, dtypes, shapes and modes depend on the installed vendor
version and device. See the
[torch_npu API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md).

For the KPool indexer query, the caller applies Hadamard rotation, reshapes
the query to `[rows, 128]` and specifies `dst_type=torch.int8`. In this
per-row symmetric case, the scale is computed from the row's maximum absolute
value and the INT8 range. The caller restores the original query shape and
converts the scale to the dtype required by the indexer.

## Parameters and return values

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `*args` | — | Required vendor input | Positional arguments passed to the selected implementation. |
| `impl` | `str` | `"auto"` | Keyword-only implementation selector, accepting `"auto"`, `"torch_npu"` or `"torch_native"`. |
| `**kwargs` | — | — | Arguments passed to the selected implementation, such as `dst_type`, `smooth_scales`, `group_index` or `quant_mode` when supported by the vendor version. |

The return value is the selected implementation's `(payload, scale)` tuple.
For the INT8 per-token example below, the payload has the input shape and
`torch.int8` dtype, and the scale has shape `[rows]` and `torch.float32`
dtype. Other modes follow the installed vendor API.

## torch_npu backend

Calls `torch_npu.npu_dynamic_quant(*args, **kwargs)`. The wrapper preserves
the input arguments and the returned tensors, including their shapes and dtypes.

## torch_native backend

The reference in `implementation/torch_native/activation/dynamic_quant.py`
implements symmetric per-row INT8 quantization along the last dimension. It
returns an INT8 payload and one FP32 scale per row, with a zero scale for an
all-zero row. It accepts `x` and `dst_type` and is intended for accuracy
comparison. Vendor options such as smoothing and other quantization modes are
outside this reference's coverage.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_quant

x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
payload, scale = dynamic_quant(x, dst_type=torch.int8)
assert payload.shape == (64, 128)
assert payload.dtype == torch.int8
assert scale.shape == (64,)
assert scale.dtype == torch.float32
```

## Limitations

- The vendor implementation's supported inputs and options depend on the
  installed `torch_npu`, CANN and device versions.
- The Torch reference covers per-row symmetric INT8 quantization.
