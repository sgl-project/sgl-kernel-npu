# dynamic_mx_quant

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | Unsupported | Supported |
| `torch_native` | Unsupported | Supported |

Both implementations are registered for arch35. Vendor support also depends
on the installed `torch_npu` and CANN versions.

## Signature

```python
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

def dynamic_mx_quant(*args, impl="auto", **kwargs):
    ...
```

`auto` and `torch_npu` call `torch_npu.npu_dynamic_mx_quant`. `torch_native`
selects the PyTorch reference. The registry dispatches the requested implementation
without falling back to a different implementation.

## Description

The `torch_npu` implementation forwards positional and keyword arguments to
`torch_npu.npu_dynamic_mx_quant` and returns its result unchanged. The vendor
operator computes a shared scale for each block along the selected axis.
Supported options, defaults, layouts and dtypes depend on the installed vendor
version. See the
[torch_npu API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md).

For the KPool query case, the caller supplies `axis=-1` and
`dst_type=torch.float8_e4m3fn`. With the vendor's per-32 block configuration,
each group of 32 values has an E8M0 scale. For an input `[rows, 128]`,
the vendor result contains four scale bytes per row, arranged as
`[rows, 2, 2]`. The caller views the returned bytes as E8M0 for the indexer.

## Parameters and return values

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `*args` | — | Required vendor input | Positional arguments passed to the selected implementation. |
| `impl` | `str` | `"auto"` | Keyword-only implementation selector, accepting `"auto"`, `"torch_npu"` or `"torch_native"`. |
| `**kwargs` | — | — | Arguments passed to the selected implementation, such as `axis`, `dst_type`, `round_mode`, `block_size` or `scale_alg` when supported by the vendor version. |

The return value is the selected implementation's `(payload, scales)` tuple.
For the E4M3FN per-32 example below, the payload has the input shape and
`torch.float8_e4m3fn` dtype. The scale tensor has dtype `torch.uint8` and
shape `[rows, ceil(D / 64), 2]`; scale bytes encode E8M0 values, and an odd
number of blocks is padded to a pair. Other options follow the installed vendor API.

## torch_npu backend

Calls `torch_npu.npu_dynamic_mx_quant(*args, **kwargs)`. The wrapper preserves
the input arguments and the returned tensors, including their shapes and dtypes.

## torch_native backend

The reference in `implementation/torch_native/activation/dynamic_mx_quant.py`
implements last-axis per-32 E4M3FN quantization with paired E8M0 scale bytes.
It accepts `x` and `dst_type` and is intended for accuracy comparison. Pass
`dst_type=torch.float8_e4m3fn` when comparing with the vendor implementation.
The reference does not implement the vendor's other axes, block sizes or modes.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.activation import dynamic_mx_quant

x = torch.randn(64, 128, dtype=torch.bfloat16, device="npu")
payload, scales = dynamic_mx_quant(x, axis=-1, dst_type=torch.float8_e4m3fn)
assert payload.shape == (64, 128)
assert payload.dtype == torch.float8_e4m3fn
assert scales.shape == (64, 2, 2)
assert scales.dtype == torch.uint8
```

## Limitations

- The registry makes both implementations available on arch35.
- The vendor implementation's supported options depend on the installed
  `torch_npu` and CANN versions.
- The Torch reference covers last-axis per-32 E4M3FN quantization.
