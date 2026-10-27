# apply_token_bitmask

## Architecture support

Generation names follow the shared operators framework: `arch22` and `arch35`.
Their architecture IDs are `2201` and `3510`, respectively; the build declaration
uses `SUPPORTED_ARCHS 2201 3510`. Implementation scope and runtime validation are
separate.

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Implemented; refactor requires NPU validation | Implemented; requires NPU validation |
| `torch_native` | Implemented; requires NPU validation | Implemented; requires NPU validation |

## Signature

```python
from sgl_kernel_npu.operators.sampler import apply_token_bitmask

apply_token_bitmask(logits, bitmask, indices=None, *, impl="auto")
```

`auto` and `ascendc` call `torch.ops.npu.apply_token_bitmask`. If the native operator is not registered, the Torch lookup fails; there is no fallback. Unknown implementation names are rejected by the shared registry. Select `torch_native` explicitly to use the reference. Install a wheel built for your device's architecture. The installed native implementation remains directly callable through `torch.ops.npu.apply_token_bitmask`.

## Parameters and semantics

| Parameter | Shape, type and constraints |
| --- | --- |
| `logits` | Contiguous NPU Tensor `[B, V]`, float32/float16/bfloat16; `B >= 0`, `V > 0`. |
| `bitmask` | Contiguous int32 Tensor `[B, W]` on the same NPU. Each word controls 32 tokens, starting at its least significant bit. Missing words act as zero; extra columns are ignored. |
| `indices` | Optional one-dimensional Tensor on CPU or the same NPU. Converted to int64 before indexing, preserving the existing behavior. Prefer int32/int64 and valid indices. Both `None` and an empty Tensor select all rows. |
| `impl` | `auto`, `ascendc`, `torch_native`; defaults to `auto`. |

Zero bits write negative infinity; one bits preserve the original value. Only selected rows change. The operator **mutates logits in place and returns a Tensor aliasing the same storage**. It does not modify bitmask or indices. The shared C++ tiling struct keeps all nine fields; the adapter pads and crops around the Kernel's 256-element alignment without requiring callers to change `V`.

The Kernel uses uint32 indices. Working dimensions, accessed offsets and tiling arithmetic must fit that range; unsupported inputs fail instead of being truncated. No new autograd implementation or validated graph-capture/compiled-mode support is claimed.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sampler import apply_token_bitmask

logits = torch.randn(4, 320, device="npu", dtype=torch.float16)
mask = torch.full((4, 10), -1, device="npu", dtype=torch.int32)
expected = apply_token_bitmask(logits.clone(), mask, impl="torch_native")
result = apply_token_bitmask(logits, mask)  # auto -> ascendc; in place
assert result.data_ptr() == logits.data_ptr()
torch.testing.assert_close(result, expected, atol=0, rtol=0)
```

Every REFACTOR wheel includes the Python interfaces, but only migrated native operators selected by `--ops` are compiled. Subset wheels share one distribution identity and replace each other; they are not additive. The complete selection is recorded in `_build_manifest.json`.
