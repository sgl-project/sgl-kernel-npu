# kv_compress_epilog

Quantize KV-cache rows and scatter them to the slots selected by
`slot_mapping`. The cache is updated in place. A slot of `-1` skips the input
row; slot `0` is valid, and unwritten slots retain their contents.

The public API supports an integer compatibility interface and an SDK-selected
DSV4 packed-cache interface. `quant_mode` determines which interface is used.

## Hardware support

| Implementation | arch22 / A3 | arch35 / A5 |
| --- | --- | --- |
| `auto` | Unsupported | Supported |
| `ascendc` | Unsupported | Supported |
| `cann_ops_transformers_cann920` | Unsupported | Supported |
| `torch_native` | Supported | Supported |

`torch_native` is an independent correctness reference. Production execution
uses A5 and requires the matching installed kernel wheel and initialized SDK.

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(
    kv_compress_cache,
    x,
    slot_mapping,
    quant_group_size=64,
    quant_mode=None,
    round_scale_flag=None,
    layout=None,
    *,
    round_scale=True,
    x_scale=1.0,
    impl="auto",
):
    ...
```

## Calling modes

### Integer compatibility mode

Pass integer `quant_mode=1` or `2`, together with `round_scale_flag` and
`layout`. This retains the original native interface:

- `1`: FP8 values with one FP32 scale per 128 NoPE values.
- `2`: MXFP8 values with one E8M0 exponent byte per group of 64 or 128 NoPE values.

The final 64 BF16 columns of `x` are the RoPE suffix. In layout `1`, FP8 packs
quantized NoPE values, RoPE bytes, then scales; MXFP8 packs RoPE bytes, quantized
NoPE values, then scales. Each row is aligned to 128 bytes. Layout `2` supports
MXFP8 only: it stores all value rows before the scale region within each block,
with scale storage rounded up to 8 bytes per slot.

`auto` selects `ascendc` for integer mode on either supported CANN version.
Keep `round_scale=True` and `x_scale=1.0`; scale rounding is controlled by
`round_scale_flag` in this interface. The call returns `None`.

### SDK-selected DSV4 packed mode

Omit `quant_mode` or pass the string matching the initialized SDK. This
interface uses BF16 input rows of width 512: 448 NoPE values and 64 RoPE values,
with a quantization group size of 64.

| Initialized SDK | Selected `quant_mode` | Scale format | Packed row width | `auto` backend |
| --- | --- | --- | --- | --- |
| CANN 9.1.x | `"fp8_e8m0"` | E8M0, 1 byte per group | 640 bytes | `ascendc` |
| CANN 9.2.x | `"fp8_bf16"` | BF16, 2 bytes per group | 608 bytes | `cann_ops_transformers_cann920` |

Allocate the cache with `get_npu_kv_quant_path()` before calling. The writer and
the matching mixed-quant attention reader must use the same format. A cache
width or explicit mode that does not match the selected SDK is rejected.
The packed row contains 128 bytes of BF16 RoPE, 448 bytes of FP8 NoPE, then
7 scales and alignment padding. Alignment padding has no specified numerical
value and should be excluded from accuracy comparisons.

Leave `round_scale_flag` and `layout` as `None` in packed mode. CANN 9.1 requires
`round_scale=True` and `x_scale=1.0`. CANN 9.2 forwards `round_scale` to the
released writer; use `x_scale=1.0` for this FP8 contract. The call returns the
same mutated cache tensor.

## Parameters

Use input tensors on the same NPU for production execution.

| Parameter | Type | Default | Contract |
| --- | --- | --- | --- |
| `kv_compress_cache` | Tensor | Required | Mutable cache. Integer mode: `float8_e4m3fn` or `float8_e5m2`, with layout-dependent dimensions. Packed mode: byte storage (`uint8` or a `float8_e4m3fn` view), shape `[num_pages, page_size, 1, row_bytes]`, with `row_bytes` selected above. |
| `x` | BF16 Tensor `[N, D]` | Required | Dense input rows with `N > 0` and a 64-column RoPE suffix. Packed mode requires `D=512`. |
| `slot_mapping` | int32 / int64 Tensor `[N]` | Required | One destination slot per input row. Valid slots must be in cache bounds and unique; `-1` skips a write. |
| `quant_group_size` | int | `64` | Integer MXFP8: `64` or `128`; integer FP8 uses fixed 128-value groups. Packed mode requires `64`. |
| `quant_mode` | int / str / None | `None` | Integer `1`/`2` selects the compatibility interface. `None` selects the SDK-specific packed format; an explicit string must match that format. |
| `round_scale_flag` | bool / None | `None` | Required in integer mode. For MXFP8, increments each nonzero biased scale exponent by one when true. Leave unset in packed mode. |
| `layout` | int / None | `None` | Required in integer mode: `1` for packed slot rows, `2` for separate block value/scale regions. Layout `2` is MXFP8 only. Leave unset in packed mode. |
| `round_scale` | bool, keyword only | `True` | Packed-mode scale rounding. CANN 9.1 requires `True`. Keep the default in integer mode. |
| `x_scale` | float, keyword only | `1.0` | Keep `1.0` for the supported FP8/MXFP8 contracts. CANN 9.1 packed mode and the integer interface enforce this value; the released FP8 writer reserves this attribute. |
| `impl` | str, keyword only | `"auto"` | `"auto"`, `"ascendc"`, `"cann_ops_transformers_cann920"`, or `"torch_native"`. |

In integer layout `1`, the cache accepts slot rows, including
`[slots, 1, width]`. Layout `2` accepts
`[blocks, block_size, width]` or `[blocks, block_size, 1, width]`.

## Return value

| Calling mode | Return value | Mutation |
| --- | --- | --- |
| Integer compatibility | `None` | Updates the supplied cache in place. |
| SDK-selected DSV4 packed | The supplied `kv_compress_cache` tensor | Updates the same cache in place. |

## Backend selection

- `auto`: integer mode uses the repository's `ascendc` implementation. Packed
  mode selects the backend from the initialized CANN 9.1/9.2 version.
- `ascendc`: provides the integer interface on A5 and the 640-byte packed
  interface on CANN 9.1. The packed interface rejects a CANN 9.2 SDK.
- `cann_ops_transformers_cann920`: provides the 608-byte packed interface on
  CANN 9.2 and rejects integer mode or another SDK version.
- `torch_native`: provides independent reference implementations of both
  interfaces. For packed mode, it selects the format from the cache width,
  allowing both encodings to be checked without switching SDKs.

Execution failures propagate to the caller; `auto` never retries another
backend or falls back to `torch_native`.

## Examples

### Integer MXFP8 interface

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog
import torch

x = torch.ones((2, 192), device="npu", dtype=torch.bfloat16)
slots = torch.tensor([1, -1], device="npu", dtype=torch.int32)
cache = torch.zeros((2, 384), device="npu", dtype=torch.uint8).view(torch.float8_e4m3fn)
result = kv_compress_epilog(
    cache, x, slots,
    quant_group_size=64,
    quant_mode=2,
    round_scale_flag=True,
    layout=1,
    impl="ascendc",
)
assert result is None
```

### SDK-selected DSV4 cache

```python
from sgl_kernel_npu.operators.mem_cache import (
    get_npu_kv_quant_path,
    kv_compress_epilog,
)
import torch

path = get_npu_kv_quant_path()
cache = torch.zeros(
    (1, 128, 1, path.packed_dim(448, 64)), device="npu", dtype=torch.uint8
)
x = torch.randn((1, 512), device="npu", dtype=torch.bfloat16)
slots = torch.tensor([0], device="npu", dtype=torch.int32)
result = kv_compress_epilog(cache, x, slots, impl="auto")
assert result is cache
```

## Testing

Run the operator tests on an A5 host after installing the current wheel and
initializing the target CANN SDK:

```bash
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog.py \
  --impl auto --impl-ref torch_native --skip-acc-check=false --enable-perf=false -v -s -rs
```

Sources: [repository Ascend C kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/26ec66efe01f8acef43db55da679107702876b23/csrc/kv_compress_epilog/op_kernel/kv_compress_epilog.cpp),
[released CANN writer](https://gitcode.com/cann/ops-transformer/blob/3baf8ac0eba25fa464260a43bab8d5bac1fefb8a/attention/kv_compress_epilog/docs/torchapi_kv_compress_epilog.md).
