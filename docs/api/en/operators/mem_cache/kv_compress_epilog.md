# kv_compress_epilog

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `ascendc` | Not supported | Supported |
| `torch_native` | Supported | Supported |
| `cpu` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import kv_compress_epilog

def kv_compress_epilog(kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode, round_scale_flag, layout, *, impl="auto"):
    ...
```

## Description

Writes the caller's cache in place and returns None. `x` is positive-size 2D BF16, `slot_mapping` is int32/int64 with one index per input row (`-1` skips a row), and cache is float8_e4m3fn/float8_e5m2 storage. Inputs must provide the dense storage required by the kernel; the entry does not cast or make them contiguous. Slot destinations must be unique for deterministic parallel writes.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `kv_compress_cache` | `torch.Tensor` | Depends on layout | float8_e4m3fn / float8_e5m2 | Required | Packed cache written in place; capacity and strides must fit the layout. |
| `x` | `torch.Tensor` | `[N, D]` | bfloat16 | Required | Input rows; the last 64 columns contain RoPE values. |
| `slot_mapping` | `torch.Tensor` | `[N]` | int32 / int64 | Required | Destination slots; -1 skips a row, valid slots must be distinct. |
| `quant_group_size` | `int` | — | — | Required | 64 or 128 for MXFP8; FP8 uses fixed groups of 128 columns. |
| `quant_mode` | `int` | — | — | Required | 1 for FP8; 2 for MXFP8. |
| `round_scale_flag` | `bool` | — | — | Required | Whether to increment a nonzero biased exponent for MXFP8. |
| `layout` | `int` | — | — | Required | 1 for slot-major storage; 2 for MXFP8 block storage. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; auto / ascendc / torch_native / cpu. auto selects ascendc without fallback. |

Returns `None`; outputs are written into caller-owned buffers in place.

The last 64 BF16 columns are copied byte-exactly. Mode 1 quantizes preceding columns in groups of 128 and stores float32 scales. Mode 2 uses groups of 64 or 128 and E8M0 scale bytes; its rounding flag advances a nonzero biased exponent by one. FP32 intermediate arithmetic and the kernel's coefficient bit patterns are retained in the PyTorch computation.

Layout 1 stores `[quant | rope | scales | zero padding]` for FP8 and `[rope | quant | scales | zero padding]` for MXFP8. Rows use cache stride(0); a `[slots,1,width]` view is supported. Layout 2 supports MXFP8 only and stores a block's value region followed by its scale region. Four-dimensional `[blocks, block_size, 1, width]` cache views are squeezed by the adapter. Cache capacity and strides must accommodate the packed storage; unwritten slots and block tail remain unchanged.

## Backend implementations

`ascendc` and `auto` are ARCH35-only; there is no automatic fallback. `torch_native` computes on the input device. `cpu` performs packing on CPU and copies bytes into the original output; the two PyTorch backends share packing arithmetic. Tests explicitly select `impl_ref="cpu"`. This backend does not make the package importable without its NPU runtime dependencies.

CANN type `SglKvCompressEpilog` receives cache stride as an internal attribute, with a Ref cache output. Host tiling computes the row/core/UB partition.

## Build and test

The 15 numeric cases cover both FP8 formats, int32/int64 slots, D=192/320/512, groups 64/128, rounding on/off, skipped/shuffled slots, zeros, 4096-row batches, partial blocks, 3D/4D caches and padded row strides. Another test rejects FP8 layout 2. Quant values use FP8/MXFP8 tolerances, FP32 scales use rtol=1e-5/atol=1e-7, and RoPE bytes, stored E8M0 exponents, layout 1 padding and unwritten slots compare exactly. In layout 2 the alignment bytes after the last exponent of each written slot are unspecified. Independent output buffers avoid shared-input mutation during accuracy checks; performance uses make_executor/check_performance.

```bash
python3 setup.py --make_clean --arch-version 3510 --ops kv_compress_epilog --artifact wheel
python3 -m pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
python3 -m pytest tests/python/operators/mem_cache/test_kv_compress_epilog.py \
  --impl=ascendc --impl-ref=cpu --enable-perf=false -v -s
```
