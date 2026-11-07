# BF16 hyper-connection epilogue (Ascend C)

Native vector kernels behind `sgl_kernel_npu.operators.norm.grouped_gemma_rmsnorm`,
`sgl_kernel_npu.operators.hyper_conn.hc_mix` and
`sgl_kernel_npu.operators.hyper_conn.hc_combine`. The public contracts,
parameter tables and architecture support matrices are in
[`docs/api/en/operators/norm/grouped_gemma_rmsnorm.md`](../../docs/api/en/operators/norm/grouped_gemma_rmsnorm.md),
[`docs/api/en/operators/hyper_conn/hc_mix.md`](../../docs/api/en/operators/hyper_conn/hc_mix.md),
[`docs/api/en/operators/hyper_conn/hc_combine.md`](../../docs/api/en/operators/hyper_conn/hc_combine.md)
and their Chinese counterparts. This file documents the native implementation,
the tiling tables and how they were measured.

`SGL_KERNEL_ENABLE_A3_ONLY_OPS` gates these four internal native helpers:
`torch.ops.npu.hyper_conn_grouped_norm`,
`torch.ops.npu.hyper_conn_scaled_silu`, `torch.ops.npu.hyper_conn_branch_mix`
and `torch.ops.npu.hyper_conn_branch_combine`. Despite its name,
`SGL_KERNEL_ENABLE_A3_ONLY_OPS` is an `arch22`-family toggle enabled for every
SoC outside the Ascend950 series (both A2 and A3); measurements exist on A3
only, so the API pages mark A2 unverified. Callers should invoke the three
public operators above rather than calling these bindings directly. Other dtypes
and widths are rejected before launch.

## Sources

| Path | Contents |
| --- | --- |
| `op_host/hyper_conn_epilogue.cpp` | The four `HOST_API` entry points: input checks, optional `out` buffer validation where supported (or output allocation), block count and column tile selection, kernel launch. |
| `op_host/hyper_conn_tiling.h` | The measured block-count and column-tile tables, the nearest-neighbour lookup rules, and the Unified Buffer budget check. |
| `op_kernel/hyper_conn_epilogue.cpp` | The four `KERNEL_TYPE_AIV_ONLY` kernels. |
| `op_kernel/hyper_conn_limits.h` | Shared `MAX_BRANCHES` constant used by host validation and the combine kernel. |
| `op_kernel/hyper_conn_round.h` | Tile/alignment constants and `SigmoidF32`, shared by the kernels. |

The kernels are wired into the legacy Torch extension build
(`cmake/legacy_kernels.cmake`), like every other Ascend C operator that has not
been migrated to the CANN operator package track under `csrc/cmake`.

## What the cluster computes

One transformer block with `hc` hyper-connection branches holds its residual as
`[rows, hc * width]`. Per sub-block:

```text
residual --grouped_gemma_rmsnorm--> normed
  └── hyper_conn_grouped_norm

normed, down_weight, up_weight --hc_mix--> block input
  └── F.linear(down) -> hyper_conn_scaled_silu -> F.linear(up) -> hyper_conn_branch_mix

block_output, residual, normed, inject_weight --hc_combine--> residual
  └── F.linear(inject) -> hyper_conn_branch_combine
```

The projections inside `hc_mix` and `hc_combine` execute through PyTorch's NPU
`F.linear` (`aclnn`), including pre-converted `FRACTAL_NZ` up-projection
weights, while the four vector stages execute as Ascend C AIV kernels.

## Numerics

Every kernel evaluates its intermediate arithmetic in fp32 throughout and rounds
to bf16 once when writing its output tensor to global memory (`activated` before
the `up_weight` projection in `hc_mix`, and each kernel's final output),
matching the upstream SGLang CUDA/Triton kernels (`hc_combine.cuh`,
`grouped_gemma_rmsnorm.cuh`, `hc_mix_triton.py`). The `torch_native` references
follow the same fp32-throughout contract and evaluate sigmoid as
`1 / (1 + exp(-x))` rather than calling `torch.sigmoid`. fp32 to bf16 uses
`CAST_RINT`; over 327803 random values it is bitwise equal to torch's
`.bfloat16()`.

Bitwise equality between `ascendc` and `torch_native` is not claimed:

1. `AscendC::Exp` and `torch.exp` are distinct fp32 transcendental
   implementations (~1 fp32 ulp), which can straddle a bf16 rounding
   tie/boundary.
2. `hyper_conn_grouped_norm` accumulates `sum(x^2)` with `ReduceSum` (tree
   reduction vs `torch.mean`'s reduction order), multiplies by a precomputed
   `invWidth = 1.0f / width` reciprocal (which can differ by 1 fp32 ulp from
   division on non-power-of-two widths such as 24 or 48), and computes the
   inverse RMS via `Sqrt` + `Div` rather than `torch.rsqrt` (`AscendC::Rsqrt` on
   `dav_c220` emits a raw `vrsqrt` approximation without Newton-Raphson
   refinement).

Public operator accuracy tests enforce a tightened `BF16_TIGHT` tolerance
(`rtol = 8e-3`, 1–2 BF16 ulp).

## Tiling tables

`op_host/hyper_conn_tiling.h` holds a measured sweep rather than a formula.
Block dispatch on this device costs about 0.088 us per block (an empty vector
kernel measures 1.48 us at 4 blocks and 5.37 us at 48), and these kernels are
vector-ALU bound, so at decode row counts the optimum sits well below the 48
available vector cores. A model fitted to the trivial kernel did not transfer.

The sweep ran on Ascend910_9382 with `hc = 4`, `width` in {2048, 2560} and rows
in {1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 2048}, taking the minimum average
device task duration over the grid of block counts and column tiles. Device
microseconds of each vector stage at the optimum, `width = 2048`:

| rows | 1 | 8 | 16 | 32 | 64 | 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| `hyper_conn_grouped_norm` | 2.16 | 3.75 | 4.92 | 6.12 | 8.44 | 89.4 |
| `hyper_conn_scaled_silu` | 1.48 | 2.08 | 2.34 | 2.73 | 3.28 | 13.0 |
| `hyper_conn_branch_mix` | 2.19 | 3.52 | 4.19 | 5.40 | 7.58 | 66.3 |
| `hyper_conn_branch_combine` | 2.31 | 3.31 | 4.14 | 5.54 | 8.22 | 69.3 |

Lookup rules, all in `hyper_conn_tiling.h`:

- An untabulated width uses the nearest tabulated column.
- An untabulated row count uses the next larger tabulated row, or the largest.
  Row counts between 65 and 1022 therefore interpolate from the 128 entry and
  have not been measured individually.
- `hyper_conn_scaled_silu` keys on rows only; its optimum does not track the
  column count.
- The block count is clamped to the task count and to the device's vector core
  count, so a device with fewer than 48 cores stays within its own range.
- `hyper_conn_branch_mix` and `hyper_conn_branch_combine` share one column tile
  ladder. The tuned tile is used when it divides `width` and fits the Unified
  Buffer budget; otherwise the fallback loop selects the largest divisor that
  fits. Because `width % 16 == 0` holds, that loop always finds at least
  `tile = 16`, so rejecting a width with no fitting divisor is a defensive guard
  kept in case alignment or minimum-tile rules change.
