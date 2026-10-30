# elementwise_add

This directory contains the Ascend C source of the `elementwise_add` sample
operator (`z = x + y`) used by the tutorial
[`docs/developer_guide/ascendc_operator_sample.md`](../../docs/developer_guide/ascendc_operator_sample.md).

| File | Role |
| --- | --- |
| `op_kernel/kernel_elementwise_add.cpp` | Device side: two Ascend C kernel entries (`elementwise_add_fp16`, `elementwise_add_fp32`), `CopyIn -> Compute -> CopyOut` with double buffering. |
| `op_host/elementwise_add.cpp` | Host side: input validation, UB-aware tiling, padding of the element count, kernel launch through `EXEC_KERNEL_CMD`. |

## Build integration

The operator is part of the ordinary `sgl_kernel_npu` extension build. Its two
sources are listed in:

- `cmake/legacy_kernels.cmake` (`OP_SRCS` and `NO_WORKSPACE_KERNEL_SRCS`);
- `csrc/pytorch_extensions.cpp` (`npu::elementwise_add` schema and PrivateUse1
  implementation);
- `include/sgl_kenel_npu_ops.h` (`sglang::npu_kernel::elementwise_add`).

Build and install it with the usual flow:

```bash
cd <repo root>
./build.sh -a kernels          # add a SOC_VERSION if auto-detection is wrong
pip install -e python/sgl_kernel_npu
```

## Layout contract

The host pads the element count to `blockDim * tileLength` before launching, so
the kernel always walks whole tiles. `tileLength` is derived from the UB size of
the current device (512-element aligned, capped at 8192 elements). If you change
the number of queues or buffers in the kernel, update `IN_FLIGHT_TILES` in the
host file accordingly, otherwise the UB allocation can overflow.
