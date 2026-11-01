# add_example

```python
from sgl_kernel_npu.operators.examples import add_example
result = add_example(x1, x2, impl="aicpu")
```

Returns a new tensor with elementwise `x1 + x2`. Both inputs remain unchanged.

| Input property | Contract |
| --- | --- |
| Shape | Identical, nonempty; broadcasting is rejected |
| Dtype | Matching `torch.float32` or `torch.int32` |
| Device | Same NPU |
| Layout | Contiguous ND storage |
| Implementation | `auto` selects `aicpu` without fallback; `torch_native` is an independent PyTorch reference |

This is a build/integration example, not a performance recommendation. The initial
accuracy case is shape `(2,)`, `x1=[2,5]`, `x2=[3,3]`, expected `[5,8]`, in both
dtypes. Integer overflow behavior is not validated by this case.

## Source and adaptation

The CpuKernel source/header come from
[ops-transformer AddExample](https://gitcode.com/cann/ops-transformer/tree/39b52ad5e252d98e13df76ed45b6c68b21b98859/examples/add_example/op_kernel_aicpu).
The registered OpType becomes `SglAddExample`; `Compute` and `AddCompute` remain
unchanged apart from repository formatting. The JSON uses the same OpType and the packaged device library name
`libsgl_kernel_npu_aicpu.so`.

Upstream AddExample has no handwritten ACLNN API. This integration adds a small
Host wrapper following
[QuantLightningIndexerV2Metadata's AICPU launcher](https://gitcode.com/cann/ops-transformer/blob/39b52ad5e252d98e13df76ed45b6c68b21b98859/attention/quant_lightning_indexer_v2_metadata/op_host/op_api/quant_lightning_indexer_v2_metadata.cpp).
It validates the restricted contract and enqueues the CpuKernel using
`AicpuTaskSpace` and `ADD_TO_LAUNCHER_LIST_AICPU`. The public API rejects empty
inputs, even though the upstream kernel itself treats them as a no-op.
No AddExample AI Core definition or Tiling source is needed for this path.
The upstream CANN license is retained in `LICENSE-CANN` and included in the RUN
vendor directory and wheel.

## Build and on-device validation

Use the same initialized CANN 9.2.0 and matching PyTorch/torch_npu environment as
PR #921. Mixed builds also require CANN's HCC compiler and AICPU development files.
The following commands use an explicit, fresh checkout location on the NPU host;
no server access or remote execution is performed by these instructions.

```bash
git clone --single-branch --branch feat/aicpu-kernel-build https://github.com/yuejiat0416/sgl-kernel-npu.git /tmp/sgl-kernel-npu-aicpu
cd /tmp/sgl-kernel-npu-aicpu
```

Choose the architecture matching the machine. For A2/A3 (`2201`):

```bash
python /tmp/sgl-kernel-npu-aicpu/setup.py --arch-version 2201 --ops apply_token_bitmask,add_example --artifact wheel --make_clean
```

For A5 (`3510`), use this build command instead:

```bash
python /tmp/sgl-kernel-npu-aicpu/setup.py --arch-version 3510 --ops apply_token_bitmask,add_example --artifact wheel --make_clean
```

Both commands select one existing AI Core operator and the new AICPU example,
producing one wheel. `--make_clean` removes this checkout's previous build
intermediates; it does not modify the installed SDK. Install the resulting wheel
into the active Python environment, keeping its existing dependencies:

```bash
python -m pip install --force-reinstall --no-deps /tmp/sgl-kernel-npu-aicpu/output/*.whl
python -c 'import sgl_kernel_npu; import torch; assert torch.npu.is_available(); print(sgl_kernel_npu.__file__)'
cd /tmp/sgl-kernel-npu-aicpu
python -m pytest /tmp/sgl-kernel-npu-aicpu/tests/python/operators/examples/test_add_example.py --impl aicpu --impl-ref torch_native --skip-acc-check=false --enable-perf=false -ra
```

Expect **15 passed**, not skipped: two accuracy cases plus shape/dtype/device,
empty and noncontiguous rejection checks. Each accuracy case compares against
`torch_native` using the shared test utilities, checks the exact known result,
and verifies that inputs were not changed and output storage is independent.
The rejection cases also exercise the direct `torch.ops.npu.sgl_add_example`
entry. The existing test option `--npu-device` can select a visible device;
omitting it preserves the current device.

To verify automatic dispatch, repeat the test command with `--impl auto`.
For a mixed-package AI Core regression run, use
`/tmp/sgl-kernel-npu-aicpu/tests/python/operators/sampler/test_apply_token_bitmask.py`
with `--impl ascendc --impl-ref torch_native --enable-perf=false`.

## Package scope

The build path is Host ACLNN + AI Core Tiling/kernel + HCC AICPU objects/library,
then one RUN package, its expanded vendor directory, and one wheel. The AICPU
files are installed under `op_impl/cpu/config/` and
`op_impl/cpu/aicpu_kernel/impl/`. The custom ACLNN header joins `op_api/include/`.
Existing wheel staging copies the complete vendor directory and verifies its
contents; the device library is not linked into the Torch extension.

Existing AI Core declarations and explicitly selected AI Core-only builds keep
the PR #921 path. `--ops all` now includes `add_example`, so it is a mixed build
and requires the AICPU toolchain. There is no separate `--aicpu_kernel` switch.

AICPU-only RUN/wheel packaging (`--ops add_example`) is rejected in this PR.
The common Host-library/install-template work is deferred to a separate PR;
see [roadmap issue #8](https://github.com/yuejiat0416/sgl-kernel-npu/issues/8).

## Validation boundary

Local checks cover CMake selection/dependencies, host/compiler separation,
registration merging, install/wheel payload checks, Python dispatch/contracts,
and the unchanged kernel's arithmetic using native C++ with test doubles.
They are not real HCC, CANN RUN/wheel, or NPU accuracy results. Both architectures
still require the on-device build and tests above before this draft is ready.
