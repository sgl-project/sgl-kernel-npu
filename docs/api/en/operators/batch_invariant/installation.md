# Batch-invariant operator installation

The batch-invariant operators are thin wrappers over an external package. The
Python modules ship with the `sgl-kernel-npu` wheel, but two external artifacts
must be installed first:

| Artifact | Provides | Required for |
| --- | --- | --- |
| CANN op package (`.run`) | aclnn kernels such as `aclnnReduceSumBatchInvariant` | every call |
| torch extension (`.zip`) | the `batch_invariant_ops` package and `batch_invariant_ops_lib` | `torch.ops.batch_invariant_ops.*` |

## Download

CANN op package, pick one by device and CPU architecture:

- Ascend 910B, x86_64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run>
- Ascend 910B, aarch64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-910b-1.0.0-linux.aarch64.run>
- A3, aarch64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run>
- A3, x86_64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-A3-1.0.0-linux.x86_64.run>
- Ascend 950, x86_64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.x86_64.run>
- Ascend 950, aarch64: <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/cann-ops-batch_invariant-950-1.0.0-linux.aarch64.run>

torch extension source package:

- <https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260901/batch_invariant-torch_ops_extension-1.0.0.zip>

## Install the CANN op package

The file name below is an example; use the package name of the version you
downloaded.

```bash
chmod +x cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run
./cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run --help     # confirm the installer options
./cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run --quiet    # use your package name
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# The vendor op_api library must be resolvable at run time.
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/batch_invariant/op_api/lib:${LD_LIBRARY_PATH}
```

`--quiet` installs without interactive prompts. Adjust `cann-9.1.0` in the
library path to the CANN version installed on the machine, and set the variable
in every shell that runs these operators.

## Build and install the torch extension

Extract the zip; the resulting directory contains `setup.py` and
`build_and_install.sh`.

```bash
cd batch_invariant_ops
bash build_and_install.sh
```

The script builds `batch_invariant_ops_lib`, packages the wheel and installs it
with pip. It requires an importable `torch_npu`, `ninja`, a C++17 compiler and
the Python development headers. The wheel is bound to the Python version and
CPU architecture, so build it on the target machine.

## Verify

```bash
python -c "import batch_invariant_ops, torch; print(torch.ops.batch_invariant_ops.npu_mm_batch_invariant)"
python -c "from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant; print(mm_batch_invariant)"
cd examples && python test_npu_mm_batch_invariant.py
```

## Notes

- The operator modules import the extension lazily. Importing
  `sgl_kernel_npu.operators.batch_invariant` works without the extension; the
  first operator call fails if it is missing.
- Use a `sgl-kernel-npu` wheel that contains the batch-invariant modules; an
  older wheel has no `sgl_kernel_npu.operators.batch_invariant` package.
- Install the `.run` package that matches the device and CPU architecture; a
  mismatched package fails at call time with an aclnn lookup error.
- Adjust the `set_env.sh` path if CANN is installed elsewhere.
- Do not copy a wheel built on one architecture to another.
