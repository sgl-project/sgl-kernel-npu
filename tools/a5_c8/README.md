# Pinned A5 C8 runtime

The Ascend 950 C8 path combines two independently sourced components:

- CopySFA is the repository-owned direct AscendC kernel under
  `csrc/attention/fused_quant_scatter_copy_sparse_flash_attention`.
- LIM is the original ACLNN/OPP implementation from vLLM-Ascend PR 18383,
  pinned to revision `7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e`.

`INTEGRATION.json` records the donor revisions and the CopySFA Git objects
after relocation into the `attention` domain, before the v2 adaptations.
`UPSTREAM_PROVENANCE.json` records hashes for the current adapted kernel files.

## Build

Initialize the selected CANN environment, then build an Ascend 950 kernel
wheel normally:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a kernels Ascend950
```

For A5 builds, `build.sh` prepares and compiles the pinned original LIM OPP in
an isolated temporary directory, installs it into a private root, verifies its
source and installed-file receipts, and stages the verified OPP into the
wheel. It does not install or overwrite a system OPP.

To reuse a completed private build, point the build at the exact result root:

```bash
export SGL_KERNEL_A5_C8_PREBUILT=/absolute/path/to/original-lim-build
bash build.sh -a kernels Ascend950
```

The prebuilt root must contain `prepared/SOURCE_MANIFEST.json`, the private
`opp/vendors/sglc8original_transformer` installation, and the recorded file
and symlink receipts. Reuse revalidates every source and installed file before
replacing the wheel staging directory.

A2/A3 builds clear any previously staged original OPP. The migration
checkpoint intentionally accepts only `SGL_KERNEL_A5_C8_BACKEND=original` for
A5; it does not silently fall back to the removed direct LIM adaptation.

Only the legacy `bash build.sh -a kernels Ascend950` route is a bootable A5 C8
wheel in this checkpoint. The standalone/refactored 3510 build registers only
CopySFA with `split_query=0` and `ring_tail=0`; it does not package the original
LIM OPP or register the 19-tensor LIM operator at all. That modern wheel is
therefore non-bootable for A5 C8 until an independently reviewed third-party
OPP staging path is added. Capability zero describes the missing original LIM
runtime as well as the unavailable v2 CopySFA interface.

## Runtime

Importing `sgl_kernel_npu` verifies the bundled manifest and hashes before
`torch_npu` is imported, prepends the private vendor to
`ASCEND_CUSTOM_OPP_PATH`, loads the normal kernel library, and confirms that
both LIM ACLNN entry points resolve from a recorded bundled DSO.

For launch scripts that need the OPP environment before Python imports the
package, emit shell exports without importing Torch:

```bash
python3 tools/a5_c8/env.py > /tmp/sgl-a5-c8.env
source /tmp/sgl-a5-c8.env
```

## Validation boundary

The CPU contracts cover archive safety, source and install receipts,
relocation, package activation, resolved-library provenance, and stale-stage
cleanup. A successful cross-build only proves build compatibility. Numerical
correctness, eager and graph execution, MTP/reuse behavior, MF P/D behavior,
and matched serving performance still require A5 device acceptance.
