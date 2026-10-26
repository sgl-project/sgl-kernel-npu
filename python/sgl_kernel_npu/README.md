# SGLang Kernels NPU

## REFACTOR builds

`bash build.sh` continues to use **LEGACY** by default, including its existing
package selection, cleanup and invocation behavior. Opt into the new build with
`SGL_BUILD_MODE=REFACTOR`. The repository-root `setup.py` builds the new
`sgl-kernel-npu` distribution; `build.sh` also schedules independently packaged
DeepEP, memory saver and attentions projects. The nested Python `setup.py`
remains the LEGACY entry during migration.

Only `apply_token_bitmask` has been migrated. REFACTOR compiles selected migrated
native operators; it never mixes in unmigrated LEGACY native operators. It ships
the complete Python interfaces, implementation sources and required shared
modules. Thus a Python operator can be present while its AscendC implementation
is absent from a custom wheel; selecting that implementation reports an error.

### Environment and architecture

Use Linux, an initialized CANN development SDK, CMake >=3.20 and a C++17 compiler.
Wheel builds also require matching Torch/torch_npu, Python development headers,
setuptools and wheel. `.run` builds do not require Torch. Runtime validation
requires a supported NPU and matching driver/firmware.

The refactor acceptance environment is CANN 9.1.0, Torch 2.10.0 and torch_npu
2.10.0.post6. The earlier engineering prototype was tested on one 2201 device;
**the latest refactor still requires SDK builds, precision and performance
validation**. Portable tests do not establish hardware compatibility.

`--arch-version` is mandatory; no machine detection or public SoC argument is
used. `2201` builds all SDK targets mapped to that architecture, covering both
hardware families in 2201. `3510` builds the corresponding newer architecture.
The mapping is centralized in `csrc/cmake/architectures.json`; declarations and
API documentation use `SUPPORTED_ARCHS 2201 3510`.

Examples use `/workspace/sgl-kernel-npu` as the checkout. Replace that prefix if
your checkout is elsewhere. Initialize the SDK using its actual installed path:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SGL_BUILD_MODE=REFACTOR bash /workspace/sgl-kernel-npu/build.sh \
  --arch-version 2201 --ops apply_token_bitmask
```

### Select packages, operators and artifacts

| Option | Default | Meaning |
| --- | --- | --- |
| `-a kernels/deepep/deepep2/memory-saver/attentions/all` | `kernels` | Independent distribution(s) to build. |
| `--arch-version 2201/3510` | Required | Architecture coverage, independent of the current device. |
| `--ops all` or `--ops ID[,ID...]` | `all` | All supported migrated operators, or a strict explicit list. Unknown, repeated or unsupported IDs fail. Requires kernels. |
| `--artifact run/wheel` | `wheel` | Native CANN installer or complete Python wheel. `run` is only valid with `-a kernels`. |
| `-j N` / `--jobs N` | Build tool default | Positive parallelism limit for the CANN and Torch builds. |
| `--make_clean` | Off | Clear REFACTOR intermediates; alone, clean and exit. With architecture/build options, clean then rebuild. |
| `--declarations-only` | Off | Validate kernels selection without compiling or producing an artifact. |

```bash
# Direct ACLNN development: one migrated operator, no Torch build.
SGL_BUILD_MODE=REFACTOR bash /workspace/sgl-kernel-npu/build.sh \
  --arch-version 2201 --ops apply_token_bitmask --artifact run

# All migrated operators supported by 3510, still a complete Python package.
SGL_BUILD_MODE=REFACTOR bash /workspace/sgl-kernel-npu/build.sh \
  --arch-version 3510 --ops all --make_clean

# Equivalent root Python entry for the kernels distribution.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 --ops all --make_clean
```

Final artifacts share `/workspace/sgl-kernel-npu/output/`. Before a wheel build,
old wheels of the selected distribution are removed so failed builds cannot
leave a misleading old wheel as the apparent result. Unselected distributions
are retained. A run build similarly refreshes its installers.

DeepEP reuses its existing builds and remains separately packaged: 2201 produces
separate variants under `output/deepep_a2/` and `output/deepep_a3/`; 3510 uses
`output/deepep_950/`. These variants share an installation identity and must not
be installed together. `deepep2` selects only the existing 2201 variant.
The existing attentions project has no 3510 build support, so requesting
attentions, deepep2 or all with 3510 fails before building. Select supported
packages explicitly. Independent projects retain their own build logic.

### Incremental builds and recovery

CANN and Torch use `build/cann/` and `build/torch/`. Unchanged inputs reuse
compiled objects; shared Tiling headers trigger metadata regeneration. Refreshing
wheel/vendor staging does not erase the compiler cache. A changed architecture,
operator set, Python/Torch environment or SDK location requires an explicit
`--make_clean` instead of silently mixing incompatible artifacts. Toolchain or
SDK upgrades should also start with `--make_clean`.

LEGACY may still clear the shared `build/` directory. Running REFACTOR again
reconstructs everything. Save any installers/wheels/baselines you need before
switching builds. Concurrent builds against the same checkout are unsupported.

### Wheel naming and installation

The distribution remains `sgl-kernel-npu`, with import name `sgl_kernel_npu`.
Standard wheel build tags identify the architecture and selection:

```text
sgl_kernel_npu-<version>-1refactor2201all-<python>-<abi>-<platform>.whl
sgl_kernel_npu-<version>-1refactor2201custom<set-id>-<python>-<abi>-<platform>.whl
```

Omitted `--ops` or `--ops all` means `all`; explicit IDs mean `custom`, even if
currently equal to the complete migrated set. `_build_manifest.json` records the
complete operator list, architecture, selection kind and SDK targets. The build
tag is not a new package version. Use explicit reinstallation and start a new
Python process when switching wheels; subset packages replace rather than extend
each other.

```bash
python3 -m pip install --force-reinstall --no-deps /workspace/sgl-kernel-npu/output/sgl_kernel_npu-*.whl
python3 /workspace/sgl-kernel-npu/tests/python/sgl_kernel_npu/test_bitmask_cann.py
python3 -m pytest /workspace/sgl-kernel-npu/tests/python/operators/sampler -q \
  --impl ascendc --impl-ref torch_native --enable-perf=false
```

The wheel contains the expanded CANN vendor, formal Torch library and Python
sources, not the SDK or the installer. Import sets up the vendor and loads the
library. Public use:

```python
from sgl_kernel_npu.operators.sampler import apply_token_bitmask
result = apply_token_bitmask(logits, bitmask, indices, impl="auto")
```

`auto` selects AscendC without fallback. Existing direct calls through
`torch.ops.npu.apply_token_bitmask` remain available when this native operator is
included. Both forms update logits in place and return its storage. See the
[English API](../../docs/api/en/operators/sampler/apply_token_bitmask.md) and
[中文 API](../../docs/api/zh/operators/sampler/apply_token_bitmask.md).

### Direct ACLNN and staged SDK checks

After a run or wheel build, the checked private vendor is under
`build/cann/installed/vendors/sgl_kernel_npu`; the system OPP tree is unchanged.

```bash
cmake -S /workspace/sgl-kernel-npu/tests/csrc \
  -B /workspace/sgl-kernel-npu/build/aclnn-test \
  -DSGL_BITMASK_INSTALLED_VENDOR=/workspace/sgl-kernel-npu/build/cann/installed/vendors/sgl_kernel_npu
cmake --build /workspace/sgl-kernel-npu/build/aclnn-test --target test_bitmask_aclnn
ASCEND_CUSTOM_OPP_PATH=/workspace/sgl-kernel-npu/build/cann/installed/vendors/sgl_kernel_npu \
  /workspace/sgl-kernel-npu/build/aclnn-test/test_bitmask_aclnn

# Optional staged diagnosis, not required for ordinary builds.
python3 /workspace/sgl-kernel-npu/tests/csrc/verify_cann_codegen.py \
  --arch-version 2201 --through package
```

The direct ACLNN suite contains 48 cases; the installed-wheel suite contains 57.
The public-interface suite additionally keeps LEGACY functional shapes. No test
silently reduces an unsupported shape or dtype.

### Precision and performance acceptance

Use separate installations/processes for old and new wheels, with the same
hardware, SDK and Torch versions. Build the old wheel from the recorded baseline
commit `ecd3f947e9ed6cd512225c5030f2fd6deb63233c` using LEGACY. Run the validation
script from the refactor checkout against that installed baseline; then install
the REFACTOR wheel and run the second phase:

```bash
python3 /workspace/sgl-kernel-npu/tests/python/sgl_kernel_npu/benchmark_bitmask_migration.py \
  --phase baseline --baseline /workspace/bitmask-baseline.pt --report /workspace/bitmask-baseline.json
# Install the freshly built REFACTOR wheel in a new process/environment first.
python3 /workspace/sgl-kernel-npu/tests/python/sgl_kernel_npu/benchmark_bitmask_migration.py \
  --phase refactor --baseline /workspace/bitmask-baseline.pt --report /workspace/bitmask-refactor.json
```

The baseline file preserves the exact inputs and old results. Both builds are
checked independently against Torch native and directly against each other.
Mask positions, returned alias and unchanged inputs are checked. Existing
atol/rtol 1e-2 is retained; unmasked values must be preserved exactly.

Nine existing performance configurations retain all shapes/dtypes/index counts.
End-to-end timing uses warmup 5, iterations 50, five independent rounds. Fresh
inputs are prepared before timing; every call sees the original data. Both direct
and public-interface latencies are reported for the refactor. No allowed slowdown
percentage is introduced. Overlapping measurements or a slower median require
retesting; they are not automatically accepted as noise.

The shared profiler can be enabled with `--enable-perf=true`. Its default 5/5
warmup/active result sums device task durations and is a diagnostic distinct from
end-to-end latency. The test framework prepares in-place inputs before profiling.

AICORE/ACLNN is the implemented CANN backend. `EXECUTION_UNIT AICPU` declarations
are permitted, but selecting them fails until that backend is implemented.
Other native operators remain on LEGACY. SDK/NPU validation of this refactor,
full independent-package builds and performance non-regression are still required.

## Legacy build: software and hardware
Supported Hardware Models: Ascend 910B, Ascend 910C, and Ascend 950 series products
Platform: aarch64/x86
Supporting Software
- Driver Ascend HDK 25.0.RC1.1, CANN 8.3.RC1 or later versions (refer to the "[CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)" to install the CANN development kit package, as well as the supporting firmware and drivers)
- Before installing CANN software, you need to install the relevant [dependency list](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)
- Python >= 3.7
- pybind11 (install via `pip install pybind11`)

## Legacy build: quick start
### Compile and Run
1. Prepare the CANN environment variables (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. Build the project
Executing the engineering build script build.sh
```bash
# Detect the local NPU and build for it.
bash build.sh -a kernels

# Or name the SoC explicitly.
bash build.sh -a kernels 910
bash build.sh -a kernels 950
bash build.sh -a kernels Ascend950PR_9599
```

Omitting the SoC queries the local device with `npu-smi`. Hosts without an NPU —
build containers, for instance — fall back to `Ascend910_9382`, which is what
every target used to default to unconditionally, so container builds are
unaffected. Detection only picks a default: an explicit argument always wins.

`910B` (A2), `910`/`910C` (A3) and `950` (Ascend 950) select the SoC family. A generic or
auto-detected Ascend 950 target compiles with `Ascend950PR_9599`, the target of the
released `950` package; pass another concrete compiler target such as
`Ascend950PR_958b` to override it. Both forms stage the same Ascend 950 target
providers for the wheel.

Operators whose implementation differs across Ascend generations live under
`target_providers/<target>/` — build input, not a runtime package. The wheel
build stages the chosen target's tree into the `sgl_kernel_npu` package by
relative module path, so the directory layout itself is the registry:
`target_providers/Ascend950/norm/gemma_rmsnorm.py` becomes
`sgl_kernel_npu/norm/gemma_rmsnorm.py` in the wheel. The 910 and 950 provider
sets must stay symmetric (CI enforces this), and conflicting or missing
providers fail the build instead of silently falling back. setup.py knows
targets, not operators: adding another cross-generation operator is just two
provider files plus tests, with no build-framework changes.

Every target exposes `sgl_kernel_npu.norm.gemma_rmsnorm`: native `torch_npu`
Gemma RMSNorm on Ascend 910, standard ACLNN RMSNorm with `1 + weight` on
Ascend 950. SGLang performs no runtime SoC detection for this operator.
Because providers are chosen at build time, the source tree ships no
`norm/gemma_rmsnorm.py` — it only exists inside a built wheel, and
`SGL_KERNEL_NPU_BUILD_TARGET` must be set when building one. Running from a
source checkout (or `pip install -e .`) raises `ImportError` rather than
silently defaulting to the 910 operator.

### Installation
1. Pip install the `.whl` file into your Python environment
```bash
pip install output/sgl_kernel_npu*.whl

# (Optional) Confirm whether the import can be successfully
python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
```

2. Execute the environment variables for CANN (modify according to the installation path)
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
3. In the Python project, import `sgl_kernel_npu`.

### Test
Execute sgl_kernel_npu test scripts, for example
```bash
python3 tests/python/sgl_kernel_npu/test_hello_world.py
```
