<h2 align="left">
SGLang Kernels NPU
</h2>

## Introduction
SGLang Kernels for Ascend NPU

## Selected CANN operator builds

The CANN build is an opt-in path alongside the existing build. It selects a
hardware family and a set of migrated operators, then produces a `.run` package,
Torch libraries, or a wheel containing the CANN operator runtime. The default
`bash build.sh` still builds the legacy repository; `bash build.sh -a kernels`
still builds the legacy kernel package.

Currently, **`apply_token_bitmask` is the only migrated operator**. `--ops all`
means all **registered CANN operators supported by the selected hardware**, not
all operators in the legacy repository. A selected CANN wheel is therefore not
yet a replacement for the full legacy package.

### 1. Get the source and prepare the environment

The examples below use `/workspace/sgl-kernel-npu` as the checkout path. If you
choose another writable directory, replace that prefix throughout the commands.

```bash
git clone https://github.com/sgl-project/sgl-kernel-npu.git /workspace/sgl-kernel-npu
cd /workspace/sgl-kernel-npu

# Until this change is merged, check out the PR implementation.
git fetch origin pull/880/head
git switch -c ascendc-refactor FETCH_HEAD
```

The selected `apply_token_bitmask` CANN build does not need the `catlass` or
`pto-isa` submodules. Before using the legacy whole-repository build, initialize
its dependencies:

```bash
git -C /workspace/sgl-kernel-npu submodule update --init --recursive
```

Use Linux with a CANN development installation, its Ascend C compiler, CMake
3.20 or later, a C++17 compiler, and Python. Torch/wheel builds additionally need
matching `torch` and `torch_npu`, Python development headers, `setuptools` and
`wheel` in the build environment. Native `.run` builds do not require Torch.
An NPU, compatible driver and firmware are required for runtime tests.

The validated environment is A3 / Ascend910_9362, Linux aarch64, CANN 9.1.0,
CMake 3.22.1, GCC 11.4, Python 3.12, Torch 2.10.0 and torch_npu 2.10.0.post6.
This is a tested combination, not a claim that every other version is supported.
Initialize the SDK in each new shell; replace the following SDK path if needed:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake --version
python3 --version

# Required only when building Torch libraries or a wheel.
python3 -c 'import torch, torch_npu, setuptools, wheel; print(torch.__version__, torch_npu.__version__)'
```

### 2. Select the hardware and operators

`--soc` takes one explicit **CANN hardware family**, not a device number or a
chip model such as `Ascend910_9362` / `Ascend910_9382`. No device auto-detection or
hardware default is used in the new path.

| Generation | `--soc` value | Current bitmask declaration | Runtime validation |
| --- | --- | --- | --- |
| A2 | `ascend910b` | Shared implementation | Pending |
| A3 | `ascend910_93` | Shared implementation | Passed |
| A5 | `ascend950` | Shared implementation | Pending |

For `--ops all`, CMake filters the registered operators by their `SUPPORTED_SOCS`
declarations. For explicit IDs, an unknown or unsupported operator fails the
build instead of being silently omitted. An empty selection also fails.

```bash
# One operator, native CANN package.
SGL_BUILD_MODE=CANN bash /workspace/sgl-kernel-npu/build.sh \
  --soc ascend910_93 --ops apply_token_bitmask --artifact run

# All migrated operators supporting A3, including shared A2/A3/A5 operators.
SGL_BUILD_MODE=CANN bash /workspace/sgl-kernel-npu/build.sh \
  --soc ascend910_93 --ops all --artifact wheel
```

Multiple IDs use comma-separated syntax: `--ops id_one,id_two`. These names are
syntax placeholders; use IDs actually registered in the checkout. There is only
one real ID in the current implementation.

### 3. Choose the final artifact

| `--artifact` | Automatically performed | Final output |
| --- | --- | --- |
| `run` (default) | CANN configure, Kernel build, `.run` packaging, private installation and content checks | `/workspace/sgl-kernel-npu/build/cann/package-stage/*.run` |
| `torch` | Everything in `run`, then compile the selected Torch adapters | `/workspace/sgl-kernel-npu/build/torch/package/sgl_kernel_npu/` |
| `wheel` | Everything in `torch`, then assemble and verify the binary wheel | `/workspace/sgl-kernel-npu/build/wheel/dist/*.whl` |

The build installs the `.run` automatically under
`/workspace/sgl-kernel-npu/build/cann/installed/`; it does not install into the
system SDK's OPP tree. Torch compilation consumes the installed headers and
libraries. The wheel contains the **expanded vendor directory and formal Torch
library**, not the `.run` installer, comparison library, CANN SDK or driver.
Installing this wheel does not require separately installing the custom `.run`.

The Bash entry delegates directly to the Python coordinator. The equivalent
Python command is:

```bash
python3 /workspace/sgl-kernel-npu/python/sgl_kernel_npu/build_tools/cann.py \
  --soc ascend910_93 --ops apply_token_bitmask --artifact wheel
```

The coordinator prints `SELECTED_OPERATORS`, `RUN_PACKAGE`, `INSTALLED_VENDOR`,
`TORCH_PACKAGE`, `COMPARE_LIBRARY` and `WHEEL` as applicable. A failure stops the
chain. Building a package does not run an NPU correctness or performance test.
`setup.py` consumes contents prepared by this coordinator; standalone
`pip wheel .`, `python -m build` and editable installs are not new-build entry
points.

### 4. Install the wheel and call it from Python

Use an environment matching the build Python, architecture, Torch and torch_npu.
The wheel records the build's Torch/torch_npu versions. The following isolated
venv reuses the prepared environment's dependencies; `--ignore-installed` places
the new package in the venv even if a legacy package is installed globally.
`--no-deps` assumes those matching dependencies are already available.

```bash
python3 -m venv --system-site-packages /workspace/sgl-kernel-npu-cann-venv
/workspace/sgl-kernel-npu-cann-venv/bin/python -m pip install \
  --no-deps --ignore-installed /workspace/sgl-kernel-npu/build/wheel/dist/*.whl
/workspace/sgl-kernel-npu-cann-venv/bin/python -c \
  'import torch, torch_npu, sgl_kernel_npu; print(sgl_kernel_npu.__file__); print(torch.ops.npu.apply_token_bitmask)'
```

The import sets up the package-local vendor path and loads the Torch extension.
Existing callers keep the interface:

```python
import torch
import torch_npu
import sgl_kernel_npu

# logits, bitmask and optional indices are caller-provided NPU tensors.
result = torch.ops.npu.apply_token_bitmask(logits, bitmask, indices)
```

`logits` is updated in place and returned. Existing input preparation and
computation are preserved; omitted or empty `indices` currently means all rows.
For a complete runnable example with inputs and CPU reference checks, run the
acceptance script below. First use `npu-smi info` to select an available device;
replace `0` if that device is not available to you:

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
/workspace/sgl-kernel-npu-cann-venv/bin/python \
  /workspace/sgl-kernel-npu/tests/python/sgl_kernel_npu/test_bitmask_cann.py
```

Success ends with `PASS: 57/57 installed CANN wheel cases`. It checks FP32, FP16
and BF16 results, in-place storage, unchanged bitmask/indices, and selected-row
behavior. It is not a performance benchmark.

### 5. Develop and test one operator

For direct ACLNN testing, build with `--artifact run` as above, then compile the
standalone C++ test against that private installation:

```bash
cmake -S /workspace/sgl-kernel-npu/tests/csrc \
  -B /workspace/sgl-kernel-npu/build/aclnn-test \
  -DSGL_BITMASK_INSTALLED_VENDOR=/workspace/sgl-kernel-npu/build/cann/installed/vendors/sgl_kernel_npu
cmake --build /workspace/sgl-kernel-npu/build/aclnn-test --target test_bitmask_aclnn
export ASCEND_CUSTOM_OPP_PATH=/workspace/sgl-kernel-npu/build/cann/installed/vendors/sgl_kernel_npu
# Choose an available physical device as in the previous section.
export ASCEND_RT_VISIBLE_DEVICES=0
/workspace/sgl-kernel-npu/build/aclnn-test/test_bitmask_aclnn
```

Success ends with `PASS: 48/48 ACLNN cases`. This covers direct calls and in-place
results without Torch; row selection is covered by the Torch tests.

For same-process old/new comparison, keep a working **legacy** package in the
current Python environment and build the new comparison library:

```bash
SGL_BUILD_MODE=CANN bash /workspace/sgl-kernel-npu/build.sh \
  --soc ascend910_93 --ops apply_token_bitmask --artifact torch
export ASCEND_RT_VISIBLE_DEVICES=0
python3 /workspace/sgl-kernel-npu/tests/python/sgl_kernel_npu/test_bitmask_cann.py \
  --compare-library /workspace/sgl-kernel-npu/build/torch/package/sgl_kernel_npu/lib/libsgl_kernel_npu_compare.so
```

Success ends with `PASS: 57/57 same-process legacy/CANN cases`. The old entry is
`torch.ops.npu.apply_token_bitmask`; the new comparison entry is
`torch.ops.sgl_kernel_npu_compare.apply_token_bitmask_cann`. The test verifies the
loaded package is legacy. Do not install both formal wheels into one Python
environment: they share the `sgl_kernel_npu` package name.

### All new CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--soc SERIES` | Required | Build for one explicit CANN family. |
| `--ops all` or `--ops ID[,ID...]` | `all` | Select supported migrated operators, or validate an explicit list. IDs are case-sensitive; duplicates and mixing `all` with IDs fail. |
| `--artifact run\|torch\|wheel` | `run` | Select where the build chain finishes. |
| `-j N`, `--jobs N` | Build tool default | Positive maximum parallel job count for Kernel, package and Torch builds. |
| `--declarations-only` | Off | Validate declarations and selection without SDK/Torch compilation; also works on macOS. Requires the default `run` artifact. |
| `-h`, `--help` | — | Print CLI help. |

`SGL_BUILD_MODE` is a Bash dispatch environment variable: `LEGACY` is the default,
`CANN` selects the new coordinator, and other values fail. Set it only on the
intended command. Legacy flags such as `-a kernels` and legacy chip aliases are
not CANN CLI options. The direct Python entry does not need `SGL_BUILD_MODE`.

```bash
SGL_BUILD_MODE=CANN bash /workspace/sgl-kernel-npu/build.sh --help
python3 /workspace/sgl-kernel-npu/python/sgl_kernel_npu/build_tools/cann.py \
  --soc ascend910_93 --ops all --declarations-only
```

For lower-level fault isolation, the existing checkpoint tool can stop after
`codegen`, `tiling`, `kernel` or `package`:

```bash
python3 /workspace/sgl-kernel-npu/scripts/verify_cann_codegen.py \
  --compute-unit ascend910_93 --through package
```

Its `--build-dir` optionally selects an absolute, separate checkpoint directory;
`--through` defaults to `codegen`. `--compute-unit` has the same hardware-family
meaning as the new `--soc`; the older `--confirmed-socs` option is deprecated and
ignored. The checkpoint tool performs no NPU execution and is not required for
normal package builds.

### Build directories, logs and current limits

- CANN logs: `/workspace/sgl-kernel-npu/build/cann/configure.log`,
  `kernel-build.log`, `build.log`, and `install.log` in that directory.
- Torch logs: `/workspace/sgl-kernel-npu/build/torch/configure.log` and
  `/workspace/sgl-kernel-npu/build/torch/build.log`.
- Wheel log: `/workspace/sgl-kernel-npu/build/wheel/build.log`.
- Legacy wheels still go to `/workspace/sgl-kernel-npu/output/`; new wheels use
  `/workspace/sgl-kernel-npu/build/wheel/dist/`.

The modes share `/workspace/sgl-kernel-npu/build/`. The legacy script clears it;
the new coordinator clears incompatible caches when switching configurations.
Torch/wheel stages currently rebuild their staging directories. Copy any
artifacts you need to keep before switching modes. Incremental build/recovery
and parallel dependency acceptance remain follow-up work.

Only AICORE/ACLNN is implemented in the new backend. A2/A5 runtime validation,
legacy whole-repository build regression, performance validation and migration
of the remaining operators are still pending. Source declarations and portable
checks do not establish NPU runtime compatibility.

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
