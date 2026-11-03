# SGLang Kernels NPU

## REFACTOR builds

The repository-root `setup.py` builds the new `sgl-kernel-npu` package. It includes
all Python interfaces and implementation sources, including Triton and Torch
native, plus the migrated AscendC operators selected for this build.

The existing `build.sh` and nested Python `setup.py` keep their original build
behavior. DeepEP, memory saver and attentions remain independent packages built
through that existing entry. The new entry does not schedule or change them.

### Build a wheel or CANN installer

Use Linux, an initialized CANN development SDK, CMake >=3.20 and a C++17 compiler.
Wheel builds also require matching Torch/torch_npu, Python development headers,
setuptools and wheel. CANN `.run` builds do not require Torch.

The examples use `/workspace/sgl-kernel-npu` as the checkout. Replace that prefix
with your checkout path and initialize the SDK from its installed location:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Complete Python package and all supported migrated native operators.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201

# One migrated native operator, packaged with the same Python sources.
python3 /workspace/sgl-kernel-npu/setup.py \
  --arch-version 2201 --ops apply_token_bitmask

# CANN installer only, for direct ACLNN use; no Torch adapter or wheel.
python3 /workspace/sgl-kernel-npu/setup.py \
  --arch-version 2201 --ops apply_token_bitmask --artifact run
```

### Command-line options

These options belong to the repository-root `setup.py`. The existing `build.sh`
keeps its own options; it does not accept `--arch-version`, `--ops` or `--artifact`.
No `SGL_BUILD_MODE` setting or `-a kernels` is needed for the new entry.

| Option | Default | Meaning |
| --- | --- | --- |
| `--arch-version 2201/3510` | Required | Architecture coverage, independent of the current device. May be omitted only for help or standalone `--make_clean`. |
| `--ops all` or `--ops ID[,ID...]` | `all` | All supported migrated native operators, or an explicit list. Unknown, repeated or unsupported IDs fail. |
| `--artifact run/wheel` | `wheel` | CANN installer or installable Python wheel. |
| `-j N` / `--jobs N` | Build tool default | Limit the number of simultaneous build jobs; must be a positive integer. |
| `--make_clean` | Off | Clear new-build intermediates; alone, clean and exit. With build options, clean then rebuild. |
| `-h` / `--help` | Off | Print the supported options and exit without building. |

`--ops` accepts comma-separated IDs; do not mix `all` with IDs or repeat an ID.
It selects native operators only, not which Python modules enter the wheel.
`-j N` and `--jobs N` are equivalent. They pass `--parallel N` to the CANN and
Torch CMake builds. More simultaneous jobs can reduce build time but consume
more CPU and memory. This setting does not change the operator's runtime
parallelism or computation. Omit it to use the build tool's default.

`run` and `wheel` are the only artifact choices; there is no separate `torch`
artifact or independent-package selector. Operator declarations are checked as
part of the normal build, without a separate check-only command.

```bash
# Print help; no SDK or architecture argument is required.
python3 /workspace/sgl-kernel-npu/setup.py --help

# Clear intermediates and build a wheel in the same command.
python3 /workspace/sgl-kernel-npu/setup.py \
  --make_clean --arch-version 2201 --ops apply_token_bitmask --artifact wheel
```

`2201` builds both CANN targets `ascend910b` and `ascend910_93`; `3510` builds
`ascend950`. The SDK mapping lives in `csrc/cmake/architectures.json` and is used
only by this build. Operator declarations use `SUPPORTED_ARCHS 2201 3510`;
API documentation follows the shared framework's `arch22` and `arch35` names.

Final artifacts go to `/workspace/sgl-kernel-npu/output/`. A wheel build removes
previous `sgl_kernel_npu-*.whl` files there before building; a run build removes
previous `sgl_kernel_npu-*.run` files. Other packages' outputs are left in place.

### CANN operator declarations

Declare an operator in `csrc/<domain>/<operator>/CMakeLists.txt` with
`sgl_register_operator()`. Optionally add `sgl_register_operator_variant()` calls
for architecture-specific implementations. Register the common declaration
before its variants. An operator with one implementation for all supported
architectures needs no variant.

#### Fields and required values

"Common" below means `sgl_register_operator()`; "variant" means
`sgl_register_operator_variant()`. The declaration accepts 13 configuration
fields, plus the `ID` and `ARCH` keys:

| Field | Purpose | Allowed declaration | Required value |
| --- | --- | --- | --- |
| `ID` | Operator identifier; associates variants with the common declaration. | Common and variant | Required in every declaration. |
| `ARCH` | Architecture selected by this variant. | Variant only | Required for a variant; must belong to the common `SUPPORTED_ARCHS`. |
| `EXECUTION_UNIT` | Execution unit. | Common only | Required. The implemented build backend currently supports `AICORE`. |
| `CANN_OP_TYPE` | CANN operator type registered by the OpDef. | Common only | Required. |
| `SUPPORTED_ARCHS` | Architectures supported by the operator. | Common only | Required and non-empty; currently accepts `2201` and `3510`, without duplicates. |
| `TORCH_ADAPTER` | Torch adapter source file. | Common only | Optional at registration; required for each selected operator when building the Torch extension and wheel. |
| `OPDEF_SOURCES` | CANN interface definition source files. | Common only | Required and non-empty in the common declaration. |
| `INFER_SOURCES` | Handwritten shape/type inference source files. | Common and variant | Optional; may be cleared explicitly. |
| `TILING_SOURCES` | Host Tiling source files. | Common and variant | The resolved list must be non-empty for each selected architecture. |
| `TILING_INCLUDE_DIRECTORIES` | Private header search directories for Tiling sources. | Common and variant | Optional; may be cleared explicitly. |
| `TILING_COMPILE_DEFINITIONS` | Private Tiling macros, written as `NAME` or `NAME=value`. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_SOURCES` | Device compilation entry. | Common and variant | The resolved list must contain exactly one entry after deduplication. |
| `KERNEL_INCLUDE_DIRECTORIES` | Private header search directories for the Kernel. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_COMPILE_DEFINITIONS` | Private Kernel macros, written as `NAME` or `NAME=value`. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_COMPILE_OPTIONS` | Kernel compiler options. | Common and variant | Optional; may be cleared explicitly. |

The common declaration therefore has **five required fields**: `ID`,
`EXECUTION_UNIT`, `CANN_OP_TYPE`, `SUPPORTED_ARCHS` and `OPDEF_SOURCES`.
`TILING_SOURCES` and `KERNEL_SOURCES` need not appear in the common declaration:
they may come from a variant, provided each selected architecture resolves to
non-empty Tiling sources and exactly one Kernel entry. A CANN-only `.run` build
does not require `TORCH_ADAPTER`.

`EXECUTION_UNIT` registration recognizes `AICORE` and `AICPU`, but selecting an
`AICPU` operator currently fails because its build backend is not implemented.
Use `SUPPORTED_ARCHS` to exclude an unsupported architecture rather than clearing
required sources. `--ops all` excludes operators that do not support the selected
architecture; explicitly requesting such an operator fails.

Source and include paths may be absolute or relative to the directory containing
the declaring `CMakeLists.txt`; relative paths are normalized at registration.
Source files and include directories in the final selected configuration must
exist. Compiler macros use `NAME` or `NAME=value`, without a `-D` prefix.

#### Variant inheritance, replacement and clearing

All eight fields allowed in variants use the same rule, including
`KERNEL_SOURCES`: a variant replaces the **entire field**, not individual list
items. It never appends the common list.

| Variant declaration | Resolved value for that architecture |
| --- | --- |
| Field omitted | Inherit the common field. |
| `FIELD value1 value2` | Replace the complete common field with this list. List every value still needed, including shared helpers or flags. |
| `FIELD ""` | Clear an optional field for this architecture. |
| Bare `FIELD` without a value argument | Fail with a missing-value error. |

Explicit clearing is allowed for `INFER_SOURCES`, private include directories,
private macros and Kernel compiler options. Clearing `TILING_SOURCES` or
`KERNEL_SOURCES` fails the final required-source checks. Clearing private settings
removes only those declared for this operator, not the build target's shared
settings. Clearing `INFER_SOURCES` omits handwritten inference from GRAPH; the
generated `op_proto.cc` is still compiled. Only omit inference when the operator's
supported call paths do not require that handwritten implementation.

The following schematic example assumes the listed source files exist under the
operator directory. `MODE` is an illustrative operator macro:

```cmake
sgl_register_operator(
    ID example
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglExample
    SUPPORTED_ARCHS 2201 3510
    TORCH_ADAPTER torch_adapter.cpp
    OPDEF_SOURCES op_host/example_def.cpp
    INFER_SOURCES op_host/example_infer.cpp
    TILING_SOURCES op_host/example_tiling.cpp
    KERNEL_SOURCES op_kernel/example.cpp
    KERNEL_COMPILE_DEFINITIONS MODE=1
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off
)

# Only 3510 overrides these fields; no 2201 variant is needed.
sgl_register_operator_variant(
    ID example
    ARCH 3510
    KERNEL_SOURCES op_kernel/arch35/example.cpp
    KERNEL_COMPILE_DEFINITIONS MODE=2
    KERNEL_COMPILE_OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
    INFER_SOURCES ""
)
```

For `2201`, the common values apply. For `3510`, only the arch35 Kernel entry is
selected, `MODE=2` replaces `MODE=1`, and handwritten inference is omitted. Tiling
sources are inherited because the variant does not declare `TILING_SOURCES`.
The variant repeats `--cce-auto-sync=off` because its compiler-option list fully
replaces the common one. Only the final Kernel list is checked for one entry;
the common and variant entries are not combined.

#### Duplicate declarations and values

| Situation | Result |
| --- | --- |
| Common and selected variant declare the same variant-capable field | Valid; the variant fully replaces the common value. |
| A field keyword appears twice in one function call | Error, even when the values match. |
| The same operator `ID` has two common declarations | Error. |
| The same `ID` + `ARCH` has two variant declarations | Error, even if they provide different fields; use one call for that variant. |
| A common-only field is supplied to a variant, or a field name is unknown | Error. |

Within the resolved field lists, duplicate sources, include directories and
identical macro definitions are removed. Conflicting definitions of the same
macro within the selected list fail. Common `MODE=1` and variant `MODE=2` are valid
because replacement happens first; a variant containing both `MODE=1 MODE=2`
fails. Kernel option order and repeated tokens are preserved.

#### Source roles and combined Host compatibility

These fields describe build roles, not mandatory file boundaries.
`OPDEF_SOURCES` defines the CANN interface, optional `INFER_SOURCES` provides
shape/type inference, and `TILING_SOURCES` computes device launch data. CGDR keeps
these three source lists common and registers `ARCH 2201` / `ARCH 3510` variants
for its two Kernel entries.

An existing combined Host file using `SetTiling` and `OP_ADD` can be listed in
both `OPDEF_SOURCES` and `TILING_SOURCES`; also list it in `INFER_SOURCES` when it
provides inference through `SetInferShape` / `SetInferDataType`.
`apply_token_bitmask` keeps this combined form, while CGDR uses separate files
and implementation-side registration. Both forms can be selected in one build.
Codegen compiles each entire OpDef input, so combined files must work with its
shared compile settings; private Tiling include paths/macros are not forwarded
to codegen. The generated `sgl_operator_config.h` is available to both TILING
and GRAPH targets. Codegen tracks headers under OpDef directories and, for
combined files, selected Kernel directories. Headers outside those trees require
an explicit clean rebuild after changes.

Use lowercase snake_case for operator IDs by convention. IDs are case-sensitive;
uppercase letters are also accepted. IDs must start with a letter and contain
only letters, digits and underscores; `all` and `ALL` are reserved for selection.
`CANN_OP_TYPE` retains the same basic letter/digit/underscore syntax check and
must match the type registered by the operator's OpDef.

### Incremental builds and cleanup

Normal builds retain the CANN and Torch compiler caches. Changing architecture,
operator selection, Python/Torch environment or SDK location requires an explicit
clean build:

```bash
python3 /workspace/sgl-kernel-npu/setup.py --make_clean
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 3510 --ops all
```

Cleanup removes `build/cann`, `build/torch` and `build/wheel`. Existing builds may
still clear the shared `build/` directory; rerunning the new entry reconstructs
its intermediates. Do not run both builds concurrently in one checkout.

### Installation and public API

The distribution name remains `sgl-kernel-npu`, imported as `sgl_kernel_npu`.
Wheel build tags distinguish architecture and native operator selection:

```text
sgl_kernel_npu-<version>-1refactor2201all-<python>-<abi>-<platform>.whl
sgl_kernel_npu-<version>-1refactor2201custom<set-id>-<python>-<abi>-<platform>.whl
```

Omitted `--ops` or `--ops all` produces `all`; explicit IDs produce `custom`.
The package's `_build_manifest.json` records the full selection and SDK targets.
Subset wheels replace one another; they do not add operators to an existing
installation. Reinstall explicitly and start a new Python process when switching:

```bash
python3 -m pip install --force-reinstall --no-deps /workspace/sgl-kernel-npu/output/sgl_kernel_npu-*.whl
python3 -m pytest /workspace/sgl-kernel-npu/tests/python/operators/sampler/test_apply_token_bitmask.py
```

The public entry is `sgl_kernel_npu.operators.sampler.apply_token_bitmask`.
`impl="auto"` and `impl="ascendc"` call `torch.ops.npu.apply_token_bitmask`;
`impl="torch_native"` selects the reference explicitly. There is no automatic
fallback when the native operator is absent. See the API documentation in
[English](../../docs/api/en/operators/sampler/apply_token_bitmask.md) or
[Chinese](../../docs/api/zh/operators/sampler/apply_token_bitmask.md).

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
