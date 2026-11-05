# SGLang Kernels NPU

## REFACTOR builds

The repository-root `setup.py` builds the new `sgl-kernel-npu` package. It includes
all Python interfaces and implementation sources, including Triton and Torch
native, plus selected CANN (AI Core/AICPU) and host C++ operators.

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
| `--arch-version 2201/3510` | Required | Architecture coverage, independent of the current device. May be omitted only for help or standalone `--make-clean`. |
| `--ops all` or `--ops ID[,ID...]` | `all` | All supported migrated native operators, or an explicit list. Unknown, repeated or unsupported IDs fail. |
| `--artifact run/wheel` | `wheel` | CANN installer or installable Python wheel. |
| `-j N` / `--jobs N` | Build tool default | Limit the number of simultaneous build jobs; must be a positive integer. |
| `--third-party NAME[,NAME...]` | All matching profiles | Select third-party profile filenames without `.json`, such as `cann_recipes_infer` or `ops_transformer`. Wheel builds only; names must be distinct and non-empty. |
| `--without-third-party` | Off | Skip third-party builds and bundling. Mutually exclusive with `--third-party`. |
| `--plan` | Off | Print the resolved third-party configuration and build commands, then exit without compiling either third-party or repository-owned operators. |
| `--cann-version X.Y.Z` | Detected SDK version | Override the profile version for `--plan` only. Actual builds always detect the initialized SDK version. |
| `--make-clean` | Off | Clear new-build intermediates; alone, clean and exit. With build options, clean then rebuild. |
| `-h` / `--help` | Off | Print the supported options and exit without building. |

`--ops` accepts comma-separated IDs; do not mix `all` with IDs or repeat an ID.
It selects repository-owned native operators only, not which Python modules enter
the wheel or which third-party profiles are built. Operator IDs determine their
build paths from the shared declarations: AI Core and AICPU use CANN, while host
C++ sources enter the Torch extension. Explicit IDs may mix all three; selection needs no additional type flag. Unsupported architectures and unknown IDs fail.

For wheels, `--ops all` selects every native operator supporting the requested
architecture. For CANN `.run` installers, `all` retains its existing CANN-only
scope and logs why CPP IDs are excluded. Explicit CPP IDs with `.run` fail before
compilation or third-party builds. Use explicit IDs to choose a subset of any type.

CPP-only wheels skip the repository CANN package, codegen, Tiling, HCC and device
Kernel compilation. They still require Linux, the initialized ACL development
SDK, matching Torch/torch_npu and a C++17 compiler. Third-party dependencies keep
their independent profile/toolchain requirements; use `--without-third-party` to
build only the selected repository operators.

```bash
# One host C++ operator, with no repository or third-party CANN package build.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops transfer_mamba_state --without-third-party

# Mix AI Core, AICPU and host C++ operators; the IDs select each build path.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops apply_token_bitmask,add_example,transfer_mamba_state --without-third-party

# All repository native operators in a wheel, including AICPU and host C++.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops all --without-third-party

# All repository CANN operators in an installer, excluding host C++.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops all --artifact run
```

`--plan` retains its existing third-party-only JSON output and does not resolve or
compile repository operators. Normal builds log the native selection and exclusions
before starting third-party compilation.

`-j N` and `--jobs N` are equivalent. They pass `--parallel N` to the CANN and
Torch CMake builds. More simultaneous jobs can reduce build time but consume
more CPU and memory. This setting does not change the operator's runtime
parallelism or computation. Omit it to use the native build tool's default;
the current third-party adapters default to 8 jobs.

`run` and `wheel` are the only artifact choices; there is no separate `torch`
artifact or independent-package selector. Operator declarations are checked as
part of the normal build, without a separate check-only command.

```bash
# Print help; no SDK or architecture argument is required.
python3 /workspace/sgl-kernel-npu/setup.py --help

# Clear intermediates and build a wheel in the same command.
python3 /workspace/sgl-kernel-npu/setup.py \
  --make-clean --arch-version 2201 --ops apply_token_bitmask --artifact wheel
```

`2201` builds both CANN targets `ascend910b` and `ascend910_93`; `3510` builds
`ascend950`. The SDK mapping lives in `csrc/cmake/architectures.json` and is used
only by this build. Operator declarations use `SUPPORTED_ARCHS 2201 3510`;
API documentation follows the shared framework's `arch22` and `arch35` names.

Final artifacts go to `/workspace/sgl-kernel-npu/output/`. A wheel build removes
previous `sgl_kernel_npu-*.whl` files there before building; a run build removes
previous `sgl_kernel_npu-*.run` files. Other packages' outputs are left in place.

### Third-party dependencies

Wheel builds read `third_party/profiles/cann-<SDK-version>/*.json`, then select
the profile's `ops.arch22` or `ops.arch35` list using `--arch-version`.
Profiles pin the source repository, commit, operators and build commands.
`--ops` selects repository-owned native operators; `--third-party` independently
selects external libraries. The profile filename, not its internal package name,
is passed to `--third-party`.

```bash
# CANN 9.1.0: build one native operator and bundle the recipes dependency.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops apply_token_bitmask --third-party cann_recipes_infer -j8

# CANN 9.2.0 / arch35: bundle the ops-transformer dependency.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 3510 \
  --ops apply_token_bitmask --third-party ops_transformer -j8

# Build repository-owned operators only, without third-party dependencies.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --ops apply_token_bitmask --without-third-party

# Preview dependency commands; no compilation or cloning is performed.
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 \
  --third-party cann_recipes_infer --plan --cann-version 9.1.0
```

Omitting `--third-party` selects all profiles for the detected SDK version.
No matching profiles, or an empty/missing operator list for the chosen
architecture, produces an INFO message and skips that dependency. Explicitly
requesting a profile that does not exist for the SDK version is an error.
`--artifact run` packages only repository-owned CANN operators.

Third-party SoC detection uses the local device's `npu-smi` board information:
`ascend910b` or `ascend910_93` for arch22, and `ascend950` for arch35. It must match
`--arch-version`. This detection also runs for `--plan` when a dependency has
operators to build; `--cann-version` changes profile selection, not the installed
SDK or local SoC.

Dependencies are built before repository-owned operators. Their vendor resources,
Python bindings and licenses are bundled into the same wheel; users do not need
to install the dependency `.run` or wheel separately. Successful dependency builds
are cached under `build/third_party/` and reused when the build fingerprint and
artifact checksums match. Changes to the profile, build scripts or relevant build
environment invalidate the cache. `--make-clean` preserves this third-party cache.

### Shared CANN and host C++ registry

See the [complete build architecture and file call graph](../../docs/build_architecture.md)
for configuration, compilation, packaging and runtime-loading phases.

All operators use `sgl_register_operator()` and declare one `EXECUTION_UNIT`:
`AICORE`, `AICPU` or `HOST`. Existing AI Core/AICPU declarations need no edits.
`HOST` identifies ordinary host C++ code compiled into the Torch extension.
Host C++ implementations declare their own source list, without dummy OpDefs,
Tiling, device Kernels or AICPU registration metadata:

```cmake
sgl_register_operator(
    ID transfer_mamba_state
    EXECUTION_UNIT HOST
    SUPPORTED_ARCHS 2201 3510
    CPP_SOURCES transfer_mamba_state.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
```

HOST requires `ID`, `EXECUTION_UNIT`, `SUPPORTED_ARCHS`, `CPP_SOURCES` and
`TORCH_ADAPTER`; CANN-only
fields are rejected. `CPP_SOURCES` may be replaced by an architecture variant
using `sgl_register_operator_variant()`. IDs are unique across all three execution units. CPP IDs
must also have unique uppercase forms because they determine build-guard names.

Add new domains to the existing `sgl_add_operator_domains()` discovery list.
Discovery only registers declarations; it never adds unselected compilation inputs.
One metadata table holds every declaration. The CANN-specific index is `SGL_CANN_OPERATOR_IDS`; the unified and HOST indexes
are `SGL_NATIVE_OPERATOR_IDS` and `SGL_HOST_OPERATOR_IDS`.
`sgl_get_registered_operators(out)` and `sgl_select_cann_operators(... OPERATORS ALL)`
retain their CANN-only meaning. Queries with `EXECUTION_UNIT HOST`,
`EXECUTION_UNIT AICORE` or `EXECUTION_UNIT AICPU` filter the registered IDs.
Direct CANN selection rejects explicit HOST IDs; the native selector combines
the CANN and HOST indexes when processing `--ops all`.

The root `setup.py::configure_native()` configures the existing
`csrc/CMakeLists.txt` once. It registers HOST declarations first, partitions the
original `SGL_NATIVE_OPERATORS` request using registered HOST IDs, then discovers
CANN declarations with a CANN-only `SGL_CANN_OPERATORS` request. Existing
declaration-local dependency expansion remains unchanged; the root no longer
needs a deferred cache restore. The original request stays intact, and final
expanded IDs are exported separately from request caches. It reads CANN and HOST IDs
from `selected-operators.txt` and `selected-host-operators.txt` in `build/cann`.
HOST-only configuration returns before SDK/compiler initialization.

`cann.py::build_cann(configured, ...)` requires an existing `CannBuild`
configuration; it derives the architecture, directory and operator IDs from
that result and only builds, installs and verifies CANN operators. It never
runs CMake configuration itself.
Its `CannBuild` result has no HOST IDs.

`torch.py::configure_torch()` configures the existing Torch CMake project once,
with HOST IDs in `SGL_HOST_OPERATORS` and CANN adapter IDs in `SGL_CANN_OPERATORS`.
It validates Python/Torch/SDK and selection caches, but does not compile.
`host.py::build_host()` accepts only HOST IDs and the configured directory,
builds the `sgl_host_operators` OBJECT target from `CPP_SOURCES`.
It takes no CANN result/vendor and compiles no adapters.
HOST-only implementation objects still need Torch/torch_npu/ACL headers and ABI flags.
`torch.py::build_torch()` compiles the selected CANN/HOST `TORCH_ADAPTER` files and
links the HOST objects into the existing, single `libsgl_kernel_npu.so`.
Object dependencies are managed by CMake; no HOST result is passed back for a
second metadata comparison. Architecture/selection validation remains at the
entry and CMake, while the original environment/cache and output checks remain.
Only this extension target uses the repository-owned CANN vendor headers and
nnopbase when CANN adapters are selected. No extra shared library is produced.
CANN-only builds skip `build_host()`; HOST-only builds skip `build_cann()`.

The selection txt files remain the CMake-to-Python result interface. The root
selection entry no longer writes an unused HOST cache entry. No additional
`SGL_NATIVE_SDK_ROOT` is needed: CANN configuration reuse checks the existing
`SGL_CANN_SDK_ROOT` before reconfiguration, and Torch checks its own SDK signature.
`SGL_NATIVE_ARTIFACT` forwards the existing `--artifact`: historically all native
selections were CANN, while HOST now belongs in wheel ALL and must be excluded
from CANN .run ALL (explicit HOST requests for .run fail).
`common.py` contains backend-independent process, cache and architecture helpers.

`torch_wheel.py::stage_package()` consumes the completed `TorchBuild` and optional
`CannBuild`, checks architecture/IDs/vendor agreement, stages Python and third-party
payloads, and writes the manifest. `build_wheel()` only packages and verifies
the staged content. Neither packaging stage compiles operators. There is no
separate selection module or selection-only CMake project.

Each selected CPP ID defines `SGL_CPP_<UPPERCASE_ID>=1`. Shared registration files
must guard each declaration, schema and binding; shared implementation files must
guard each public entrypoint. Shared source files are deduplicated. The two direct
state-transfer operators share helpers while compiling only selected entrypoints.
The ID macro is generated automatically for every selected HOST operator;
there is currently no opt-out switch. Independent implementation/adapter
files may ignore it entirely and be selected by their source lists alone.

CPP-only packaging can bundle third-party vendors without a repository-owned CANN
vendor. Runtime initialization reads `operators_by_type` to distinguish this from
a broken CANN build; declared third-party vendors and legacy required vendors are
still validated. Unselected native implementations do not automatically fall back.

No dedicated `build_tools` test suite is added because main has none for this
build entry. Development checks run outside the repository; existing operator
functional cases remain unchanged. Real SDK/HCC compilation, installation and NPU
correctness need validation in their target environments.

Example builds on Linux after initializing CANN and installing matching Torch/torch_npu:

```bash
# HOST-only; no repository CANN vendor or device compilation.
python setup.py --arch-version 2201 --ops transfer_mamba_state --artifact wheel --without-third-party --make-clean
# Mixed AICORE + AICPU + HOST; still one Torch extension.
python setup.py --arch-version 2201 --ops apply_token_bitmask,add_example,transfer_mamba_state --artifact wheel --without-third-party --make-clean
# CANN-only RUN: ALL retains the CANN scope and logs excluded HOST IDs.
python setup.py --arch-version 2201 --ops all --artifact run --make-clean
```

Replace `2201` with `3510` for that architecture. Install the resulting wheel
before running the existing matching operator cases. `--without-third-party`
only skips third-party profiles; it does not change native operator selection.

### CANN operator declarations

Declare an operator in `csrc/<domain>/<operator>/CMakeLists.txt` with
`sgl_register_operator()`. Optionally add `sgl_register_operator_variant()` calls
for architecture-specific implementations. Register the common declaration
before its variants. An operator with one implementation for all supported
architectures needs no variant.

#### Fields and required values

"Common" below means `sgl_register_operator()`; "variant" means
`sgl_register_operator_variant()`. The shared declaration accepts 17 configuration
fields, plus the `ID` and `ARCH` keys:

| Field | Purpose | Allowed declaration | Required value |
| --- | --- | --- | --- |
| `ID` | Operator identifier; associates variants with the common declaration. | Common and variant | Required in every declaration. |
| `ARCH` | Architecture selected by this variant. | Variant only | Required for a variant; must belong to the common `SUPPORTED_ARCHS`. |
| `CPP_SOURCES` | Host C++ implementation sources. | Common and variant | Required for `HOST`; rejected for `AICORE`/`AICPU`. |
| `EXECUTION_UNIT` | Execution unit. | Common only | Required: `AICORE`, CpuKernel-based `AICPU`, or host C++ `HOST`. |
| `CANN_OP_TYPE` | CANN operator type used by the OpDef or AICPU registration. | Common only | Required for `AICORE`/`AICPU`; rejected for `HOST`. |
| `SUPPORTED_ARCHS` | Architectures supported by the operator. | Common only | Required and non-empty; currently accepts `2201` and `3510`, without duplicates. |
| `TORCH_ADAPTER` | Torch adapter source file. | Common only | Required for `HOST`; optional for CANN registration, but required for each selected wheel operator. |
| `OPDEF_SOURCES` | CANN interface definition source files for codegen. | Common only | Required and non-empty for AI Core; optional for AICPU. |
| `INFER_SOURCES` | Handwritten shape/type inference source files. | Common and variant | Optional; may be cleared explicitly. |
| `ACLNN_SOURCES` | Explicit Host ACLNN implementations, alongside any generated interfaces. | Common only | Optional; used by the AICPU AddExample wrapper. |
| `ACLNN_HEADERS` | Explicit ACLNN headers installed in `op_api/include/`. | Common only | Optional; declare the public headers of explicit interfaces. |
| `AICPU_JSON` | CpuKernel registration JSON. | Common only | Required when packaging selected AICPU operators; not needed by AI Core. |
| `TILING_SOURCES` | Host Tiling source files. | Common and variant | The resolved list must be non-empty for each selected AI Core architecture; not required for AICPU. |
| `TILING_INCLUDE_DIRECTORIES` | Private header search directories for Tiling sources. | Common and variant | Optional; may be cleared explicitly. |
| `TILING_COMPILE_DEFINITIONS` | Private Tiling macros, written as `NAME` or `NAME=value`. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_SOURCES` | Device compilation sources. | Common and variant | Required for CANN after resolution; rejected for `HOST`. AI Core requires exactly one entry; AICPU accepts multiple `.cpp` files. |
| `KERNEL_INCLUDE_DIRECTORIES` | Private header search directories for the Kernel. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_COMPILE_DEFINITIONS` | Private Kernel macros, written as `NAME` or `NAME=value`. | Common and variant | Optional; may be cleared explicitly. |
| `KERNEL_COMPILE_OPTIONS` | Kernel compiler options. | Common and variant | Optional; may be cleared explicitly. |

CANN declarations require `ID`, `EXECUTION_UNIT`, `CANN_OP_TYPE` and
`SUPPORTED_ARCHS`; AI Core additionally requires common `OPDEF_SOURCES`.
`TILING_SOURCES` and `KERNEL_SOURCES` may come from a variant. Each selected AI
Core architecture still requires non-empty Tiling sources and exactly one
Kernel entry. AICPU requires non-empty Kernel sources, without requiring an
AI Core OpDef or Tiling function. A CANN-only `.run` build does not require
`TORCH_ADAPTER`.

The same `--ops` selection supports AI Core-only, AICPU-only and mixed packages.

Use `SUPPORTED_ARCHS` to exclude an unsupported architecture rather than clearing
required sources. `--ops all` excludes operators that do not support the selected
architecture; explicitly requesting such an operator fails.

Source and include paths may be absolute or relative to the directory containing
the declaring `CMakeLists.txt`; relative paths are normalized at registration.
Source files and include directories in the final selected configuration must
exist. Compiler macros use `NAME` or `NAME=value`, without a `-D` prefix.

#### Variant inheritance, replacement and clearing

All CANN fields allowed in variants use the same rule, including
`KERNEL_SOURCES`: a variant replaces the **entire field**, not individual list
items. It never appends the common list.

| Variant declaration | Resolved value for that architecture |
| --- | --- |
| Field omitted | Inherit the common field. |
| `FIELD value1 value2` | Replace the complete common field with this list. List every value still needed, including shared helpers or flags. |
| `FIELD ""` | Clear an optional field for this architecture. |
| Bare `FIELD` without a value argument | Fail with a missing-value error. |

Explicit clearing is allowed for `INFER_SOURCES`, private include directories,
private macros and Kernel compiler options. Clearing `TILING_SOURCES` for AI Core,
or `KERNEL_SOURCES` for either execution unit, fails the final required-source
checks. Clearing private settings removes only those declared for this operator,
not the build target's shared settings. Clearing `INFER_SOURCES` omits handwritten inference from GRAPH; the
generated `op_proto.cc`, when present, is still compiled. Only omit inference
when the operator's supported call paths do not require that handwritten
implementation.

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

#### Building and installing AICPU operators

Use an initialized CANN 9.2.0 SDK with its HCC compiler and AICPU development
files, plus matching PyTorch and torch_npu. Select AICPU operators with the same
`--ops` option used for AI Core operators. For a repository-owned AICPU-only
wheel on A2/A3:

```bash
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 2201 --ops add_example --artifact wheel --without-third-party --make-clean
```

On A5, use `--arch-version 3510`. For a mixed wheel, select
`--ops apply_token_bitmask,add_example`.

When installing the RUN package manually, pass `--force` to the generated installer. For the A2/A3 custom build above with `--artifact run`, install it by:

```bash
bash /workspace/sgl-kernel-npu/output/sgl_kernel_npu-2201-custom.run --install-path=/workspace/sgl-kernel-npu/installed --force
```

After installing the wheel, run the example through the public API:

```bash
python3 -m pytest /workspace/sgl-kernel-npu/tests/python/operators/examples/test_add_example.py --impl aicpu --impl-ref torch_native --skip-acc-check=false --enable-perf=false -ra
```

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
python3 /workspace/sgl-kernel-npu/setup.py --make-clean
python3 /workspace/sgl-kernel-npu/setup.py --arch-version 3510 --ops all
```

Cleanup removes `build/cann`, `build/torch` and `build/wheel`. Existing builds may
still clear the shared `build/` directory; rerunning the new entry reconstructs
its intermediates. Do not run both builds concurrently in one checkout.

### Installation and public API

The distribution name remains `sgl-kernel-npu`, imported as `sgl_kernel_npu`.
Wheel build tags distinguish architecture, native operator selection, and bundled third-party dependencies:

```text
sgl_kernel_npu-<version>-1_refactor_2201_all_<set-id>-<python>-<abi>-<platform>.whl
sgl_kernel_npu-<version>-1_refactor_2201_custom_<set-id>-<python>-<abi>-<platform>.whl
sgl_kernel_npu-<version>-1_refactor_2201_custom_<set-id>_tp_<dependency-id>-<python>-<abi>-<platform>.whl
```

Omitted `--ops` or `--ops all` produces `all`; explicit IDs produce `custom`.
Build-tag fields use underscores. `<set-id>` is the first 10 hexadecimal characters
of the selected native operator list's SHA-256 hash. `tp` means `third_party`;
the optional `tp_<dependency-id>` identifies bundled third-party contents with a
10-character SHA-256 fingerprint. Neither ID is a source Git commit.
Schema-2 wheels hash the actual native IDs for `all` as well as `custom`, so
different actual native sets have distinct build tags, including when selected by `all`.
The package's `_build_manifest.json` records the actual full native selection,
`operators_by_type`, SDK targets and third-party payloads.
CPP-only builds have no repository device compute units. The native cache checks
include both subsets and the CANN vendor path; changing the native selection requires
`--make-clean`. The third-party cache keeps its existing independent rules.
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
