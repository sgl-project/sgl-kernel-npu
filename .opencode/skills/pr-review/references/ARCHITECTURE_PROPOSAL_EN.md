# Architecture and Dependency Direction

The layers distinguish internal components, external compilation and execution dependencies, and external operator providers. `csrc` is a source location; runtime calls go through the compiled bindings. Transitive dependencies are not expanded.

The NPU dependencies are TritonAscend, TileLangAscend, and torch_npu; directory names and `impl` values remain unchanged. torch_npu extends PyTorch and does not replace the torch dependency. [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend) exists but has not yet been integrated into this repository.

External operators are divided into **CANN operators** and **third-party operator libraries**, such as NPU-compatible implementations from DeepSeek and FLA. The latter is a reserved integration path. Implementations ported into and maintained by this repository remain internal components, regardless of their original source. Build dependency versions are maintained separately from this architecture reference.

- Higher layers call lower layers; lower layers do not depend on higher layers.
- SGLang can call `functional/modules` or `operators` directly. These are the framework entry points in this design.
- `functional/modules` uses operators through `operators`, which selects and dispatches implementations.
- `operators` has three call paths: `implementation`, repository-native extensions, and external operator libraries. The latter two bypass `implementation`.
- Implementation backends do not call one another.
- Each layer can use utilities within its scope. Utilities do not depend on the business modules that call them.

# Directory Structure

This is the target structure. `norm/rmsnorm_bias` is the migration example; not all domains have been refactored. The choice between `functional` and `modules` remains open. Domain directories are added as needed. They do not have their own `utils/` directories; shared helpers belong to the corresponding backend or layer.

```text
sgl_kernel_npu/
├── implementation/
│   ├── triton/
│   │   ├── attention/
│   │   │   └── <operator>.py    # Kernel and first-level wrapper
│   │   ├── moe/
│   │   │   └── <operator>.py
│   │   └── utils/
│   ├── tilelang/
│   │   ├── attention/
│   │   │   └── <operator>.py
│   │   ├── moe/
│   │   │   └── <operator>.py
│   │   └── utils/
│   ├── torch_native/
│   │   ├── attention/
│   │   │   └── <operator>.py    # Native PyTorch implementation
│   │   ├── moe/
│   │   │   └── <operator>.py
│   │   └── utils/
│   └── utils/
├── operators/
│   ├── attention/
│   │   └── <operator>.py
│   ├── moe/
│   │   └── <operator>.py
│   └── utils/
│       ├── registry.py         # Operator registration and implementation lookup
│       └── generation.py       # Architecture mapping and hardware detection
└── functional/                 # Or modules; to be decided
    ├── attention/
    │   └── <module>.py
    ├── fla/
    │   └── <module>.py
    ├── kda/
    │   └── <module>.py
    └── utils/
```

# Core Responsibilities

| Layer | Core responsibilities |
| --- | --- |
| `implementation` | Contains Triton, TileLang, and torch_native implementations and their first-level wrappers. Handles data preparation, output allocation, and kernel launches. |
| `operators` | Registers public operators and implementations, manages availability for the current hardware generation, and provides unified lookup and dispatch. Calls `implementation` or directly adapts repository-native extensions and external operator libraries. |
| `functional/modules` | Composes operators into complete algorithms or modules, such as attention, FLA, and KDA. Manages intermediate results and state transitions. |
| `csrc` | Contains repository-native Ascend C operators and host/binding code. Builds native extensions and registers callable operator interfaces. |
| External operator libraries | Includes CANN operators and third-party operator libraries, integrated through public Python APIs or existing native bindings. |
| Scoped `utils` | Contains helpers shared within the corresponding backend or layer. |

Open question: the ownership of input constraints and validation remains undecided. The current preference is to keep implementation-specific checks close to the implementation. For native and external operators that bypass `implementation`, their bindings and private adapters in `operators` handle these checks.

# Key Design

## Registration Objects and State

Registration and dispatch live in `operators`, with three tables:

| Table | Contents |
| --- | --- |
| `_OPERATOR_NAMES` | Public operator function object → operator name, populated by `register_operator`. |
| `_IMPLS` | `(operator name, impl)` → implementation entry point available on the current generation. |
| `_UNSUPPORTED_IMPLS` | `(operator name, impl)` → known generations for which no entry point has been registered. Undeclared implementations have no key; a key with an empty set remains when all known generations are covered. |

Callers can query implementations using the public operator function without repeating its string name. An operator can still dispatch by name internally.

`register_operator` associates a public function with its name. `register_impl` records an implementation declaration and classifies it by generation. `get_impl` resolves the operator name from the public function, then returns the selected entry point for the caller to execute. Operator registration and generation filtering are detailed here and in Hardware Generation Management below. RMSNorm declares `auto`, `triton`, and `torch_native`. The first two refer to the same entry point, and each call executes only the selected backend.

| Function | Purpose and usage |
| --- | --- |
| `register_operator(op_name)` | Decorator associating a public function object with a unique operator name. Rejects duplicate functions or names. Returns the original function without adding a wrapper. |
| `get_op_name(operator)` | Looks up the name of a registered public function. Rejects unregistered functions rather than inferring names from paths or function names. |
| `register_impl(op_name, impl, *, supported_generations=None, excluded_generations=None)` | Decorator allowing different entry points under the same implementation name for non-overlapping generation sets. Rejects overlapping registrations. Support ranges are independent across names. |
| `list_impls(operator)` | Accepts a public function and returns a sorted list of all declared implementation names, including those unsupported on the current generation. |
| `get_impl(operator, impl)` | Accepts a public function, resolves its operator name, and returns an entry point available on the current generation. The caller executes it. |
| `list_impls_by_name(op_name)` | String-based lookup. Returns `[]` for an unknown operator or one whose module has not yet been imported. |
| `get_impl_by_name(op_name, impl)` | String-based resolution. Distinguishes an unsupported generation from an implementation that was not found. Does not silently fall back. |

## Public Operator Registration and Lookup

The code lives in `operators/utils/registry.py`. The following excerpts show public function registration and lookup. Implementation registration and classification are described in Hardware Generation Management.

```python
def register_operator(op_name):
    def decorator(fn):
        with _LOCK:
            if fn in _OPERATOR_NAMES:
                raise ValueError("Public operator function already registered: {}".format(fn))
            if op_name in _OPERATOR_NAMES.values():
                raise ValueError("Duplicate public operator name: {}".format(op_name))
            _OPERATOR_NAMES[fn] = op_name
        return fn

    return decorator


def get_op_name(operator):
    if operator not in _OPERATOR_NAMES:
        raise ValueError("Unregistered public operator: {!r}".format(operator))
    return _OPERATOR_NAMES[operator]


def list_impls_by_name(op_name):
    with _LOCK:
        keys = tuple(_UNSUPPORTED_IMPLS)
    return sorted(name for op, name in keys if op == op_name)


def get_impl_by_name(op_name, impl):
    key = (op_name, impl)
    try:
        return _IMPLS[key]
    except KeyError:
        pass
    if key in _UNSUPPORTED_IMPLS:
        raise UnsupportedGenerationError(
            op_name, impl, _HARDWARE, frozenset(Generation) - _UNSUPPORTED_IMPLS[key]
        )
    raise ImplementationNotFoundError(
        "Unknown impl {!r} for {}; declared: {}".format(
            impl, op_name, list_impls_by_name(op_name)
        )
    )


def get_impl(operator, impl):
    return get_impl_by_name(get_op_name(operator), impl)


def list_impls(operator):
    return list_impls_by_name(get_op_name(operator))
```

## Registry-Based Dispatch and Lazy Loading

`operators/norm/rmsnorm_bias.py` registers two entry points under three implementation names. The public interface dispatches through a registry lookup. Each backend is imported on first use, and subsequent calls reuse a globally cached function:

```python
from ..utils.registry import get_impl_by_name, register_impl, register_operator


_triton_fn = None
_torch_native_fn = None


@register_impl("norm.rmsnorm_bias", "auto")
@register_impl("norm.rmsnorm_bias", "triton")
def _triton_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.rmsnorm_bias import rmsnorm_bias

        _triton_fn = rmsnorm_bias
    return _triton_fn(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)


@register_impl("norm.rmsnorm_bias", "torch_native")
def _torch_native_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.rmsnorm_bias import rmsnorm_bias

        _torch_native_fn = rmsnorm_bias
    return _torch_native_fn(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)


@register_operator("norm.rmsnorm_bias")
def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    impl="auto",
):
    impl_fn = get_impl_by_name("norm.rmsnorm_bias", impl)
    return impl_fn(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)
```

Hardware detection runs when the registry module is imported. Decorators execute and classify implementations when the operator module is imported. Every operator call still performs a registry lookup; there is no last-used implementation cache or dedicated `auto` cache. Only loaded backend functions are cached. A failed import does not populate the cache. Concurrent first calls are not guaranteed to execute the import statement exactly once.

The default is `impl="auto"`. It currently shares the Triton entry point and loading cache with `triton`; a dedicated selection function can be registered as `auto` later. `list_impls` lists all declarations. Successful `get_impl` resolution means the generation declaration permits the implementation, not that its dependencies are installed or its input constraints are satisfied.

Public operator registration and implementation registration are independent and are associated through the same operator name. Public function lookup relies on function object identity, so the operator module must be imported before calling `get_impl` or `list_impls`. Internally, operators use `get_impl_by_name` to avoid relying on their own global function name, which may be replaced or rebound.

Package entry points export the interfaces through `__init__.py`:

```python
# operators/__init__.py
from .utils.registry import get_impl, list_impls

__all__ = ["get_impl", "list_impls"]

# operators/norm/__init__.py
from .rmsnorm_bias import rmsnorm_bias

__all__ = ["rmsnorm_bias"]
```

Import the operator module before querying or calling it:

```python
from sgl_kernel_npu.operators import get_impl, list_impls
from sgl_kernel_npu.operators.norm import rmsnorm_bias

print(list_impls(rmsnorm_bias))
# ['auto', 'torch_native', 'triton']

impl_fn = get_impl(rmsnorm_bias, "triton")  # Resolve only; do not execute.

y_triton = rmsnorm_bias(x, weight, bias, 1e-6, impl="triton")
y_native = rmsnorm_bias(x, weight, bias, 1e-6, impl="torch_native")
```

## Concurrency Contract

- `_LOCK` protects public operator registration, duplicate implementation checks, updates to remaining generation sets, and writes to the current-generation entry-point table.
- `get_op_name` and successful `get_impl_by_name` lookups do not acquire the lock. Callers must not query a target implementation before its registration has completed.
- List queries acquire the lock only to copy a snapshot of keys; filtering and sorting happen outside the lock. Building the candidate list for an implementation-not-found error also uses this lock. The unsupported-generation error path does not acquire it.

# API Documentation Management

## Documentation Locations

The following paths are relative to the `sgl-kernel-npu` repository root:

- Chinese index: `docs/api/zh/API_list.md`.
- English index: `docs/api/en/API_list.md`.
- Operator documentation: `docs/api/<zh|en>/operators/<domain>/<operator>.md`, mirroring the `operators` code layout.
- Example: `docs/api/zh/operators/norm/rmsnorm_bias.md` and its English counterpart.

The index covers public operators and contains only two columns: interface name and supported backends. Each interface name links to its documentation in the same language using a relative path.

## Operator Documentation Structure

Use the operator name as the level-one heading and organize level-two headings in this order:

| Section | Requirements |
| --- | --- |
| Hardware support | One table only: columns represent hardware generations, rows represent implementations, and cells indicate supported, unsupported, pending verification, or not implemented. |
| Signature | Public import path and complete signature, including defaults and keyword-only parameters. State the default implementation and the current selection of `auto`, consistent with the code. |
| Description | Briefly describe the computation or operation. Computational operators must include mathematical formulas and define their symbols. Quantized operators must specify scaling, offsets, rounding, and saturation. |
| Parameters and return value | Separate tables for parameters and return values. Each parameter must include its type, description, and constraints. Tensor parameters must also specify shape, dtype, device, and layout requirements. Include parameter defaults and describe outputs and side effects. Keep additional notes below the tables concise. |
| `<backend> backend` | A separate level-two heading for each implemented backend. Describe its implementation, supported features, and important differences, such as layout constraints, quantization support, and numerical behavior. |
| Example | A minimal runnable example showing imports, input construction, and implementation selection. |

# Hardware Generation Management

## Hardware Detection and Initialization

`operators/utils/generation.py` defines `Generation.ARCH22/ARCH35` and the exact `ARCH_TO_GENERATION` mapping. It queries the architecture ID of visible device 0 with `acl.rt.get_device_info(0, 601)`: `2201 → arch22`, `3510 → arch35`. Detection no longer relies on chip names.

The assumption is that a process runs on one machine and all cards on that machine belong to the same generation. Importing the registry module executes `_HARDWARE = get_hardware()` to detect and cache hardware information. Selecting a different device later does not change the detected generation. Detection failures, nonzero return codes, and unknown architecture IDs raise `HardwareDetectionError` and abort the import. There is no runtime `_initialize()` call or repeated hardware detection.

## Declaring Supported Generations

The two generation arguments to `register_impl` are mutually exclusive, and their members must be `Generation` enum values:

| Configuration | Meaning |
| --- | --- |
| Both arguments are `None` | Supports all known generations. |
| `supported_generations={Generation.ARCH22}` | Supports only arch22. An empty set supports none. |
| `excluded_generations={Generation.ARCH35}` | Supports all known generations except arch35. An empty set excludes none. |
| Both arguments are specified, or a generation value is invalid | Registration fails. |

The same `(op_name, impl)` can have multiple entry points for different generations, but their support sets must not overlap. Overlaps are rejected even if they concern a generation other than the current machine's. When one function is registered under multiple names, each name maintains its own support range; no consistency check is imposed across names. This example illustrates the API and does not imply that production RMSNorm has these restrictions:

```python
from sgl_kernel_npu.operators.utils.generation import Generation
from sgl_kernel_npu.operators.utils.registry import register_impl


@register_impl("example.op", "auto", supported_generations={Generation.ARCH22})
@register_impl("example.op", "triton", supported_generations={Generation.ARCH22})
def _example_arch22_impl(*args, **kwargs):
    ...


@register_impl("example.op", "auto", supported_generations={Generation.ARCH35})
@register_impl("example.op", "triton", supported_generations={Generation.ARCH35})
def _example_arch35_impl(*args, **kwargs):
    ...
```

Registration selects the current-generation entry point immediately rather than retaining a complete history of declarations. `_UNSUPPORTED_IMPLS` tracks both declared names and generations not yet covered. The core logic is:

```python
# Inside the register_impl decorator; supported has already been validated.
known = frozenset(Generation)
key = (op_name, impl)
with _LOCK:
    unsupported = _UNSUPPORTED_IMPLS.get(key, known)
    overlap = (known - unsupported) & supported
    if overlap:
        raise ValueError(
            "Duplicate impl: {}; overlapping generations={}".format(
                key, sorted(g.value for g in overlap)
            )
        )
    _UNSUPPORTED_IMPLS[key] = unsupported - supported
    if _HARDWARE.generation in supported:
        _IMPLS[key] = fn
```

For example, when the known architectures are arch22 and arch35, registering arch22 leaves `{Generation.ARCH35}` uncovered. Registering arch35 afterward leaves an empty set. The same key can exist in both `_IMPLS` and `_UNSUPPORTED_IMPLS`, representing the current entry point and coverage gaps for other generations, respectively.

Lookup first checks `_IMPLS` and returns the entry point directly on a hit, without rechecking the generation. Otherwise, if the key exists in `_UNSUPPORTED_IMPLS`, the supported set is derived as all known generations minus uncovered generations, and `UnsupportedGenerationError` is raised. If the key does not exist, `ImplementationNotFoundError` is raised. There is no silent fallback. Registration updates are protected by the lock; successful lookups are lock-free.
