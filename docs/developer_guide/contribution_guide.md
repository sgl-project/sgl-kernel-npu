# Contributing New Operators to sgl-kernel-npu

This guide describes the shared contribution workflow and API conventions for new sgl-kernel-npu operators across backends. A complete contribution includes **an implementation, a public API, correctness tests, performance data, and bilingual API documentation**. Code paths are relative to the sgl-kernel-npu repository root. Replace `example_op` and `<module>` with actual names.

## Development Setup

Fork the repository to your GitHub account, then clone it and create a development branch:

```bash
git clone "https://github.com/<your_user_name>/sgl-kernel-npu.git"
cd sgl-kernel-npu
git switch -c feature/new-operator
```

Follow the [build and installation guide](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) to prepare matching CANN, PyTorch, torch_npu, and backend dependencies. Build and install the package with your changes before testing, and confirm that tests load the code from that build. Use the target branch's `python/sgl_kernel_npu/README.md` for its specific build options.

## 1. Define the Use Case and API

Describe the operator's purpose and its model or SGLang use case. Check for an existing API with equivalent semantics first; extend it with a new implementation where appropriate.

Define inputs and outputs, shapes, dtypes, layouts, optional arguments, in-place behavior, and target architectures. Specify the mathematical semantics of computational operators, including scaling, offsets, rounding, and saturation for quantization. Preserve source attribution, licenses, and required copyright notices when reusing external code.

## 2. Provide an Implementation and an Independent Reference

Suggested layout (Triton and torch_native illustrate the directory conventions; adapt it to the backends provided):

```text
python/sgl_kernel_npu/sgl_kernel_npu/
├── implementation/
│   ├── triton/<module>/example_op.py
│   ├── torch_native/<module>/example_op.py
│   └── cpu_reference/<module>/example_op.py  # Optional existing CPU reference
└── operators/<module>/
    ├── __init__.py
    └── example_op.py

tests/python/operators/<module>/test_example_op.py
docs/api/zh/operators/<module>/example_op.md
docs/api/en/operators/<module>/example_op.md
```

- Place kernels and first-level wrappers in `implementation/<backend>`. Wrappers allocate outputs, configure launches, and invoke kernels.
- Check implementation-specific constraints before launching kernels. Prefer metadata checks such as shape, dtype, and stride, with explicit error messages.
- Provide a semantically equivalent, independently computed `torch_native` implementation for comparison; it should not call the kernel under test.
- If torch_native is impractical, choose another trusted independent reference and explain the choice in the PR.
- Preserve existing CPU references in `implementation/cpu_reference/<module>` and register them as `impl="cpu_reference"`. Keep their computation independent of NPU backends. Match the public input/output contract, including in-place updates and the returned device; document any changes to the original reference's semantics. For example, `sampler.apply_token_bitmask` copies NPU inputs to CPU for calculation and writes the result back to the original logits. Use these references for accuracy checks, not NPU performance measurements.
- Wrap exposed native bindings for Ascend C operators in the operators layer, as shown in Section 3. Integrate other external operators through their public APIs as well.
- For buffers, handles, or communication state, define resource lifetimes and the module-layer design first.

## 3. Register Implementations and Expose the Public API

**Computational operators must provide at least two independent implementations for accuracy comparison. Use `torch_native` as the recommended reference implementation.** Different aliases of the same implementation, such as `auto` and its selected backend, do not count as separate implementations. If exceptional circumstances prevent this, explain the reason and the alternative correctness validation method in the PR.

Register implementations and the public function in `operators/<module>/example_op.py`. Add keyword-only `impl="auto"` to the public API. Register the default backend as `auto` as well. Lazily import and cache Python backend functions; Ascend C entry points call registered native bindings directly.

The following skeleton registers Triton, Ascend C, and torch_native under one public API. Triton is the default, and torch_native provides the accuracy reference. Keep the entry points for the backends actually provided:

```python
import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.example_op", "auto")
@register_impl("norm.example_op", "triton")
def _triton_impl(x):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.example_op import example_op

        _triton_fn = example_op
    return _triton_fn(x)


@register_impl("norm.example_op", "ascendc")
def _ascendc_impl(x):
    return torch.ops.npu.example_op(x)


@register_impl("norm.example_op", "torch_native")
def _torch_native_impl(x):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.example_op import example_op

        _torch_native_fn = example_op
    return _torch_native_fn(x)


@register_operator("norm.example_op")
def example_op(x, *, impl="auto"):
    """auto selects Triton. Describe the operation and x's type and shape."""
    impl_fn = get_impl_by_name("norm.example_op", impl)
    return impl_fn(x)
```

Here, `auto` is registered only on `_triton_impl`. To default to Ascend C, move the `auto` decorator to `_ascendc_impl`, avoiding duplicate registration for the same architecture. The Ascend C example assumes package loading has registered `torch.ops.npu.example_op`; adapt the namespace, arguments, and return values to the actual interface. No `implementation/ascendc` forwarding layer is needed. Keep `import torch` at module scope rather than executing the import statement on every call.

Export the function from `operators/<module>/__init__.py`. Keep its docstring concise: default implementation, behavior, argument meanings and types, and tensor shapes.

For architecture restrictions, replace the corresponding registration decorators above with declarations using `Generation` enum members. Keep the function body unchanged; do not append duplicate registrations:

```python
from ..utils.generation import Generation


@register_impl(
    "norm.example_op", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "norm.example_op", "triton", supported_generations={Generation.ARCH22}
)
def _triton_impl(x):
    ...  # Use the same loading and invocation logic.
```

Alternatively, use `excluded_generations={Generation.ARCH35}`; the two options are mutually exclusive. Omitting both declares support for all known architectures, so choose declarations that reflect actual support. Keep auto and the concrete implementation consistent when they alias the same default entry point. An auto alias denotes a default implementation, not automatic fallback.

## 4. Test Through the Public API

Add tests for new functionality. For bug fixes, add regression cases that reproduce the original problem.

Place tests in `tests/python/operators/<module>/test_<operator>.py` and call the public API. Follow these four conventions:

1. **Manage all test parameters through pytest.** Use `pytest.mark.parametrize` for shapes, dtypes, modes, and operator arguments. Include both **production shapes** and **edge-case shapes**. Match production shapes to actual SGLang calls as closely as possible, accounting for reshaping and sharding, and identify the model or scenario. Choose edge cases such as minimum inputs, unaligned dimensions, and tile tails within the supported range.
2. **Generate data in a separate function.** Centralize input construction in `generate_data`, driven by the parametrized configuration. The existing conftest supplies the default NPU device context.
3. **Use a consistent test flow.** Prefer `generate_data → check_accuracy → check_performance`, collecting performance after accuracy passes. Explain the reason and test method in the PR when a special scenario requires a custom flow.
4. **Explicitly declare performance-monitoring shapes.** List shapes to profile in a separate `perf_shape` list and use it to set `enable_perf`. Add dtype or mode conditions where appropriate.

The example below uses the RMSNorm public API to demonstrate test organization, rather than operator-specific functionality. Replace the illustrative shapes with production and edge cases for your operator; the model dimensions shown do not represent captured production calls.

```python
import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [
    (1, 4096),
    (16, 7168),
]


def generate_data(shape, dtype, mode):
    # conftest.py supplies the default NPU device context.
    torch.npu.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    bias = torch.randn(shape[-1], dtype=dtype)
    if mode == "zero":
        x.zero_()
    elif mode != "normal":
        raise ValueError(f"Unknown mode: {mode}")
    return x, weight, bias


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-width"),
        pytest.param((16, 7168), id="deepseek-v3-width"),
        pytest.param((1, 1), id="edge-minimum"),
        pytest.param((7, 1025), id="edge-tail"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias

    x, weight, bias = generate_data(shape, dtype, mode)
    executor = check_accuracy(
        rmsnorm_bias,
        args=(x, weight, bias, eps),
    )
    # An accuracy failure raises before performance collection.
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.float16
            and mode == "normal"
        ),
    )
```

Define modes according to the operator's test objectives and explain them in comments or parameter IDs. Here, `normal` means random data and `zero` means zero inputs. Pass keyword-only operator arguments through `check_accuracy(..., kwargs={...})`.

References, relative to the sgl-kernel-npu repository root:

- RMSNorm test: `tests/python/operators/norm/test_rmsnorm_bias.py`.
- Shared test utility guide: `tests/python/README.md`.

## 5. Validate Correctness, Input Coverage, and Performance

Run API tests on the declared hardware and software configurations. Pass accuracy checks before collecting performance. Use a separate `perf_shape` list for representative shapes, optionally combined with dtype and mode conditions.

```bash
# Replace the test path with the new operator's test.
python -m pytest "tests/python/operators/<module>/test_example_op.py" \
  --skip-acc-check=false --enable-perf=false -v -s -x -rs

python -m pytest "tests/python/operators/<module>/test_example_op.py" \
  --skip-acc-check=false \
  --prof-dir=results/profilings --csv-path=results/example_op.csv -v -s
```

Omitting `--enable-perf` uses per-case settings; explicit true/false overrides all cases. CSV output requires `--csv-path`. Current shared profiling utilities support serial/single-worker execution and report average summed NPU task durations, not end-to-end latency.

Report the baseline implementation, shapes, dtypes, hardware, software versions, and measurement method. Recheck accuracy and performance when extending shape, dtype, or layout support. Reject unsupported inputs before kernel launch. Keep raw profiling and build artifacts local.

## 6. Complete Bilingual API Documentation

Update both `docs/api/{zh,en}/API_list.md` indexes. Each index contains only linked API names and supported backends. Organize each operator document as follows:

| Section | Required content |
| --- | --- |
| Hardware support | One table with implementations as rows and architectures such as arch22/arch35 as columns, marking supported, unsupported, or unverified. |
| Signature | Public import path, complete signature, defaults, and the implementation selected by auto. |
| Description | Behavior, mathematical formulas for computational operators, and necessary numerical conversion rules. |
| Parameters and return value | Separate tables covering types, descriptions, and constraints; tensors include shape, dtype, device, and layout requirements. List parameter defaults. |
| Backend descriptions | A separate second-level heading per backend, describing support and important differences. |
| Example | A minimal runnable example showing input creation, the public API, and impl selection. |

Keep Chinese and English versions consistent. Add module documentation for new module APIs. Put detailed test logs and performance results in the PR.

The repository includes a complete bilingual example:

```text
docs/api/
├── zh/operators/norm/rmsnorm_bias.md
└── en/operators/norm/rmsnorm_bias.md
```

Its indexes are `docs/api/zh/API_list.md` and `docs/api/en/API_list.md`. Use these documents as references for section structure, parameter tables, architecture support, and backend descriptions.

## 7. Submit the PR

Install pre-commit and enable its Git hook when setting up the development environment, then run all checks before submitting:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

Review the diff after automatic formatting, fix remaining issues, and rerun the checks until they pass. Commit on your development branch, push it to your fork, and open a PR against the upstream repository. Do not commit directly on `main`.

Include in the PR description:

- **Operator purpose and API**: describe the computation and the model or SGLang scenario it serves. Provide the public import path, function signature, and a usage example, and state which implementation `auto` selects by default.
- **Correctness**: include the test commands, results, and validation environment (chip and software versions).
- **Performance**: for a new operator, provide baseline timings for the cases selected for performance monitoring. For an optimization to an existing operator, provide before-and-after timings under the same test conditions.

Submission checklist:

- [ ] Kernel/wrapper and an independent comparison implementation are ready.
- [ ] Operator, impl, auto, architecture declarations, and public exports are complete.
- [ ] Tests cover production shapes, edge cases, argument branches, and input restrictions.
- [ ] Accuracy checks pass and representative performance measurements are available.
- [ ] Chinese and English API indexes and detail pages are updated.
- [ ] Build/dependency configuration is complete and pre-commit passes.

See `python/sgl_kernel_npu/sgl_kernel_npu/operators/norm/rmsnorm_bias.py`, `tests/python/operators/norm/test_rmsnorm_bias.py`, and `tests/python/README.md` in the repository for examples.
