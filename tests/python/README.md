# Operator tests

## Run tests

Run from the repository root with the current `sgl_kernel_npu` installed,
`torch_npu`, Triton-Ascend and an available NPU.

```bash
# Accuracy only.
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py \
  --skip-acc-check=false --enable-perf=false -v -s -x -rs

# Use each case's performance selection and save results.
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py \
  --prof-dir=results/profilings --csv-path=results/performance.csv -v -s
```

| Option | Usage |
| --- | --- |
| `--impl`, `--impl-ref` | Tested and reference implementations; defaults: `auto`, `torch_native`. Only `impl` is benchmarked. |
| `--enable-perf=true/false` | Enable/disable performance for all cases; overrides code settings. Omit to use each case's `enable_perf`, defaulting to False. |
| `--skip-acc-check=true/false` | Skip/require accuracy checks; overrides code settings. Default: False. |
| `--prof-dir` | Profiling root; overrides `prof_dir`. Default: `./profilings`. |
| `--csv-path` | Append summary CSV; overrides `csv_path`. No CSV is written unless a path is supplied. |
| `--npu-device=0` | Select a visible logical NPU; omitted means the current NPU. |

Paths alone do not enable performance. Skipping accuracy does not enable it either;
if both checks are disabled, the operator is not executed. Use serial pytest or
`-n 1`, and avoid concurrent profiling on the same device or writing the same CSV.

## Write a test

Place tests in `operators/<module>/test_<operator>.py`, matching the public API.
Import operators from `sgl_kernel_npu.operators`, not backend implementations.
The shared `conftest.py` configures the default NPU device for these tests;
no device, impl, `tmp_path` or `record_property` fixture arguments are needed.

- Generate inputs in a separate function; parameterize shape, dtype, mode and operator options.
- Include production shapes from actual SGLang calls, accounting for reshaping and sharding. Label the model and scenario. If only model dimensions are borrowed, say so.
- Add uncommon legal edge cases: minimum inputs, unaligned dimensions and block tails. Explain the purpose of each mode.
- Check accuracy first, then performance. Select performance shapes with a separate `perf_shape` list.

Minimal non-quantized example:

```python
import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 4096)]


def generate_test_data(shape, dtype):
    torch.npu.manual_seed(42)
    return (
        torch.randn(shape, dtype=dtype),
        torch.randn(shape[1], dtype=dtype),
        torch.randn(shape[1], dtype=dtype),
    )


@pytest.mark.parametrize("shape", [
    pytest.param((1, 4096), id="qwen3-8b-width"),  # Borrowed model width, not a traced call.
    pytest.param((7, 1025), id="edge-tail"),  # Unaligned reduction and odd rows.
])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("eps", [1e-6])
def test_rmsnorm_bias(
    shape,
    dtype,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias

    x, weight, bias = generate_test_data(shape, dtype)
    executor = check_accuracy(rmsnorm_bias, args=(x, weight, bias, eps))
    check_performance(executor, enable_perf=shape in perf_shape)
```

See [the RMSNorm test](operators/norm/test_rmsnorm_bias.py) for additional modes.

## Core helpers

### `check_accuracy` and accuracy standards

`check_accuracy(operator, args=(), kwargs=None, ...)` compares two implementations
and returns a zero-argument executor for performance collection. Use `kwargs` for
keyword-only operator parameters. Explicit `impl` / `impl_ref` arguments override
the corresponding test option; otherwise the configured defaults are used.

The reference must be independent: aliases such as RMSNorm's `auto` and `triton`
cannot be compared. If `torch_native` is unavailable, select another registered
reference with `--impl-ref` or `impl_ref=`, or use the lower-level helpers below.

Tensor, tuple/list and dict outputs are compared recursively; corresponding
Tensors must have matching shape, dtype and device. Inputs are shared, so in-place
or stateful operators require a separate input-isolation/reset strategy.

Default atol/rtol: FP32 `1e-5`, FP16 `1e-3`, BF16 `5e-2`; integers compare exactly.
Override selected output dtypes with a dictionary; omitted dtypes keep defaults:

```python
from tests.python.utils.standards import AccuracyStandard

executor = check_accuracy(
    operator,
    args=inputs,
    kwargs=keyword_inputs,
    standard={"torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4)},
)
```

Use `skip_acc_check=True` to skip comparison unless overridden by the CLI.
An unsupported tested architecture (`arch22` or `arch35`) skips the case; a missing
implementation or unavailable reference fails. Inspect skip reasons with `-rs`.
Architecture detection runs when the operator registry is imported and uses
`acl.rt.get_device_info(0, 601)`. Keep the CANN environment configured so Python
can import `acl`; detection failures and unknown architectures fail rather than skip.
All visible devices must share an architecture; `--npu-device` selects the test
device but does not change the cached architecture.

### `check_performance`

Pass the executor returned by `check_accuracy`:

```python
result = check_performance(
    executor,
    enable_perf=True,
    prof_dir="results/profilings",
    csv_path="results/performance.csv",
)
```

This helper handles profiling, printing, pytest properties and optional CSV output.
It returns a result dictionary, or None when disabled. Profiling files are retained
under `<prof_dir>/prof_<function>/<case-run>/`; they are not automatically deleted.
CSV times use three decimal places. Existing CSV columns must match; use a new file
after schema changes. `accuracy_checked` records whether accuracy passed.

The timing metric is average summed NPU task duration in microseconds, not
end-to-end latency. `warmup` and `active` both default to 5 and can be overridden.
Both must be positive Python integers; booleans and non-integers raise TypeError,
and non-positive integers raise ValueError before the operator executes.

### `assert_accuracy` and `benchmark_npu`

For a custom comparison flow, call a trusted reference yourself, then use:

```python
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.performance import benchmark_npu

assert_accuracy(result, result_ref)  # Optional standard= uses the same dtype mapping.
average_us = benchmark_npu(executor, directory="results/custom-profile-run-001")
```

`executor` can be any zero-argument callable; `directory` must not already exist.
These lower-level helpers do not handle implementation selection, generation skips,
CLI switches or CSV reporting. Keep accuracy before profiling in custom flows too.

# Framework tests

Framework tests are separate from production operator accuracy/performance tests:

```text
framework/
  conftest.py                  Shared NPU runtime and isolated source-module loading
  operator_framework/          Registration, architecture, probe dispatch and imports
    conftest.py                Registry/generation fixtures and probe/backend setup
    probe.py                   Test-only public entry and lazy backend adapters
  testing_framework/           Accuracy standards, comparison, performance and reporting
    conftest.py                Accuracy/performance fixtures and integration marker
```

Run from the repository root with pytest, a working NPU/torch_npu/ACL environment
and an importable `sgl_kernel_npu` native runtime. Python framework modules are
loaded from this checkout; the installed native package may predate `operators`.

```bash
# All 225 cases, including one real profiler integration.
python -m pytest tests/python/framework -v -rs --npu-device=0

# Run either area separately.
python -m pytest tests/python/framework/operator_framework -v -rs --npu-device=0
python -m pytest tests/python/framework/testing_framework -v -rs --npu-device=0

# Unit tests only, or real profiler integration only.
python -m pytest tests/python/framework -m 'not npu_integration' -v -rs --npu-device=0
python -m pytest tests/python/framework -m npu_integration -v -s -rs --npu-device=0
```

Both subdirectories inherit `npu_runtime`, `package_factory` and `operator_modules`
from `framework/conftest.py`. Area-specific fixtures live in each subdirectory's
own `conftest.py`. Non-autouse fixtures run only when requested.

There are 75 operator-framework cases, 149 testing-helper unit cases and one real
NPU integration case. Use a single pytest process without parallel workers; add
`-x` to stop on the first failure, or `--junitxml=framework-ut.xml` to save a report.
These tests set their own implementation/accuracy/performance contexts; operator
CLI selections do not rewrite their scenarios. Missing NPU/ACL/native prerequisites
are errors rather than blanket skips.

Operator-framework tests use a test-only probe, registered with the checkout's
real registry, rather than a production RMSNorm operator. Only its backend leaves
are replaced in routing tests. The probe remains a separate module so import tests
can exercise module execution, registration and lazy loading. The public entry,
lazy adapters and their caches are test support: they exercise an integration
pattern, not a shared production
dispatcher or production-kernel correctness. The mainline currently implements
those adapters separately in each operator.

Testing-helper cases register lightweight Python implementations using small NPU
Tensors. Accuracy comparisons, registry lookups and report/CSV logic remain real.
Most performance unit cases replace measurement or profiler boundaries; the
`npu_integration` case uses real NPU execution and profiler collection. None of
these cases establishes a historical performance-regression baseline.
