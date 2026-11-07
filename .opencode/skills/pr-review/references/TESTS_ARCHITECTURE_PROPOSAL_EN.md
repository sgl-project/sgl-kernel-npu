# Test Framework Refactoring: Architecture Draft

Tests exercise public APIs and share utilities for data generation patterns, cross-backend comparison, performance collection, and reporting. This document describes the current RMSNorm example; features not yet implemented are tracked separately as TODOs. All paths are relative to the `sgl-kernel-npu` repository root.

## 1. Directory Structure

```text
tests/
├── __init__.py
└── python/
    ├── __init__.py
    ├── README.md
    ├── conftest.py                 # CLI options; fixtures set and restore device, backend, and reporting contexts
    ├── operators/                  # Public operator tests: parameters, data generation, accuracy, and performance
    │   ├── __init__.py
    │   └── norm/
    │       ├── __init__.py
    │       └── test_rmsnorm_bias.py
    ├── functional/                 # Reserved for module API tests mirroring production layout; name TBD
    ├── utils/                      # Shared test utilities
    │   ├── __init__.py
    │   ├── accuracy.py             # Recursive cross-backend comparison; returns an executor for benchmarking
    │   ├── standards.py            # Per-dtype accuracy tolerances with per-test overrides
    │   ├── performance.py          # Profiling, timing, optional CSV, pytest properties, and console output
    │   ├── common.py               # make_executor, test configuration, implementation identity checks, generation skip/error policy
    │   └── device.py               # NPU device context, device configuration checks, and concurrency restrictions
    └── sgl_kernel_npu/             # Existing tests and the original utils.py remain in place
```

New utilities are imported through `tests.python.utils` to avoid conflicts with the existing `utils.py`. The `__init__.py` files define package paths and do not perform application initialization. Other existing test directories remain unchanged.

## 2. Dependency Direction

```text
pytest CLI
    │
    ▼
conftest.py ── configuration / device / reporting ──► Shared test utilities
                                                            ▲
                                                            │ call
                                               operators / functional tests
                                                            │
                                                            ▼
                                                Public operator / module APIs
                                                            │
                                                            ▼
                                      operators dispatch to backend or external operators
```

## 3. Key Design

### Test Organization

Shapes, dtypes, eps, and modes are managed with `pytest.mark.parametrize`, with one test function parameter per line. Data generation is isolated in `generate_test_data`. Tensors are created directly on the NPU by default, without a device argument in the test function.

Run important accuracy and performance regression cases by default. Mark supplementary cases, such as extended parameter combinations, noncritical generalization, and input validation, with `@pytest.mark.extra_checks`. Keep critical production scenarios and essential edge checks in the default set. Mark an entire test function or an individual parameter set with `pytest.param(..., marks=pytest.mark.extra_checks)`.

The repository's `pytest.ini` defaults to `-m "not extra_checks"`. Pass `-m extra_checks` for supplementary cases only, or `-m ""` for all cases. Selecting a supplementary case by function path or `-k` still requires overriding this default filter. `--enable-perf` does not select excluded cases.

```bash
python -m pytest tests/python/operators -v                    # Default core cases
python -m pytest tests/python/operators -m extra_checks -v    # Supplementary cases only
python -m pytest tests/python/operators -m "" -v              # All cases
```

Accuracy and performance checks share one test case and run in an explicit order. The RMSNorm example below omits data-generation details.

Shapes cover model-dimension examples and edge cases. No direct SGLang call to this standalone `rmsnorm_bias` API has been identified. The first two shapes use the hidden dimensions of Qwen3-8B and DeepSeek-V3; their row counts are example workloads, not captured production calls. When migrating other operators, prioritize shapes from actual call sites.

```python
import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Profile selected shapes by default; other shapes still run accuracy checks.
perf_shape = [(1, 4096), (16, 7168)]


def generate_test_data(shape, dtype, mode):
    ...  # Generate data by mode; conftest.py provides the default NPU device context.
    return x, weight, bias, scale, offset


@pytest.mark.parametrize("shape", [(1, 4096), (16, 7168), (1, 1), (7, 1025)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero", "offset_only", "quantized"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias

    x, weight, bias, scale, offset = generate_test_data(shape, dtype, mode)

    # Compare auto with torch_native by default; utilities read test configuration.
    executor = check_accuracy(
        rmsnorm_bias,
        args=(x, weight, bias, eps, scale, offset),
    )

    # An accuracy failure raises an exception, preventing performance collection.
    # Without a CLI override, profile only selected shapes in FP16 quantized mode.
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.float16
            and mode == "quantized"
        ),
    )
```

Test functions receive only parametrized operator input settings, not `impl`, `impl_ref`, a device, `tmp_path`, or `record_property`. Explicit CLI options `--enable-perf=true` and `--enable-perf=false` override the code condition to enable or disable performance collection for all cases. Without a CLI override, the condition above applies.

Accuracy is checked first by default; failures prevent benchmarking. When accuracy is explicitly skipped, the helper returns an executor without invoking either implementation or comparing outputs. It still validates the implementation under test but does not require an available reference implementation. Skipping is not a successful accuracy check: `executor.accuracy_checked` is False. If performance is also disabled, the case performs no operator computation.

### Accuracy Comparison

Tests compare two backends of the same public API instead of defining a separate reference function. The default pair is `auto` and `torch_native`; RMSNorm currently maps `auto` to Triton. When accuracy checks are enabled, both names must resolve to different registered entry points. For example, `auto` versus `triton` is rejected.

`check_accuracy(operator, args=(), *, kwargs=None, impl=None, impl_ref=None, standard=None, skip_acc_check=False)` accepts keyword-only operator arguments through `kwargs`. Tests import this function directly; neither the function nor implementation settings need to be fixture parameters.

Outputs may be tensors, tuples/lists, dictionaries, or supported Python scalars. Structures and values are compared recursively. Corresponding tensors must have the same shape, dtype, and device; outputs do not have to reside on an NPU.

| dtype | atol | rtol |
| --- | --- | --- |
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 5e-2 | 5e-2 |
| int8 / int32 / int64 | 0 | 0 |

Custom tolerances use the same mapping format as `DEFAULT_STANDARDS`. Unspecified dtypes fall back to their defaults:

```python
standard = {"torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4)}
executor = check_accuracy(operator, args=inputs, standard=standard)
```

Cross-backend comparison only checks whether results agree; each implementation must still follow the operator's functional definition. `check_accuracy` shares input objects between implementations. When it cannot cover a case, such as an in-place operator that modifies inputs without returning outputs, prepare independent inputs, call each implementation manually, compare the modified tensors or relevant results with `assert_accuracy`, and use `make_executor` to construct the callable for performance collection.

`make_executor` lives in `tests/python/utils/common.py` and is also used internally by `check_accuracy`. It binds the operator, inputs, and implementation and attaches `impl`, `impl_ref`, and `accuracy_checked` attributes. It does not execute the operator, validate registry entries, or copy or refresh inputs. Omitted implementation names use the current test configuration. `accuracy_checked` defaults to False; set it to True only after a successful manual accuracy comparison.

The following example assumes a public `operator` that modifies `x` in place and accepts the `impl` keyword. Data generation and parametrization follow the same conventions:

```python
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check, make_executor, require_distinct_impls, require_test_impl,
)
from tests.python.utils.performance import check_performance

executor = make_executor(operator, args=(x.clone(),))
if get_skip_acc_check():
    require_test_impl(operator, executor.impl)
else:
    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    x_test, x_ref = x.clone(), x.clone()
    operator(x_test, impl=executor.impl)
    operator(x_ref, impl=executor.impl_ref)
    assert_accuracy(x_test, x_ref)
    executor.accuracy_checked = True

check_performance(executor, enable_perf=True)
```

Use independent performance inputs so they do not inherit mutations from the accuracy check. If repeated execution changes the workload, numeric range, or state, the test author must provide reset logic and define whether its cost is included in profiling. Manual accuracy paths must also honor accuracy-skip configuration and generation skip/error policy.

### Configuration

`conftest.py` stores configuration in a per-test ContextVar, which shared utilities read. Fixtures restore the context on completion or failure. Boolean CLI options retain None when omitted, distinguishing an unspecified value from explicit False.

| Code argument | CLI option | Precedence and default |
| --- | --- | --- |
| `impl` | `--impl` | Explicit code value > CLI > `auto`. |
| `impl_ref` | `--impl-ref` | Explicit code value > CLI > `torch_native`. |
| `skip_acc_check` | `--skip-acc-check[=true\|false]` | CLI > code > False. |
| `enable_perf` | `--enable-perf[=true\|false]` | CLI > code > False. |
| `prof_dir` | `--prof-dir` | CLI > code > `profilings/` under the current working directory. |
| `csv_path` | `--csv-path` | CLI > code > None (no summary CSV). |
| Not passed to test functions | `--npu-device` | Optional visible logical device index; uses the current NPU if omitted. |

Either boolean flag without a value means True; `=false` overrides a True value in code. Configuring output paths does not enable performance collection.

```bash
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py \
  --impl=auto --impl-ref=torch_native --enable-perf \
  --prof-dir=results/profilings --csv-path=results/performance.csv -v -s -x

# Run performance collection without accuracy comparison:
python -m pytest tests/python/operators --skip-acc-check --enable-perf -v -s
```

### Performance Collection and Reporting

`check_performance` accepts an executor and handles compilation warmup, NPU synchronization, profiling, statistics, recording, and console output. Test functions do not receive `tmp_path` or `record_property`.

- Defaults are warmup=5 and active=5. Only the implementation under test is profiled, not the reference.
- Timing is the sum of device-task durations divided by the number of active iterations, not end-to-end latency. No performance pass/fail threshold is currently enforced.
- The summary CSV appends one record per case, including the operator, implementation, parameters, device, paths, and timing. Durations are in μs with three decimal places; returned values retain their original precision.
- Profiling artifacts are retained and not automatically cleaned up. Relative paths are resolved from the working directory where pytest runs.
- Only serial execution or a single worker is currently supported. There is no automatic allocation of idle devices, and concurrent writes to the same CSV from multiple processes are unsupported.

```text
results/profilings/prof_rmsnorm_bias/
└── 000001-float16-1e-05-normal-1x1024-auto-a1b2c3d4/
    └── ASCEND_PROFILER_OUTPUT/
```

Directory names follow `process-local sequence number-parameter values-impl-unique ID`. Parameter values are ordered by parameter name, excluding the internal `impl_pair`. Sorting by directory name preserves creation order within a run. After collection, a single `*_ascend_pt` host-directory layer is removed. If multiple directories or naming conflicts exist, the original structure is retained to avoid overwriting files.

## 4. Test Policy

| Condition | Behavior |
| --- | --- |
| The implementation under test does not support the current generation, and the reference resolves successfully | Skip and record the operator, impl, SoC, generation, and supported generations. |
| The implementation under test or reference implementation is not found | Raise an error. |
| The reference implementation does not support the current generation | Raise an error; this takes precedence over skipping the implementation under test. |
| Hardware detection, backend import, operator execution, or accuracy comparison fails | Raise an error; do not convert it to a skip. |

`require_distinct_impls` resolves both implementations before a normal accuracy comparison. With `skip_acc_check=True`, `require_test_impl` checks only the implementation under test and does not require an available reference. An unsupported generation skips the entire case, including performance collection. Use `pytest -rs` to display skip reasons. See `tests/python/utils/common.py` for the implementation.
