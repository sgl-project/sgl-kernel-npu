# Public-interface tests

```text
operators/             # Mirror exported operators; test public entry points only
  norm/test_rmsnorm_bias.py
functional/            # Reserved for functional/modules tests
utils/
  accuracy.py          # Recursive assertions and check_accuracy
  standards.py         # Explicit accuracy thresholds
  performance.py       # NPU profiling and per-case CSV output
  common.py            # Shared impl-selection checks
  device.py            # Scoped NPU defaults and serial execution policy
```

Legacy tests and their original utils.py remain under sgl_kernel_npu/ unchanged.
New tests import helpers through tests.python.utils to avoid legacy utils imports.
The shared conftest.py applies automatic device and reporting setup only to
operators/ and functional/. New tests must not import implementations directly.

## RMSNorm example

```bash
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py -v -s -x

# Override either side without rebuilding.
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py -v -s -x \
  --impl=triton --impl-ref=torch_native
```

Requires a built/installed sgl_kernel_npu, torch_npu, Triton-Ascend and an NPU.
No NPU means skip; a broken installed operator or profiler fails the test.
Default comparison: auto (currently Triton) against torch_native. The two pytest
options select one pair per run, included in case IDs and performance reports.
Only --impl is benchmarked; --impl-ref provides the
comparison output. Unknown/unavailable implementations fail without fallback;
aliases of the same registered entry (such as auto vs triton) are rejected.
The identity check cannot detect a future dynamic dispatcher selecting the same
backend internally; that will require resolved-backend metadata.
Cases cover FP32/FP16/BF16, multiple shapes, eps=1e-6/1e-5, zero input, ignored
offset and int8 quantization.

`generate_test_data` creates every input tensor directly on NPU. Each case compares
the selected backends through the public interface. No standalone reference is used.
Performance is disabled when neither CLI nor code enables it. When enabled, collection runs **only after
accuracy passes**. Use `-x` to stop the
entire run on the first failure. Shared atol/rtol values in `utils/standards.py`
are both 1e-5 for FP32, 1e-3 for FP16 and 5e-2 for BF16; int8 comparison is exact.
INT8/INT32/INT64 defaults are explicitly listed with atol=rtol=0.
Custom `standard` uses the same dictionary format, not a single threshold object:

```python
from tests.python.utils.standards import AccuracyStandard

standard = {
    "torch.float32": AccuracyStandard(atol=1e-4, rtol=1e-4),
    "torch.float16": AccuracyStandard(atol=2e-3, rtol=2e-3),
    "torch.int32": AccuracyStandard(atol=0, rtol=0),
}
executor = check_accuracy(operator, args=inputs, standard=standard)
```

Overrides apply per output Tensor dtype, including nested outputs; missing keys
retain defaults without mutating DEFAULT_STANDARDS. Python float/complex values
use the float32 entry; Python integers and other discrete scalars stay exact.
Backend agreement is a consistency check, not an independent correctness proof.
`check_accuracy(operator, args, kwargs=..., impl=None, impl_ref=None)`
checks registration, executes both public calls, validates nested outputs and
returns a zero-argument executor only on success. Data generation and
`check_performance(executor, ...)` remain explicit in each test. Inputs must not be
mutated by the operator; in-place/stateful APIs need a separate input-reset design
before using this helper. Accuracy checks only require corresponding output
Tensors to be on the same device; they do not enforce an NPU output.
Tests import check_accuracy directly; neither it nor impl/impl_ref is a test
parameter. An autouse fixture in conftest.py scopes the CLI settings
to each test using a ContextVar and restores them on teardown. check_accuracy
reads those settings for each omitted name; explicitly supplied impl/impl_ref
override the respective side. Its returned executor exposes impl/impl_ref for
reporting without re-reading configuration in the test.
Public interfaces use `@register_operator("domain.operator")` to associate the
function object with its operator name. get_impl(operator, impl) and
list_impls(operator) resolve that name internally; get_impl_by_name and
list_impls_by_name retain the string-based APIs. Tests do not repeat the name.
Unknown functions fail explicitly.
register_impl still registers backend entry points independently. The public
registration decorator returns the original function, without a wrapper.
Quantized inputs deliberately
avoid rounding boundaries while testing truncation and saturation.

Profiling follows the NPU profiler approach in the supplied apply_attn_res.py:
compile first, 5 warmup calls, 5 active calls, explicit synchronization, CSV export.
The metric is summed device task duration / active calls, including all kernels
of a composite impl. It is not end-to-end latency; no speed threshold
is imposed yet. Only one export schema is summed (summary, otherwise statistic).

Performance tests require no tmp_path or record_property argument:

```python
executor = check_accuracy(operator, args=inputs)
check_performance(executor)  # Follow --enable-perf; otherwise disabled.
# Per-case defaults, used only when the CLI switch is omitted:
check_performance(executor, enable_perf=True)
check_performance(executor, enable_perf=False)
# Alternatively, explicitly request persistent CSV output:
check_performance(executor, enable_perf=True, csv_path="results/performance.csv")
# Or select the profiling root explicitly (independent of summary CSV output):
check_performance(executor, enable_perf=True, prof_dir="results/profiles")
```

Use one of these performance calls. Priority: explicit pytest --enable-perf >
code enable_perf > False. CLI --enable-perf (or --enable-perf=true) enables all
cases; --enable-perf=false disables all cases, overriding per-shape settings.
Omitting the CLI switch preserves each case's code setting.
A disabled call returns None, does not run the executor,
does not record or print performance results, and creates no profiler/CSV files.
Supplying csv_path or prof_dir alone does not enable performance collection.
Paths can also be configured with --prof-dir and --csv-path. Each explicitly
provided CLI path overrides its code argument; otherwise the code argument is
used. Without either setting, profiling uses cwd/profilings and no summary CSV
is written. Relative paths are resolved from the pytest working directory.

```bash
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py --enable-perf -v -s -x
# Disable performance for every case, even those enabled in code:
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py --enable-perf=false -v -s -x
# Configure output paths without changing the test:
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py --enable-perf \
  --prof-dir=results/profilings --csv-path=results/performance.csv -v -s -x
```

When enabled, profiler files are retained under
`<current working directory>/profilings/prof_<function name>/<sequence>-<values>-<impl>-<unique-id>/`.
The six-digit sequence starts at 000001 per process and preserves creation order
when sorted by name within one run. Parameter values follow parameter-name order,
without labels (for example, `000001-float16-1e-05-normal-1x1024-auto-a1b2c3d4`).
After profiling completes, the single automatically generated `*_ascend_pt`
host directory is flattened, so `ASCEND_PROFILER_OUTPUT` sits directly inside
the case directory. All profiler contents are retained. Unexpected multiple
host directories or name collisions are left intact to avoid overwriting data.
Passing `prof_dir` replaces the `profilings` root, while retaining the function
and run subdirectories. Relative paths resolve against the current working directory.
Each call uses a fresh subdirectory to avoid mixing old
measurements. Files are not automatically removed, including on failure.
The profile path is printed before collection and included in successful reports.
csv_path=None writes no summary CSV; a supplied path appends one row per case
with a single header. Profiler files are saved regardless of csv_path.
CSV device task durations are formatted to three decimal places (microseconds);
the returned result retains the original numeric precision.
Mismatched existing CSV columns raise an error instead of overwriting data.
An autouse fixture supplies case IDs, parameters and pytest property recording.
The helper returns the result dictionary and prints a concise summary (`-s`).
Do not write the same CSV concurrently from separate pytest processes.
Do not run performance tests concurrently on the same NPU.
The accuracy-helper separation and dtype-specific standards follow the mojo test
organization, without importing mojo or inheriting its global device settings.

## Skipping accuracy checks

`check_accuracy(..., skip_acc_check=True)` returns the executor without running
either implementation or comparing outputs. The selected impl is validated;
the reference impl is not required. The default is False (perform accuracy checks).
An explicit CLI setting overrides the code setting in either direction:

```bash
# Profile without running the accuracy comparison:
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py --skip-acc-check --enable-perf -v -s
# Force accuracy checks even when code requests skipping:
python -m pytest tests/python/operators/norm/test_rmsnorm_bias.py --skip-acc-check=false -v -s
```

Skipping does not enable profiling. If profiling is also disabled, neither
implementation executes. A skipped check is logged, not treated as accuracy
validation; `executor.accuracy_checked` is False. Otherwise accuracy must pass
before profiling can start. Data generation and NPU setup still occur.

## Device selection

New operator tests automatically set the default device to `"npu"`, preserving
the process's current NPU selection. No device argument is needed in tests,
data generators or accuracy checks. Use `--npu-device=1` to explicitly select
visible logical device 1. Invalid indices fail; a missing NPU skips the tests.
The default device and any explicit selection are restored after each case,
including failures, so legacy and CPU helper tests are not affected.
Multiple xdist workers are rejected; use serial pytest or `-n 1`. There is no
round-robin allocation or automatic selection of an idle card. Separate pytest
processes still need external coordination to avoid sharing a busy NPU.
