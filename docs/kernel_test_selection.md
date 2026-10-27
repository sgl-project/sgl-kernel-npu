# PR kernel test selection

`pr-test-kernels.yml` runs a CPU selector for every PR targeting main. A docs-only
PR completes this check without allocating an NPU. There is deliberately no
workflow-level path filter: GitHub can truncate changed-path lists on large PRs,
and a newly introduced build dependency must not bypass the selector.

## Change detection and execution

The selector compares `git merge-base <event base SHA> <event head SHA>` to the
**event head SHA**, with NUL-delimited `git diff --name-status --find-renames`.
This covers all commits in the PR, including additions, deletions and both names
of renames. The upstream `refs/pull/<number>/head` ref makes fork commits available
without assuming that the fork branch exists in upstream. The workflow uses
`pull_request`, not `pull_request_target`.

Tests execute against the normal checkout merge commit. Dependency edges are
collected from both the merge base and the checkout, so removing an import does
not hide its former consumers. Only test files present in the checkout enter the
plan. A test that disappears between planning and execution is an error.

The selection JSON is uploaded once and downloaded by selected jobs. Job Summary
and logs include changed paths, status/rename information, test files, reasons,
job matrices, skipped modules and existing test exclusions. Each selected
module retains its CANN 9.0.0/9.1.0 × A3 matrix; selection never shrinks a matrix.
`workflow_dispatch` explicitly selects the complete existing kernel suite.

## Mapping rules

`scripts/kernel_test_manifest.json` is the reviewable mapping:

- `operators`: a public operator name, owning CI module, exact tests and source
  paths relative to `python/sgl_kernel_npu/sgl_kernel_npu/`. List the public API,
  each backend and its reference implementation. Names need not match: for
  example `norm.fused_variance` uses Triton's `norm/rmsnorm_split.py`.
- `modules`: the existing enabled legacy test files and legacy source prefixes.
  Public `operators/<module>/`, `implementation/<backend>/<module>/` and
  `tests/python/operators/<module>/` ownership follows the CI module name.
- `disabled_legacy_tests` and `cann_exclusions`: pre-existing exclusions, with
  reasons. This change does not re-enable these tests or change accuracy criteria.

Static Python imports (including relative and function-local imports) extend the
explicit map transitively, without importing torch or any operator. Shared input
helpers and reference helpers select their actual consumers. Tests also depend on
ancestor `conftest.py` files when executed with pytest. Legacy script tests retain
the original script execution mode and working directory. Package initializers
are dependencies; their shared imports are followed, but mapped sibling operator
re-exports are not expanded into every operator. Changing an initializer broadens
to its consumers and owning module. Unresolved dynamic or wildcard imports and
Python parse errors conservatively expand the suite and explain why. The explicit
operator map is required for dynamic backend dispatch; it must be updated when
adding a public operator or a backend.

Examples:

| Change | Selection |
| --- | --- |
| New Triton/reference/public `norm/l1_norm.py` | Public `norm/test_l1_norm.py` |
| Legacy `norm/l1_norm.py` | Legacy `test_l1_norm.py` |
| A single public/legacy test | That file |
| Public norm `_qkv_inputs.py` | Four QKV/TP public tests, via transitive imports |
| Reference norm `_qkv_compute.py` | Three reference consumers |
| Legacy norm `split_qkv_rmsnorm_rope.py` | Its attention test, despite the source directory name |
| Registry or test framework helper | Public tests and any legacy pytest users of that framework |
| Unknown helper in a known module | Full owning module, with a fallback reason |
| Unknown impact outside known modules | Full existing suite, with a fallback reason |

An unmapped new public test in a known module is discovered automatically. An
unmapped changed test outside known public modules conservatively runs alongside
the full suite until assigned to a module. Existing disabled legacy tests remain excluded and are
reported explicitly. New modules must be assigned a CI job and manifest entry;
they cannot inherit hardware/backend support merely from their directory names.

## Global dependencies

Build/install/selection scripts, CMake files, shared C++ headers, wheel packaging,
submodules and unclassified configuration changes select all modules. Every job
builds and imports the same shared `libsgl_kernel_npu.so`, so these can break any
job even if the Python test does not call the changed native operator. In
particular the build also consumes headers under DeepEP; those paths are not
blindly excluded. Known individual native operator directories expand to their
module; unclassified native sources conservatively expand to all modules.

`prepare_kernel_tests.sh` keeps the existing build/install procedure. Importing
`sgl_kernel_npu` loads the shared library, so a pure Triton selection still needs
the wheel. Unselected modules do not allocate a runner, install dependencies or
build a wheel. This patch does not attempt partial native compilation.

## Workflow changes

The selector compares parsed YAML objects, not textual lines or the filename:

- A change only under `jobs.test-<module>-ops` selects that entire module. This
  includes its image, matrix, environment, steps and conditions.
- Changes to multiple such jobs select their union.
- Top-level triggers, permissions, concurrency, shared environment/defaults, the
  selection job, the finish job, or an unknown job select all modules.
- YAML anchors are resolved before comparison, so changing a shared value affects
  every job that uses it. Comments and formatting alone select no modules.
- An unreadable base workflow causes a full selection with a reason; an invalid
  current workflow or selector failure fails CI rather than reporting success.

The initial rollout of this selector changes global selection/execution logic and
therefore intentionally runs the full suite. Subsequent norm-only changes or
norm-job-only edits select norm. There are no PR-number or branch-name exceptions.

## Failures and reports

The runner preserves public accuracy options and legacy script/pytest semantics,
continues to the legacy checks after a public failure, and returns failure if any
selected check fails. Pytest uses `--maxfail=0` instead of `-x`, so assertion or
fixture failures do not suppress subsequent cases or parameter combinations.
`--continue-on-collection-errors` also runs valid collected tests when another
file cannot be collected; the collection error still fails CI. This applies to
both selected execution and the original positional group interface. Standalone
scripts continue to the next file after a nonzero exit; their internal control
flow is unchanged. Cancellation, process crashes and job timeouts can still
interrupt execution. Pytest writes JUnit XML; standalone scripts get one JUnit
case recording their process exit status. All seven jobs upload reports with
`always()`, including partial reports on failure. The finish job depends on both
the selector and test jobs, and fails on failure/cancellation. No test execution
uses `continue-on-error`.

The daily/manual positional group interface in `run_kernel_tests.sh` is retained.
A CPU regression check prevents its enabled legacy lists from drifting from the
manifest (the existing smoke test is included in the PR fused group).

## Local validation

```bash
python3 -m pip install PyYAML==6.0.2 pytest==9.1.1
python3 -m unittest discover -s tests/ci -v
python3 scripts/select_kernel_tests.py --base upstream/main --head HEAD
python3 scripts/select_kernel_tests.py --full --output /tmp/kernel-tests.json
```

The CPU tests are outside `tests/python` to avoid loading NPU conftest fixtures.
They cover mapping and shared dependencies, workflow scoping, full PR history,
upstream divergence, deletion/rename, matrix retention, failure propagation and
report creation. Add mapping regressions alongside new operator/backend entries.
