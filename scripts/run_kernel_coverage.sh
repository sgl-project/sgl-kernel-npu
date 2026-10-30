#!/usr/bin/env bash
# Collect per-test-file Python coverage for sgl-kernel-npu.
#
# Output:
#   coverage-out/test/<test-file>.py/covdata/coverage
#
# This script reuses the existing test list and execution logic in
# scripts/run_kernel_tests.sh. It only wraps python3 test invocations with
# coverage.py; it does not build test_case_map.json or upload anything.

set -uo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/coverage-out}"
TEST_ROOT="${REPO_ROOT}/tests/python/sgl_kernel_npu"
REAL_PYTHON="$(command -v python3 || true)"
WRAPPER_DIR="${OUT_DIR}/bin"
CONFIG_FILE="${OUT_DIR}/.coveragerc"

if [[ -z "${REAL_PYTHON}" ]]; then
    echo "ERROR: python3 not found" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/scripts/run_kernel_tests.sh" ]]; then
    echo "ERROR: scripts/run_kernel_tests.sh not found" >&2
    exit 2
fi

if [[ ! -d "${TEST_ROOT}" ]]; then
    echo "ERROR: Python test directory not found: ${TEST_ROOT}" >&2
    exit 2
fi

if [[ "${OUT_DIR}" == "${REPO_ROOT}" || "${OUT_DIR}" == "/" ]]; then
    echo "ERROR: unsafe output directory: ${OUT_DIR}" >&2
    exit 2
fi

# Start with a clean output directory for this sampling run.
rm -rf "${OUT_DIR}"
mkdir -p "${WRAPPER_DIR}" "${OUT_DIR}/test"

# Coverage.py is the only additional dependency required by this script.
"${REAL_PYTHON}" -m pip install coverage

# Collect only the project's Python package, not the test files themselves.
cat > "${CONFIG_FILE}" <<'RC'
[run]
branch = True
source =
    sgl_kernel_npu
RC

export KERNEL_COVERAGE_REPO_ROOT="${REPO_ROOT}"
export KERNEL_COVERAGE_OUT_DIR="${OUT_DIR}"
export KERNEL_COVERAGE_TEST_ROOT="${TEST_ROOT}"
export KERNEL_COVERAGE_CONFIG="${CONFIG_FILE}"
export KERNEL_COVERAGE_REAL_PYTHON="${REAL_PYTHON}"

# Intercept python3 commands that execute a test file under TEST_ROOT.
# Other Python commands pass through unchanged.
cat > "${WRAPPER_DIR}/python3" <<'WRAPPER'
#!/usr/bin/env bash
set -uo pipefail

REAL_PYTHON="${KERNEL_COVERAGE_REAL_PYTHON:?}"
REPO_ROOT="${KERNEL_COVERAGE_REPO_ROOT:?}"
OUT_DIR="${KERNEL_COVERAGE_OUT_DIR:?}"
TEST_ROOT="${KERNEL_COVERAGE_TEST_ROOT:?}"
CONFIG="${KERNEL_COVERAGE_CONFIG:?}"

args=("$@")
test_index=-1

for ((i=0; i<${#args[@]}; i++)); do
    arg="${args[$i]}"
    if [[ "${arg}" == *.py && -f "${arg}" ]]; then
        test_index=$i
        break
    fi
done

# Not a Python-file invocation (for example, a setup/helper command).
if ((test_index < 0)); then
    exec "${REAL_PYTHON}" "${args[@]}"
fi

test_path="${args[$test_index]}"
if [[ "${test_path}" = /* ]]; then
    absolute_test="${test_path}"
else
    absolute_test="$(cd "$(dirname "${test_path}")" && pwd)/$(basename "${test_path}")"
fi

# Only collect tests under tests/python/sgl_kernel_npu.
case "${absolute_test}" in
    "${TEST_ROOT}"/*) ;;
    *) exec "${REAL_PYTHON}" "${args[@]}" ;;
esac

relative_test="${absolute_test#"${TEST_ROOT}"/}"
case_dir="${OUT_DIR}/test/${relative_test}/covdata"
mkdir -p "${case_dir}"
data_file="${case_dir}/coverage"

# Preserve the original invocation style:
#   python3 test_file.py
#   python3 -m pytest test_file.py
if [[ "${args[0]}" == "-m" && "${args[1]:-}" == "pytest" ]]; then
    pytest_args=("${args[@]:2}")
    exec "${REAL_PYTHON}" -m coverage run \
        --rcfile="${CONFIG}" \
        --data-file="${data_file}" \
        -m pytest "${pytest_args[@]}"
else
    exec "${REAL_PYTHON}" -m coverage run \
        --rcfile="${CONFIG}" \
        --data-file="${data_file}" \
        "${args[@]}"
fi
WRAPPER
chmod +x "${WRAPPER_DIR}/python3"

export PATH="${WRAPPER_DIR}:${PATH}"

echo "Coverage output: ${OUT_DIR}/test/<test-file>/covdata/coverage"
echo "Test group: ${TEST_GROUP:-all}"

set +e
(
    cd "${REPO_ROOT}"
    bash scripts/run_kernel_tests.sh "${TEST_GROUP:-all}"
)
TEST_STATUS=$?
set -e

# Do not hide test failures behind a successful coverage step.
exit "${TEST_STATUS}"
