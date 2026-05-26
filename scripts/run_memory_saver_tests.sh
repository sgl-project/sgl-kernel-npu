#!/bin/bash

TEST_DIR="${GITHUB_WORKSPACE}/contrib/torch_memory_saver/test"
cd "$TEST_DIR" || { echo "Directory not found: $TEST_DIR"; exit 1; }

HOOK_MODE="${1:-torch}"

PASSED=()
FAILED=()

run_test() {
    local test_file="$1"
    echo "=========================================="
    echo "Running: $test_file (hook_mode=$HOOK_MODE)"
    echo "=========================================="
    if python3 "$test_file" "$HOOK_MODE"; then
        PASSED+=("$test_file")
        echo "PASSED: $test_file"
    else
        FAILED+=("$test_file")
        echo "FAILED: $test_file"
    fi
    echo ""
}

SINGLE_DEVICE_TESTS=(
    simple.py
    cpu_backup.py
    rl_example.py
)

MULTI_DEVICE_TESTS=(
    multi_device.py
)

echo "Running memory saver tests (hook_mode=$HOOK_MODE)"
echo ""

for test in "${SINGLE_DEVICE_TESTS[@]}"; do
    run_test "$test"
done

if [ "${2:-}" = "with-multi-device" ]; then
    echo "Running multi-device tests..."
    for test in "${MULTI_DEVICE_TESTS[@]}"; do
        run_test "$test"
    done
fi

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Passed: ${#PASSED[@]}"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED[@]}"; do
        echo "  - $test"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
