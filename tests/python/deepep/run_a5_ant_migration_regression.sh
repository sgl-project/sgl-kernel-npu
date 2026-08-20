#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_intranode.py"

NUM_PROCESSES="${A5_ANT_NUM_PROCESSES:-4}"
HIDDEN="${A5_ANT_HIDDEN:-7168}"
NUM_EXPERTS="${A5_ANT_NUM_EXPERTS:-256}"
HCCL_BUFFER_MB="${HCCL_BUFFSIZE:-2300}"
LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}"

RUN_MAX_BOUNDARY="${A5_ANT_RUN_MAX_BOUNDARY:-0}"
RUN_ALL_QUANT="${A5_ANT_RUN_ALL_QUANT:-0}"
RUN_EXPERT_MATRIX="${A5_ANT_RUN_EXPERT_MATRIX:-0}"
SKIP_DEVICE_CHECK="${A5_ANT_SKIP_DEVICE_CHECK:-0}"

require_positive_integer()
{
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "${name} must be a positive integer, got '${value}'." >&2
        exit 1
    fi
}

require_boolean()
{
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1, got '${value}'." >&2
        exit 1
    fi
}

require_positive_integer "A5_ANT_NUM_PROCESSES" "${NUM_PROCESSES}"
require_positive_integer "A5_ANT_HIDDEN" "${HIDDEN}"
require_positive_integer "A5_ANT_NUM_EXPERTS" "${NUM_EXPERTS}"
require_positive_integer "HCCL_BUFFSIZE" "${HCCL_BUFFER_MB}"
require_boolean "ASCEND_LAUNCH_BLOCKING" "${LAUNCH_BLOCKING}"
require_boolean "A5_ANT_RUN_MAX_BOUNDARY" "${RUN_MAX_BOUNDARY}"
require_boolean "A5_ANT_RUN_ALL_QUANT" "${RUN_ALL_QUANT}"
require_boolean "A5_ANT_RUN_EXPERT_MATRIX" "${RUN_EXPERT_MATRIX}"
require_boolean "A5_ANT_SKIP_DEVICE_CHECK" "${SKIP_DEVICE_CHECK}"

if (( NUM_PROCESSES < 2 )); then
    echo "A5_ANT_NUM_PROCESSES must be at least 2." >&2
    exit 1
fi

if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "Cannot find test script: ${TEST_SCRIPT}" >&2
    exit 1
fi

if [[ "${SKIP_DEVICE_CHECK}" != "1" ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "npu-smi is required to verify the A5 test target." >&2
        echo "Set A5_ANT_SKIP_DEVICE_CHECK=1 only when the target is already known to be A5." >&2
        exit 1
    fi

    BOARD_INFO="$(npu-smi info -t board -i 0 2>/dev/null || true)"
    if ! printf '%s\n' "${BOARD_INFO}" | grep -Eiq 'Chip Name[[:space:]]*:[[:space:]]*Ascend[[:space:]_-]*950'; then
        echo "This regression must run on Ascend950 (A5)." >&2
        echo "Detected board information:" >&2
        printf '%s\n' "${BOARD_INFO}" >&2
        exit 1
    fi
fi

run_case()
{
    local name="$1"
    local rounds="$2"
    local per_round_tokens="$3"
    local num_tokens="$4"
    local quant_type="$5"
    local combine_long_seq="$6"
    local hidden="${7:-${HIDDEN}}"
    local num_experts="${8:-${NUM_EXPERTS}}"
    local num_topk="${9:-8}"

    require_positive_integer "${name}: rounds" "${rounds}"
    require_positive_integer "${name}: per_round_tokens" "${per_round_tokens}"
    require_positive_integer "${name}: num_tokens" "${num_tokens}"
    require_positive_integer "${name}: hidden" "${hidden}"
    require_positive_integer "${name}: num_experts" "${num_experts}"
    require_positive_integer "${name}: num_topk" "${num_topk}"
    require_boolean "${name}: combine_long_seq" "${combine_long_seq}"

    if (( rounds > 256 )); then
        echo "${name}: rounds=${rounds} exceeds the supported maximum of 256." >&2
        exit 1
    fi
    if (( per_round_tokens < 32 || per_round_tokens > 8192 )); then
        echo "${name}: per_round_tokens must be in [32, 8192]." >&2
        exit 1
    fi
    if (( rounds * per_round_tokens > 131072 )); then
        echo "${name}: rounds * per_round_tokens exceeds 131072." >&2
        exit 1
    fi
    if (( num_tokens > rounds * per_round_tokens )); then
        echo "${name}: ${num_tokens} tokens cannot fit in ${rounds} rounds." >&2
        exit 1
    fi
    if (( num_experts % NUM_PROCESSES != 0 )); then
        echo "${name}: num_experts=${num_experts} must be divisible by num_processes=${NUM_PROCESSES}." >&2
        exit 1
    fi

    local actual_rounds=$(( (num_tokens + per_round_tokens - 1) / per_round_tokens ))
    echo
    echo "===== ${name} ====="
    echo "configured_rounds=${rounds}, actual_rounds=${actual_rounds}, per_round_tokens=${per_round_tokens}"
    echo "tokens=${num_tokens}, hidden=${hidden}, experts=${num_experts}, topk=${num_topk}, quant=${quant_type}, combine_long_seq=${combine_long_seq}"

    ASCEND_LAUNCH_BLOCKING="${LAUNCH_BLOCKING}" \
    DEEPEP_NORMAL_LONG_SEQ_ROUND="${rounds}" \
    DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${per_round_tokens}" \
    DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ="${combine_long_seq}" \
    HCCL_BUFFSIZE="${HCCL_BUFFER_MB}" \
        python3 "${TEST_SCRIPT}" \
            --num-processes="${NUM_PROCESSES}" \
            --num-tokens="${num_tokens}" \
            --hidden="${hidden}" \
            --num-experts="${num_experts}" \
            --num-topk="${num_topk}" \
            --quant-type="${quant_type}"
}

# Legacy single-round path.
run_case "single-round-regression" 1 8192 4096 bf16 0

# Five configured dispatch rounds with an empty padding round.
run_case "multi-round-padding" 5 512 2048 bf16 1

# Five full dispatch rounds.
run_case "multi-round-exact" 5 512 2560 bf16 1

# Four full dispatch rounds and a 74-token tail.
run_case "multi-round-partial-tail" 5 512 2122 bf16 1

# Notify's largest single UB round batch.
run_case "notify-32-round-boundary" 32 32 1024 bf16 1 1024

# Cross the 32-round UB batch boundary and process a partial final round.
run_case "notify-33-round-cross-batch" 33 32 1040 bf16 1 1024

# Cover Notify's smaller expert-matrix batch modes introduced by A5 migration.
# With four ranks these select batchRounds 16 and 8 respectively.
run_case "notify-batch-16" 17 128 2176 bf16 1 1024 512
run_case "notify-batch-8" 9 256 2304 bf16 1 1024 1024

if [[ "${RUN_MAX_BOUNDARY}" == "1" ]]; then
    # Maximum supported runtime round count.
    run_case "max-256-round-boundary" 256 32 8192 bf16 1 1024

    # 8192 * topk(8) * 32 bytes = 2 MiB, exactly one combine state ping-pong slot.
    run_case "max-combine-state-slot" 2 8192 8193 bf16 1 1024 256 8
fi

if [[ "${RUN_ALL_QUANT}" == "1" ]]; then
    for quant_type in int8 pertoken_fp8_e4m3 mx_fp8_e4m3 mx_fp8_e5m2 mx_fp4_e2m1; do
        run_case "multi-round-${quant_type}" 5 512 2122 "${quant_type}" 1
    done
fi

if [[ "${RUN_EXPERT_MATRIX}" == "1" ]]; then
    # With four ranks these select Notify batch sizes 32, 16, and 8.
    # The 16/8 cases are part of the default regression because they cover
    # the new expert-dependent batching paths in NotifyDispatchA5.
    run_case "notify-batch-32" 33 64 2112 bf16 1 1024 256
fi

echo
echo "All A5 ant-migration regression cases passed."
