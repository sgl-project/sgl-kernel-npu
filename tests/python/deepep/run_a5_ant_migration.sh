#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_intranode.py"
NUM_PROCESSES="${A5_ANT_NUM_PROCESSES:-4}"
HIDDEN="${A5_ANT_HIDDEN:-7168}"
NUM_EXPERTS="${A5_ANT_NUM_EXPERTS:-256}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2300}"
RUN_ALL_QUANT="${A5_ANT_RUN_ALL_QUANT:-0}"
RUN_BOUNDARY="${A5_ANT_RUN_BOUNDARY:-1}"
RUN_EXPERT_MATRIX="${A5_ANT_RUN_EXPERT_MATRIX:-0}"

if ! [[ "${NUM_PROCESSES}" =~ ^[0-9]+$ ]] || (( NUM_PROCESSES < 2 )); then
    echo "A5_ANT_NUM_PROCESSES must be an integer greater than or equal to 2." >&2
    exit 1
fi

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi is required to verify the A5 test target." >&2
    exit 1
fi

BOARD_INFO="$(npu-smi info -t board -i 0 2>/dev/null || true)"
if ! printf '%s\n' "${BOARD_INFO}" | grep -Eiq 'Chip Name[[:space:]]*:[[:space:]]*Ascend950'; then
    echo "This regression must run on Ascend950 (A5)." >&2
    exit 1
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

    if (( num_experts % NUM_PROCESSES != 0 )); then
        echo "${name}: num_experts=${num_experts} must be divisible by num_processes=${NUM_PROCESSES}." >&2
        exit 1
    fi

    local actual_rounds=$(( (num_tokens + per_round_tokens - 1) / per_round_tokens ))
    echo "===== ${name}: configured_rounds=${rounds}, actual_rounds=${actual_rounds}, per_round_tokens=${per_round_tokens}, tokens=${num_tokens}, hidden=${hidden}, experts=${num_experts}, topk=${num_topk}, quant=${quant_type}"
    DEEPEP_NORMAL_LONG_SEQ_ROUND="${rounds}" \
    DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${per_round_tokens}" \
    DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ="${combine_long_seq}" \
    HCCL_BUFFSIZE="${HCCL_BUFFSIZE}" \
        python3 "${TEST_SCRIPT}" \
            --num-processes="${NUM_PROCESSES}" \
            --num-tokens="${num_tokens}" \
            --hidden="${hidden}" \
            --num-experts="${num_experts}" \
            --num-topk="${num_topk}" \
            --quant-type="${quant_type}"
}

# 1. Single-round regression keeps the legacy A5 path covered.
# 2. 2048 is an exact four-round payload with one configured padding round.
# 3. 2122 enters the fifth round and verifies the partial-tail path.
# 4. 33 rounds crosses Notify's largest 32-round UB batch.
# 5. The boundary case executes all 256 Dispatch/Combine rounds.
run_case "single-round-regression" 1 8192 4096 bf16 0
run_case "multi-round-exact" 5 512 2048 bf16 1
run_case "multi-round-partial-tail" 5 512 2122 bf16 1
run_case "multi-batch-rounds" 33 64 2112 bf16 1 1024

if [[ "${RUN_BOUNDARY}" == "1" ]]; then
    run_case "max-runtime-round-boundary" 256 32 8192 bf16 1 1024
    # Fill the complete 4 MiB state slot in round 0, then switch to the
    # second slot. This catches overlap between state ping-pong slots and data.
    run_case "max-combine-state-slot" 2 8192 8193 bf16 1 1024 256 16
fi

if [[ "${RUN_ALL_QUANT}" == "1" ]]; then
    for quant_type in int8 pertoken_fp8_e4m3 mx_fp8_e4m3 mx_fp8_e5m2 mx_fp4_e2m1; do
        run_case "multi-round-${quant_type}" 5 512 2122 "${quant_type}" 1
    done
fi

if [[ "${RUN_EXPERT_MATRIX}" == "1" ]]; then
    # These expert counts select Notify batchRounds 32, 16, and 8 respectively
    # for the default four-rank topology.
    run_case "notify-batch-32" 33 64 2112 bf16 1 1024 256
    run_case "notify-batch-16" 17 128 2176 bf16 1 1024 512
    run_case "notify-batch-8" 9 256 2304 bf16 1 1024 1024
fi

echo "All A5 ant-migration regression cases passed."
